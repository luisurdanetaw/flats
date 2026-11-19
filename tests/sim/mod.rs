//! A seeded workload simulator with a reference model: the oracle for
//! differential testing (`tests/differential.rs`) and crash testing
//! (`tests/crash_kill.rs`).
//!
//! Flats does EXACT search, so every statement has one right answer, and a
//! few hundred lines of obviously-correct Rust can compute it: a map of rows
//! per collection, predicates evaluated row by row, scores as f64 dot
//! products. The simulator generates a random workload of structured [`Op`]s,
//! the model predicts each one's outcome (including every error the binder
//! must raise), the op is rendered to V-SQL text, with each value inline or
//! as a `?` parameter, and the database's answer is compared with the
//! model's.
//!
//! Ops are self-contained (they name collections and columns, not positions
//! in the model), so a failing workload can be SHRUNK by deleting ops and
//! re-running: what remains is still a valid program, and some ops just turn
//! into expected errors.
//!
//! Every comparison is exact except scores, which the engine computes in f32
//! with SIMD and the model in f64; see [`check_search`].

#![allow(dead_code)] // each test binary uses a different slice of this module

use std::collections::{BTreeMap, BTreeSet};
use std::fmt::Write as _;
use std::path::{Path, PathBuf};
use std::time::Duration;

use flats::{CompactionOptions, Db, DbOptions, Param, RegValue};

// ---------------------------------------------------------------------------
// Randomness
// ---------------------------------------------------------------------------

/// xorshift64*: tiny, deterministic, good enough for workloads. Every failure
/// names its seed, and the seed replays it exactly.
#[derive(Clone)]
pub struct Rng(u64);

impl Rng {
    pub fn new(seed: u64) -> Rng {
        Rng(seed.wrapping_mul(0x9E37_79B9_7F4A_7C15) ^ 0xD1B5_4A32_D192_ED03 | 1)
    }
    pub fn next(&mut self) -> u64 {
        self.0 ^= self.0 >> 12;
        self.0 ^= self.0 << 25;
        self.0 ^= self.0 >> 27;
        self.0.wrapping_mul(0x2545_F491_4F6C_DD1D)
    }
    pub fn below(&mut self, n: u64) -> u64 {
        if n == 0 { 0 } else { self.next() % n }
    }
    pub fn range(&mut self, lo: i64, hi: i64) -> i64 {
        lo + self.below((hi - lo + 1) as u64) as i64
    }
    /// True with probability `pct`%.
    pub fn chance(&mut self, pct: u64) -> bool {
        self.below(100) < pct
    }
    pub fn pick<'a, T>(&mut self, xs: &'a [T]) -> &'a T {
        &xs[self.below(xs.len() as u64) as usize]
    }
    pub fn shuffle<T>(&mut self, xs: &mut [T]) {
        for i in (1..xs.len()).rev() {
            let j = self.below(i as u64 + 1) as usize;
            xs.swap(i, j);
        }
    }
}

/// An env var parsed as a number, or `default`.
pub fn env_u64(name: &str, default: u64) -> u64 {
    std::env::var(name)
        .ok()
        .and_then(|s| s.parse().ok())
        .unwrap_or(default)
}

// ---------------------------------------------------------------------------
// Schemas, values, rows
// ---------------------------------------------------------------------------

#[derive(Clone, Copy, Debug, PartialEq)]
pub enum Ty {
    Int,
    Float,
    Text,
    Vector(usize),
}

impl Ty {
    fn sql(self) -> String {
        match self {
            Ty::Int => "INT".into(),
            Ty::Float => "FLOAT".into(),
            Ty::Text => "TEXT".into(),
            Ty::Vector(d) => format!("VECTOR({d})"),
        }
    }
}

#[derive(Clone, Debug, PartialEq)]
pub struct Col {
    pub name: String,
    pub ty: Ty,
    pub pk: bool,
}

#[derive(Clone, Debug, PartialEq)]
pub struct SchemaDef {
    pub name: String,
    /// Declaration order, vector column included.
    pub cols: Vec<Col>,
    pub capacity: u64,
}

impl SchemaDef {
    fn col(&self, name: &str) -> Option<&Col> {
        self.cols.iter().find(|c| c.name == name)
    }
    fn idx(&self, name: &str) -> Option<usize> {
        self.cols.iter().position(|c| c.name == name)
    }
    fn pk(&self) -> usize {
        self.cols
            .iter()
            .position(|c| c.pk)
            .expect("valid schema has a key")
    }
    fn vector(&self) -> usize {
        self.cols
            .iter()
            .position(|c| matches!(c.ty, Ty::Vector(_)))
            .expect("valid schema has a vector")
    }
    fn dim(&self) -> usize {
        match self.cols[self.vector()].ty {
            Ty::Vector(d) => d,
            _ => unreachable!(),
        }
    }
    fn scalars(&self) -> impl Iterator<Item = (usize, &Col)> {
        self.cols
            .iter()
            .enumerate()
            .filter(|(_, c)| !matches!(c.ty, Ty::Vector(_)))
    }
    /// What the binder accepts for CREATE.
    fn valid(&self) -> bool {
        let vectors = self
            .cols
            .iter()
            .filter(|c| matches!(c.ty, Ty::Vector(_)))
            .count();
        let pks: Vec<&Col> = self.cols.iter().filter(|c| c.pk).collect();
        let names: BTreeSet<&str> = self.cols.iter().map(|c| c.name.as_str()).collect();
        vectors == 1
            && pks.len() == 1
            && matches!(pks[0].ty, Ty::Int | Ty::Text)
            && names.len() == self.cols.len()
            && self.capacity > 0
            && self.cols.iter().all(|c| !matches!(c.ty, Ty::Vector(0)))
    }
}

/// A value as a statement carries it. `BadFloat` is a non-finite float, only
/// expressible as a parameter, and always refused.
#[derive(Clone, Debug)]
pub enum Lit {
    Int(i64),
    Float(f64),
    Text(String),
    Vector(Vec<f32>),
    BadFloat(f64),
}

impl Lit {
    /// Exact equality, -0.0 and 0.0 distinguished: a stored value must come
    /// back bit for bit.
    fn same(&self, other: &Lit) -> bool {
        match (self, other) {
            (Lit::Int(a), Lit::Int(b)) => a == b,
            (Lit::Float(a), Lit::Float(b)) => a.to_bits() == b.to_bits(),
            (Lit::Text(a), Lit::Text(b)) => a == b,
            (Lit::Vector(a), Lit::Vector(b)) => {
                a.len() == b.len() && a.iter().zip(b).all(|(x, y)| x.to_bits() == y.to_bits())
            }
            _ => false,
        }
    }
    /// A total, printable rendering for sorting result multisets.
    fn canon(&self) -> String {
        match self {
            Lit::Int(i) => format!("i{i}"),
            Lit::Float(f) => format!("f{:016x}", f.to_bits()),
            Lit::Text(s) => format!("t{s:?}"),
            Lit::Vector(v) => {
                let bits: Vec<String> = v.iter().map(|x| format!("{:08x}", x.to_bits())).collect();
                format!("v[{}]", bits.join(","))
            }
            Lit::BadFloat(f) => format!("bad{f}"),
        }
    }
}

#[derive(Clone, Debug, PartialEq, Eq, PartialOrd, Ord)]
pub enum Key {
    Int(i64),
    Text(String),
}

fn key_of(lit: &Lit) -> Key {
    match lit {
        Lit::Int(i) => Key::Int(*i),
        Lit::Text(s) => Key::Text(s.clone()),
        other => panic!("not a key value: {other:?}"),
    }
}

/// A stored row: one value per declared column, vector included.
pub type ModelRow = Vec<Lit>;

#[derive(Clone, Debug)]
pub struct Coll {
    pub schema: SchemaDef,
    pub rows: BTreeMap<Key, ModelRow>,
}

/// The reference model: every collection and every row.
#[derive(Clone, Debug, Default)]
pub struct Model {
    pub colls: BTreeMap<String, Coll>,
}

// ---------------------------------------------------------------------------
// Predicates
// ---------------------------------------------------------------------------

#[derive(Clone, Copy, Debug, PartialEq)]
pub enum CmpOp {
    Eq,
    Ne,
    Lt,
    Le,
    Gt,
    Ge,
}

impl CmpOp {
    const ALL: [CmpOp; 6] = [
        CmpOp::Eq,
        CmpOp::Ne,
        CmpOp::Lt,
        CmpOp::Le,
        CmpOp::Gt,
        CmpOp::Ge,
    ];
    fn sql(self) -> &'static str {
        match self {
            CmpOp::Eq => "=",
            CmpOp::Ne => "!=",
            CmpOp::Lt => "<",
            CmpOp::Le => "<=",
            CmpOp::Gt => ">",
            CmpOp::Ge => ">=",
        }
    }
    /// `lit OP col` means `col FLIPPED lit`.
    fn flipped(self) -> CmpOp {
        match self {
            CmpOp::Lt => CmpOp::Gt,
            CmpOp::Le => CmpOp::Ge,
            CmpOp::Gt => CmpOp::Lt,
            CmpOp::Ge => CmpOp::Le,
            other => other,
        }
    }
    fn holds(self, ord: std::cmp::Ordering) -> bool {
        use std::cmp::Ordering::*;
        match self {
            CmpOp::Eq => ord == Equal,
            CmpOp::Ne => ord != Equal,
            CmpOp::Lt => ord == Less,
            CmpOp::Le => ord != Greater,
            CmpOp::Gt => ord == Greater,
            CmpOp::Ge => ord != Less,
        }
    }
}

#[derive(Clone, Debug)]
pub enum Pred {
    /// `col op lit`, or written `lit op' col` when `flipped`.
    Cmp {
        col: String,
        op: CmpOp,
        lit: Lit,
        flipped: bool,
    },
    And(Box<Pred>, Box<Pred>),
    Or(Box<Pred>, Box<Pred>),
}

/// The binder's typecheck: `Some(coerced)` when `lit` may go in a column of
/// type `ty`. INT coerces into FLOAT; nothing else coerces.
fn typecheck(ty: Ty, lit: &Lit) -> Option<Lit> {
    match (ty, lit) {
        (Ty::Int, Lit::Int(i)) => Some(Lit::Int(*i)),
        (Ty::Float, Lit::Float(f)) if f.is_finite() => Some(Lit::Float(*f)),
        (Ty::Float, Lit::Int(i)) => Some(Lit::Float(*i as f64)),
        (Ty::Text, Lit::Text(s)) => Some(Lit::Text(s.clone())),
        (Ty::Vector(d), Lit::Vector(v)) if v.len() == d && v.iter().all(|x| x.is_finite()) => {
            Some(Lit::Vector(v.clone()))
        }
        _ => None,
    }
}

/// A predicate with every literal coerced, or `None` if the binder refuses
/// it.
#[derive(Clone, Debug)]
enum Bound {
    Cmp(usize, CmpOp, Lit),
    And(Box<Bound>, Box<Bound>),
    Or(Box<Bound>, Box<Bound>),
}

fn bind(pred: &Pred, schema: &SchemaDef) -> Option<Bound> {
    Some(match pred {
        Pred::Cmp { col, op, lit, .. } => {
            let i = schema.idx(col)?;
            let c = &schema.cols[i];
            if matches!(c.ty, Ty::Vector(_)) {
                return None;
            }
            if c.ty == Ty::Text && !matches!(op, CmpOp::Eq | CmpOp::Ne) {
                return None;
            }
            Bound::Cmp(i, *op, typecheck(c.ty, lit)?)
        }
        Pred::And(a, b) => Bound::And(Box::new(bind(a, schema)?), Box::new(bind(b, schema)?)),
        Pred::Or(a, b) => Bound::Or(Box::new(bind(a, schema)?), Box::new(bind(b, schema)?)),
    })
}

fn eval(b: &Bound, row: &ModelRow) -> bool {
    match b {
        Bound::Cmp(i, op, lit) => {
            let ord = match (&row[*i], lit) {
                (Lit::Int(a), Lit::Int(b)) => a.cmp(b),
                // IEEE comparison, as SQL defines it: -0.0 == 0.0. (NaN can't
                // be stored or written.)
                (Lit::Float(a), Lit::Float(b)) => a.partial_cmp(b).expect("no NaN"),
                (Lit::Text(a), Lit::Text(b)) => a.as_bytes().cmp(b.as_bytes()),
                other => panic!("model: ill-typed comparison {other:?}"),
            };
            op.holds(ord)
        }
        Bound::And(a, b) => eval(a, row) && eval(b, row),
        Bound::Or(a, b) => eval(a, row) || eval(b, row),
    }
}

// ---------------------------------------------------------------------------
// Ops
// ---------------------------------------------------------------------------

/// Where a value goes: inline in the statement text, or as a `?` parameter.
/// Each literal picks independently, so both paths see every value shape.
#[derive(Clone, Debug)]
pub enum Projection {
    Star,
    Cols(Vec<String>),
}

#[derive(Clone, Debug)]
pub enum Op {
    Create(SchemaDef),
    Drop(String),
    Insert {
        coll: String,
        /// The column list, in the order the statement names them.
        cols: Vec<String>,
        rows: Vec<Vec<Lit>>,
    },
    Update {
        coll: String,
        sets: Vec<(String, Lit)>,
        pred: Pred,
    },
    Delete {
        coll: String,
        pred: Pred,
    },
    Search {
        coll: String,
        k: i64,
        query: Lit,
        pred: Option<Pred>,
        returning: Option<Vec<String>>,
    },
    Select {
        coll: String,
        proj: Projection,
        pred: Option<Pred>,
        window: Option<(i64, i64)>,
    },
    Checkpoint,
    Reopen,
    Crash,
    Compact(String),
    Backup,
}

impl Op {
    /// Does this op change the database (so a crash may tear it)?
    pub fn is_write(&self) -> bool {
        matches!(
            self,
            Op::Create(_) | Op::Drop(_) | Op::Insert { .. } | Op::Update { .. } | Op::Delete { .. }
        )
    }
}

// ---------------------------------------------------------------------------
// Rendering ops to V-SQL
// ---------------------------------------------------------------------------

/// Statement text plus its parameters. `inline` decides, per literal, whether
/// it is written into the text.
struct Sql {
    text: String,
    params: Vec<Param>,
    rng: Rng,
    param_pct: u64,
}

fn f64_sql(f: f64) -> String {
    // `{:?}` is the shortest text that round-trips, with an exponent when
    // that's shorter: `1e-7`, `1.7976931348623157e308`, `-0.0`.
    format!("{f:?}")
}

impl Sql {
    fn new(rng: Rng, param_pct: u64) -> Sql {
        Sql {
            text: String::new(),
            params: Vec::new(),
            rng,
            param_pct,
        }
    }

    fn push(&mut self, s: &str) {
        self.text.push_str(s);
    }

    fn lit(&mut self, lit: &Lit) {
        let as_param = match lit {
            // Not writable as text: the lexer has no NaN/inf, and the
            // magnitude of i64::MIN does not fit in i64.
            Lit::BadFloat(_) => true,
            Lit::Int(i64::MIN) => true,
            _ => self.rng.chance(self.param_pct),
        };
        if as_param {
            self.text.push('?');
            self.params.push(match lit {
                Lit::Int(i) => Param::Int(*i),
                Lit::Float(f) | Lit::BadFloat(f) => Param::Float(*f),
                Lit::Text(s) => Param::Text(s.clone()),
                Lit::Vector(v) => Param::Vector(v.clone()),
            });
            return;
        }
        match lit {
            Lit::Int(i) => {
                let _ = write!(self.text, "{i}");
            }
            Lit::Float(f) => self.text.push_str(&f64_sql(*f)),
            Lit::Text(s) => {
                self.text.push('\'');
                self.text.push_str(&s.replace('\'', "''"));
                self.text.push('\'');
            }
            Lit::Vector(v) => {
                let parts: Vec<String> = v.iter().map(|x| format!("{x:?}")).collect();
                let _ = write!(self.text, "[{}]", parts.join(", "));
            }
            Lit::BadFloat(_) => unreachable!(),
        }
    }

    /// Minimal parentheses: AND binds tighter than OR, so an AND under an OR
    /// needs none, and the parser's precedence is exercised. Extra parens are
    /// sprinkled in at random.
    fn pred(&mut self, p: &Pred, parent_and: bool) {
        let extra = self.rng.chance(10);
        let needs = matches!(p, Pred::Or(..)) && parent_and;
        let paren = needs || extra;
        if paren {
            self.push("(");
        }
        match p {
            Pred::Cmp {
                col,
                op,
                lit,
                flipped,
            } => {
                if *flipped {
                    self.lit(lit);
                    let _ = write!(self.text, " {} {col}", op.flipped().sql());
                } else {
                    let _ = write!(self.text, "{col} {} ", op.sql());
                    self.lit(lit);
                }
            }
            Pred::And(a, b) => {
                self.pred(a, true);
                self.push(" AND ");
                self.pred(b, true);
            }
            Pred::Or(a, b) => {
                self.pred(a, false);
                self.push(" OR ");
                self.pred(b, false);
            }
        }
        if paren {
            self.push(")");
        }
    }
}

/// The statement for a data or DDL op; `None` for an engine op.
pub fn render(op: &Op, rng: Rng, param_pct: u64) -> Option<(String, Vec<Param>)> {
    let mut s = Sql::new(rng, param_pct);
    match op {
        Op::Create(schema) => {
            let cols: Vec<String> = schema
                .cols
                .iter()
                .map(|c| {
                    format!(
                        "{} {}{}",
                        c.name,
                        c.ty.sql(),
                        if c.pk { " PRIMARY KEY" } else { "" }
                    )
                })
                .collect();
            let _ = write!(
                s.text,
                "CREATE COLLECTION {} ({}) WITH (capacity = {});",
                schema.name,
                cols.join(", "),
                schema.capacity
            );
        }
        Op::Drop(name) => {
            let _ = write!(s.text, "DROP COLLECTION {name};");
        }
        Op::Insert { coll, cols, rows } => {
            let _ = write!(s.text, "INSERT INTO {coll} ({}) VALUES ", cols.join(", "));
            for (r, row) in rows.iter().enumerate() {
                if r > 0 {
                    s.push(", ");
                }
                s.push("(");
                for (i, v) in row.iter().enumerate() {
                    if i > 0 {
                        s.push(", ");
                    }
                    s.lit(v);
                }
                s.push(")");
            }
            s.push(";");
        }
        Op::Update { coll, sets, pred } => {
            let _ = write!(s.text, "UPDATE {coll} SET ");
            for (i, (col, v)) in sets.iter().enumerate() {
                if i > 0 {
                    s.push(", ");
                }
                let _ = write!(s.text, "{col} = ");
                s.lit(v);
            }
            s.push(" WHERE ");
            s.pred(pred, false);
            s.push(";");
        }
        Op::Delete { coll, pred } => {
            let _ = write!(s.text, "DELETE FROM {coll} WHERE ");
            s.pred(pred, false);
            s.push(";");
        }
        Op::Search {
            coll,
            k,
            query,
            pred,
            returning,
        } => {
            s.push("SEARCH TOP ");
            s.lit(&Lit::Int(*k));
            s.push(" NEAREST TO ");
            s.lit(query);
            let _ = write!(s.text, " FROM {coll}");
            if let Some(p) = pred {
                s.push(" WHERE ");
                s.pred(p, false);
            }
            if let Some(cols) = returning {
                let _ = write!(s.text, " RETURNING {}", cols.join(", "));
            }
            s.push(";");
        }
        Op::Select {
            coll,
            proj,
            pred,
            window,
        } => {
            s.push("SELECT ");
            match proj {
                Projection::Star => s.push("*"),
                Projection::Cols(c) => s.push(&c.join(", ")),
            }
            let _ = write!(s.text, " FROM {coll}");
            if let Some(p) = pred {
                s.push(" WHERE ");
                s.pred(p, false);
            }
            if let Some((limit, offset)) = window {
                s.push(" LIMIT ");
                s.lit(&Lit::Int(*limit));
                s.push(" OFFSET ");
                s.lit(&Lit::Int(*offset));
            }
            s.push(";");
        }
        _ => return None,
    }
    Some((s.text, s.params))
}

// ---------------------------------------------------------------------------
// The model's verdicts
// ---------------------------------------------------------------------------

/// What a read must return, according to the model.
#[derive(Debug)]
pub enum ReadExpect {
    /// Exactly these labels, and exactly these rows in some order.
    Rows {
        labels: Vec<String>,
        rows: Vec<Vec<Lit>>,
    },
    /// A ranked read: see `check_search`.
    Ranked {
        labels: Vec<String>,
        k: usize,
        /// Every row passing the filter: (projected values, model score,
        /// error bound on that score).
        candidates: Vec<(Vec<Lit>, f64, f64)>,
        /// Where the key and the score sit in `labels`.
        key_at: usize,
        score_at: Option<usize>,
    },
}

#[derive(Debug)]
pub enum Expect {
    /// The statement must fail and change nothing.
    Error,
    /// A write that must succeed.
    Ok,
    /// A read that must succeed with this answer.
    Read(ReadExpect),
}

fn score(q: &[f32], v: &[f32]) -> (f64, f64) {
    let mut s = 0.0f64;
    let mut mag = 0.0f64;
    for (a, b) in q.iter().zip(v) {
        let p = f64::from(*a) * f64::from(*b);
        s += p;
        mag += p.abs();
    }
    // The engine sums f32 products in a SIMD order: each partial sum carries
    // a relative error of about 2^-24, times the dimension. 1e-4 of the
    // magnitude is far looser than that and far tighter than a wrong answer.
    (s, mag * 1e-4 + 1e-30)
}

impl Model {
    /// Predict `op`'s outcome and apply it. Engine ops change nothing here.
    pub fn apply(&mut self, op: &Op) -> Expect {
        match op {
            Op::Create(schema) => {
                if !schema.valid() || self.colls.contains_key(&schema.name) {
                    return Expect::Error;
                }
                self.colls.insert(
                    schema.name.clone(),
                    Coll {
                        schema: schema.clone(),
                        rows: BTreeMap::new(),
                    },
                );
                Expect::Ok
            }
            Op::Drop(name) => {
                if self.colls.remove(name).is_some() {
                    Expect::Ok
                } else {
                    Expect::Error
                }
            }
            Op::Insert { coll, cols, rows } => {
                let Some(c) = self.colls.get_mut(coll) else {
                    return Expect::Error;
                };
                let schema = &c.schema;
                // Every column exactly once.
                let named: BTreeSet<&str> = cols.iter().map(|s| s.as_str()).collect();
                if named.len() != cols.len()
                    || cols.len() != schema.cols.len()
                    || cols.iter().any(|n| schema.idx(n).is_none())
                {
                    return Expect::Error;
                }
                let mut new_rows = Vec::new();
                for row in rows {
                    if row.len() != cols.len() {
                        return Expect::Error;
                    }
                    let mut stored: Vec<Option<Lit>> = vec![None; schema.cols.len()];
                    for (name, v) in cols.iter().zip(row) {
                        let i = schema.idx(name).unwrap();
                        let Some(v) = typecheck(schema.cols[i].ty, v) else {
                            return Expect::Error;
                        };
                        stored[i] = Some(v);
                    }
                    new_rows.push(stored.into_iter().map(Option::unwrap).collect::<Vec<_>>());
                }
                // All-or-nothing on keys: one already live, or two in the
                // statement, refuses every row.
                let pk = schema.pk();
                let mut keys = BTreeSet::new();
                for row in &new_rows {
                    let k = key_of(&row[pk]);
                    if c.rows.contains_key(&k) || !keys.insert(k) {
                        return Expect::Error;
                    }
                }
                for row in new_rows {
                    c.rows.insert(key_of(&row[pk]), row);
                }
                Expect::Ok
            }
            Op::Update { coll, sets, pred } => {
                let Some(c) = self.colls.get_mut(coll) else {
                    return Expect::Error;
                };
                let schema = &c.schema;
                let Some(bound) = bind(pred, schema) else {
                    return Expect::Error;
                };
                let mut assign = Vec::new();
                let mut seen = BTreeSet::new();
                for (name, v) in sets {
                    let Some(i) = schema.idx(name) else {
                        return Expect::Error;
                    };
                    if schema.cols[i].pk || !seen.insert(i) {
                        return Expect::Error;
                    }
                    let Some(v) = typecheck(schema.cols[i].ty, v) else {
                        return Expect::Error;
                    };
                    assign.push((i, v));
                }
                // A new embedding needs the statement to name ONE row: the
                // predicate must be exactly `key = literal`.
                let sets_vector = assign.iter().any(|(i, _)| *i == schema.vector());
                if sets_vector
                    && !matches!(&bound, Bound::Cmp(i, CmpOp::Eq, _) if *i == schema.pk())
                {
                    return Expect::Error;
                }
                for row in c.rows.values_mut() {
                    if eval(&bound, row) {
                        for (i, v) in &assign {
                            row[*i] = v.clone();
                        }
                    }
                }
                Expect::Ok
            }
            Op::Delete { coll, pred } => {
                let Some(c) = self.colls.get_mut(coll) else {
                    return Expect::Error;
                };
                let Some(bound) = bind(pred, &c.schema) else {
                    return Expect::Error;
                };
                c.rows.retain(|_, row| !eval(&bound, row));
                Expect::Ok
            }
            Op::Search {
                coll,
                k,
                query,
                pred,
                returning,
            } => {
                let Some(c) = self.colls.get(coll) else {
                    return Expect::Error;
                };
                let schema = &c.schema;
                if *k < 1 {
                    return Expect::Error;
                }
                let Some(Lit::Vector(q)) = typecheck(Ty::Vector(schema.dim()), query) else {
                    return Expect::Error;
                };
                let bound = match pred {
                    Some(p) => match bind(p, schema) {
                        Some(b) => Some(b),
                        None => return Expect::Error,
                    },
                    None => None,
                };
                // RETURNING: columns (the embedding included) and `score`.
                let labels: Vec<String> = match returning {
                    Some(cols) => {
                        let distinct: BTreeSet<&str> = cols.iter().map(|s| s.as_str()).collect();
                        if distinct.len() != cols.len() {
                            return Expect::Error;
                        }
                        for n in cols {
                            if n != "score" && schema.idx(n).is_none() {
                                return Expect::Error;
                            }
                        }
                        cols.clone()
                    }
                    None => schema.scalars().map(|(_, c)| c.name.clone()).collect(),
                };
                let pk_name = &schema.cols[schema.pk()].name;
                let key_at = labels
                    .iter()
                    .position(|l| l == pk_name)
                    .expect("generator always returns the key");
                let score_at = labels.iter().position(|l| l == "score");
                let vec_i = schema.vector();
                let candidates = c
                    .rows
                    .values()
                    .filter(|row| bound.as_ref().is_none_or(|b| eval(b, row)))
                    .map(|row| {
                        let Lit::Vector(v) = &row[vec_i] else {
                            unreachable!()
                        };
                        let (s, tol) = score(&q, v);
                        let projected = labels
                            .iter()
                            .map(|l| match schema.idx(l) {
                                Some(i) => row[i].clone(),
                                None => Lit::Float(s), // the score slot, checked separately
                            })
                            .collect();
                        (projected, s, tol)
                    })
                    .collect();
                Expect::Read(ReadExpect::Ranked {
                    labels,
                    k: (*k).min(i64::from(u32::MAX)) as usize,
                    candidates,
                    key_at,
                    score_at,
                })
            }
            Op::Select {
                coll,
                proj,
                pred,
                window,
            } => {
                let Some(c) = self.colls.get(coll) else {
                    return Expect::Error;
                };
                let schema = &c.schema;
                let bound = match pred {
                    Some(p) => match bind(p, schema) {
                        Some(b) => Some(b),
                        None => return Expect::Error,
                    },
                    None => None,
                };
                let idxs: Vec<usize> = match proj {
                    Projection::Star => schema.scalars().map(|(i, _)| i).collect(),
                    Projection::Cols(cols) => {
                        let mut v = Vec::new();
                        for n in cols {
                            match schema.idx(n) {
                                Some(i) => v.push(i),
                                None => return Expect::Error,
                            }
                        }
                        v
                    }
                };
                if let Some((limit, offset)) = window
                    && (*limit < 0 || *offset < 0)
                {
                    return Expect::Error;
                }
                let labels = idxs.iter().map(|i| schema.cols[*i].name.clone()).collect();
                let rows = c
                    .rows
                    .values()
                    .filter(|row| bound.as_ref().is_none_or(|b| eval(b, row)))
                    .map(|row| idxs.iter().map(|i| row[*i].clone()).collect())
                    .collect();
                // A window over an unordered SELECT is checked against the
                // database's own unwindowed answer (see `run_select`); here
                // the model gives the full multiset.
                Expect::Read(ReadExpect::Rows { labels, rows })
            }
            Op::Checkpoint | Op::Reopen | Op::Crash | Op::Compact(_) | Op::Backup => Expect::Ok,
        }
    }
}

// ---------------------------------------------------------------------------
// Generation
// ---------------------------------------------------------------------------

const COLL_NAMES: [&str; 4] = ["c0", "c1", "c2", "c3"];
const COL_NAMES: [&str; 12] = [
    "id", "a", "b", "n", "m", "x", "y", "title", "year", "pk", "w", "u",
];
const VEC_NAMES: [&str; 3] = ["vector", "emb", "v"];
const TEXTS: [&str; 16] = [
    "",
    "a",
    "b",
    "A",
    "a ",
    "it's",
    "''",
    "ünï",
    "日本語",
    "SELECT",
    "a;b",
    "\\",
    "x\ny",
    "?",
    "-- c",
    "%",
];

fn gen_int(rng: &mut Rng) -> i64 {
    match rng.below(10) {
        0 => *rng.pick(&[
            0,
            1,
            -1,
            i64::MAX,
            i64::MIN,
            i64::MIN + 1,
            1 << 53,
            (1 << 53) + 1,
        ]),
        1 => rng.next() as i64,
        _ => rng.range(-12, 12),
    }
}

fn gen_float(rng: &mut Rng) -> f64 {
    match rng.below(10) {
        0 => *rng.pick(&[
            0.0,
            -0.0,
            1.0,
            -1.0,
            0.1,
            1e-7,
            5e-324,
            f64::MIN_POSITIVE,
            1e300,
            -1e300,
            f64::MAX,
            9007199254740993.0,
        ]),
        1 => {
            // Any finite bit pattern (`clamp` would let a NaN through).
            let f = f64::from_bits(rng.next());
            if f.is_finite() { f } else { 0.75 }
        }
        2 => rng.range(-1000, 1000) as f64 / 7.0,
        _ => rng.range(-12, 12) as f64 / 2.0,
    }
}

fn gen_text(rng: &mut Rng) -> String {
    if rng.chance(70) {
        (*rng.pick(&TEXTS)).to_string()
    } else {
        let n = rng.below(6);
        (0..n)
            .map(|_| *rng.pick(&['a', 'b', 'z', ' ', '\'', 'é']))
            .collect()
    }
}

fn gen_vector(rng: &mut Rng, dim: usize) -> Vec<f32> {
    let style = rng.below(4);
    (0..dim)
        .map(|_| match style {
            // Small integers: plenty of exact ties in the ranking.
            0 => rng.range(-2, 2) as f32,
            1 => *rng.pick(&[0.0, -0.0, 1.0, 1e-20, -3.5, 1e3]),
            _ => rng.range(-1000, 1000) as f32 / 100.0,
        })
        .collect()
}

fn gen_value(rng: &mut Rng, ty: Ty) -> Lit {
    match ty {
        Ty::Int => Lit::Int(gen_int(rng)),
        Ty::Float => {
            // An INT literal into a FLOAT column is legal, and coerces.
            if rng.chance(15) {
                Lit::Int(gen_int(rng))
            } else {
                Lit::Float(gen_float(rng))
            }
        }
        Ty::Text => Lit::Text(gen_text(rng)),
        Ty::Vector(d) => Lit::Vector(gen_vector(rng, d)),
    }
}

/// A value of the WRONG shape for `ty`, which the binder must refuse.
fn gen_bad_value(rng: &mut Rng, ty: Ty) -> Lit {
    match (ty, rng.below(3)) {
        (_, 0) => Lit::BadFloat(*rng.pick(&[f64::NAN, f64::INFINITY, f64::NEG_INFINITY])),
        (Ty::Int, _) => Lit::Float(0.5),
        (Ty::Float, _) => Lit::Text("1.5".into()),
        (Ty::Text, _) => Lit::Int(1),
        (Ty::Vector(d), 1) => Lit::Vector(vec![1.0; d + 1]),
        (Ty::Vector(_), _) => Lit::Int(3),
    }
}

fn gen_schema(rng: &mut Rng, name: &str) -> SchemaDef {
    let mut names: Vec<&str> = COL_NAMES.to_vec();
    rng.shuffle(&mut names);
    let n_scalars = 1 + rng.below(4) as usize;
    let mut cols: Vec<Col> = names[..n_scalars]
        .iter()
        .map(|n| Col {
            name: n.to_string(),
            ty: *rng.pick(&[Ty::Int, Ty::Float, Ty::Text]),
            pk: false,
        })
        .collect();
    let pk = rng.below(n_scalars as u64) as usize;
    cols[pk].pk = true;
    cols[pk].ty = if rng.chance(50) { Ty::Int } else { Ty::Text };
    let at = rng.below(cols.len() as u64 + 1) as usize;
    cols.insert(
        at,
        Col {
            name: (*rng.pick(&VEC_NAMES)).to_string(),
            ty: Ty::Vector(1 + rng.below(8) as usize),
            pk: false,
        },
    );
    let mut schema = SchemaDef {
        name: name.to_string(),
        cols,
        capacity: 100_000,
    };
    // Sometimes a schema the binder must refuse.
    if rng.chance(8) {
        match rng.below(4) {
            0 => {
                let pk = pk_index(&schema);
                schema.cols[pk].ty = Ty::Float;
            }
            1 => {
                let c = schema.cols[0].clone();
                schema.cols.push(c);
            }
            2 => schema.cols.retain(|c| !matches!(c.ty, Ty::Vector(_))),
            _ => {
                for c in &mut schema.cols {
                    c.pk = false;
                }
            }
        }
    }
    schema
}

fn pk_index(s: &SchemaDef) -> usize {
    s.cols.iter().position(|c| c.pk).unwrap_or(0)
}

/// A value for column `i` that some live row actually holds, when there is
/// one: equality predicates should match something.
fn existing_value(rng: &mut Rng, c: &Coll, i: usize) -> Option<Lit> {
    if c.rows.is_empty() {
        return None;
    }
    let n = rng.below(c.rows.len() as u64) as usize;
    c.rows.values().nth(n).map(|r| r[i].clone())
}

fn gen_leaf(rng: &mut Rng, c: &Coll, noisy: bool) -> Pred {
    let schema = &c.schema;
    let scalars: Vec<(usize, Col)> = schema.scalars().map(|(i, c)| (i, c.clone())).collect();
    let roll = if noisy { rng.below(100) } else { 100 };
    let (col, ty, i) = match roll {
        0..=8 => {
            let v = schema.vector();
            (schema.cols[v].name.clone(), schema.cols[v].ty, Some(v))
        }
        9..=15 => ("score".to_string(), Ty::Float, None),
        16..=22 => ("nope".to_string(), Ty::Int, None),
        _ => {
            let (i, c) = rng.pick(&scalars).clone();
            (c.name, c.ty, Some(i))
        }
    };
    let op = if ty == Ty::Text && !(noisy && rng.chance(25)) {
        *rng.pick(&[CmpOp::Eq, CmpOp::Ne])
    } else {
        *rng.pick(&CmpOp::ALL)
    };
    let lit = match i {
        _ if noisy && rng.chance(15) => gen_bad_value(rng, ty),
        Some(i) if rng.chance(50) => {
            existing_value(rng, c, i).unwrap_or_else(|| gen_value(rng, ty))
        }
        _ => gen_value(rng, ty),
    };
    Pred::Cmp {
        col,
        op,
        lit,
        flipped: rng.chance(20),
    }
}

fn gen_pred(rng: &mut Rng, c: &Coll, depth: u32, noisy: bool) -> Pred {
    if rng.chance(3) {
        // A long OR chain, like an `IN` list: exercises the parser's
        // rebalancing above 32 terms.
        let n = 2 + rng.below(45);
        let mut p = gen_leaf(rng, c, false);
        for _ in 1..n {
            let bad = noisy && rng.chance(5);
            p = Pred::Or(Box::new(p), Box::new(gen_leaf(rng, c, bad)));
        }
        return p;
    }
    if depth == 0 || rng.chance(45) {
        return gen_leaf(rng, c, noisy);
    }
    let bad_a = noisy && rng.chance(50);
    let bad_b = noisy && !bad_a;
    let a = Box::new(gen_pred(rng, c, depth - 1, bad_a));
    let b = Box::new(gen_pred(rng, c, depth - 1, bad_b));
    if rng.chance(50) {
        Pred::And(a, b)
    } else {
        Pred::Or(a, b)
    }
}

/// A key for a new row. A noisy op may reuse a live key (the insert must
/// then be refused); otherwise the key is fresh against the collection AND
/// the rest of the batch (`taken`), so big batches really go in.
fn gen_key(rng: &mut Rng, c: &Coll, noisy: bool, taken: &mut BTreeSet<Key>) -> Lit {
    let pk = c.schema.pk();
    if noisy
        && rng.chance(30)
        && let Some(v) = existing_value(rng, c, pk)
    {
        return v;
    }
    let ty = c.schema.cols[pk].ty;
    for attempt in 0.. {
        let lit = if attempt < 8 {
            gen_value(rng, ty)
        } else if ty == Ty::Int {
            Lit::Int(rng.next() as i64)
        } else {
            Lit::Text(format!("{}{}", gen_text(rng), rng.next() % 100_000))
        };
        let lit = typecheck(ty, &lit).unwrap_or(lit);
        let k = key_of(&lit);
        if !c.rows.contains_key(&k) && taken.insert(k) {
            return lit;
        }
    }
    unreachable!()
}

fn gen_insert(rng: &mut Rng, c: &Coll, noisy: bool) -> Op {
    let schema = &c.schema;
    let mut order: Vec<usize> = (0..schema.cols.len()).collect();
    rng.shuffle(&mut order);
    let mut cols: Vec<String> = order.iter().map(|i| schema.cols[*i].name.clone()).collect();
    let n = match rng.below(10) {
        0..=2 => 2 + rng.below(60),
        3 => 2 + rng.below(6),
        _ => 1,
    };
    let mut rows = Vec::new();
    let mut taken = BTreeSet::new();
    for _ in 0..n {
        rows.push(
            order
                .iter()
                .map(|i| {
                    let col = &schema.cols[*i];
                    if col.pk {
                        gen_key(rng, c, noisy, &mut taken)
                    } else if noisy && rng.chance(10) {
                        gen_bad_value(rng, col.ty)
                    } else {
                        gen_value(rng, col.ty)
                    }
                })
                .collect::<Vec<_>>(),
        );
    }
    match if noisy { rng.below(12) } else { 99 } {
        0 => {
            cols.pop(); // a missing column
            for r in &mut rows {
                r.pop();
            }
        }
        1 => {
            let dup = cols[0].clone(); // a column named twice
            cols.push(dup);
            for r in &mut rows {
                let v = r[0].clone();
                r.push(v);
            }
        }
        2 => cols[0] = "nope".into(),
        _ => {}
    }
    Op::Insert {
        coll: schema.name.clone(),
        cols,
        rows,
    }
}

fn gen_update(rng: &mut Rng, c: &Coll, noisy: bool) -> Op {
    let schema = &c.schema;
    let pk = schema.pk();
    let mut sets = Vec::new();
    let mut scalars: Vec<usize> = schema
        .scalars()
        .map(|(i, _)| i)
        .filter(|i| *i != pk)
        .collect();
    rng.shuffle(&mut scalars);
    let n = rng.below(scalars.len() as u64 + 1) as usize;
    for &i in &scalars[..n] {
        let col = &schema.cols[i];
        let v = if noisy && rng.chance(15) {
            gen_bad_value(rng, col.ty)
        } else {
            gen_value(rng, col.ty)
        };
        sets.push((col.name.clone(), v));
    }
    let vector = rng.chance(35);
    if vector || sets.is_empty() {
        let v = schema.vector();
        sets.push((
            schema.cols[v].name.clone(),
            gen_value(rng, schema.cols[v].ty),
        ));
    }
    if noisy && rng.chance(15) {
        sets.push((
            schema.cols[pk].name.clone(),
            gen_value(rng, schema.cols[pk].ty),
        ));
    }
    rng.shuffle(&mut sets);
    // A row-naming predicate (needed to set the vector), or a general one.
    let pred = if rng.chance(60) {
        Pred::Cmp {
            col: schema.cols[pk].name.clone(),
            op: CmpOp::Eq,
            lit: existing_value(rng, c, pk).unwrap_or_else(|| gen_value(rng, schema.cols[pk].ty)),
            flipped: rng.chance(15),
        }
    } else {
        gen_pred(rng, c, 2, noisy)
    };
    Op::Update {
        coll: schema.name.clone(),
        sets,
        pred,
    }
}

fn gen_search(rng: &mut Rng, c: &Coll, noisy: bool) -> Op {
    let schema = &c.schema;
    // Mostly well below the row count, so the ranking decides what's in.
    let n = c.rows.len() as u64;
    let k = match rng.below(20) {
        0 if noisy => *rng.pick(&[0, -1]),
        1 => 1_000_000,
        2..=6 => 1 + rng.below(12) as i64,
        _ => 1 + rng.below((n / 3).max(1)) as i64,
    };
    let query = if noisy && rng.chance(20) {
        gen_bad_value(rng, schema.cols[schema.vector()].ty)
    } else {
        Lit::Vector(gen_vector(rng, schema.dim()))
    };
    let pred = rng.chance(50).then(|| gen_pred(rng, c, 3, noisy));
    let returning = rng.chance(70).then(|| {
        let mut cols = vec![schema.cols[schema.pk()].name.clone()];
        if rng.chance(70) {
            cols.push("score".into());
        }
        for (i, col) in schema.cols.iter().enumerate() {
            if i != schema.pk() && rng.chance(40) {
                cols.push(col.name.clone());
            }
        }
        if noisy && rng.chance(15) {
            cols.push("nope".into());
        }
        rng.shuffle(&mut cols);
        cols
    });
    Op::Search {
        coll: schema.name.clone(),
        k,
        query,
        pred,
        returning,
    }
}

fn gen_select(rng: &mut Rng, c: &Coll, noisy: bool) -> Op {
    let schema = &c.schema;
    let proj = if rng.chance(30) {
        Projection::Star
    } else {
        let mut cols: Vec<String> = schema
            .cols
            .iter()
            .filter(|_| rng.chance(60))
            .map(|c| c.name.clone())
            .collect();
        if cols.is_empty() {
            cols.push(schema.cols[schema.pk()].name.clone());
        }
        if noisy && rng.chance(15) {
            cols.push("nope".into());
        }
        rng.shuffle(&mut cols);
        Projection::Cols(cols)
    };
    let pred = rng.chance(60).then(|| gen_pred(rng, c, 3, noisy));
    let window = rng.chance(25).then(|| {
        if noisy && rng.chance(20) {
            (-1, 0)
        } else {
            (rng.range(0, 8), rng.range(0, 10))
        }
    });
    Op::Select {
        coll: schema.name.clone(),
        proj,
        pred,
        window,
    }
}

/// One op, chosen against the model's current state.
pub fn gen_op(rng: &mut Rng, model: &Model) -> Op {
    let colls: Vec<&Coll> = model.colls.values().collect();
    if colls.is_empty() || (colls.len() < 3 && rng.chance(2)) || rng.chance(1) {
        let name = rng.pick(&COLL_NAMES);
        return Op::Create(gen_schema(rng, name));
    }
    let c = *rng.pick(&colls);
    // About one op in five may be malformed somewhere; the rest are valid
    // all the way down, so long predicates and big batches really run.
    let noisy = rng.chance(20);
    let mut op = match rng.below(1000) {
        0..=299 => gen_insert(rng, c, noisy),
        300..=419 => gen_update(rng, c, noisy),
        420..=489 => Op::Delete {
            coll: c.schema.name.clone(),
            pred: gen_pred(rng, c, 2, noisy),
        },
        490..=719 => gen_search(rng, c, noisy),
        720..=889 => gen_select(rng, c, noisy),
        890..=897 => Op::Drop(c.schema.name.clone()),
        898..=929 => Op::Checkpoint,
        930..=949 => Op::Reopen,
        950..=974 => Op::Crash,
        975..=989 => Op::Compact(c.schema.name.clone()),
        990..=995 => Op::Backup,
        _ => Op::Drop("nope".into()),
    };
    if noisy && rng.chance(5) {
        match &mut op {
            Op::Insert { coll, .. }
            | Op::Update { coll, .. }
            | Op::Delete { coll, .. }
            | Op::Search { coll, .. }
            | Op::Select { coll, .. } => *coll = "missing".into(),
            _ => {}
        }
    }
    op
}

/// A whole workload: `n` ops, each generated against the model as the ops
/// before it left it.
pub fn gen_workload(seed: u64, n: usize) -> Vec<Op> {
    let mut rng = Rng::new(seed);
    let mut model = Model::default();
    let mut ops = Vec::with_capacity(n);
    for _ in 0..n {
        let op = gen_op(&mut rng, &model);
        model.apply(&op);
        ops.push(op);
    }
    ops
}

// ---------------------------------------------------------------------------
// Running against the database
// ---------------------------------------------------------------------------

pub fn opts() -> DbOptions {
    DbOptions {
        checkpoint_interval: Duration::from_secs(3600),
        compaction: CompactionOptions {
            enabled: false,
            ..Default::default()
        },
    }
}

fn to_lit(v: &RegValue) -> Lit {
    match v {
        RegValue::Int(i) => Lit::Int(*i),
        RegValue::Real(f) => Lit::Float(*f),
        RegValue::Str(s) => Lit::Text(s.clone()),
        RegValue::Vector(v) => Lit::Vector(v.to_vec()),
        other => panic!("unexpected register value in a result: {other:?}"),
    }
}

fn row_canon(row: &[Lit]) -> String {
    row.iter().map(Lit::canon).collect::<Vec<_>>().join("|")
}

/// Run one statement: `Ok((labels, rows))` or the error text.
fn query(db: &Db, sql: &str, params: &[Param]) -> Result<(Vec<String>, Vec<Vec<Lit>>), String> {
    let stream = db.execute_with(sql, params).map_err(|e| e.to_string())?;
    let labels = stream.labels().to_vec();
    let mut rows = Vec::new();
    for row in stream {
        let row = row.map_err(|e| e.to_string())?;
        rows.push(row.0.iter().map(to_lit).collect());
    }
    Ok((labels, rows))
}

fn check_rows(
    want_labels: &[String],
    want: &[Vec<Lit>],
    labels: &[String],
    got: &[Vec<Lit>],
) -> Result<(), String> {
    if labels != want_labels {
        return Err(format!("labels {labels:?}, model says {want_labels:?}"));
    }
    let mut a: Vec<String> = got.iter().map(|r| row_canon(r)).collect();
    let mut b: Vec<String> = want.iter().map(|r| row_canon(r)).collect();
    a.sort();
    b.sort();
    if a != b {
        let extra: Vec<&String> = a.iter().filter(|x| !b.contains(x)).take(5).collect();
        let missing: Vec<&String> = b.iter().filter(|x| !a.contains(x)).take(5).collect();
        return Err(format!(
            "{} rows, model says {}; unexpected {extra:?}; missing {missing:?}",
            a.len(),
            b.len()
        ));
    }
    Ok(())
}

/// A ranked answer is right when it is the right SIZE, holds only rows that
/// pass the filter with their exact values, comes out in score order, and
/// no row left out scores higher than one let in. Each of those allows the
/// f32-vs-f64 score error and nothing more, so ties may break either way.
fn check_search(expect: &ReadExpect, labels: &[String], got: &[Vec<Lit>]) -> Result<(), String> {
    let ReadExpect::Ranked {
        labels: want_labels,
        k,
        candidates,
        key_at,
        score_at,
    } = expect
    else {
        unreachable!()
    };
    if labels != want_labels.as_slice() {
        return Err(format!("labels {labels:?}, model says {want_labels:?}"));
    }
    let want_len = (*k).min(candidates.len());
    if got.len() != want_len {
        return Err(format!(
            "{} rows, model says {want_len} (k = {k}, {} pass the filter)",
            got.len(),
            candidates.len()
        ));
    }
    let mut chosen = Vec::new();
    for row in got {
        let Some(at) = candidates
            .iter()
            .position(|(c, ..)| c[*key_at].same(&row[*key_at]))
        else {
            return Err(format!(
                "returned {:?}, which does not pass the filter",
                row[*key_at]
            ));
        };
        if chosen.contains(&at) {
            return Err(format!("returned {:?} twice", row[*key_at]));
        }
        let (want, s, tol) = &candidates[at];
        for (i, (g, w)) in row.iter().zip(want).enumerate() {
            if Some(i) == *score_at {
                let Lit::Float(g) = g else {
                    return Err(format!("score is {g:?}"));
                };
                if (g - s).abs() > *tol {
                    return Err(format!(
                        "{:?} scored {g}, model says {s} (± {tol})",
                        row[*key_at]
                    ));
                }
            } else if !g.same(w) {
                return Err(format!(
                    "{:?}: {} is {g:?}, model says {w:?}",
                    row[*key_at], labels[i]
                ));
            }
        }
        chosen.push(at);
    }
    for pair in chosen.windows(2) {
        let (a, b) = (&candidates[pair[0]], &candidates[pair[1]]);
        if a.1 < b.1 - a.2.max(b.2) {
            return Err(format!("out of order: {} ranked above {}", a.1, b.1));
        }
    }
    let lowest = chosen
        .iter()
        .map(|i| (candidates[*i].1, candidates[*i].2))
        .fold(
            (f64::INFINITY, 0.0),
            |acc, x| if x.0 < acc.0 { x } else { acc },
        );
    for (i, (row, s, tol)) in candidates.iter().enumerate() {
        if !chosen.contains(&i) && *s > lowest.0 + tol.max(lowest.1) {
            return Err(format!(
                "{:?} (score {s}) was left out, but {} made the top {k}",
                row[*key_at], lowest.0
            ));
        }
    }
    Ok(())
}

/// A database under test, its directory, and the model it must match.
pub struct Sim {
    root: tempfile::TempDir,
    db: Option<Db>,
    pub model: Model,
    rng: Rng,
    backups: u32,
    /// Percent of literals sent as `?` parameters.
    pub param_pct: u64,
    /// What the workload actually exercised; see [`Stats`].
    pub stats: Stats,
}

/// Coverage of a run, so a generator that drifts into testing nothing shows
/// up as numbers rather than as green tests.
#[derive(Debug, Default, Clone)]
pub struct Stats {
    pub ok_writes: u64,
    pub refused: u64,
    pub searches: u64,
    /// Searches where `k` cut the candidates: the ranking actually mattered.
    pub ranked_cuts: u64,
    pub nonempty_reads: u64,
    pub max_rows: usize,
}

impl Sim {
    pub fn new(seed: u64) -> Sim {
        let root = tempfile::tempdir().expect("tempdir");
        let db = Db::open(root.path().join("db"), &[], opts()).expect("open");
        Sim {
            root,
            db: Some(db),
            model: Model::default(),
            rng: Rng::new(seed ^ 0xA5A5),
            backups: 0,
            param_pct: 40,
            stats: Stats::default(),
        }
    }

    pub fn path(&self) -> PathBuf {
        self.root.path().join("db")
    }

    fn db(&self) -> &Db {
        self.db.as_ref().expect("open")
    }

    fn reopen(&mut self, clean: bool) -> Result<(), String> {
        let db = self.db.take().expect("open");
        if clean {
            db.close().map_err(|e| format!("close: {e}"))?;
        } else {
            drop(db);
        }
        self.db = Some(Db::open(self.path(), &[], opts()).map_err(|e| format!("reopen: {e}"))?);
        self.verify_all()
    }

    /// Every collection and every row, compared with the model.
    pub fn verify_all(&self) -> Result<(), String> {
        verify_db(self.db(), &self.model)
    }

    /// Run one op and compare the outcome with the model's prediction.
    pub fn step(&mut self, op: &Op) -> Result<(), String> {
        let rng = Rng(self.rng.next() | 1);
        let expect = self.model.apply(op);
        let rows_now = self
            .model
            .colls
            .values()
            .map(|c| c.rows.len())
            .max()
            .unwrap_or(0);
        self.stats.max_rows = self.stats.max_rows.max(rows_now);
        match &expect {
            Expect::Error => self.stats.refused += 1,
            Expect::Ok if op.is_write() => self.stats.ok_writes += 1,
            Expect::Read(ReadExpect::Ranked { k, candidates, .. }) => {
                self.stats.searches += 1;
                if *k < candidates.len() {
                    self.stats.ranked_cuts += 1;
                }
                if !candidates.is_empty() {
                    self.stats.nonempty_reads += 1;
                }
            }
            Expect::Read(ReadExpect::Rows { rows, .. }) if !rows.is_empty() => {
                self.stats.nonempty_reads += 1;
            }
            _ => {}
        }
        match op {
            Op::Checkpoint => self
                .db()
                .checkpoint()
                .map_err(|e| format!("checkpoint: {e}")),
            Op::Reopen => self.reopen(true),
            Op::Crash => self.reopen(false),
            Op::Compact(name) => {
                let id = self
                    .db()
                    .collections()
                    .into_iter()
                    .find(|c| &c.name == name)
                    .map(|c| c.id);
                match id {
                    Some(id) => {
                        self.db()
                            .compact_now(id)
                            .map_err(|e| format!("compact: {e}"))?;
                        self.verify_all()
                    }
                    None => Ok(()),
                }
            }
            Op::Backup => {
                self.backups += 1;
                let dest = self.root.path().join(format!("backup-{}", self.backups));
                self.db()
                    .backup(&dest)
                    .map_err(|e| format!("backup: {e}"))?;
                let copy = Db::open(&dest, &[], opts()).map_err(|e| format!("open backup: {e}"))?;
                let verdict = verify_db(&copy, &self.model).map_err(|e| format!("backup: {e}"));
                copy.close().map_err(|e| format!("close backup: {e}"))?;
                let _ = std::fs::remove_dir_all(&dest);
                verdict
            }
            _ => {
                let (sql, params) = render(op, rng, self.param_pct).expect("a statement");
                let got = query(self.db(), &sql, &params);
                let ctx =
                    |msg: String| format!("{msg}\n    statement: {sql}\n    params: {params:?}");
                match (expect, got) {
                    (Expect::Error, Err(_)) => Ok(()),
                    (Expect::Error, Ok(_)) => Err(ctx("succeeded, model says it must fail".into())),
                    (_, Err(e)) => Err(ctx(format!("failed: {e}"))),
                    (Expect::Ok, Ok(_)) => Ok(()),
                    (Expect::Read(r @ ReadExpect::Ranked { .. }), Ok((labels, rows))) => {
                        check_search(&r, &labels, &rows).map_err(ctx)?;
                        self.metamorphic_search(op, &rows).map_err(ctx)
                    }
                    (
                        Expect::Read(ReadExpect::Rows {
                            labels: wl,
                            rows: wr,
                        }),
                        Ok((labels, rows)),
                    ) => match op {
                        Op::Select {
                            window: Some(_), ..
                        } => self.check_window(op, &wl, &wr, &labels, &rows).map_err(ctx),
                        _ => check_rows(&wl, &wr, &labels, &rows).map_err(ctx),
                    },
                }
            }
        }
    }

    /// A windowed SELECT must be exactly a slice of the same SELECT without
    /// the window, run against the same state — and that one must match the
    /// model.
    fn check_window(
        &mut self,
        op: &Op,
        wl: &[String],
        wr: &[Vec<Lit>],
        labels: &[String],
        rows: &[Vec<Lit>],
    ) -> Result<(), String> {
        let Op::Select {
            coll,
            proj,
            pred,
            window: Some((limit, offset)),
        } = op
        else {
            unreachable!()
        };
        let full_op = Op::Select {
            coll: coll.clone(),
            proj: proj.clone(),
            pred: pred.clone(),
            window: None,
        };
        let (sql, params) = render(&full_op, Rng(self.rng.next() | 1), self.param_pct).unwrap();
        let (_, full) = query(self.db(), &sql, &params).map_err(|e| format!("unwindowed: {e}"))?;
        check_rows(wl, wr, labels, &full).map_err(|e| format!("unwindowed: {e}"))?;
        let from = (*offset as usize).min(full.len());
        let to = from.saturating_add(*limit as usize).min(full.len());
        let want: Vec<String> = full[from..to].iter().map(|r| row_canon(r)).collect();
        let got: Vec<String> = rows.iter().map(|r| row_canon(r)).collect();
        if want != got {
            return Err(format!(
                "LIMIT {limit} OFFSET {offset} gave {} rows that are not rows {from}..{to} of the same query",
                got.len()
            ));
        }
        Ok(())
    }

    /// Asking for more neighbours must not change the ones already found:
    /// the top k's scores are the first k scores of the top k + more.
    fn metamorphic_search(&mut self, op: &Op, rows: &[Vec<Lit>]) -> Result<(), String> {
        let Op::Search {
            coll,
            k,
            query: q,
            pred,
            returning: Some(ret),
        } = op
        else {
            return Ok(());
        };
        let Some(score_at) = ret.iter().position(|c| c == "score") else {
            return Ok(());
        };
        if *k > 50 {
            return Ok(());
        }
        let wider = Op::Search {
            coll: coll.clone(),
            k: k + 1 + self.rng.below(5) as i64,
            query: q.clone(),
            pred: pred.clone(),
            returning: Some(ret.clone()),
        };
        let (sql, params) = render(&wider, Rng(self.rng.next() | 1), self.param_pct).unwrap();
        let (_, more) =
            query(self.db(), &sql, &params).map_err(|e| format!("wider search: {e}"))?;
        let scores =
            |rs: &[Vec<Lit>]| -> Vec<String> { rs.iter().map(|r| r[score_at].canon()).collect() };
        let a = scores(rows);
        let b = scores(&more);
        if b.len() < a.len() || b[..a.len()] != a[..] {
            return Err(format!(
                "TOP {k} scores {a:?} are not a prefix of TOP {} scores {b:?}",
                k + 1
            ));
        }
        Ok(())
    }

    /// Run a whole workload, verifying every row every `every` ops and at the
    /// end. `Err((index, message))` names the op that went wrong.
    pub fn run(&mut self, ops: &[Op], every: usize) -> Result<(), (usize, String)> {
        for (i, op) in ops.iter().enumerate() {
            self.step(op).map_err(|e| (i, e))?;
            if every > 0 && i % every == every - 1 {
                self.verify_all()
                    .map_err(|e| (i, format!("after this op: {e}")))?;
            }
        }
        self.verify_all().map_err(|e| (ops.len(), e))?;
        self.reopen(false)
            .map_err(|e| (ops.len(), format!("final crash-reopen: {e}")))
    }
}

/// `db`'s collections and every row of each, compared with `model`.
pub fn verify_db(db: &Db, model: &Model) -> Result<(), String> {
    let names: BTreeSet<String> = db.collections().into_iter().map(|c| c.name).collect();
    let want: BTreeSet<String> = model.colls.keys().cloned().collect();
    if names != want {
        return Err(format!("collections {names:?}, model says {want:?}"));
    }
    for c in model.colls.values() {
        let cols: Vec<String> = c.schema.cols.iter().map(|c| c.name.clone()).collect();
        let sql = format!("SELECT {} FROM {};", cols.join(", "), c.schema.name);
        let (labels, rows) = query(db, &sql, &[]).map_err(|e| format!("{sql}: {e}"))?;
        let want: Vec<Vec<Lit>> = c.rows.values().cloned().collect();
        check_rows(&cols, &want, &labels, &rows).map_err(|e| format!("{}: {e}", c.schema.name))?;
    }
    Ok(())
}

// ---------------------------------------------------------------------------
// Shrinking and reporting
// ---------------------------------------------------------------------------

/// Run `ops` on a fresh database: `Some(message)` if it fails.
pub fn fails(seed: u64, ops: &[Op]) -> Option<String> {
    let result = std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
        let mut sim = Sim::new(seed);
        sim.run(ops, 0).err()
    }));
    match result {
        Ok(Some((_, msg))) => Some(msg),
        Ok(None) => None,
        Err(p) => Some(format!(
            "PANIC: {}",
            p.downcast_ref::<String>()
                .cloned()
                .or_else(|| p.downcast_ref::<&str>().map(|s| s.to_string()))
                .unwrap_or_default()
        )),
    }
}

/// Delta debugging: drop chunks of ops, halving the chunk size, keeping each
/// removal that still fails. Bounded so a CI job always finishes.
pub fn shrink(seed: u64, mut ops: Vec<Op>, budget: usize) -> Vec<Op> {
    let mut tries = 0;
    let mut chunk = ops.len().div_ceil(2).max(1);
    while chunk >= 1 && tries < budget {
        let mut at = 0;
        let mut progressed = false;
        while at < ops.len() && tries < budget {
            let mut candidate = ops.clone();
            let end = (at + chunk).min(candidate.len());
            candidate.drain(at..end);
            tries += 1;
            if !candidate.is_empty() && fails(seed, &candidate).is_some() {
                ops = candidate;
                progressed = true;
            } else {
                at += chunk;
            }
        }
        if chunk == 1 && !progressed {
            break;
        }
        if !progressed {
            chunk /= 2;
        }
    }
    ops
}

/// A replayable description of `ops`: each statement as the simulator would
/// render it (literals inline), engine ops by name.
pub fn script(ops: &[Op]) -> String {
    let mut out = String::new();
    for (i, op) in ops.iter().enumerate() {
        let line = match render(op, Rng(1), 0) {
            Some((sql, _)) => sql,
            None => format!("-- {op:?}"),
        };
        let _ = writeln!(out, "{i:4}: {line}");
    }
    out
}

/// Where failure reports go (CI uploads this directory).
pub fn report_dir() -> PathBuf {
    std::env::var_os("FLATS_SIM_OUT")
        .map(PathBuf::from)
        .unwrap_or_else(|| Path::new(env!("CARGO_MANIFEST_DIR")).join("target/sim-failures"))
}
