//! Floats in scientific notation, end to end.
//!
//! A client builds V-SQL by formatting its floats with its own language's
//! printer, and every mainstream printer emits an exponent for small or large
//! magnitudes (Rust `{:?}` -> `1e-5`, Python `repr` -> `1e-05`, JavaScript ->
//! `1e-7`). Embedding components are routinely that small, so the round trip
//! "format with `{:?}`, INSERT, SELECT back" has to be EXACT — bit for bit —
//! or every SDK silently perturbs the vectors it stores.

use flats::vm::exec::RegValue;
use flats::{Db, DbOptions};

fn open(dir: &std::path::Path) -> Db {
    Db::open(dir, &[], DbOptions::default()).expect("open")
}

fn vector_sql(v: &[f32]) -> String {
    let parts: Vec<String> = v.iter().map(|x| format!("{x:?}")).collect();
    format!("[{}]", parts.join(", "))
}

/// xorshift; deterministic so a failure names a reproducible vector.
struct Rng(u64);
impl Rng {
    fn next_u32(&mut self) -> u32 {
        self.0 ^= self.0 << 13;
        self.0 ^= self.0 >> 7;
        self.0 ^= self.0 << 17;
        (self.0 >> 32) as u32
    }
    /// Any FINITE f32 — drawn from raw bits, so subnormals, tiny and huge
    /// magnitudes all show up, not just the friendly middle of the range.
    fn finite_f32(&mut self) -> f32 {
        loop {
            let x = f32::from_bits(self.next_u32());
            if x.is_finite() {
                return x;
            }
        }
    }
}

fn select_vectors(db: &Db) -> Vec<(i64, Vec<f32>)> {
    db.execute("SELECT id, vector FROM docs;")
        .expect("select")
        .map(|row| match row.expect("row").0.as_slice() {
            [RegValue::Int(id), RegValue::Vector(v)] => (*id, v.to_vec()),
            other => panic!("unexpected row {other:?}"),
        })
        .collect()
}

#[test]
fn every_finite_f32_round_trips_through_insert_and_select_bit_for_bit() {
    const DIM: usize = 16;
    const ROWS: i64 = 200;
    let dir = tempfile::tempdir().unwrap();
    let db = open(dir.path());
    db.execute(&format!(
        "CREATE COLLECTION docs (id INT PRIMARY KEY, vector VECTOR({DIM})) \
         WITH (capacity = {ROWS});"
    ))
    .unwrap();

    let mut rng = Rng(0x5EED_F10A7);
    let mut want = Vec::new();
    for id in 0..ROWS {
        let v: Vec<f32> = (0..DIM).map(|_| rng.finite_f32()).collect();
        db.execute(&format!(
            "INSERT INTO docs (id, vector) VALUES ({id}, {});",
            vector_sql(&v)
        ))
        .unwrap_or_else(|e| panic!("insert {id} {v:?}: {e}"));
        want.push((id, v));
    }

    let mut got = select_vectors(&db);
    got.sort_by_key(|(id, _)| *id);
    assert_eq!(got.len(), want.len());
    for ((gid, g), (wid, w)) in got.iter().zip(&want) {
        assert_eq!(gid, wid);
        let gb: Vec<u32> = g.iter().map(|x| x.to_bits()).collect();
        let wb: Vec<u32> = w.iter().map(|x| x.to_bits()).collect();
        assert_eq!(gb, wb, "row {gid} changed in the round trip");
    }
}

#[test]
fn python_and_javascript_spellings_are_accepted_everywhere_a_float_is() {
    let dir = tempfile::tempdir().unwrap();
    let db = open(dir.path());
    db.execute(
        "CREATE COLLECTION docs (id INT PRIMARY KEY, vector VECTOR(3), w FLOAT) \
         WITH (capacity = 8);",
    )
    .unwrap();
    // Python repr spellings in the vector AND in a FLOAT column.
    db.execute("INSERT INTO docs (id, vector, w) VALUES (1, [1e-05, -2.5e-07, 1.0], 1e-05);")
        .unwrap();
    // JavaScript spellings.
    db.execute("INSERT INTO docs (id, vector, w) VALUES (2, [1e-7, 1e+21, -0.5], 2e+3);")
        .unwrap();

    // WHERE takes one too.
    let ids: Vec<i64> = db
        .execute("SELECT id FROM docs WHERE w < 1E-4;")
        .unwrap()
        .map(|r| match r.unwrap().0.as_slice() {
            [RegValue::Int(id)] => *id,
            other => panic!("unexpected row {other:?}"),
        })
        .collect();
    assert_eq!(ids, vec![1]);

    // And the query vector of a SEARCH.
    let hits = db
        .execute("SEARCH TOP 1 NEAREST TO [1e-05, 0e0, 1E+0] FROM docs RETURNING id;")
        .unwrap()
        .count();
    assert_eq!(hits, 1);

    let mut got = select_vectors(&db);
    got.sort_by_key(|(id, _)| *id);
    assert_eq!(got[0].1, vec![1e-5_f32, -2.5e-7, 1.0]);
    assert_eq!(got[1].1, vec![1e-7_f32, 1e21, -0.5]);
}

#[test]
fn an_overflowing_literal_is_refused_not_stored_as_infinity() {
    let dir = tempfile::tempdir().unwrap();
    let db = open(dir.path());
    db.execute(
        "CREATE COLLECTION docs (id INT PRIMARY KEY, vector VECTOR(2)) WITH (capacity = 4);",
    )
    .unwrap();
    let err = db
        .execute("INSERT INTO docs (id, vector) VALUES (1, [1e999, 0.0]);")
        .err()
        .expect("an infinite component must be refused");
    assert!(
        err.to_string().contains("1e999"),
        "error names the literal: {err}"
    );
    assert!(select_vectors(&db).is_empty(), "nothing was stored");
}

#[test]
fn a_finite_literal_with_no_f32_to_become_is_refused_in_a_vector() {
    let dir = tempfile::tempdir().unwrap();
    let db = open(dir.path());
    db.execute(
        "CREATE COLLECTION docs (id INT PRIMARY KEY, vector VECTOR(2)) WITH (capacity = 4);",
    )
    .unwrap();
    // 3.5e38 is an ordinary f64 but past f32::MAX: storing it would store inf.
    let err = db
        .execute("INSERT INTO docs (id, vector) VALUES (1, [3.5e38, 0.0]);")
        .err()
        .expect("an element beyond f32 range must be refused");
    assert!(
        err.to_string().contains("f32"),
        "error names the range: {err}"
    );
    assert!(select_vectors(&db).is_empty(), "nothing was stored");
}
