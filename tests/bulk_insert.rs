//! Bulk insert: many rows, ONE statement, ONE durable record.
//!
//! Two front doors reach the same engine path:
//!   * `Db::insert_many` — the Rust API, no ordinals in the caller's hands;
//!   * `INSERT … VALUES (…), (…), …` — the V-SQL spelling.
//!
//! The contract both inherit: a batch lands WHOLE or not at all. Every refusal
//! (wrong dimension, malformed row, a key taken by a live row or repeated in
//! the batch, capacity) leaves the collection exactly as it was — no row, no
//! burned capacity — and a batch that returned Ok survives a crash.
//!
//! The authoritative primary-key check lives on the commit thread, so it also
//! guards `Db::mutate`, which has no caller-side check at all; the tests that
//! drive `mutate` directly are the deterministic way to reach it.

use flats::vm::exec::RegValue;
use flats::wal::wal::{Install, Outcome, RejectReason};
use flats::{Db, DbOptions, Error, Row, Value};
use roaring::RoaringBitmap;

const DIM: usize = 4;

fn open(dir: &std::path::Path) -> Db {
    Db::open(dir, &[], DbOptions::default()).expect("open")
}

/// `docs (id INT PRIMARY KEY, vector VECTOR(4), tag TEXT)`: scalar ColumnIds
/// are id = 0, tag = 1.
fn create(db: &Db, capacity: usize) -> u32 {
    db.execute(&format!(
        "CREATE COLLECTION docs (id INT PRIMARY KEY, vector VECTOR({DIM}), tag TEXT) \
         WITH (capacity = {capacity});"
    ))
    .expect("create");
    db.collection_id("docs").expect("id")
}

fn vector(n: i64) -> Vec<f32> {
    vec![n as f32, 1.0, -0.5, 0.25]
}

fn row(id: i64) -> Row {
    vec![(0, Value::Int(id)), (1, Value::Text(format!("t{id}")))]
}

fn batch(ids: impl IntoIterator<Item = i64>) -> Vec<(Vec<f32>, Row)> {
    ids.into_iter().map(|id| (vector(id), row(id))).collect()
}

/// Every live row as (id, tag, vector), sorted by id.
fn rows(db: &Db) -> Vec<(i64, String, Vec<f32>)> {
    let mut out: Vec<_> = db
        .execute("SELECT id, tag, vector FROM docs;")
        .expect("select")
        .map(|r| match r.expect("row").0.as_slice() {
            [RegValue::Int(id), RegValue::Str(tag), RegValue::Vector(v)] => {
                (*id, tag.to_string(), v.to_vec())
            }
            other => panic!("unexpected row {other:?}"),
        })
        .collect();
    out.sort_by_key(|(id, _, _)| *id);
    out
}

fn ids(db: &Db) -> Vec<i64> {
    rows(db).into_iter().map(|(id, _, _)| id).collect()
}

// -- Db::insert_many ---------------------------------------------------------

#[test]
fn insert_many_lands_every_row_readable_through_every_path() {
    let dir = tempfile::tempdir().unwrap();
    let db = open(dir.path());
    let c = create(&db, 100);
    let ordinals = db.insert_many(c, batch(0..50)).expect("insert_many");

    assert_eq!(ordinals.len(), 50);
    let got = rows(&db);
    assert_eq!(got.len(), 50);
    for (i, (id, tag, v)) in got.iter().enumerate() {
        assert_eq!(*id, i as i64);
        assert_eq!(tag, &format!("t{id}"));
        assert_eq!(v, &vector(*id));
    }
    // Searchable as soon as it returns, like a single insert.
    let hits = db.search(c, &vector(49), 1).expect("search");
    assert_eq!(hits.len(), 1);
    // And WHERE sees the metadata.
    let n = db
        .execute("SELECT id FROM docs WHERE id >= 40;")
        .unwrap()
        .count();
    assert_eq!(n, 10);
}

#[test]
fn insert_many_returns_distinct_ordinals_in_input_order() {
    let dir = tempfile::tempdir().unwrap();
    let db = open(dir.path());
    let c = create(&db, 100);
    db.insert(c, &vector(1000), row(1000)).unwrap();
    let ordinals = db.insert_many(c, batch(0..10)).unwrap();
    let raw: Vec<u32> = ordinals.iter().map(|o| o.0).collect();
    assert_eq!(
        raw,
        (1..11).collect::<Vec<u32>>(),
        "consecutive, after the existing row"
    );
    // The ordinal handed back for position i IS row i's.
    let reader = db.reader(c).unwrap();
    for (i, o) in ordinals.iter().enumerate() {
        assert_eq!(reader.vector_at(*o).unwrap(), vector(i as i64).as_slice());
    }
}

#[test]
fn an_empty_batch_is_a_no_op() {
    let dir = tempfile::tempdir().unwrap();
    let db = open(dir.path());
    let c = create(&db, 4);
    assert!(db.insert_many(c, Vec::new()).unwrap().is_empty());
    assert!(ids(&db).is_empty());
}

#[test]
fn a_key_already_live_refuses_the_whole_batch() {
    let dir = tempfile::tempdir().unwrap();
    let db = open(dir.path());
    let c = create(&db, 100);
    db.insert(c, &vector(7), row(7)).unwrap();
    match db.insert_many(c, batch(0..10)) {
        Err(Error::DuplicateKey { column, value }) => {
            assert_eq!(column, "id");
            assert_eq!(value, "7");
        }
        other => panic!("expected DuplicateKey, got {other:?}"),
    }
    assert_eq!(ids(&db), vec![7], "none of the batch landed");
}

#[test]
fn a_key_repeated_inside_the_batch_refuses_the_whole_batch() {
    let dir = tempfile::tempdir().unwrap();
    let db = open(dir.path());
    let c = create(&db, 100);
    match db.insert_many(c, batch([1, 2, 3, 2, 5])) {
        Err(Error::DuplicateKey { column, value }) => {
            assert_eq!(column, "id");
            assert_eq!(value, "2");
        }
        other => panic!("expected DuplicateKey, got {other:?}"),
    }
    assert!(ids(&db).is_empty());
}

#[test]
fn a_wrong_dimension_anywhere_refuses_the_whole_batch() {
    let dir = tempfile::tempdir().unwrap();
    let db = open(dir.path());
    let c = create(&db, 100);
    let mut rows_in = batch(0..5);
    rows_in[3].0 = vec![1.0; DIM + 1];
    match db.insert_many(c, rows_in) {
        Err(Error::DimensionMismatch { expected, got }) => {
            assert_eq!((expected, got), (DIM, DIM + 1));
        }
        other => panic!("expected DimensionMismatch, got {other:?}"),
    }
    assert!(ids(&db).is_empty());
}

#[test]
fn a_malformed_row_anywhere_refuses_the_whole_batch() {
    let dir = tempfile::tempdir().unwrap();
    let db = open(dir.path());
    let c = create(&db, 100);
    let mut rows_in = batch(0..5);
    rows_in[2].1 = vec![(0, Value::Int(2))]; // `tag` missing
    assert!(db.insert_many(c, rows_in).is_err());
    assert!(ids(&db).is_empty());
}

#[test]
fn a_batch_past_capacity_is_refused_and_burns_nothing() {
    let dir = tempfile::tempdir().unwrap();
    let db = open(dir.path());
    let c = create(&db, 10);
    db.insert_many(c, batch(0..8)).unwrap();
    match db.insert_many(c, batch(100..103)) {
        Err(Error::CapacityExceeded { capacity }) => assert_eq!(capacity, 10),
        other => panic!("expected CapacityExceeded, got {other:?}"),
    }
    // The two slots that remain are still there: the refused batch reserved
    // nothing.
    db.insert_many(c, batch(200..202))
        .expect("exactly fills capacity");
    assert_eq!(ids(&db).len(), 10);
}

#[test]
fn refusals_burn_no_capacity() {
    // Dimension, row, and key refusals all happen before any ordinal is
    // claimed, so a collection at its last slots can still be filled.
    let dir = tempfile::tempdir().unwrap();
    let db = open(dir.path());
    let c = create(&db, 3);
    db.insert(c, &vector(0), row(0)).unwrap();
    let mut bad_dim = batch(1..3);
    bad_dim[1].0.push(0.0);
    assert!(db.insert_many(c, bad_dim).is_err());
    assert!(db.insert_many(c, batch([0, 1])).is_err()); // 0 is taken
    assert!(db.insert_many(c, batch([1, 1])).is_err()); // repeated
    db.insert_many(c, batch(1..3))
        .expect("the last two slots are intact");
    assert_eq!(ids(&db), vec![0, 1, 2]);
}

#[test]
fn an_acknowledged_batch_survives_a_crash() {
    let dir = tempfile::tempdir().unwrap();
    {
        let db = open(dir.path());
        let c = create(&db, 100);
        db.insert_many(c, batch(0..40)).unwrap();
        // Dropped without close(): no final checkpoint, so the rows come back
        // from WAL replay alone.
    }
    let db = open(dir.path());
    assert_eq!(ids(&db), (0..40).collect::<Vec<_>>());
    let c = db.collection_id("docs").unwrap();
    // And the allocator resumed past them.
    let more = db.insert_many(c, batch(40..42)).unwrap();
    assert_eq!(more.iter().map(|o| o.0).collect::<Vec<_>>(), vec![40, 41]);
}

#[test]
fn a_keyless_collection_takes_a_batch_too() {
    let dir = tempfile::tempdir().unwrap();
    let db = open(dir.path());
    // V-SQL insists on a key; the engine API does not.
    let schema = flats::Schema::from_columns(vec![flats::ColumnSpec::Vector {
        name: "vector".into(),
        dim: std::num::NonZeroUsize::new(2).unwrap(),
    }])
    .unwrap();
    let c = db.create_collection("v", 8, schema).unwrap();
    let ordinals = db
        .insert_many(c, vec![(vec![1.0, 0.0], vec![]), (vec![0.0, 1.0], vec![])])
        .unwrap();
    assert_eq!(ordinals.len(), 2);
    assert_eq!(db.search(c, &[0.0, 1.0], 2).unwrap().len(), 2);
}

#[test]
fn concurrent_batches_racing_on_the_same_keys_never_both_land() {
    // Whichever way the race falls — the caller-side check catching it, or
    // both passing it and the commit thread deciding — a key ends up held by
    // exactly one live row, and each batch is all-or-nothing.
    let dir = tempfile::tempdir().unwrap();
    let db = open(dir.path());
    let c = create(&db, 10_000);
    for round in 0..40i64 {
        let keys: Vec<i64> = (round * 10..round * 10 + 10).collect();
        let barrier = std::sync::Barrier::new(2);
        let wins: usize = std::thread::scope(|s| {
            let a = s.spawn(|| {
                barrier.wait();
                db.insert_many(c, batch(keys.clone())).is_ok() as usize
            });
            let b = s.spawn(|| {
                barrier.wait();
                db.insert_many(c, batch(keys.clone())).is_ok() as usize
            });
            a.join().unwrap() + b.join().unwrap()
        });
        assert_eq!(wins, 1, "round {round}: exactly one batch wins");
    }
    let got = ids(&db);
    assert_eq!(
        got,
        (0..400).collect::<Vec<_>>(),
        "every key held exactly once"
    );
}

// -- the commit thread's check, reached through Db::mutate -------------------

#[test]
fn mutate_installing_a_live_key_is_rejected_on_the_commit_thread() {
    let dir = tempfile::tempdir().unwrap();
    let db = open(dir.path());
    let c = create(&db, 100);
    db.insert(c, &vector(5), row(5)).unwrap();
    let install = vec![Install {
        ordinal: 1,
        vector: vector(5),
        metadata: row(5),
    }];
    let outcome = db.mutate(c, RoaringBitmap::new(), install).expect("logged");
    assert_eq!(outcome, Outcome::Rejected(RejectReason::DuplicateKey));
    assert_eq!(ids(&db), vec![5], "the key is still held exactly once");
    assert!(!db.is_poisoned(), "a refusal is not an apply failure");
    db.insert(c, &vector(6), row(6))
        .expect("the log keeps going");
}

#[test]
fn mutate_installing_the_same_key_twice_is_rejected() {
    let dir = tempfile::tempdir().unwrap();
    let db = open(dir.path());
    let c = create(&db, 100);
    let install = vec![
        Install {
            ordinal: 0,
            vector: vector(9),
            metadata: row(9),
        },
        Install {
            ordinal: 1,
            vector: vector(9),
            metadata: row(9),
        },
    ];
    let outcome = db.mutate(c, RoaringBitmap::new(), install).expect("logged");
    assert_eq!(outcome, Outcome::Rejected(RejectReason::DuplicateKey));
    assert!(ids(&db).is_empty());
}

#[test]
fn mutate_reinstalling_a_key_it_retires_is_still_an_update() {
    // What UPDATE logs: retire the old row, install its copy under the same
    // key. The key is taken only by a row the record itself retires, so this
    // is not a duplicate.
    let dir = tempfile::tempdir().unwrap();
    let db = open(dir.path());
    let c = create(&db, 100);
    let o = db.insert(c, &vector(3), row(3)).unwrap();
    let mut retire = RoaringBitmap::new();
    retire.insert(o.0);
    let install = vec![Install {
        ordinal: 1,
        vector: vector(33),
        metadata: row(3),
    }];
    assert_eq!(db.mutate(c, retire, install).unwrap(), Outcome::Applied);
    let got = rows(&db);
    assert_eq!(got.len(), 1);
    assert_eq!(got[0].2, vector(33));
}

#[test]
fn a_rejected_mutate_is_rejected_again_on_replay() {
    let dir = tempfile::tempdir().unwrap();
    {
        let db = open(dir.path());
        let c = create(&db, 100);
        db.insert(c, &vector(5), row(5)).unwrap();
        let install = vec![Install {
            ordinal: 1,
            vector: vector(55),
            metadata: row(5),
        }];
        let outcome = db.mutate(c, RoaringBitmap::new(), install).unwrap();
        assert_eq!(outcome, Outcome::Rejected(RejectReason::DuplicateKey));
    } // no checkpoint: both records replay on open
    let db = open(dir.path());
    let got = rows(&db);
    assert_eq!(got.len(), 1, "replay reached the same verdict");
    assert_eq!(got[0].2, vector(5), "and kept the original row");
}

// -- INSERT … VALUES (…), (…) ------------------------------------------------

fn vector_sql(v: &[f32]) -> String {
    let parts: Vec<String> = v.iter().map(|x| format!("{x:?}")).collect();
    format!("[{}]", parts.join(", "))
}

fn values_sql(ids: impl IntoIterator<Item = i64>) -> String {
    ids.into_iter()
        .map(|id| format!("({id}, {}, 't{id}')", vector_sql(&vector(id))))
        .collect::<Vec<_>>()
        .join(", ")
}

#[test]
fn a_multi_row_insert_lands_every_row() {
    let dir = tempfile::tempdir().unwrap();
    let db = open(dir.path());
    create(&db, 100);
    db.execute(&format!(
        "INSERT INTO docs (id, vector, tag) VALUES {};",
        values_sql(0..25)
    ))
    .expect("multi-row insert");
    let got = rows(&db);
    assert_eq!(got.len(), 25);
    for (id, tag, v) in got {
        assert_eq!(tag, format!("t{id}"));
        assert_eq!(v, vector(id));
    }
}

#[test]
fn a_multi_row_insert_may_name_its_columns_in_any_order() {
    let dir = tempfile::tempdir().unwrap();
    let db = open(dir.path());
    create(&db, 100);
    db.execute(
        "INSERT INTO docs (tag, id, vector) VALUES \
         ('a', 1, [1, 0, 0, 0]), ('b', 2, [0, 1, 0, 0]);",
    )
    .unwrap();
    let got = rows(&db);
    assert_eq!(
        got.iter()
            .map(|(id, tag, _)| (*id, tag.as_str()))
            .collect::<Vec<_>>(),
        vec![(1, "a"), (2, "b")]
    );
}

#[test]
fn a_multi_row_insert_with_a_repeated_key_lands_nothing() {
    let dir = tempfile::tempdir().unwrap();
    let db = open(dir.path());
    create(&db, 100);
    let err = db
        .execute(&format!(
            "INSERT INTO docs (id, vector, tag) VALUES {};",
            values_sql([1, 2, 1])
        ))
        .err()
        .expect("a repeated key must be refused");
    assert!(err.to_string().contains('1'), "names the key: {err}");
    assert!(ids(&db).is_empty());
}

#[test]
fn a_multi_row_insert_with_one_bad_row_lands_nothing() {
    let dir = tempfile::tempdir().unwrap();
    let db = open(dir.path());
    create(&db, 100);
    // Row 2 has the wrong arity, row 3 the wrong type: each is a bind error.
    for bad in [
        "(2, [0, 1, 0, 0])",
        "(2, [0, 1, 0, 0], 7)",
        "(2, [0, 1, 0], 'x')",
    ] {
        let sql =
            format!("INSERT INTO docs (id, vector, tag) VALUES (1, [1, 0, 0, 0], 'a'), {bad};");
        assert!(db.execute(&sql).is_err(), "{sql} must be refused");
    }
    assert!(ids(&db).is_empty());
}

#[test]
fn a_multi_row_insert_past_capacity_lands_nothing() {
    let dir = tempfile::tempdir().unwrap();
    let db = open(dir.path());
    create(&db, 3);
    let sql = format!(
        "INSERT INTO docs (id, vector, tag) VALUES {};",
        values_sql(0..4)
    );
    assert!(db.execute(&sql).is_err());
    assert!(ids(&db).is_empty());
    let sql = format!(
        "INSERT INTO docs (id, vector, tag) VALUES {};",
        values_sql(0..3)
    );
    db.execute(&sql).expect("three fit");
    assert_eq!(ids(&db), vec![0, 1, 2]);
}
