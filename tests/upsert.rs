//! `UPSERT INTO … VALUES …`: insert each row, or REPLACE the live row that
//! holds its primary key — every column and the vector — as ONE statement.

use std::collections::BTreeMap;
use std::path::Path;
use std::sync::Arc;
use std::time::Duration;

use flats::{CompactionOptions, Db, DbOptions, Param, RegValue};

fn opts() -> DbOptions {
    DbOptions {
        checkpoint_interval: Duration::from_secs(3600),
        compaction: CompactionOptions {
            enabled: false,
            ..Default::default()
        },
    }
}

fn open(dir: &Path) -> Db {
    let db = Db::open(dir, &[], opts()).unwrap();
    if db.collections().is_empty() {
        db.execute("CREATE COLLECTION t (id TEXT PRIMARY KEY, vector VECTOR(2), n INT, s TEXT) WITH (capacity = 10000);")
            .unwrap();
    }
    db
}

/// id → (n, s, vector)
fn rows(db: &Db) -> BTreeMap<String, (i64, String, Vec<f32>)> {
    db.execute("SELECT id, n, s, vector FROM t;")
        .unwrap()
        .map(|r| match r.unwrap().0.as_slice() {
            [
                RegValue::Str(id),
                RegValue::Int(n),
                RegValue::Str(s),
                RegValue::Vector(v),
            ] => (id.clone(), (*n, s.clone(), v.to_vec())),
            other => panic!("{other:?}"),
        })
        .collect()
}

fn row(n: i64, s: &str, v: [f32; 2]) -> (i64, String, Vec<f32>) {
    (n, s.to_string(), v.to_vec())
}

#[test]
fn upsert_inserts_rows_whose_keys_are_new() {
    let dir = tempfile::tempdir().unwrap();
    let db = open(dir.path());
    db.execute("UPSERT INTO t (id, vector, n, s) VALUES ('a', [1, 0], 1, 'x');")
        .unwrap();
    db.execute(
        "UPSERT INTO t (id, vector, n, s) VALUES ('b', [0, 1], 2, 'y'), ('c', [1, 1], 3, 'z');",
    )
    .unwrap();
    assert_eq!(
        rows(&db),
        BTreeMap::from([
            ("a".into(), row(1, "x", [1.0, 0.0])),
            ("b".into(), row(2, "y", [0.0, 1.0])),
            ("c".into(), row(3, "z", [1.0, 1.0])),
        ])
    );
}

#[test]
fn upsert_replaces_the_whole_row_that_holds_the_key() {
    let dir = tempfile::tempdir().unwrap();
    let db = open(dir.path());
    db.execute("INSERT INTO t (id, vector, n, s) VALUES ('a', [1, 0], 1, 'old');")
        .unwrap();
    db.execute("UPSERT INTO t (s, id, n, vector) VALUES ('new', 'a', 9, [0, 1]);")
        .unwrap();
    assert_eq!(
        rows(&db),
        BTreeMap::from([("a".into(), row(9, "new", [0.0, 1.0]))])
    );
    // The old version is gone from the filter index and from the ranking.
    assert_eq!(
        db.execute("SELECT id FROM t WHERE s = 'old';")
            .unwrap()
            .count(),
        0
    );
    assert_eq!(
        db.execute("SELECT id FROM t WHERE n = 9;").unwrap().count(),
        1
    );
    let top: Vec<_> = db
        .execute("SEARCH TOP 5 NEAREST TO [0, 1] FROM t RETURNING id, score;")
        .unwrap()
        .map(|r| format!("{:?}", r.unwrap().0))
        .collect();
    assert_eq!(top, vec![r#"[Str("a"), Real(1.0)]"#.to_string()]);
}

#[test]
fn a_mixed_batch_replaces_some_and_inserts_others_in_one_statement() {
    let dir = tempfile::tempdir().unwrap();
    let db = open(dir.path());
    db.execute(
        "INSERT INTO t (id, vector, n, s) VALUES ('a', [1, 0], 1, 'x'), ('b', [0, 1], 2, 'y');",
    )
    .unwrap();
    db.execute(
        "UPSERT INTO t (id, vector, n, s) VALUES ('b', [2, 2], 20, 'B'), ('c', [3, 3], 30, 'C');",
    )
    .unwrap();
    assert_eq!(
        rows(&db),
        BTreeMap::from([
            ("a".into(), row(1, "x", [1.0, 0.0])),
            ("b".into(), row(20, "B", [2.0, 2.0])),
            ("c".into(), row(30, "C", [3.0, 3.0])),
        ])
    );
}

#[test]
fn a_key_repeated_in_one_upsert_is_refused_and_changes_nothing() {
    let dir = tempfile::tempdir().unwrap();
    let db = open(dir.path());
    db.execute("INSERT INTO t (id, vector, n, s) VALUES ('a', [1, 0], 1, 'x');")
        .unwrap();
    let before = rows(&db);
    assert!(
        db.execute(
            "UPSERT INTO t (id, vector, n, s) VALUES ('a', [0, 1], 2, 'y'), ('a', [1, 1], 3, 'z');"
        )
        .is_err()
    );
    assert!(
        db.execute(
            "UPSERT INTO t (id, vector, n, s) VALUES ('q', [0, 1], 2, 'y'), ('q', [1, 1], 3, 'z');"
        )
        .is_err()
    );
    assert_eq!(rows(&db), before);
}

#[test]
fn upsert_refuses_what_insert_refuses() {
    let dir = tempfile::tempdir().unwrap();
    let db = open(dir.path());
    for bad in [
        "UPSERT INTO t (id, vector, n) VALUES ('a', [1, 0], 1);", // missing column
        "UPSERT INTO t (id, vector, n, s) VALUES ('a', [1, 0, 0], 1, 'x');", // wrong dimension
        "UPSERT INTO t (id, vector, n, s) VALUES ('a', [1, 0], 1.5, 'x');", // wrong type
        "UPSERT INTO nope (id, vector, n, s) VALUES ('a', [1, 0], 1, 'x');", // no collection
        "UPSERT t (id, vector, n, s) VALUES ('a', [1, 0], 1, 'x');", // no INTO
    ] {
        assert!(db.execute(bad).is_err(), "{bad}");
    }
    assert!(rows(&db).is_empty());
}

#[test]
fn upsert_takes_parameters() {
    let dir = tempfile::tempdir().unwrap();
    let db = open(dir.path());
    for n in 0..3 {
        db.execute_with(
            "UPSERT INTO t (id, vector, n, s) VALUES (?, ?, ?, ?);",
            &[
                Param::Text("k".into()),
                Param::Vector(vec![n as f32, 1.0]),
                Param::Int(n),
                Param::Text(format!("v{n}")),
            ],
        )
        .unwrap();
    }
    assert_eq!(
        rows(&db),
        BTreeMap::from([("k".into(), row(2, "v2", [2.0, 1.0]))])
    );
}

#[test]
fn upserts_survive_checkpoints_compaction_and_crashes() {
    let dir = tempfile::tempdir().unwrap();
    let want;
    {
        let db = open(dir.path());
        for round in 0..5i64 {
            for k in 0..20i64 {
                db.execute_with(
                    "UPSERT INTO t (id, vector, n, s) VALUES (?, ?, ?, ?);",
                    &[
                        Param::Text(format!("k{k}")),
                        Param::Vector(vec![round as f32, k as f32]),
                        Param::Int(round * 100 + k),
                        Param::Text(format!("r{round}")),
                    ],
                )
                .unwrap();
            }
            if round == 2 {
                db.checkpoint().unwrap();
            }
            if round == 3 {
                let id = db.collections()[0].id;
                assert!(db.compact_now(id).unwrap(), "upserts left dead rows behind");
            }
        }
        want = rows(&db);
        assert_eq!(want.len(), 20);
        assert!(want.values().all(|(n, s, _)| *n >= 400 && s == "r4"));
        drop(db); // crash
    }
    let db = open(dir.path());
    assert_eq!(rows(&db), want);
}

#[test]
fn racing_upserts_of_one_key_leave_exactly_one_row() {
    let dir = tempfile::tempdir().unwrap();
    let db = Arc::new(open(dir.path()));
    let threads: Vec<_> = (0..4)
        .map(|t| {
            let db = db.clone();
            std::thread::spawn(move || {
                for i in 0..50i64 {
                    let n = t * 1000 + i;
                    db.execute_with(
                        "UPSERT INTO t (id, vector, n, s) VALUES ('hot', ?, ?, ?);",
                        &[
                            Param::Vector(vec![n as f32, 0.0]),
                            Param::Int(n),
                            Param::Text(format!("t{t}")),
                        ],
                    )
                    .unwrap_or_else(|e| panic!("thread {t}: {e}"));
                }
            })
        })
        .collect();
    for t in threads {
        t.join().unwrap();
    }
    let all = rows(&db);
    assert_eq!(all.len(), 1, "one key, one row: {all:?}");
    let (n, s, v) = &all["hot"];
    // Whole rows, never a mix of two upserts.
    assert_eq!(s, &format!("t{}", n / 1000));
    assert_eq!(v, &vec![*n as f32, 0.0]);
    assert_eq!(n % 1000, 49, "the last write of some thread won");
}

/// `Record::Upsert` is APPENDED to the WAL's record enum: every older record
/// keeps its positional tag (pinned by the WAL's own tests), so a v5 or v6 log
/// replays unchanged, and the new record takes the next tag.
#[test]
fn the_upsert_record_takes_the_next_tag_in_the_log() {
    use flats::wal::wal::Record;
    let rec = Record::Upsert {
        collection: 3,
        install: vec![],
    };
    let mut buf = Vec::new();
    bincode::serialize_into(&mut buf, &rec).unwrap();
    assert_eq!(
        &buf[..4],
        &5u32.to_le_bytes(),
        "bincode tags variants positionally"
    );
}
