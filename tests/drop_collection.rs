//! DROP COLLECTION, end to end.
//!
//! A drop is DDL riding the WAL like CREATE: one durable record, applied on
//! the commit thread. What it promises:
//!
//!   * the collection is gone — from the catalog, from every read path, and
//!     from disk — and stays gone across a crash and replay;
//!   * its id is never handed out again, so a log that still holds the dropped
//!     collection's records can never apply them to a newcomer;
//!   * its NAME is free at once;
//!   * nothing else moves: other collections, their rows, the log's health.

use std::path::Path;
use std::time::Duration;

use flats::{Db, DbOptions, Error, RegValue, Value};

fn opts() -> DbOptions {
    // Checkpoints only when a test asks, so "restart" means WAL replay.
    DbOptions {
        checkpoint_interval: Duration::from_secs(3600),
        ..DbOptions::default()
    }
}

fn open(dir: &Path) -> Db {
    Db::open(dir, &[], opts()).expect("open")
}

fn create(db: &Db, name: &str) -> u32 {
    db.execute(&format!(
        "CREATE COLLECTION {name} (id INT PRIMARY KEY, vector VECTOR(2), n INT) \
         WITH (capacity = 64);"
    ))
    .expect("create");
    db.collection_id(name).expect("id")
}

fn insert(db: &Db, name: &str, id: i64) {
    db.execute(&format!(
        "INSERT INTO {name} (id, vector, n) VALUES ({id}, [{id}, 1], {id});"
    ))
    .expect("insert");
}

fn ids(db: &Db, name: &str) -> Vec<i64> {
    let mut out: Vec<i64> = db
        .execute(&format!("SELECT id FROM {name};"))
        .expect("select")
        .map(|r| match r.expect("row").0.as_slice() {
            [RegValue::Int(id)] => *id,
            other => panic!("unexpected row {other:?}"),
        })
        .collect();
    out.sort_unstable();
    out
}

fn names(db: &Db) -> Vec<String> {
    let mut out: Vec<String> = db.collections().into_iter().map(|c| c.name).collect();
    out.sort();
    out
}

/// Every `collection-*` entry in the database directory.
fn collection_dirs(dir: &Path) -> Vec<String> {
    let mut out: Vec<String> = std::fs::read_dir(dir)
        .unwrap()
        .map(|e| e.unwrap().file_name().to_string_lossy().into_owned())
        .filter(|n| n.starts_with("collection-"))
        .collect();
    out.sort();
    out
}

#[test]
fn drop_collection_removes_it_from_every_read_path_and_from_disk() {
    let dir = tempfile::tempdir().unwrap();
    let db = open(dir.path());
    let id = create(&db, "docs");
    insert(&db, "docs", 1);
    assert_eq!(
        collection_dirs(dir.path()),
        vec![format!("collection-{id}")]
    );

    db.execute("DROP COLLECTION docs;").expect("drop");

    assert!(names(&db).is_empty());
    assert!(db.collection_id("docs").is_err());
    assert!(db.execute("SELECT id FROM docs;").is_err());
    assert!(db.search(id, &[1.0, 0.0], 1).is_err());
    assert!(collection_dirs(dir.path()).is_empty(), "its files are gone");
}

#[test]
fn drop_collection_through_the_engine_api() {
    let dir = tempfile::tempdir().unwrap();
    let db = open(dir.path());
    create(&db, "docs");
    db.drop_collection("docs").expect("drop");
    assert!(names(&db).is_empty());
    match db.drop_collection("docs") {
        Err(Error::UnknownCollectionName { name }) => assert_eq!(name, "docs"),
        other => panic!("expected UnknownCollectionName, got {other:?}"),
    }
}

#[test]
fn dropping_an_unknown_collection_is_an_error() {
    let dir = tempfile::tempdir().unwrap();
    let db = open(dir.path());
    let err = db
        .execute("DROP COLLECTION nope;")
        .err()
        .expect("unknown collection");
    assert!(err.to_string().contains("nope"), "names it: {err}");
}

#[test]
fn other_collections_are_untouched() {
    let dir = tempfile::tempdir().unwrap();
    {
        let db = open(dir.path());
        create(&db, "a");
        create(&db, "b");
        create(&db, "c");
        for i in 0..5 {
            insert(&db, "a", i);
            insert(&db, "b", i);
            insert(&db, "c", i);
        }
        db.execute("DROP COLLECTION b;").unwrap();
        assert_eq!(names(&db), vec!["a", "c"]);
        assert_eq!(ids(&db, "a"), (0..5).collect::<Vec<_>>());
        assert_eq!(ids(&db, "c"), (0..5).collect::<Vec<_>>());
        // Writes to the survivors keep working.
        insert(&db, "a", 5);
    }
    let db = open(dir.path());
    assert_eq!(names(&db), vec!["a", "c"]);
    assert_eq!(ids(&db, "a"), (0..6).collect::<Vec<_>>());
    assert_eq!(ids(&db, "c"), (0..5).collect::<Vec<_>>());
}

#[test]
fn a_drop_survives_a_crash_and_replay_does_not_resurrect_it() {
    let dir = tempfile::tempdir().unwrap();
    {
        let db = open(dir.path());
        create(&db, "docs");
        for i in 0..10 {
            insert(&db, "docs", i);
        }
        db.execute("DROP COLLECTION docs;").unwrap();
        // Dropped without close(): the create, the inserts and the drop are
        // all still in the log and replay on open.
    }
    let db = open(dir.path());
    assert!(names(&db).is_empty());
    assert!(collection_dirs(dir.path()).is_empty());
    assert!(!db.is_poisoned());
}

#[test]
fn the_name_is_free_at_once_and_the_newcomer_starts_empty() {
    let dir = tempfile::tempdir().unwrap();
    {
        let db = open(dir.path());
        let old = create(&db, "docs");
        for i in 0..10 {
            insert(&db, "docs", i);
        }
        db.execute("DROP COLLECTION docs;").unwrap();
        let new = create(&db, "docs");
        assert_ne!(old, new, "a dropped id is never handed out again");
        assert!(ids(&db, "docs").is_empty());
        insert(&db, "docs", 100);
        assert_eq!(ids(&db, "docs"), vec![100]);
    }
    // The log now holds the OLD collection's create and inserts, its drop,
    // and the newcomer under the same name. Replay must keep them apart.
    let db = open(dir.path());
    assert_eq!(names(&db), vec!["docs"]);
    assert_eq!(ids(&db, "docs"), vec![100]);
}

#[test]
fn ids_are_never_reused_even_when_the_highest_is_dropped() {
    let dir = tempfile::tempdir().unwrap();
    {
        let db = open(dir.path());
        assert_eq!(create(&db, "a"), 0);
        assert_eq!(create(&db, "b"), 1);
        db.execute("DROP COLLECTION b;").unwrap();
        assert_eq!(create(&db, "c"), 2);
        db.execute("DROP COLLECTION c;").unwrap();
        db.close().unwrap();
    }
    // ...across a CLEAN restart too, where the log has been checkpointed away
    // and only the catalog remembers that ids 1 and 2 were ever issued.
    let db = open(dir.path());
    assert_eq!(create(&db, "d"), 3);
}

#[test]
fn a_write_holding_a_dropped_collection_id_is_refused_not_fatal() {
    let dir = tempfile::tempdir().unwrap();
    let db = open(dir.path());
    let id = create(&db, "docs");
    db.drop_collection("docs").unwrap();
    let row = vec![(0, Value::Int(1)), (1, Value::Int(1))];
    match db.insert(id, &[1.0, 0.0], row.clone()) {
        Err(Error::UnknownCollection { id: got }) => assert_eq!(got, id),
        other => panic!("expected UnknownCollection, got {other:?}"),
    }
    assert!(db.insert_many(id, vec![(vec![1.0, 0.0], row)]).is_err());
    assert!(!db.is_poisoned());
}

#[test]
fn writers_racing_a_drop_never_poison_the_log() {
    // Inserts in flight when the drop lands either commit before it (and are
    // dropped with the collection) or are refused after it. Neither path may
    // be an apply failure.
    let dir = tempfile::tempdir().unwrap();
    {
        let db = open(dir.path());
        create(&db, "docs");
        create(&db, "keep");
        std::thread::scope(|s| {
            for t in 0..4i64 {
                let db = &db;
                s.spawn(move || {
                    for i in 0..200 {
                        let id = t * 1000 + i;
                        let _ = db.execute(&format!(
                            "INSERT INTO docs (id, vector, n) VALUES ({id}, [1, 0], 0);"
                        ));
                    }
                });
            }
            std::thread::sleep(Duration::from_millis(20));
            db.execute("DROP COLLECTION docs;").unwrap();
        });
        assert!(!db.is_poisoned());
        insert(&db, "keep", 1);
    }
    let db = open(dir.path());
    assert!(!db.is_poisoned());
    assert_eq!(names(&db), vec!["keep"]);
    assert_eq!(ids(&db, "keep"), vec![1]);
}

#[test]
fn a_read_already_running_finishes_against_what_it_started_with() {
    let dir = tempfile::tempdir().unwrap();
    let db = open(dir.path());
    create(&db, "docs");
    for i in 0..20 {
        insert(&db, "docs", i);
    }
    let mut stream = db.execute("SELECT id FROM docs;").unwrap();
    let first = stream.next().expect("a row").expect("ok");
    db.execute("DROP COLLECTION docs;").unwrap();
    // The stream holds its own handles on the stores; unlinked files stay
    // mapped until the last of them lets go.
    let rest: Vec<_> = stream.map(|r| r.expect("row")).collect();
    assert_eq!(1 + rest.len(), 20);
    drop(first);
}

#[test]
fn leftovers_of_an_interrupted_drop_are_cleaned_up_on_open() {
    // A crash after the catalog forgot the collection but before its files
    // were deleted leaves the directory behind (and compaction may have left
    // a staging copy). Open must clear them rather than trip over them.
    let dir = tempfile::tempdir().unwrap();
    let id;
    {
        let db = open(dir.path());
        id = create(&db, "docs");
        create(&db, "keep");
        db.execute("DROP COLLECTION docs;").unwrap();
        // Move every store past the drop so the clean close truncates the log
        // beyond it: replaying the drop would clean up too, and this test is
        // about the open that has nothing left to replay.
        insert(&db, "keep", 1);
        db.close().unwrap();
    }
    let wal = std::fs::read(dir.path().join("wal.log")).unwrap();
    assert!(
        wal.len() <= 8,
        "the log was truncated to its header ({} bytes)",
        wal.len()
    );
    for leftover in [
        format!("collection-{id}"),
        format!("collection-{id}.compact"),
    ] {
        let p = dir.path().join(leftover);
        std::fs::create_dir_all(&p).unwrap();
        std::fs::write(p.join("vectors.idx"), b"junk").unwrap();
    }
    let db = open(dir.path());
    assert_eq!(names(&db), vec!["keep"]);
    assert_eq!(
        collection_dirs(dir.path()),
        vec!["collection-1".to_string()]
    );
}

#[test]
fn drop_and_compaction_on_the_same_collection_do_not_collide() {
    let dir = tempfile::tempdir().unwrap();
    {
        let db = open(dir.path());
        for round in 0..10 {
            let name = format!("c{round}");
            let id = create(&db, &name);
            for i in 0..40 {
                insert(&db, &name, i);
            }
            db.execute(&format!("DELETE FROM {name} WHERE n < 30;"))
                .unwrap();
            std::thread::scope(|s| {
                let db = &db;
                s.spawn(move || {
                    let _ = db.compact_now(id);
                });
                let _ = db.drop_collection(&name);
            });
            assert!(!db.is_poisoned(), "round {round}");
        }
        assert!(names(&db).is_empty());
    }
    let db = open(dir.path());
    assert!(names(&db).is_empty());
    assert!(collection_dirs(dir.path()).is_empty());
}

#[test]
fn an_open_time_config_cannot_revive_a_dropped_id() {
    let dir = tempfile::tempdir().unwrap();
    let config;
    {
        let db = open(dir.path());
        create(&db, "docs");
        config = db.collections().remove(0);
        db.drop_collection("docs").unwrap();
        db.close().unwrap();
    }
    match Db::open(dir.path(), &[config], opts()) {
        Err(Error::CollectionConfigMismatch { id }) => assert_eq!(id, 0),
        Err(other) => panic!("expected CollectionConfigMismatch, got {other:?}"),
        Ok(_) => panic!("a dropped id must not be re-registered"),
    }
}
