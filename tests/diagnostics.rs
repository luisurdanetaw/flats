//! The logging hook and `Db::stats`.

use std::sync::{Arc, Mutex};

use flats::diagnostics::{self, Level};
use flats::{Db, DbOptions};

/// The hook is process-wide: one test at a time installs it.
static SERIAL: Mutex<()> = Mutex::new(());

type Seen = Arc<Mutex<Vec<(Level, String)>>>;

fn capture() -> (std::sync::MutexGuard<'static, ()>, Seen) {
    let g = SERIAL.lock().unwrap_or_else(|e| e.into_inner());
    let seen = Arc::new(Mutex::new(Vec::new()));
    let sink = seen.clone();
    diagnostics::set_log_hook(move |level, msg| {
        sink.lock().unwrap().push((level, msg.to_string()))
    });
    (g, seen)
}

#[test]
fn a_store_rebuilt_from_the_log_is_reported() {
    let (_g, seen) = capture();
    let dir = tempfile::tempdir().unwrap();
    {
        let db = Db::open(dir.path(), &[], DbOptions::default()).unwrap();
        db.execute(
            "CREATE COLLECTION t (id INT PRIMARY KEY, vector VECTOR(2)) WITH (capacity = 10);",
        )
        .unwrap();
        db.execute("INSERT INTO t (id, vector) VALUES (1, [1, 0]);")
            .unwrap();
        drop(db); // nothing checkpointed: the log can rebuild a lost store
    }
    std::fs::write(dir.path().join("collection-0/tuples.snap"), b"garbage").unwrap();
    let db = Db::open(dir.path(), &[], DbOptions::default()).unwrap();
    assert_eq!(db.execute("SELECT id FROM t;").unwrap().count(), 1);
    db.close().unwrap();
    diagnostics::reset_log_hook();
    let seen = seen.lock().unwrap();
    assert!(
        seen.iter()
            .any(|(l, m)| *l == Level::Warn && m.contains("tuples.snap") && m.contains("rebuilt")),
        "{seen:?}"
    );
}

#[test]
fn stats_count_rows_space_checkpoints_and_compactions() {
    let _g = SERIAL.lock().unwrap_or_else(|e| e.into_inner());
    let dir = tempfile::tempdir().unwrap();
    let db = Db::open(
        dir.path(),
        &[],
        DbOptions {
            checkpoint_interval: std::time::Duration::from_secs(3600),
            ..DbOptions::default()
        },
    )
    .unwrap();
    db.execute("CREATE COLLECTION t (id INT PRIMARY KEY, vector VECTOR(2)) WITH (capacity = 100);")
        .unwrap();
    db.execute("INSERT INTO t (id, vector) VALUES (1, [1, 0]), (2, [0, 1]), (3, [1, 1]);")
        .unwrap();
    db.execute("DELETE FROM t WHERE id = 2;").unwrap();

    let s = db.stats();
    assert_eq!(s.collections.len(), 1);
    let c = &s.collections[0];
    assert_eq!(
        (c.name.as_str(), c.rows, c.allocated, c.capacity),
        ("t", 2, 3, 100)
    );
    assert!(s.wal_bytes > 8, "writes are in the log");
    assert_eq!(s.checkpoints, 0);
    assert!(s.halted.is_none());

    db.checkpoint().unwrap();
    let id = db.collections()[0].id;
    assert!(db.compact_now(id).unwrap());
    let s = db.stats();
    assert_eq!(s.checkpoints, 1);
    assert_eq!(s.compactions, 1);
    assert_eq!(
        s.collections[0].allocated, 2,
        "compaction reclaimed the dead row"
    );
    assert_eq!(s.wal_bytes, 8, "checkpoint shed the log");
    db.close().unwrap();
}

#[test]
fn the_default_hook_still_prints_and_reset_restores_it() {
    let (_g, seen) = capture();
    diagnostics::reset_log_hook();
    // With the hook reset, an event goes to stderr, not to the old sink.
    let dir = tempfile::tempdir().unwrap();
    {
        let db = Db::open(dir.path(), &[], DbOptions::default()).unwrap();
        db.execute(
            "CREATE COLLECTION t (id INT PRIMARY KEY, vector VECTOR(2)) WITH (capacity = 10);",
        )
        .unwrap();
        drop(db);
    }
    std::fs::write(dir.path().join("collection-0/tuples.snap"), b"garbage").unwrap();
    Db::open(dir.path(), &[], DbOptions::default())
        .unwrap()
        .close()
        .unwrap();
    assert!(seen.lock().unwrap().is_empty());
}
