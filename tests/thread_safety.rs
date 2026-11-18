//! The thread-safety guarantees the bindings rely on, pinned at compile time.
//!
//! * `Db: Send + Sync` — one handle shared by every thread (the C ABI's
//!   "a flats_db may be shared across threads", Python's GIL-free calls).
//! * `RowStream: Send` — a cursor made on one thread and read on another
//!   (a C cursor passed between threads, an async call finishing on a pool).
//!
//! These are compile-time facts, so the "test" is that this file builds: a
//! non-`Send` field added anywhere under them fails here, by name, instead of
//! surfacing as a soundness bug in a binding.

use flats::{Cursor, Db, RowStream};

fn assert_send<T: Send>() {}
fn assert_sync<T: Sync>() {}

#[test]
fn the_handles_cross_threads() {
    assert_send::<Db>();
    assert_sync::<Db>();
    assert_send::<RowStream>();
    assert_send::<Cursor<'static>>();
}

#[test]
fn a_stream_made_on_one_thread_reads_on_another() {
    let dir = tempfile::tempdir().unwrap();
    let db = Db::open(dir.path(), &[], flats::DbOptions::default()).unwrap();
    db.execute("CREATE COLLECTION t (id INT PRIMARY KEY, vector VECTOR(2)) WITH (capacity = 8);")
        .unwrap();
    db.execute("INSERT INTO t (id, vector) VALUES (1, [1, 0]), (2, [0, 1]);")
        .unwrap();
    let stream = db.execute("SELECT id FROM t;").unwrap();
    let n = std::thread::spawn(move || stream.count()).join().unwrap();
    assert_eq!(n, 2);
}
