//! One process, one handle, per database directory.
//!
//! Two handles writing the same files would each believe they are the single
//! writer — two WAL appenders, two appliers, two checkpointers — and corrupt
//! the database between them. `Db::open` takes an exclusive OS lock on the
//! directory and holds it for the handle's life; the OS drops it when the
//! process ends, so a crash never leaves a database locked.

use std::io::{BufRead, BufReader, Write};
use std::process::{Command, Stdio};
use std::time::{Duration, Instant};

use flats::{Db, DbOptions, Error};

fn open(dir: &std::path::Path) -> flats::Result<Db> {
    Db::open(dir, &[], DbOptions::default())
}

#[test]
fn a_second_handle_in_the_same_process_is_refused() {
    let dir = tempfile::tempdir().unwrap();
    let first = open(dir.path()).unwrap();
    match open(dir.path()) {
        Err(Error::DatabaseLocked { path }) => {
            assert!(path.contains(&dir.path().display().to_string()), "{path}");
        }
        Err(other) => panic!("expected DatabaseLocked, got {other:?}"),
        Ok(_) => panic!("two handles on one directory"),
    }
    // The refused open changed nothing: the first handle still works.
    first
        .execute("CREATE COLLECTION c (id INT PRIMARY KEY, vector VECTOR(2)) WITH (capacity = 4);")
        .unwrap();
}

#[test]
fn the_lock_is_released_by_close_and_by_drop() {
    let dir = tempfile::tempdir().unwrap();
    open(dir.path()).unwrap().close().unwrap();
    let again = open(dir.path()).expect("free after close");
    drop(again);
    open(dir.path()).expect("free after drop");
}

#[test]
fn a_failed_open_does_not_keep_the_lock() {
    let dir = tempfile::tempdir().unwrap();
    open(dir.path()).unwrap().close().unwrap();
    let snap = dir.path().join("catalog.snap");
    let good = std::fs::read(&snap).unwrap();
    std::fs::write(&snap, b"garbage").unwrap();
    assert!(matches!(open(dir.path()), Err(Error::CorruptSnapshot(_))));
    std::fs::write(&snap, good).unwrap();
    open(dir.path()).expect("the failed open released the lock");
}

/// The holder process for the test below: this same test binary, re-run with
/// only this (normally ignored) test selected. It opens the database, says so,
/// and waits to be killed.
#[test]
#[ignore = "helper: run by another_process_holding_the_database_is_refused_until_it_exits"]
fn hold_the_database_until_killed() {
    let Ok(dir) = std::env::var("FLATS_LOCK_HOLDER_DIR") else {
        return;
    };
    let _db = open(std::path::Path::new(&dir)).expect("holder opens");
    println!("HOLDER-READY");
    std::io::stdout().flush().unwrap();
    std::thread::sleep(Duration::from_secs(120));
}

#[test]
fn another_process_holding_the_database_is_refused_until_it_exits() {
    let dir = tempfile::tempdir().unwrap();
    let mut child = Command::new(std::env::current_exe().unwrap())
        .args([
            "--ignored",
            "--exact",
            "hold_the_database_until_killed",
            "--nocapture",
        ])
        .env("FLATS_LOCK_HOLDER_DIR", dir.path())
        .stdout(Stdio::piped())
        .stderr(Stdio::null())
        .spawn()
        .expect("spawn the holder");
    // Wait until it has the database open.
    let mut out = BufReader::new(child.stdout.take().unwrap());
    let mut line = String::new();
    loop {
        line.clear();
        if out.read_line(&mut line).unwrap() == 0 {
            panic!("the holder exited before opening the database");
        }
        if line.contains("HOLDER-READY") {
            break;
        }
    }

    match open(dir.path()) {
        Err(Error::DatabaseLocked { .. }) => {}
        Err(other) => panic!("expected DatabaseLocked, got {other:?}"),
        Ok(_) => panic!("opened a database another process holds"),
    }

    // Kill it — no clean shutdown. The OS releases the lock with the process.
    child.kill().unwrap();
    child.wait().unwrap();
    let deadline = Instant::now() + Duration::from_secs(5);
    loop {
        match open(dir.path()) {
            Ok(_) => break,
            Err(Error::DatabaseLocked { .. }) if Instant::now() < deadline => {
                std::thread::sleep(Duration::from_millis(20))
            }
            Err(e) => panic!("still locked after the holder died: {e:?}"),
        }
    }
}

#[test]
fn the_lock_file_is_not_part_of_a_backup() {
    let dir = tempfile::tempdir().unwrap();
    let dst = tempfile::tempdir().unwrap();
    let copy = dst.path().join("copy");
    let db = open(dir.path()).unwrap();
    db.backup(&copy).unwrap();
    assert!(!copy.join("LOCK").exists());
    // And the copy opens while the source is still held.
    open(&copy).expect("the backup is its own database");
}
