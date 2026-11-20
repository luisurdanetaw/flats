//! Resource limits and cancellation.
//!
//! Every limit here exists because exceeding it used to do real harm: deep or
//! long predicates overflowed the stack and killed the process, and an
//! unsatisfiable CREATE was logged, failed to apply, and poisoned the
//! database on every reopen.

use std::time::Duration;

use flats::{Db, DbOptions, Error, Limits, Param, QueryError};

fn open(dir: &std::path::Path) -> Db {
    Db::open(dir, &[], DbOptions::default()).unwrap()
}

fn engine_error(r: Result<flats::RowStream, QueryError>) -> Error {
    match r {
        Err(QueryError::Engine(e)) => e,
        Err(other) => panic!("expected an engine error, got {other:?}"),
        Ok(_) => panic!("expected an error"),
    }
}

/// A collection big enough that one search takes a while, so a deadline can
/// land in the middle of it.
fn big(db: &Db) -> u32 {
    db.execute(
        "CREATE COLLECTION big (id INT PRIMARY KEY, vector VECTOR(64)) WITH (capacity = 300000);",
    )
    .unwrap();
    let c = db.collection_id("big").unwrap();
    let v = vec![0.5f32; 64];
    for chunk in (0..200_000i64).collect::<Vec<_>>().chunks(20_000) {
        db.insert_many(
            c,
            chunk
                .iter()
                .map(|&i| (v.clone(), vec![(0, flats::Value::Int(i))]))
                .collect(),
        )
        .unwrap();
    }
    c
}

#[test]
fn deep_and_long_predicates_are_errors_not_crashes() {
    let dir = tempfile::tempdir().unwrap();
    let db = open(dir.path());
    db.execute(
        "CREATE COLLECTION t (id INT PRIMARY KEY, vector VECTOR(2), n INT) WITH (capacity = 10);",
    )
    .unwrap();
    let deep = format!(
        "SELECT id FROM t WHERE {}n = 1{};",
        "(".repeat(200_000),
        ")".repeat(200_000)
    );
    assert!(matches!(db.execute(&deep), Err(QueryError::Parse(_))));
    let long = format!(
        "SELECT id FROM t WHERE {};",
        (0..200_000)
            .map(|i| format!("n = {i}"))
            .collect::<Vec<_>>()
            .join(" OR ")
    );
    assert!(matches!(db.execute(&long), Err(QueryError::Parse(_))));
}

#[test]
fn the_largest_allowed_predicate_runs_correctly_on_a_small_stack() {
    // The caps are only worth something if what they admit is safe on a
    // thread pool's stack: run the biggest legal chain on 256 KiB.
    let dir = tempfile::tempdir().unwrap();
    let db = open(dir.path());
    db.execute(
        "CREATE COLLECTION t (id INT PRIMARY KEY, vector VECTOR(2), n INT) WITH (capacity = 100);",
    )
    .unwrap();
    for i in 0..50 {
        db.execute_with(
            "INSERT INTO t (id, vector, n) VALUES (?, [1, 0], ?);",
            &[Param::Int(i), Param::Int(i)],
        )
        .unwrap();
    }
    let chain = (0..flats::MAX_EXPR_TERMS)
        .map(|i| format!("n = {}", i * 2))
        .collect::<Vec<_>>()
        .join(" OR ");
    let sql = format!("SELECT id FROM t WHERE {chain};");
    let n = std::thread::scope(|s| {
        std::thread::Builder::new()
            .stack_size(256 * 1024)
            .spawn_scoped(s, || db.execute(&sql).unwrap().count())
            .unwrap()
            .join()
            .unwrap()
    });
    assert_eq!(n, 25, "the even ids below 50");
}

#[test]
fn an_impossible_collection_is_refused_before_it_is_logged() {
    let dir = tempfile::tempdir().unwrap();
    {
        let db = open(dir.path());
        for sql in [
            // Past the dimension cap.
            "CREATE COLLECTION a (id INT PRIMARY KEY, vector VECTOR(1000000000)) WITH (capacity = 10);",
            // Past the ordinal space.
            "CREATE COLLECTION b (id INT PRIMARY KEY, vector VECTOR(4)) WITH (capacity = 5000000000);",
            // Past the per-collection size cap.
            "CREATE COLLECTION c (id INT PRIMARY KEY, vector VECTOR(65536)) WITH (capacity = 4000000000);",
        ] {
            let e = engine_error(db.execute(sql));
            assert!(
                matches!(
                    e,
                    Error::CollectionTooLarge { .. }
                        | Error::InvalidDimension
                        | Error::InvalidCapacity
                ),
                "{sql}: {e:?}"
            );
            assert!(!db.is_poisoned(), "{sql} poisoned the database");
        }
        // Still healthy.
        db.execute(
            "CREATE COLLECTION ok (id INT PRIMARY KEY, vector VECTOR(4)) WITH (capacity = 10);",
        )
        .unwrap();
    }
    // And nothing was logged that would fail on replay.
    let db = open(dir.path());
    assert!(!db.is_poisoned());
    assert_eq!(db.collections().len(), 1);
}

#[test]
fn a_statement_over_the_size_limit_is_refused() {
    let dir = tempfile::tempdir().unwrap();
    let db = open(dir.path());
    assert_eq!(
        db.limits().max_statement_bytes,
        Limits::default().max_statement_bytes
    );
    db.set_limits(Limits {
        max_statement_bytes: 32,
        ..Limits::default()
    });
    assert!(matches!(
        db.execute("SELECT id FROM some_collection_name_long_enough;"),
        Err(QueryError::Parse(_))
    ));
}

#[test]
fn a_timeout_stops_a_long_search() {
    let dir = tempfile::tempdir().unwrap();
    let db = open(dir.path());
    let c = big(&db);
    // A deadline already past fails at the first check — deterministic.
    db.set_limits(Limits {
        statement_timeout: Some(Duration::ZERO),
        ..Limits::default()
    });
    let e = engine_error(db.execute_with(
        "SEARCH TOP 5 NEAREST TO ? FROM big;",
        &[vec![0.5f32; 64].into()],
    ));
    assert!(matches!(e, Error::Interrupted { .. }), "{e:?}");
    assert!(e.to_string().contains("timeout"), "{e}");
    assert!(matches!(
        db.search(c, &[0.5; 64], 5),
        Err(Error::Interrupted { .. })
    ));
    // A generous one lets it finish; and none at all is the default.
    db.set_limits(Limits {
        statement_timeout: Some(Duration::from_secs(60)),
        ..Limits::default()
    });
    assert_eq!(db.search(c, &[0.5; 64], 5).unwrap().len(), 5);
    db.set_limits(Limits::default());
    assert_eq!(db.search(c, &[0.5; 64], 5).unwrap().len(), 5);
    assert!(!db.is_poisoned());
}

#[test]
fn a_timeout_stops_a_long_scan_between_rows() {
    let dir = tempfile::tempdir().unwrap();
    let db = open(dir.path());
    big(&db);
    let mut stream = db.execute("SELECT id FROM big;").unwrap();
    assert!(stream.next().unwrap().is_ok());
    // The deadline belongs to the statement: shortening the limit now does
    // not reach back into a stream already running...
    db.set_limits(Limits {
        statement_timeout: Some(Duration::ZERO),
        ..Limits::default()
    });
    assert!(stream.next().unwrap().is_ok());
    // ...but a statement started under it stops at its first check.
    let mut stream = db.execute("SELECT id FROM big;").unwrap();
    let first_err = stream.find_map(|r| r.err()).expect("the scan is stopped");
    assert!(
        matches!(first_err, QueryError::Engine(Error::Interrupted { .. })),
        "{first_err:?}"
    );
}

#[test]
fn interrupt_stops_statements_running_now_and_no_others() {
    let dir = tempfile::tempdir().unwrap();
    let db = open(dir.path());
    big(&db);
    let mut running = db.execute("SELECT id FROM big;").unwrap();
    assert!(running.next().unwrap().is_ok());
    db.interrupt();
    let stopped = running
        .find_map(|r| r.err())
        .expect("the running scan is interrupted");
    assert!(
        matches!(stopped, QueryError::Engine(Error::Interrupted { .. })),
        "{stopped:?}"
    );
    assert!(stopped.to_string().contains("interrupt"), "{stopped}");
    // A statement started after the interrupt runs normally.
    assert_eq!(
        db.execute("SELECT id FROM big LIMIT 10;").unwrap().count(),
        10
    );
}

#[test]
fn interrupt_from_another_thread_stops_a_search_in_flight() {
    let dir = tempfile::tempdir().unwrap();
    let db = open(dir.path());
    let c = big(&db);
    let q = vec![0.5f32; 64];
    // Keep searching until one is caught mid-flight; each search is long
    // enough (200k x 64) that an interrupt lands inside one almost at once.
    let stop = std::sync::atomic::AtomicBool::new(false);
    let interrupted = std::thread::scope(|s| {
        let searcher = s.spawn(|| {
            for _ in 0..10_000 {
                if let Err(e) = db.search(c, &q, 5) {
                    stop.store(true, std::sync::atomic::Ordering::Relaxed);
                    return Some(e);
                }
            }
            None
        });
        while !stop.load(std::sync::atomic::Ordering::Relaxed) && !searcher.is_finished() {
            db.interrupt();
            std::thread::sleep(Duration::from_millis(1));
        }
        searcher.join().unwrap()
    });
    assert!(
        matches!(interrupted, Some(Error::Interrupted { .. })),
        "{interrupted:?}"
    );
    assert!(!db.is_poisoned());
}
