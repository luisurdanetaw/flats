//! `Db::backup`: a consistent, openable copy of a live database.
//!
//! The copy is taken on the commit thread, between records, so it is exactly
//! the database as of one point in the log: every statement acknowledged
//! before that point is in it whole, none after it is in it at all, and no
//! statement is in it by halves. Restoring is just `Db::open` on the copy.

use std::collections::BTreeSet;
use std::path::Path;
use std::time::Duration;

use flats::{Db, DbOptions, RegValue, Value};

fn opts() -> DbOptions {
    DbOptions {
        // No background checkpoint: un-checkpointed writes must make it into
        // the copy on their own.
        checkpoint_interval: Duration::from_secs(3600),
        ..DbOptions::default()
    }
}

fn open(dir: &Path) -> Db {
    Db::open(dir, &[], opts()).expect("open")
}

fn create(db: &Db, name: &str) -> u32 {
    db.execute(&format!(
        "CREATE COLLECTION {name} (id INT PRIMARY KEY, vector VECTOR(3), tag TEXT) \
         WITH (capacity = 100000);"
    ))
    .expect("create");
    db.collection_id(name).expect("id")
}

fn vector(id: i64) -> Vec<f32> {
    vec![id as f32, 1.0 / (id as f32 + 1.0), -0.5]
}

fn row(id: i64) -> flats::Row {
    vec![(0, Value::Int(id)), (1, Value::Text(format!("t{id}")))]
}

/// Every row as (id, tag, vector), sorted.
fn dump(db: &Db, name: &str) -> Vec<(i64, String, Vec<f32>)> {
    let mut out: Vec<_> = db
        .execute(&format!("SELECT id, tag, vector FROM {name};"))
        .expect("select")
        .map(|r| match r.expect("row").0.as_slice() {
            [RegValue::Int(id), RegValue::Str(tag), RegValue::Vector(v)] => {
                (*id, tag.clone(), v.to_vec())
            }
            other => panic!("unexpected row {other:?}"),
        })
        .collect();
    out.sort_by_key(|(id, _, _)| *id);
    out
}

fn names(db: &Db) -> Vec<String> {
    let mut out: Vec<String> = db.collections().into_iter().map(|c| c.name).collect();
    out.sort();
    out
}

#[test]
fn a_backup_opens_to_the_same_database() {
    let src = tempfile::tempdir().unwrap();
    let dst = tempfile::tempdir().unwrap();
    let copy = dst.path().join("copy");
    let db = open(src.path());
    let a = create(&db, "a");
    let b = create(&db, "b");
    db.insert_many(a, (0..300).map(|i| (vector(i), row(i))).collect())
        .unwrap();
    db.insert_many(b, (0..50).map(|i| (vector(i), row(i))).collect())
        .unwrap();
    db.execute("DELETE FROM a WHERE id < 10;").unwrap();
    db.execute("UPDATE b SET tag = 'changed' WHERE id = 7;")
        .unwrap();

    db.backup(&copy).expect("backup");

    let restored = open(&copy);
    assert_eq!(names(&restored), names(&db));
    assert_eq!(dump(&restored, "a"), dump(&db, "a"));
    assert_eq!(dump(&restored, "b"), dump(&db, "b"));
    let ra = restored.collection_id("a").unwrap();
    assert_eq!(
        restored.search(ra, &vector(42), 5).unwrap().len(),
        db.search(a, &vector(42), 5).unwrap().len()
    );
    assert!(
        dump(&restored, "b")
            .iter()
            .any(|(id, tag, _)| *id == 7 && tag == "changed")
    );
}

#[test]
fn the_source_keeps_working_and_the_two_are_independent() {
    let src = tempfile::tempdir().unwrap();
    let dst = tempfile::tempdir().unwrap();
    let copy = dst.path().join("copy");
    let db = open(src.path());
    let a = create(&db, "a");
    db.insert(a, &vector(1), row(1)).unwrap();
    db.backup(&copy).unwrap();

    db.insert(a, &vector(2), row(2))
        .expect("the source still takes writes");
    let restored = open(&copy);
    let ra = restored.collection_id("a").unwrap();
    restored.insert(ra, &vector(3), row(3)).unwrap();

    assert_eq!(
        dump(&db, "a").iter().map(|r| r.0).collect::<Vec<_>>(),
        vec![1, 2]
    );
    assert_eq!(
        dump(&restored, "a").iter().map(|r| r.0).collect::<Vec<_>>(),
        vec![1, 3]
    );
}

#[test]
fn a_backup_survives_the_source_being_deleted() {
    let dst = tempfile::tempdir().unwrap();
    let copy = dst.path().join("copy");
    let want;
    {
        let src = tempfile::tempdir().unwrap();
        let db = open(src.path());
        let a = create(&db, "a");
        db.insert_many(a, (0..20).map(|i| (vector(i), row(i))).collect())
            .unwrap();
        db.backup(&copy).unwrap();
        want = dump(&db, "a");
        db.close().unwrap();
    } // source directory removed here
    let restored = open(&copy);
    assert_eq!(dump(&restored, "a"), want);
}

#[test]
fn a_non_empty_destination_is_refused_and_left_alone() {
    let src = tempfile::tempdir().unwrap();
    let dst = tempfile::tempdir().unwrap();
    std::fs::write(dst.path().join("precious.txt"), b"keep me").unwrap();
    let db = open(src.path());
    create(&db, "a");
    assert!(db.backup(dst.path()).is_err());
    let entries: Vec<_> = std::fs::read_dir(dst.path()).unwrap().collect();
    assert_eq!(entries.len(), 1, "nothing was written into it");
    assert_eq!(
        std::fs::read(dst.path().join("precious.txt")).unwrap(),
        b"keep me"
    );
}

#[test]
fn an_existing_empty_destination_is_fine() {
    let src = tempfile::tempdir().unwrap();
    let dst = tempfile::tempdir().unwrap();
    let db = open(src.path());
    let a = create(&db, "a");
    db.insert(a, &vector(1), row(1)).unwrap();
    db.backup(dst.path())
        .expect("an empty directory is a valid target");
    assert_eq!(dump(&open(dst.path()), "a").len(), 1);
}

#[test]
fn a_dropped_collection_is_not_in_the_backup() {
    let src = tempfile::tempdir().unwrap();
    let dst = tempfile::tempdir().unwrap();
    let copy = dst.path().join("copy");
    let db = open(src.path());
    create(&db, "gone");
    create(&db, "kept");
    db.drop_collection("gone").unwrap();
    db.backup(&copy).unwrap();
    let restored = open(&copy);
    assert_eq!(names(&restored), vec!["kept"]);
    let dirs: Vec<String> = std::fs::read_dir(&copy)
        .unwrap()
        .map(|e| e.unwrap().file_name().to_string_lossy().into_owned())
        .filter(|n| n.starts_with("collection-"))
        .collect();
    assert_eq!(dirs, vec!["collection-1".to_string()]);
}

#[test]
fn a_backup_taken_under_load_holds_whole_statements_only() {
    // A writer lands batches of 10 consecutive keys; the copy is taken while
    // it runs. Whatever the copy holds must be a prefix of whole batches: no
    // batch split by the copy, no batch after a missing one.
    const BATCH: i64 = 10;
    let src = tempfile::tempdir().unwrap();
    let dst = tempfile::tempdir().unwrap();
    let db = open(src.path());
    let a = create(&db, "a");
    let copies: Vec<std::path::PathBuf> =
        (0..5).map(|i| dst.path().join(format!("c{i}"))).collect();
    std::thread::scope(|s| {
        let db = &db;
        let writer = s.spawn(move || {
            for batch in 0..300i64 {
                let rows = (batch * BATCH..(batch + 1) * BATCH)
                    .map(|i| (vector(i), row(i)))
                    .collect();
                db.insert_many(a, rows).unwrap();
            }
        });
        for copy in &copies {
            std::thread::sleep(Duration::from_millis(5));
            db.backup(copy).unwrap();
        }
        writer.join().unwrap();
    });
    let mut sizes = Vec::new();
    for copy in &copies {
        let restored = open(copy);
        let got: BTreeSet<i64> = dump(&restored, "a").into_iter().map(|r| r.0).collect();
        let n = got.len() as i64;
        assert_eq!(n % BATCH, 0, "a batch was split: {n} rows");
        assert_eq!(got, (0..n).collect(), "the copy is a prefix of the log");
        sizes.push(n);
    }
    assert!(
        sizes.windows(2).all(|w| w[0] <= w[1]),
        "later copies hold more: {sizes:?}"
    );
}

#[test]
fn a_backup_of_an_empty_database_opens() {
    let src = tempfile::tempdir().unwrap();
    let dst = tempfile::tempdir().unwrap();
    let copy = dst.path().join("copy");
    let db = open(src.path());
    db.backup(&copy).unwrap();
    assert!(names(&open(&copy)).is_empty());
}
