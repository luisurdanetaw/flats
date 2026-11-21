//! A collection the catalog lists, whose whole directory is gone (deleted, a
//! bad copy, an rsync with the wrong excludes), used to reopen EMPTY with no
//! error: with no files left it looked exactly like a collection being created
//! for the first time. Now a collection's files are created BEFORE it is
//! registered, so "listed but no directory" can only mean lost — and opening
//! refuses, the same way it refuses a damaged store file.

use std::path::Path;

use flats::{Db, DbOptions, Error, RegValue};

fn ids(db: &Db, coll: &str) -> Vec<i64> {
    let mut v: Vec<i64> = db
        .execute(&format!("SELECT id FROM {coll};"))
        .unwrap()
        .map(|r| match r.unwrap().0[0] {
            RegValue::Int(i) => i,
            ref o => panic!("{o:?}"),
        })
        .collect();
    v.sort_unstable();
    v
}

fn populated(dir: &Path) {
    let db = Db::open(dir, &[], DbOptions::default()).unwrap();
    db.execute("CREATE COLLECTION t (id INT PRIMARY KEY, vector VECTOR(2)) WITH (capacity = 100);")
        .unwrap();
    db.execute(
        "CREATE COLLECTION keep (id INT PRIMARY KEY, vector VECTOR(2)) WITH (capacity = 100);",
    )
    .unwrap();
    db.execute("INSERT INTO t (id, vector) VALUES (1, [1, 0]), (2, [0, 1]);")
        .unwrap();
    db.execute("INSERT INTO keep (id, vector) VALUES (9, [1, 1]);")
        .unwrap();
    db.close().unwrap();
}

#[test]
fn a_collection_whose_directory_is_gone_refuses_to_open() {
    let dir = tempfile::tempdir().unwrap();
    populated(dir.path());
    std::fs::remove_dir_all(dir.path().join("collection-0")).unwrap();
    match Db::open(dir.path(), &[], DbOptions::default()) {
        Err(Error::CorruptSnapshot(why)) => {
            assert!(why.contains("'t'"), "{why}");
            assert!(why.contains("missing"), "{why}");
        }
        Err(other) => panic!("expected CorruptSnapshot, got {other:?}"),
        Ok(db) => panic!("opened with t = {:?} instead of refusing", ids(&db, "t")),
    }
    assert!(
        !dir.path().join("collection-0").exists(),
        "a refusal changes nothing"
    );
}

#[test]
fn salvage_recreates_it_empty_and_the_rest_is_untouched() {
    let dir = tempfile::tempdir().unwrap();
    populated(dir.path());
    std::fs::remove_dir_all(dir.path().join("collection-0")).unwrap();
    {
        let db = Db::open_salvage(dir.path(), &[], DbOptions::default()).unwrap();
        assert_eq!(ids(&db, "t"), Vec::<i64>::new());
        assert_eq!(ids(&db, "keep"), vec![9]);
        db.execute("INSERT INTO t (id, vector) VALUES (5, [1, 0]);")
            .unwrap();
        drop(db); // crash: the new row must come back from the log alone
    }
    let db = Db::open(dir.path(), &[], DbOptions::default()).unwrap();
    assert_eq!(ids(&db, "t"), vec![5]);
    assert_eq!(ids(&db, "keep"), vec![9]);
    db.close().unwrap();
}

#[test]
fn salvage_after_a_crash_does_not_replay_old_rows_into_the_new_files() {
    // The log still holds rows for the lost collection (no checkpoint since).
    // Replaying them into fresh files would resurrect some rows of a
    // directory the operator chose to give up — and, across a compaction,
    // land them on the wrong ordinals. Salvage starts the collection clean.
    let dir = tempfile::tempdir().unwrap();
    {
        let db = Db::open(dir.path(), &[], DbOptions::default()).unwrap();
        db.execute(
            "CREATE COLLECTION t (id INT PRIMARY KEY, vector VECTOR(2)) WITH (capacity = 100);",
        )
        .unwrap();
        db.checkpoint().unwrap();
        db.execute("INSERT INTO t (id, vector) VALUES (1, [1, 0]);")
            .unwrap();
        drop(db); // the insert is only in the log
    }
    std::fs::remove_dir_all(dir.path().join("collection-0")).unwrap();
    let db = Db::open_salvage(dir.path(), &[], DbOptions::default()).unwrap();
    assert_eq!(ids(&db, "t"), Vec::<i64>::new());
    db.execute("INSERT INTO t (id, vector) VALUES (1, [0, 1]);")
        .unwrap();
    assert_eq!(ids(&db, "t"), vec![1]);
    db.close().unwrap();
}

#[test]
fn collections_registered_by_open_still_open_on_first_use() {
    use flats::{CollectionConfig, ColumnSpec, ColumnType, Schema};
    let dir = tempfile::tempdir().unwrap();
    let schema = Schema::from_columns(vec![
        ColumnSpec::Key {
            name: "id".into(),
            ty: ColumnType::Int,
        },
        ColumnSpec::Vector {
            name: "vector".into(),
            dim: std::num::NonZeroUsize::new(2).unwrap(),
        },
    ])
    .unwrap();
    let cfg = CollectionConfig {
        id: 0,
        name: "pre".into(),
        capacity: 10,
        schema,
    };
    for _ in 0..3 {
        let db = Db::open(dir.path(), std::slice::from_ref(&cfg), DbOptions::default()).unwrap();
        db.close().unwrap();
    }
    let db = Db::open(dir.path(), &[], DbOptions::default()).unwrap();
    assert_eq!(db.collections().len(), 1);
    db.close().unwrap();
}
