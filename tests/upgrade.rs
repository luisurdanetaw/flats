//! Opening databases written by EARLIER builds.
//!
//! `tests/fixtures/wal6-catalog4/` was written by the build before this one
//! (WAL v6, catalog v4) and left un-closed — a crash — so the WAL still holds
//! a tail after the last checkpoint: an insert, a delete and a DROP
//! COLLECTION. This build must open it exactly, keep working on it, and
//! upgrade its files on the way.
//!
//! When a format changes again, generate a fixture with the build before the
//! change (see `tests/fixtures/README.md` for how)
//! and add a test like this one.

use std::collections::BTreeMap;
use std::path::Path;

use flats::{Db, DbOptions, RegValue};

fn copy_dir(from: &Path, to: &Path) {
    std::fs::create_dir_all(to).unwrap();
    for e in std::fs::read_dir(from).unwrap() {
        let e = e.unwrap();
        let dest = to.join(e.file_name());
        if e.file_type().unwrap().is_dir() {
            copy_dir(&e.path(), &dest);
        } else {
            std::fs::copy(e.path(), dest).unwrap();
        }
    }
}

fn fixture(name: &str) -> tempfile::TempDir {
    let dir = tempfile::tempdir().unwrap();
    copy_dir(
        &Path::new(env!("CARGO_MANIFEST_DIR"))
            .join("tests/fixtures")
            .join(name),
        dir.path(),
    );
    dir
}

/// id → (year, rating, title, vector)
fn docs(db: &Db) -> BTreeMap<String, (i64, f64, String, Vec<f32>)> {
    db.execute("SELECT id, year, rating, title, vector FROM docs;")
        .unwrap()
        .map(|r| match r.unwrap().0.as_slice() {
            [
                RegValue::Str(id),
                RegValue::Int(y),
                RegValue::Real(r),
                RegValue::Str(t),
                RegValue::Vector(v),
            ] => (id.clone(), (*y, *r, t.clone(), v.to_vec())),
            other => panic!("{other:?}"),
        })
        .collect()
}

/// What the generator wrote, row by row.
fn expected() -> BTreeMap<String, (i64, f64, String, Vec<f32>)> {
    let mut want = BTreeMap::new();
    for i in 0..18i64 {
        if i == 3 {
            continue; // deleted after the checkpoint
        }
        let title = if 2000 + i < 2005 {
            "edited".to_string()
        } else {
            format!("title {i} it's")
        };
        let vector = if i == 7 {
            vec![9.0, 9.0, 9.0]
        } else {
            vec![i as f32, 1.0, -0.5]
        };
        want.insert(format!("d{i}"), (2000 + i, i as f64 / 4.0, title, vector));
    }
    want.insert(
        "late".into(),
        (2030, 0.5, "after".into(), vec![0.0, 0.0, 1.0]),
    );
    want
}

fn version(path: &Path, at: usize) -> u32 {
    let b = std::fs::read(path).unwrap();
    u32::from_le_bytes(b[at..at + 4].try_into().unwrap())
}

#[test]
fn a_database_from_the_previous_build_opens_exactly() {
    let dir = fixture("wal6-catalog4");
    assert_eq!(
        version(&dir.path().join("wal.log"), 4),
        6,
        "the fixture is what it claims"
    );
    assert_eq!(version(&dir.path().join("catalog.snap"), 4), 4);

    let db = Db::open(dir.path(), &[], DbOptions::default()).expect("opens");
    assert_eq!(docs(&db), expected());
    // The DROP in the WAL tail took effect.
    let names: Vec<String> = db.collections().into_iter().map(|c| c.name).collect();
    assert_eq!(names, vec!["docs".to_string()]);
    // Search works on the replayed state.
    let top = db
        .execute("SEARCH TOP 1 NEAREST TO [1, 1, 1] FROM docs RETURNING id;")
        .unwrap()
        .next()
        .unwrap()
        .unwrap();
    assert_eq!(format!("{:?}", top.0), r#"[Str("d7")]"#);
}

#[test]
fn it_keeps_working_and_its_files_are_upgraded() {
    let dir = fixture("wal6-catalog4");
    {
        let db = Db::open(dir.path(), &[], DbOptions::default()).unwrap();
        db.execute("UPSERT INTO docs (id, vector, year, rating, title) VALUES ('d1', [1, 1, 1], 1999, 0.0, 'new');")
            .unwrap();
        db.execute(
            "CREATE COLLECTION fresh (k INT PRIMARY KEY, vector VECTOR(2)) WITH (capacity = 8);",
        )
        .unwrap();
        db.close().unwrap();
    }
    assert_eq!(
        version(&dir.path().join("wal.log"), 4),
        7,
        "the log is rewritten at the current version"
    );
    assert_eq!(
        version(&dir.path().join("catalog.snap"), 4),
        5,
        "so is the catalog"
    );

    let db = Db::open(dir.path(), &[], DbOptions::default()).unwrap();
    let mut want = expected();
    want.insert("d1".into(), (1999, 0.0, "new".into(), vec![1.0, 1.0, 1.0]));
    assert_eq!(docs(&db), want);
    assert_eq!(db.collections().len(), 2);
    db.close().unwrap();
}
