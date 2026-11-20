//! `LIMIT` / `OFFSET` on SELECT, end to end.

use flats::{Db, DbOptions, Param, RegValue};

fn setup(n: i64) -> (tempfile::TempDir, Db) {
    let dir = tempfile::tempdir().unwrap();
    let db = Db::open(dir.path(), &[], DbOptions::default()).unwrap();
    db.execute("CREATE COLLECTION t (id INT PRIMARY KEY, vector VECTOR(2), g INT) WITH (capacity = 100000);")
        .unwrap();
    db.insert_many(
        0,
        (0..n)
            .map(|i| {
                (
                    vec![i as f32, 1.0],
                    vec![(0, flats::Value::Int(i)), (1, flats::Value::Int(i % 3))],
                )
            })
            .collect(),
    )
    .unwrap();
    (dir, db)
}

fn ids(db: &Db, sql: &str, params: &[Param]) -> Vec<i64> {
    db.execute_with(sql, params)
        .unwrap()
        .map(|r| match r.unwrap().0.as_slice() {
            [RegValue::Int(n), ..] => *n,
            other => panic!("{other:?}"),
        })
        .collect()
}

#[test]
fn limit_caps_the_rows_and_offset_skips_them() {
    let (_d, db) = setup(100);
    let all = ids(&db, "SELECT id FROM t;", &[]);
    assert_eq!(all.len(), 100);
    assert_eq!(ids(&db, "SELECT id FROM t LIMIT 10;", &[]), all[..10]);
    assert_eq!(
        ids(&db, "SELECT id FROM t LIMIT 10 OFFSET 95;", &[]),
        all[95..]
    );
    assert_eq!(ids(&db, "SELECT id FROM t OFFSET 90;", &[]), all[90..]);
    assert!(ids(&db, "SELECT id FROM t LIMIT 0;", &[]).is_empty());
    assert!(ids(&db, "SELECT id FROM t OFFSET 1000;", &[]).is_empty());
    assert_eq!(ids(&db, "SELECT id FROM t LIMIT 1000;", &[]), all);
}

#[test]
fn pages_tile_the_whole_result_without_gaps_or_repeats() {
    let (_d, db) = setup(1000);
    let all = ids(&db, "SELECT id FROM t WHERE g = ?;", &[Param::Int(1)]);
    let mut paged = Vec::new();
    for page in 0.. {
        let rows = ids(
            &db,
            "SELECT id FROM t WHERE g = ? LIMIT ? OFFSET ?;",
            &[Param::Int(1), Param::Int(37), Param::Int(page * 37)],
        );
        if rows.is_empty() {
            break;
        }
        paged.extend(rows);
    }
    assert_eq!(paged, all);
}

#[test]
fn limit_applies_after_where() {
    let (_d, db) = setup(30);
    let got = ids(&db, "SELECT id FROM t WHERE g = 2 LIMIT 3;", &[]);
    assert_eq!(got.len(), 3);
    assert!(got.iter().all(|i| i % 3 == 2));
}

#[test]
fn a_limited_stream_keeps_its_labels() {
    let (_d, db) = setup(5);
    let stream = db.execute("SELECT id, g FROM t LIMIT 2;").unwrap();
    assert_eq!(stream.labels(), ["id", "g"]);
    assert_eq!(stream.count(), 2);
}

#[test]
fn limit_on_search_is_refused_top_k_says_how_many() {
    let (_d, db) = setup(5);
    assert!(
        db.execute("SEARCH TOP 5 NEAREST TO [1, 0] FROM t LIMIT 2;")
            .is_err()
    );
}
