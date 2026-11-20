//! A column named `score` (common: a quality score, a rating) next to the
//! `score` pseudo-column a SEARCH returns. The pseudo-column can never be
//! filtered on — it is computed by the ranking — so in a WHERE clause the
//! name can only mean the real column, and it does. `RETURNING score` in a
//! SEARCH still means the similarity.

use flats::{Db, DbOptions, RegValue};

fn db(dir: &std::path::Path) -> Db {
    let db = Db::open(dir, &[], DbOptions::default()).unwrap();
    db.execute("CREATE COLLECTION d (id INT PRIMARY KEY, vector VECTOR(2), score FLOAT) WITH (capacity = 10);")
        .unwrap();
    db.execute("INSERT INTO d (id, vector, score) VALUES (1, [1, 0], 0.9), (2, [0, 1], 0.1), (3, [1, 1], 0.6);")
        .unwrap();
    db
}

fn ints(db: &Db, sql: &str) -> Vec<i64> {
    let mut v: Vec<i64> = db
        .execute(sql)
        .unwrap_or_else(|e| panic!("{sql}: {e}"))
        .map(|r| match r.unwrap().0[0] {
            RegValue::Int(i) => i,
            ref other => panic!("{other:?}"),
        })
        .collect();
    v.sort_unstable();
    v
}

#[test]
fn where_score_filters_on_a_real_score_column() {
    let dir = tempfile::tempdir().unwrap();
    let db = db(dir.path());
    assert_eq!(ints(&db, "SELECT id FROM d WHERE score > 0.5;"), vec![1, 3]);
    assert_eq!(
        ints(
            &db,
            "SEARCH TOP 5 NEAREST TO [1, 0] FROM d WHERE score < 0.7 RETURNING id;"
        ),
        vec![2, 3]
    );
    assert_eq!(ints(&db, "DELETE FROM d WHERE score < 0.2;").len(), 0);
    assert_eq!(ints(&db, "SELECT id FROM d;"), vec![1, 3]);
}

#[test]
fn returning_score_in_a_search_is_still_the_similarity() {
    let dir = tempfile::tempdir().unwrap();
    let db = db(dir.path());
    let first = db
        .execute("SEARCH TOP 1 NEAREST TO [2, 0] FROM d RETURNING id, score;")
        .unwrap()
        .next()
        .unwrap()
        .unwrap();
    // Row 1 or 3 both score 2.0 against [2, 0]; either way it's the dot
    // product, not the stored 0.9 or 0.6.
    assert!(
        matches!(first.0[1], RegValue::Real(s) if s == 2.0),
        "{first:?}"
    );
    // And SELECT reads the column.
    assert_eq!(
        format!(
            "{:?}",
            db.execute("SELECT score FROM d WHERE id = 2;")
                .unwrap()
                .next()
                .unwrap()
                .unwrap()
                .0
        ),
        "[Real(0.1)]"
    );
}

#[test]
fn where_score_without_a_score_column_is_still_refused() {
    let dir = tempfile::tempdir().unwrap();
    let db = Db::open(dir.path(), &[], DbOptions::default()).unwrap();
    db.execute("CREATE COLLECTION d (id INT PRIMARY KEY, vector VECTOR(2)) WITH (capacity = 10);")
        .unwrap();
    let err = db
        .execute("SEARCH TOP 5 NEAREST TO [1, 0] FROM d WHERE score > 0.5;")
        .err()
        .expect("refused");
    assert!(err.to_string().contains("score"), "{err}");
}
