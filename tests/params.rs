//! Parameterized statements, end to end: `Db::execute_with(sql, &[Param])`.
//!
//! The guarantee under test is the one that matters: a parameter is a VALUE,
//! never syntax. Every attempt to smuggle structure through one — quotes,
//! semicolons, keywords, comment markers — must arrive in the database as the
//! exact string that was passed, and change nothing else.

use flats::{Db, DbOptions, Param, QueryError, RegValue};

fn open(dir: &std::path::Path) -> Db {
    Db::open(dir, &[], DbOptions::default()).expect("open")
}

fn setup(db: &Db) {
    db.execute(
        "CREATE COLLECTION docs (id TEXT PRIMARY KEY, vector VECTOR(3), lang TEXT, \
         year INT, rating FLOAT) WITH (capacity = 1000);",
    )
    .unwrap();
}

fn insert(db: &Db, id: &str, v: [f32; 3], lang: &str, year: i64, rating: f64) {
    db.execute_with(
        "INSERT INTO docs (id, vector, lang, year, rating) VALUES (?, ?, ?, ?, ?);",
        &[
            Param::Text(id.into()),
            Param::Vector(v.to_vec()),
            Param::Text(lang.into()),
            Param::Int(year),
            Param::Float(rating),
        ],
    )
    .expect("insert");
}

fn ids(db: &Db, sql: &str, params: &[Param]) -> Vec<String> {
    let mut out: Vec<String> = db
        .execute_with(sql, params)
        .expect("query")
        .map(|r| match r.unwrap().0.as_slice() {
            [RegValue::Str(id), ..] => id.clone(),
            other => panic!("unexpected row {other:?}"),
        })
        .collect();
    out.sort();
    out
}

#[test]
fn every_statement_kind_takes_parameters() {
    let dir = tempfile::tempdir().unwrap();
    let db = open(dir.path());
    setup(&db);
    insert(&db, "a", [1.0, 0.0, 0.0], "en", 2020, 4.5);
    insert(&db, "b", [0.0, 1.0, 0.0], "es", 2021, 3.0);
    insert(&db, "c", [0.9, 0.1, 0.0], "en", 2022, 2.0);

    // SELECT with typed comparands, an INT literal into a FLOAT column too.
    assert_eq!(
        ids(
            &db,
            "SELECT id FROM docs WHERE lang = ? AND rating > ?;",
            &["en".into(), Param::Int(3)]
        ),
        vec!["a"]
    );
    // SEARCH: the query vector, TOP, and the filter all from parameters.
    let hits = ids(
        &db,
        "SEARCH TOP ? NEAREST TO ? FROM docs WHERE year >= ? RETURNING id, score;",
        &[
            Param::Int(1),
            Param::Vector(vec![1.0, 0.0, 0.0]),
            Param::Int(2021),
        ],
    );
    assert_eq!(hits, vec!["c"]);
    // UPDATE, including a replacement embedding.
    db.execute_with(
        "UPDATE docs SET rating = ?, vector = ? WHERE id = ?;",
        &[
            Param::Float(5.0),
            Param::Vector(vec![0.0, 0.0, 1.0]),
            "b".into(),
        ],
    )
    .unwrap();
    assert_eq!(
        ids(
            &db,
            "SEARCH TOP 1 NEAREST TO ? FROM docs RETURNING id;",
            &[Param::Vector(vec![0.0, 0.0, 1.0])]
        ),
        vec!["b"]
    );
    // DELETE.
    db.execute_with("DELETE FROM docs WHERE year < ?;", &[Param::Int(2021)])
        .unwrap();
    assert_eq!(
        ids(&db, "SELECT id FROM docs WHERE year > ?;", &[Param::Int(0)]),
        vec!["b", "c"]
    );
}

#[test]
fn injection_payloads_are_stored_and_matched_as_plain_strings() {
    let dir = tempfile::tempdir().unwrap();
    let db = open(dir.path());
    setup(&db);
    let payloads = [
        "x'; DROP COLLECTION docs; --",
        "' OR '1'='1",
        "a' OR lang != 'zz",
        "); DELETE FROM docs WHERE year > (0",
        "?",
        "??",
        "-- comment",
        "\\'",
        "'",
        "''",
        "\u{0}nul",
        "ñ 🦤 \n\t;",
    ];
    for (i, p) in payloads.iter().enumerate() {
        insert(&db, p, [1.0, 0.0, 0.0], p, i as i64, 0.0);
    }
    // The collection survived and holds every payload, unchanged.
    let all = ids(&db, "SELECT id FROM docs;", &[]);
    let mut want: Vec<String> = payloads.iter().map(|s| s.to_string()).collect();
    want.sort();
    assert_eq!(all, want);
    // A payload used as a comparand matches exactly its own row, no more.
    for p in payloads {
        assert_eq!(
            ids(&db, "SELECT id FROM docs WHERE lang = ?;", &[p.into()]),
            vec![p.to_string()],
            "{p:?}"
        );
        assert!(
            ids(
                &db,
                "SELECT id FROM docs WHERE id = ? AND lang != ?;",
                &[p.into(), p.into()]
            )
            .is_empty()
        );
    }
}

#[test]
fn vector_parameters_are_stored_bit_for_bit() {
    let dir = tempfile::tempdir().unwrap();
    let db = open(dir.path());
    setup(&db);
    let v = [f32::MIN_POSITIVE, 1.0e-45, -3.4e38];
    insert(&db, "tiny", v, "en", 1, 0.0);
    let got = db
        .execute_with("SELECT vector FROM docs WHERE id = ?;", &["tiny".into()])
        .unwrap()
        .map(|r| match r.unwrap().0.as_slice() {
            [RegValue::Vector(v)] => v.to_vec(),
            other => panic!("{other:?}"),
        })
        .next()
        .unwrap();
    let bits = |x: &[f32]| x.iter().map(|f| f.to_bits()).collect::<Vec<_>>();
    assert_eq!(bits(&got), bits(&v));
}

#[test]
fn a_parameter_count_mismatch_is_a_parse_error_and_changes_nothing() {
    let dir = tempfile::tempdir().unwrap();
    let db = open(dir.path());
    setup(&db);
    let short = db.execute_with(
        "INSERT INTO docs (id, vector, lang, year, rating) VALUES (?, ?, ?, ?, ?);",
        &["a".into(), Param::Vector(vec![1.0, 0.0, 0.0])],
    );
    assert!(
        matches!(short, Err(QueryError::Parse(_))),
        "{:?}",
        short.err()
    );
    // A `?` with no parameters at all, through plain `execute`.
    let bare = db.execute("SELECT id FROM docs WHERE lang = ?;");
    assert!(matches!(bare, Err(QueryError::Parse(_))));
    assert!(ids(&db, "SELECT id FROM docs;", &[]).is_empty());
}

#[test]
fn a_parameter_of_the_wrong_type_is_the_same_error_a_literal_gets() {
    let dir = tempfile::tempdir().unwrap();
    let db = open(dir.path());
    setup(&db);
    let via_param = db
        .execute_with("SELECT id FROM docs WHERE year = ?;", &["2020".into()])
        .err()
        .expect("TEXT against INT");
    let via_literal = db
        .execute("SELECT id FROM docs WHERE year = '2020';")
        .err()
        .expect("TEXT against INT");
    assert_eq!(via_param.to_string(), via_literal.to_string());
    // And a wrong-dimension vector.
    assert!(
        db.execute_with(
            "SEARCH TOP 1 NEAREST TO ? FROM docs;",
            &[Param::Vector(vec![1.0])]
        )
        .is_err()
    );
}

#[test]
fn a_parameter_cannot_name_a_collection_or_column() {
    let dir = tempfile::tempdir().unwrap();
    let db = open(dir.path());
    setup(&db);
    assert!(
        db.execute_with("SELECT id FROM ?;", &["docs".into()])
            .is_err()
    );
    assert!(
        db.execute_with("DROP COLLECTION ?;", &["docs".into()])
            .is_err()
    );
    assert_eq!(db.collections().len(), 1, "nothing was dropped");
}
