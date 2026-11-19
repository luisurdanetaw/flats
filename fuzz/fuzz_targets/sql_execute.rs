//! The whole query pipeline — lex, parse, bind, plan, compile, run — on a
//! live database with data in it. Reads only (SELECT, SEARCH), so every input
//! sees the same database and a crash replays exactly; writes are covered by
//! the differential harness (tests/differential.rs).
#![no_main]
use std::sync::OnceLock;

use flats::{Db, DbOptions};
use libfuzzer_sys::fuzz_target;

fn db() -> &'static Db {
    static DB: OnceLock<Db> = OnceLock::new();
    DB.get_or_init(|| {
        let dir = std::env::temp_dir().join(format!("flats-fuzz-execute-{}", std::process::id()));
        let _ = std::fs::remove_dir_all(&dir);
        let db = Db::open(&dir, &[], DbOptions::default()).expect("open");
        db.execute(
            "CREATE COLLECTION docs (id INT PRIMARY KEY, vector VECTOR(3), t TEXT, f FLOAT, n INT) \
             WITH (capacity = 1000);",
        )
        .expect("create");
        for i in 0..40 {
            db.execute(&format!(
                "INSERT INTO docs (id, vector, t, f, n) VALUES ({i}, [{i}, 1, -1], 'r{}', {}.5, {});",
                i % 7,
                i - 20,
                i % 5
            ))
            .expect("insert");
        }
        db.execute("DELETE FROM docs WHERE n = 3;").expect("delete");
        db
    })
}

fuzz_target!(|data: &[u8]| {
    let Ok(sql) = std::str::from_utf8(data) else {
        return;
    };
    let head = sql.trim_start().get(..6).unwrap_or("").to_ascii_uppercase();
    if head != "SELECT" && head != "SEARCH" {
        return;
    }
    if let Ok(rows) = db().execute(sql) {
        for row in rows.take(10_000) {
            let _ = row;
        }
    }
});
