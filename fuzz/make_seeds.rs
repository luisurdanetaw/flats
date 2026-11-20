//! Writes a seed corpus of REAL inputs for every fuzz target: statements
//! from the V-SQL grammar, and the snapshot bodies, catalog and WAL payloads
//! of a populated database. Starting from valid inputs, libFuzzer reaches
//! the deep decoding paths in seconds instead of hours.
//!
//!     cargo run --release --bin make_seeds -- fuzz/corpus

use std::path::Path;

use flats::{Db, DbOptions};

const STATEMENTS: &[&str] = &[
    "CREATE COLLECTION docs (id INT PRIMARY KEY, vector VECTOR(3), t TEXT, f FLOAT, n INT) WITH (capacity = 1000);",
    "INSERT INTO docs (id, vector, t, f, n) VALUES (1, [1, 0, -1.5e-3], 'it''s', 0.25, 7);",
    "INSERT INTO docs (id, vector, t, f, n) VALUES (2, [0, 1, 0], 'a', -1, 0), (3, [1, 1, 1], 'b', 2.5, 1);",
    "UPDATE docs SET t = 'x', f = 1e300 WHERE id = 1;",
    "UPDATE docs SET vector = [0.5, 0.5, 0.5] WHERE id = 2;",
    "DELETE FROM docs WHERE n >= 3 OR t = 'b' AND f < 0;",
    "SELECT * FROM docs;",
    "SELECT id, vector FROM docs WHERE (n = 1 OR n = 2) AND f != 0.0 LIMIT 5 OFFSET 1;",
    "SELECT t, f FROM docs WHERE 3 > n AND t != 'r1';",
    "SEARCH TOP 5 NEAREST TO [1, 0, 0] FROM docs;",
    "SEARCH TOP 2 NEAREST TO [0.1, -0.2, 0.3] FROM docs WHERE n < 4 RETURNING id, score, t;",
    "SEARCH TOP 40 NEAREST TO [1, 1, 1] FROM docs WHERE t = 'r1' OR t = 'r2' OR t = 'r3' RETURNING vector, score;",
    "DROP COLLECTION docs;",
];

/// The fuzz hooks take `lsn (8 bytes) ++ body` and rebuild the framing:
/// strip magic, version, length and CRC from a real file to get that.
fn snapshot_seed(bytes: &[u8]) -> Vec<u8> {
    let mut out = bytes[8..16].to_vec();
    out.extend_from_slice(&bytes[20..bytes.len() - 4]);
    out
}

fn write(dir: &Path, name: &str, data: &[u8]) {
    std::fs::create_dir_all(dir).unwrap();
    std::fs::write(dir.join(name), data).unwrap();
}

fn main() {
    let out = std::env::args()
        .nth(1)
        .unwrap_or_else(|| "fuzz/corpus".into());
    let out = Path::new(&out);
    for (i, s) in STATEMENTS.iter().enumerate() {
        write(&out.join("sql_parse"), &format!("seed-{i}"), s.as_bytes());
        write(&out.join("sql_execute"), &format!("seed-{i}"), s.as_bytes());
    }

    let dir = std::env::temp_dir().join(format!("flats-seeds-{}", std::process::id()));
    let _ = std::fs::remove_dir_all(&dir);
    let opts = DbOptions {
        checkpoint_interval: std::time::Duration::from_secs(3600),
        ..DbOptions::default()
    };
    {
        let db = Db::open(&dir, &[], opts).unwrap();
        for s in &STATEMENTS[..6] {
            db.execute(s).unwrap();
        }
        db.close().unwrap(); // checkpoint: every snapshot written
    }
    for (file, target) in [
        ("collection-0/metadata.snap", "metadata_snapshot"),
        ("collection-0/tuples.snap", "tuple_snapshot"),
        ("catalog.snap", "catalog"),
    ] {
        let bytes = std::fs::read(dir.join(file)).unwrap();
        write(&out.join(target), "seed-0", &snapshot_seed(&bytes));
    }

    // WAL payloads: a session left un-checkpointed so the log holds frames
    // of every record kind (8-byte file header; 16-byte frame headers of
    // len, crc, lsn).
    {
        let db = Db::open(&dir, &[], opts).unwrap();
        for s in &STATEMENTS[1..6] {
            let _ = db.execute(
                &s.replace("(1,", "(11,")
                    .replace("(2,", "(12,")
                    .replace("(3,", "(13,"),
            );
        }
        db.execute(
            "CREATE COLLECTION other (k TEXT PRIMARY KEY, vector VECTOR(2)) WITH (capacity = 10);",
        )
        .unwrap();
        db.execute("DROP COLLECTION other;").unwrap();
        drop(db);
    }
    let wal = std::fs::read(dir.join("wal.log")).unwrap();
    let mut at = 8;
    let mut n = 0;
    while at + 16 <= wal.len() {
        let len = u32::from_le_bytes(wal[at..at + 4].try_into().unwrap()) as usize;
        let start = at + 16;
        if start + len > wal.len() {
            break;
        }
        write(
            &out.join("wal_record"),
            &format!("seed-{n}"),
            &wal[start..start + len],
        );
        n += 1;
        at = start + len;
    }
    let _ = std::fs::remove_dir_all(&dir);
    println!("seeds written to {} ({n} WAL records)", out.display());
}
