//! A fixed performance probe for regression checks (`scripts/perf_compare.sh`).
//! Not a benchmark to quote numbers from: it measures a few hot paths the same
//! way every time, so two builds can be compared on the SAME machine in the
//! SAME job. Prints one `PERF <name> <milliseconds>` line per measurement.
//!
//!     cargo test --release --test perf_probe -- --ignored --nocapture

use std::time::Instant;

use flats::{Db, DbOptions, Param, RegValue};

const ROWS: usize = 50_000;
const DIM: usize = 384;

fn ms(t: Instant) -> f64 {
    t.elapsed().as_secs_f64() * 1000.0
}

#[test]
#[ignore]
fn perf_probe() {
    let dir = tempfile::tempdir().unwrap();
    let db = Db::open(
        dir.path(),
        &[],
        DbOptions {
            checkpoint_interval: std::time::Duration::from_secs(3600),
            ..DbOptions::default()
        },
    )
    .unwrap();
    db.execute(&format!(
        "CREATE COLLECTION t (id INT PRIMARY KEY, vector VECTOR({DIM}), tag INT) WITH (capacity = {ROWS});"
    ))
    .unwrap();
    let id = db.collections()[0].id;

    // Bulk insert through the engine's batch path.
    let t = Instant::now();
    for chunk in 0..(ROWS / 5000) {
        let rows = (0..5000)
            .map(|i| {
                let n = (chunk * 5000 + i) as i64;
                let v: Vec<f32> = (0..DIM)
                    .map(|d| ((n as usize * 31 + d) % 97) as f32 / 97.0)
                    .collect();
                (
                    v,
                    vec![
                        (0u32, flats::Value::Int(n)),
                        (1u32, flats::Value::Int(n % 10)),
                    ],
                )
            })
            .collect();
        db.insert_many(id, rows).unwrap();
    }
    println!("PERF insert_50k {:.1}", ms(t));

    let q: Vec<f32> = (0..DIM).map(|d| (d % 13) as f32 / 13.0).collect();
    let search = |sql: &str, reps: usize| {
        for _ in 0..2 {
            let _: Vec<_> = db
                .execute_with(sql, &[Param::Vector(q.clone())])
                .unwrap()
                .collect();
        }
        let t = Instant::now();
        for _ in 0..reps {
            let rows: Vec<_> = db
                .execute_with(sql, &[Param::Vector(q.clone())])
                .unwrap()
                .collect();
            assert!(!rows.is_empty());
        }
        ms(t) / reps as f64
    };
    println!(
        "PERF search_top10 {:.3}",
        search("SEARCH TOP 10 NEAREST TO ? FROM t RETURNING id, score;", 20)
    );
    println!(
        "PERF search_filtered {:.3}",
        search(
            "SEARCH TOP 10 NEAREST TO ? FROM t WHERE tag = 3 RETURNING id, score;",
            20
        )
    );

    let t = Instant::now();
    let mut n = 0;
    for row in db.execute("SELECT id, tag FROM t WHERE tag < 5;").unwrap() {
        if let [RegValue::Int(_), RegValue::Int(_)] = row.unwrap().0.as_slice() {
            n += 1;
        }
    }
    assert_eq!(n, ROWS / 2);
    println!("PERF select_scan {:.1}", ms(t));

    let t = Instant::now();
    db.checkpoint().unwrap();
    println!("PERF checkpoint {:.1}", ms(t));
    db.close().unwrap();
}
