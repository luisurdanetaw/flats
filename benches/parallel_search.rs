//! Parallel-read throughput microbenchmark (CLAUDE.md §13 acceptance for the
//! SWMR phase). Populates one index and reports searches/sec as the number of
//! concurrent reader threads grows. With the old `Arc<Mutex<FlatIndex>>` this
//! plateaued at ~1x (reads serialized); with the lock-free `(Writer, Reader)`
//! split it should scale roughly with cores.
//!
//! Run: `cargo bench --bench parallel_search`. No `criterion` dep (custom
//! harness; see CLAUDE.md §2, §11).

use std::num::NonZeroUsize;
use std::sync::Arc;
use std::sync::atomic::{AtomicBool, Ordering};
use std::time::{Duration, Instant};

use flats::index::index::{FlatIndex, Reader};

const DIM: usize = 64;
const COUNT: usize = 8000;
const K: usize = 10;
const DURATION: Duration = Duration::from_millis(750);

fn measure_qps(reader: &Reader, query: &[f32], threads: usize) -> f64 {
    let stop = Arc::new(AtomicBool::new(false));
    let start = Instant::now();
    let mut handles = Vec::new();
    for _ in 0..threads {
        let r = reader.clone();
        let q = query.to_vec();
        let stop = stop.clone();
        handles.push(std::thread::spawn(move || {
            let mut c = 0u64;
            while !stop.load(Ordering::Relaxed) {
                let _ = r.search(&q, K).expect("search");
                c += 1;
            }
            c
        }));
    }
    std::thread::sleep(DURATION);
    stop.store(true, Ordering::Relaxed);
    let total: u64 = handles.into_iter().map(|h| h.join().unwrap()).sum();
    total as f64 / start.elapsed().as_secs_f64()
}

fn main() {
    let dir = tempfile::tempdir().expect("tempdir");
    let path = dir.path().join("bench.idx");
    let (mut w, reader) = FlatIndex::create(&path, NonZeroUsize::new(DIM).unwrap(), COUNT).unwrap();
    for i in 0..COUNT {
        let v: Vec<f32> = (0..DIM).map(|j| ((i + j) % 13) as f32).collect();
        w.write_at(i as u64, &v).unwrap();
    }
    drop(w); // read-only

    let query: Vec<f32> = (0..DIM).map(|j| (j % 13) as f32).collect();
    let cores = std::thread::available_parallelism()
        .map(|n| n.get())
        .unwrap_or(4);

    println!(
        "parallel_search: dim={DIM} count={COUNT} k={K} cores={cores} (each search = {COUNT} dot products)"
    );
    let baseline = measure_qps(&reader, &query, 1);
    println!("  1 thread : {baseline:>10.0} q/s   (1.00x)");

    let mut threads = 2;
    while threads <= cores {
        let qps = measure_qps(&reader, &query, threads);
        println!(
            "  {threads:>2} threads: {qps:>10.0} q/s   ({:.2}x)",
            qps / baseline
        );
        threads *= 2;
    }
    if cores > 1 && !cores.is_power_of_two() {
        let qps = measure_qps(&reader, &query, cores);
        println!(
            "  {cores:>2} threads: {qps:>10.0} q/s   ({:.2}x)",
            qps / baseline
        );
    }
}
