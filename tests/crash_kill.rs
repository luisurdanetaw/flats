//! Real crashes: a child process runs a seeded workload and is killed with
//! SIGKILL at a random moment — mid-statement, mid-fsync, mid-checkpoint,
//! mid-compaction. No destructor runs, no buffer is flushed.
//!
//! The child prints `ACK <i>` after op `i` returns. Every acknowledged write
//! must survive, and nothing else may appear: after recovery the database
//! equals the model after ops `0..=last_ack`, or after `0..=last_ack + 1` if
//! the op in flight had reached its commit point when the kill landed. And
//! it must still work: the workload continues on the recovered database.
//!
//! The child is this same test binary re-executed with `--ignored --exact
//! crash_child`. Knobs: `FLATS_CRASH_ITERS` (kills per run, default 6),
//! `FLATS_SIM_SEED_BASE`.

mod sim;

use std::io::{BufRead, BufReader, Write};
use std::process::{Command, Stdio};
use std::time::{Duration, Instant};

use sim::{Model, Op, Rng, Sim, env_u64, gen_workload, opts, verify_db};

const OPS: usize = 400;

/// The ops a crash run uses: the usual workload, minus backups (they'd only
/// add directories for the kill to land in, which `tests/backup.rs` covers).
fn workload(seed: u64) -> Vec<Op> {
    gen_workload(seed, OPS)
        .into_iter()
        .filter(|op| !matches!(op, Op::Backup))
        .collect()
}

/// The child half. Not a real test: `#[ignore]`d so a normal run skips it.
#[test]
#[ignore]
fn crash_child() {
    let (Ok(dir), Ok(seed)) = (
        std::env::var("FLATS_CRASH_DIR"),
        std::env::var("FLATS_CRASH_SEED"),
    ) else {
        return;
    };
    let seed: u64 = seed.parse().unwrap();
    let ops = workload(seed);
    let mut sim = Sim::at(std::path::Path::new(&dir), seed);
    let stdout = std::io::stdout();
    for (i, op) in ops.iter().enumerate() {
        if let Err(e) = sim.step(op) {
            // A disagreement before any crash is a plain bug: report it.
            eprintln!("CHILD-MISMATCH op {i}: {e}");
            std::process::exit(3);
        }
        let mut out = stdout.lock();
        let _ = writeln!(out, "ACK {i}");
        let _ = out.flush();
    }
    let mut out = stdout.lock();
    let _ = writeln!(out, "DONE");
    let _ = out.flush();
    // Wait to be killed, so "the child finished first" is never the case
    // being tested by accident.
    std::thread::sleep(Duration::from_secs(60));
}

fn model_after(ops: &[Op], n: usize) -> Model {
    let mut m = Model::default();
    for op in &ops[..n.min(ops.len())] {
        m.apply(op);
    }
    m
}

fn crash_once(seed: u64) {
    let ops = workload(seed);
    let root = tempfile::tempdir().unwrap();
    let mut rng = Rng::new(seed ^ 0xC0FFEE);
    // Kill after a random number of acknowledgements, then a random beat
    // more, so the kill lands anywhere inside the next op.
    let kill_after = rng.below(ops.len() as u64 - 1) as usize;
    let beat = Duration::from_micros(rng.below(3000));

    let mut child = Command::new(std::env::current_exe().unwrap())
        .args([
            "--ignored",
            "--exact",
            "crash_child",
            "--nocapture",
            "--test-threads=1",
        ])
        .env("FLATS_CRASH_DIR", root.path())
        .env("FLATS_CRASH_SEED", seed.to_string())
        .stdout(Stdio::piped())
        .stderr(Stdio::piped())
        .spawn()
        .expect("spawn the child");
    let mut lines = BufReader::new(child.stdout.take().unwrap()).lines();
    let mut last_ack: Option<usize> = None;
    let deadline = Instant::now() + Duration::from_secs(300);
    loop {
        let Some(Ok(line)) = lines.next() else {
            let mut err = String::new();
            let _ = std::io::Read::read_to_string(child.stderr.as_mut().unwrap(), &mut err);
            panic!("seed {seed}: the child died before the kill: {err}");
        };
        if let Some(i) = line.strip_prefix("ACK ") {
            let i: usize = i.parse().unwrap();
            last_ack = Some(i);
            if i >= kill_after {
                break;
            }
        } else if line == "DONE" {
            break;
        }
        assert!(Instant::now() < deadline, "seed {seed}: the child is stuck");
    }
    std::thread::sleep(beat);
    child.kill().expect("SIGKILL");
    child.wait().unwrap();
    // Acks the child wrote before it died.
    for line in lines.map_while(Result::ok) {
        if let Some(i) = line.strip_prefix("ACK ") {
            last_ack = Some(i.parse().unwrap());
        }
    }
    let acked = last_ack.map_or(0, |i| i + 1);

    let db = flats::Db::open(root.path().join("db"), &[], opts())
        .unwrap_or_else(|e| panic!("seed {seed}: recovery after a kill at op {acked} failed: {e}"));
    let exact = verify_db(&db, &model_after(&ops, acked));
    let with_inflight = verify_db(&db, &model_after(&ops, acked + 1));
    let survivors = match (exact, with_inflight) {
        (Ok(()), _) => acked,
        (_, Ok(())) => acked + 1,
        (Err(a), Err(b)) => panic!(
            "seed {seed}: killed after {acked} acknowledged ops; the recovered database \
             matches neither {acked} ops ({a}) nor {} ({b})\n\
             replay: FLATS_SIM_SEED_BASE={seed} FLATS_CRASH_ITERS=1 cargo test --release --test crash_kill",
            acked + 1
        ),
    };
    drop(db);

    // And it still works: finish the workload on the recovered database.
    let mut sim = Sim::at(root.path(), seed ^ 1);
    sim.model = model_after(&ops, survivors);
    if let Err((i, e)) = sim.run(&ops[survivors.min(ops.len())..], 25) {
        panic!(
            "seed {seed}: after recovering {survivors} ops, op {} failed: {e}",
            survivors + i
        );
    }
}

#[test]
fn a_kill_at_any_moment_loses_no_acknowledged_write() {
    let base = env_u64("FLATS_SIM_SEED_BASE", 0);
    for seed in base..base + env_u64("FLATS_CRASH_ITERS", 6) {
        crash_once(seed);
    }
}

/// A debugging aid, not a test: print a seed's workload as statements.
/// `FLATS_SIM_SEED_BASE=20 FROM=260 TO=280 cargo test --release --test
/// crash_kill print_workload -- --ignored --nocapture`
#[test]
#[ignore]
fn print_workload() {
    let seed = env_u64("FLATS_SIM_SEED_BASE", 0);
    let ops = workload(seed);
    let from = env_u64("FROM", 0) as usize;
    let to = env_u64("TO", ops.len() as u64) as usize;
    print!("{}", sim::script(&ops[from..to]));
}
