//! Differential testing: random workloads run against the database and
//! against a reference model (`tests/sim`), compared statement by statement.
//!
//! Every statement's outcome must match the model: the same rows, the same
//! ranking (up to f32 rounding), the same errors. Metamorphic checks ride
//! along: a windowed SELECT is a slice of the unwindowed one, a wider SEARCH
//! extends the narrower one, and backups, checkpoints, compactions, clean
//! reopens and crashes leave every row exactly where the model says.
//!
//! Knobs (all optional):
//!
//! * `FLATS_SIM_SEEDS` — workloads per run (default 6)
//! * `FLATS_SIM_SEED_BASE` — first seed (default 0); CI shards set it
//! * `FLATS_SIM_OPS` — ops per workload (default 250)
//! * `FLATS_SIM_SEED` — run just this seed (to replay a failure)
//!
//! A failure is shrunk to a minimal workload and written to
//! `target/sim-failures/` (or `FLATS_SIM_OUT`) with the command to replay it.

mod sim;

use sim::{Sim, env_u64, fails, gen_workload, report_dir, script, shrink};

fn run_seed(seed: u64, n_ops: usize) {
    let ops = gen_workload(seed, n_ops);
    let mut sim = Sim::new(seed);
    let outcome = std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| sim.run(&ops, 50)));
    if std::env::var_os("FLATS_SIM_STATS").is_some() {
        eprintln!("seed {seed}: {:?}", sim.stats);
    }
    let (at, msg) = match outcome {
        Ok(Ok(())) => return,
        Ok(Err((at, msg))) => (at, msg),
        Err(_) => (ops.len(), "PANIC (see above)".to_string()),
    };
    let prefix: Vec<_> = ops[..(at + 1).min(ops.len())].to_vec();
    let minimal = if fails(seed, &prefix).is_some() {
        shrink(seed, prefix, env_u64("FLATS_SIM_SHRINK", 400) as usize)
    } else {
        prefix
    };
    let why = fails(seed, &minimal).unwrap_or_else(|| msg.clone());
    let report = format!(
        "seed {seed}: op {at} of {n_ops}: {msg}\n\n\
         minimal workload ({} ops) fails with:\n{why}\n\n{}\n\
         replay: FLATS_SIM_SEED={seed} FLATS_SIM_OPS={n_ops} cargo test --release --test differential\n",
        minimal.len(),
        script(&minimal)
    );
    let dir = report_dir();
    let _ = std::fs::create_dir_all(&dir);
    let _ = std::fs::write(dir.join(format!("differential-seed-{seed}.txt")), &report);
    panic!("{report}");
}

#[test]
fn random_workloads_match_the_model() {
    let n_ops = env_u64("FLATS_SIM_OPS", 250) as usize;
    if let Ok(seed) = std::env::var("FLATS_SIM_SEED") {
        run_seed(seed.parse().expect("FLATS_SIM_SEED is a number"), n_ops);
        return;
    }
    let base = env_u64("FLATS_SIM_SEED_BASE", 0);
    for seed in base..base + env_u64("FLATS_SIM_SEEDS", 6) {
        run_seed(seed, n_ops);
    }
}

#[test]
fn the_generator_exercises_every_kind_of_op() {
    // A guard on the guard: if a generator change stopped producing some
    // kind of op, the differential test would quietly stop covering it.
    let ops = gen_workload(1, 3000);
    let mut model = sim::Model::default();
    let mut seen = std::collections::BTreeMap::<&str, (u32, u32)>::new();
    for op in &ops {
        let kind = match op {
            sim::Op::Create(_) => "create",
            sim::Op::Drop(_) => "drop",
            sim::Op::Insert { .. } => "insert",
            sim::Op::Update { .. } => "update",
            sim::Op::Delete { .. } => "delete",
            sim::Op::Search { .. } => "search",
            sim::Op::Select { .. } => "select",
            sim::Op::Checkpoint => "checkpoint",
            sim::Op::Reopen => "reopen",
            sim::Op::Crash => "crash",
            sim::Op::Compact(_) => "compact",
            sim::Op::Backup => "backup",
        };
        let ok = !matches!(model.apply(op), sim::Expect::Error);
        let e = seen.entry(kind).or_default();
        if ok { e.0 += 1 } else { e.1 += 1 }
    }
    for kind in [
        "create",
        "drop",
        "insert",
        "update",
        "delete",
        "search",
        "select",
        "checkpoint",
        "reopen",
        "crash",
        "compact",
        "backup",
    ] {
        let (ok, err) = seen.get(kind).copied().unwrap_or_default();
        assert!(ok > 0, "{kind}: never generated a valid one ({seen:?})");
        if !matches!(
            kind,
            "checkpoint" | "reopen" | "crash" | "compact" | "backup"
        ) {
            assert!(err > 0, "{kind}: never generated an invalid one ({seen:?})");
        }
    }
}
