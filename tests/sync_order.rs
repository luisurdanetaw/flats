//! Power-loss safety, checked from the syscall trace.
//!
//! A SIGKILL keeps the OS page cache, so `crash_kill` can't tell whether an
//! `fsync` is missing; only a power cut would. Instead, run a workload under
//! `strace` and check the ORDER of the durability syscalls against the rules
//! a power cut enforces (the approach of ALICE, OSDI '14):
//!
//! * **R1** — when a write is acknowledged, the WAL holds no un-fsynced bytes.
//! * **R2** — a file is fsynced after its last write and before it is renamed
//!   into place (otherwise the rename can land before its contents).
//! * **R3** — after a rename, its directory is fsynced before the next
//!   acknowledgment (otherwise the rename itself can vanish).
//! * **R4** — after a file or directory is created, its parent directory is
//!   fsynced before the next acknowledgment (otherwise the file can vanish).
//!
//! Skipped (with a note) when `strace` isn't installed. `FLATS_SYNC_SEEDS`
//! sets the number of workloads.

mod sim;

use std::collections::{BTreeMap, BTreeSet};
use std::io::Write;
use std::path::Path;
use std::process::Command;

use sim::{Op, Sim, env_u64, gen_workload};

/// The child: run a workload, printing `ACK <i>` after each op returns.
#[test]
#[ignore]
fn sync_child() {
    let (Ok(dir), Ok(seed)) = (
        std::env::var("FLATS_SYNC_DIR"),
        std::env::var("FLATS_SYNC_SEED"),
    ) else {
        return;
    };
    let seed: u64 = seed.parse().unwrap();
    // Quiet background (even seed): every sync the trace shows belongs to
    // the op before the next ACK.
    let ops: Vec<Op> = gen_workload(seed * 2, 150)
        .into_iter()
        .filter(|op| !matches!(op, Op::Backup | Op::Crash))
        .collect();
    let mut sim = Sim::at(Path::new(&dir), seed * 2);
    let out = std::io::stdout();
    for (i, op) in ops.iter().enumerate() {
        if let Err(e) = sim.step(op) {
            eprintln!("CHILD-MISMATCH op {i}: {e}");
            std::process::exit(3);
        }
        let mut o = out.lock();
        let _ = writeln!(o, "ACK {i}");
        let _ = o.flush();
    }
}

/// One traced syscall: the call, its path arguments, and whether it failed.
#[derive(Debug)]
struct Call {
    line: usize,
    name: String,
    /// Paths named in the arguments: `<…>` fd annotations and quoted paths.
    paths: Vec<String>,
    raw: String,
}

fn parse(trace: &str) -> Vec<Call> {
    let mut out = Vec::new();
    for (line, raw) in trace.lines().enumerate() {
        // "<pid> name(args) = ret"; skip unfinished/resumed halves and
        // failed calls.
        let body = raw
            .split_once(' ')
            .map_or(raw, |(_, rest)| rest)
            .trim_start();
        let Some((name, args)) = body.split_once('(') else {
            continue;
        };
        if body.contains("<unfinished") || body.starts_with("<...") || body.contains("= -1") {
            continue;
        }
        let mut paths = Vec::new();
        // fd annotations: 5</abs/path>
        let mut rest = args;
        while let Some(i) = rest.find('<') {
            let after = &rest[i + 1..];
            let Some(j) = after.find('>') else { break };
            let p = &after[..j];
            if p.starts_with('/') {
                paths.push(p.to_string());
            }
            rest = &after[j + 1..];
        }
        // quoted string arguments that are absolute paths
        for part in args.split('"').skip(1).step_by(2) {
            if part.starts_with('/') {
                paths.push(part.to_string());
            }
        }
        out.push(Call {
            line: line + 1,
            name: name.to_string(),
            paths,
            raw: raw.to_string(),
        });
    }
    out
}

fn parent(p: &str) -> String {
    Path::new(p)
        .parent()
        .map(|d| d.display().to_string())
        .unwrap_or_default()
}

/// Check R1–R4 over a trace; `Err` names the rule, the line and the call.
fn check(calls: &[Call], db: &str) -> Result<usize, String> {
    let ours = |p: &str| p.starts_with(db);
    let mut dirty: BTreeSet<String> = BTreeSet::new(); // written, not yet fsynced
    let mut dir_pending: BTreeMap<String, String> = BTreeMap::new(); // dir -> why
    let mut acks = 0;
    for c in calls {
        let fail = |rule: &str, why: String| {
            Err(format!(
                "{rule} at trace line {}: {why}\n    {}",
                c.line, c.raw
            ))
        };
        match c.name.as_str() {
            "write" | "pwrite64" | "writev" | "pwritev" | "ftruncate" | "fallocate" => {
                if c.raw.contains("\"ACK ") {
                    acks += 1;
                    if let Some(w) = dirty.iter().find(|p| p.ends_with("/wal.log")) {
                        return fail("R1", format!("acknowledged with un-fsynced bytes in {w}"));
                    }
                    if let Some((d, why)) = dir_pending.iter().next() {
                        return fail("R3/R4", format!("acknowledged before fsync of {d} ({why})"));
                    }
                    continue;
                }
                for p in c.paths.iter().filter(|p| ours(p)) {
                    dirty.insert(p.clone());
                }
            }
            "fsync" | "fdatasync" => {
                for p in &c.paths {
                    dirty.remove(p);
                    dir_pending.remove(p);
                }
            }
            "rename" | "renameat" | "renameat2" => {
                let [src, dst] = match c.paths.as_slice() {
                    [s, d, ..] => [s.clone(), d.clone()],
                    _ => continue,
                };
                if !ours(&dst) {
                    continue;
                }
                if dirty.contains(&src) {
                    return fail(
                        "R2",
                        format!("{src} renamed over {dst} with un-fsynced contents"),
                    );
                }
                // The contents now live under the new name.
                dirty.remove(&src);
                dir_pending.insert(parent(&dst), format!("rename to {dst}"));
            }
            "openat" | "open" if c.raw.contains("O_CREAT") => {
                for p in c.paths.iter().filter(|p| ours(p)) {
                    // strace can't tell "created" from "already there", so
                    // every O_CREAT open counts as a creation (strict). A
                    // temp file is made durable by the rename that follows;
                    // LOCK holds no data.
                    if !p.ends_with(".tmp") && !p.ends_with("/LOCK") && !p.contains("/backup-") {
                        dir_pending
                            .entry(parent(p))
                            .or_insert_with(|| format!("create {p}"));
                    }
                }
            }
            "mkdir" | "mkdirat" => {
                for p in c.paths.iter().filter(|p| ours(p)) {
                    dir_pending.insert(parent(p), format!("mkdir {p}"));
                }
            }
            _ => {}
        }
    }
    Ok(acks)
}

#[test]
fn durability_syscalls_are_ordered_for_power_loss() {
    if Command::new("strace").arg("-V").output().is_err() {
        eprintln!("strace not installed: skipping the sync-order check");
        return;
    }
    for seed in 0..env_u64("FLATS_SYNC_SEEDS", 3) {
        let root = tempfile::tempdir().unwrap();
        let root_path = root.path().canonicalize().unwrap();
        let trace = root_path.join("trace.txt");
        let out = Command::new("strace")
            .args(["-f", "-y", "-s", "24", "-o"])
            .arg(&trace)
            .args([
                "-e",
                "trace=openat,write,pwrite64,writev,ftruncate,fallocate,fsync,fdatasync,rename,renameat,renameat2,mkdir,mkdirat",
            ])
            .arg(std::env::current_exe().unwrap())
            .args(["--ignored", "--exact", "sync_child", "--nocapture", "--test-threads=1"])
            .env("FLATS_SYNC_DIR", &root_path)
            .env("FLATS_SYNC_SEED", seed.to_string())
            .output()
            .expect("run strace");
        assert!(
            out.status.success(),
            "seed {seed}: child failed: {}",
            String::from_utf8_lossy(&out.stderr)
        );
        let text = std::fs::read_to_string(&trace).unwrap();
        let calls = parse(&text);
        let db = root_path.join("db").display().to_string();
        match check(&calls, &db) {
            Ok(acks) => assert!(acks > 50, "seed {seed}: only {acks} acknowledgments traced"),
            Err(e) => panic!("seed {seed}: {e}"),
        }
    }
}
