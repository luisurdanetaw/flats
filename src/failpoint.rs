//! Fault injection for tests: named points in the durability paths that can
//! be made to fail on demand. Compiled only with the `failpoints` feature;
//! without it, every [`fail_point!`](crate::fail_point) expands to nothing.
//!
//! A failpoint returns an I/O error exactly where a real one could happen: a
//! full disk at a write, `EIO` at an fsync, a failed rename. Failing at a
//! point and then dropping the database without a clean close leaves the
//! files exactly as a crash at that point would, which is how
//! `tests/failpoints.rs` crashes the engine at every step of a checkpoint, a
//! commit, a compaction flip, a WAL truncation and a catalog write.
//!
//! One point is armed at a time, process-wide. Tests that arm one must not
//! run concurrently with each other.

use std::collections::BTreeMap;
use std::io;
use std::sync::Mutex;
use std::sync::atomic::{AtomicU64, Ordering};

/// Every failpoint in the engine. A test can assert each one was reached.
pub const SITES: &[&str] = &[
    "wal.before_write",
    "wal.torn_write",
    "wal.before_fsync",
    "wal.after_fsync",
    "wal.truncate.before_rename",
    "checkpoint.before_flat_header",
    "checkpoint.before_metadata",
    "checkpoint.before_tuples",
    "checkpoint.before_catalog",
    "snapshot.before_rename",
    "catalog.before_write",
    "compact.before_flip",
    "compact.between_renames",
];

struct State {
    /// The armed point, and how many hits to let through before failing.
    armed: Option<(&'static str, u64)>,
    hits: BTreeMap<&'static str, u64>,
}

static STATE: Mutex<State> = Mutex::new(State {
    armed: None,
    hits: BTreeMap::new(),
});
static FIRED: AtomicU64 = AtomicU64::new(0);

fn state() -> std::sync::MutexGuard<'static, State> {
    STATE.lock().unwrap_or_else(|e| e.into_inner())
}

/// Make `site` fail on its `skip + 1`-th hit from now (and only then).
/// Replaces any point already armed.
pub fn arm(site: &'static str, skip: u64) {
    state().armed = Some((site, skip));
}

/// Disarm the armed point, if any.
pub fn disarm() {
    state().armed = None;
}

/// How many times any failpoint has fired in this process.
pub fn fired() -> u64 {
    FIRED.load(Ordering::SeqCst)
}

/// How many times `site` has been reached in this process, firing or not.
pub fn hits(site: &str) -> u64 {
    state().hits.get(site).copied().unwrap_or(0)
}

/// Called at each failpoint (through [`fail_point!`](crate::fail_point)).
pub fn hit(site: &'static str) -> io::Result<()> {
    let mut s = state();
    *s.hits.entry(site).or_insert(0) += 1;
    match &mut s.armed {
        Some((armed, skip)) if *armed == site => {
            if *skip > 0 {
                *skip -= 1;
                return Ok(());
            }
            s.armed = None;
            FIRED.fetch_add(1, Ordering::SeqCst);
            Err(io::Error::other(format!("failpoint {site}")))
        }
        _ => Ok(()),
    }
}
