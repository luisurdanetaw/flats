//! Stopping a statement: `Db::interrupt` and statement timeouts.
//!
//! A statement runs under a [`Watch`] — the interrupt generation it started
//! in, and its deadline. The watch is installed in a THREAD-LOCAL around the
//! code that does the statement's work (opening its cursors, pulling a row),
//! and the long loops — the search kernel, a scan — call [`check`] between
//! blocks of work. That keeps every signature between the query layer and the
//! kernel unchanged: code that runs outside a statement sees no watch, and
//! `check` is a no-op there.
//!
//! An interrupt bumps a per-database generation; a statement is stopped when
//! the generation moved since it started, so an interrupt reaches exactly the
//! statements running when it was called, and none started after.

use std::cell::RefCell;
use std::sync::Arc;
use std::sync::atomic::{AtomicU64, Ordering};
use std::time::Instant;

use crate::error::{Error, Result};

/// What a running statement checks itself against.
#[derive(Clone)]
pub(crate) struct Watch {
    generation: Arc<AtomicU64>,
    started_in: u64,
    deadline: Option<Instant>,
}

impl Watch {
    /// A watch for a statement starting now.
    pub(crate) fn start(generation: &Arc<AtomicU64>, deadline: Option<Instant>) -> Watch {
        Watch {
            generation: Arc::clone(generation),
            started_in: generation.load(Ordering::Acquire),
            deadline,
        }
    }

    fn check(&self) -> Result<()> {
        if self.generation.load(Ordering::Acquire) != self.started_in {
            return Err(Error::Interrupted {
                reason: "interrupted by Db::interrupt".to_string(),
            });
        }
        if let Some(deadline) = self.deadline
            && Instant::now() >= deadline
        {
            return Err(Error::Interrupted {
                reason: "statement timeout".to_string(),
            });
        }
        Ok(())
    }
}

thread_local! {
    static CURRENT: RefCell<Option<Watch>> = const { RefCell::new(None) };
}

/// Run `f` with `watch` installed for this thread, restoring whatever was
/// installed before (statements do not nest today, but nothing here assumes
/// they cannot).
pub(crate) fn under<T>(watch: &Watch, f: impl FnOnce() -> T) -> T {
    let previous = CURRENT.with(|c| c.replace(Some(watch.clone())));
    struct Restore(Option<Watch>);
    impl Drop for Restore {
        fn drop(&mut self) {
            let previous = self.0.take();
            CURRENT.with(|c| *c.borrow_mut() = previous);
        }
    }
    let _restore = Restore(previous);
    f()
}

/// Run `f` under the watch already installed on this thread, or — when none
/// is, as for a direct `Db::search` call — under a fresh one from `make`. A
/// search issued BY a statement must keep that statement's clock and
/// interrupt generation, not start its own.
pub(crate) fn under_current_or<T>(make: impl FnOnce() -> Watch, f: impl FnOnce() -> T) -> T {
    if CURRENT.with(|c| c.borrow().is_some()) {
        f()
    } else {
        under(&make(), f)
    }
}

/// Stop if the statement running on this thread was interrupted or ran past
/// its deadline. A no-op outside a statement.
pub(crate) fn check() -> Result<()> {
    CURRENT.with(|c| match c.borrow().as_ref() {
        Some(watch) => watch.check(),
        None => Ok(()),
    })
}

/// How many rows the search kernel scores between checks: small enough that a
/// stop lands within a fraction of a millisecond, large enough that the check
/// (a thread-local read, an atomic load, a clock read) is noise.
pub(crate) const KERNEL_CHECK_EVERY: usize = 4096;
