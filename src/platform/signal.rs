//! Turning Ctrl-C into a flag instead of a death.
//!
//! By default `SIGINT` kills the process where it stands. For a database shell
//! that is the wrong behavior twice over:
//!
//! 1. It skips [`Db::close`](crate::engine::Db::close), so the final
//!    checkpoint never runs. Nothing is *lost* — every write was fsync'd
//!    through the WAL before it was acked — but the next open has to replay a
//!    WAL that should have been truncated.
//! 2. It is far too blunt. Ctrl-C during a million-row scan should stop the
//!    scan, not the session.
//!
//! [`install`] replaces that with a handler that does one async-signal-safe
//! thing — set an atomic — leaving the program to notice at a point where it
//! can react properly. Callers poll with [`take`].
//!
//! Note this covers Ctrl-C *while work is running*. While the line editor
//! holds the terminal, `ISIG` is off (see [`super::tty`]) and no signal is
//! generated at all: Ctrl-C is simply a byte the editor reads.

use std::io;
use std::sync::atomic::{AtomicBool, Ordering};

/// Set by the signal handler, cleared by [`take`].
///
/// A plain `bool` would be a data race — the handler runs on whatever thread
/// the kernel interrupts, at whatever instruction. An `AtomicBool` is both
/// race-free and, unlike almost everything else, safe to touch from a signal
/// handler: the store compiles to a single instruction that allocates nothing
/// and takes no lock.
static INTERRUPTED: AtomicBool = AtomicBool::new(false);

/// Whether a `SIGINT` has arrived since the last [`take`], clearing it.
///
/// Named for the fact that it CONSUMES the interrupt: two callers cannot both
/// see the same Ctrl-C, which is what stops one keypress from cancelling both
/// the running statement and the line typed after it.
pub fn take() -> bool {
    INTERRUPTED.swap(false, Ordering::SeqCst)
}

/// Forget any pending interrupt.
///
/// Called before starting work that should not inherit a Ctrl-C pressed while
/// something else was running.
pub fn clear() {
    INTERRUPTED.store(false, Ordering::SeqCst);
}

// ---------------------------------------------------------------------------
// FFI
// ---------------------------------------------------------------------------

#[cfg(target_os = "linux")]
mod imp {
    use super::{INTERRUPTED, Ordering};
    use std::ffi::c_int;
    use std::io;
    use std::ptr;

    const SIGINT: c_int = 2;

    /// `sigset_t` is 1024 bits on Linux, whatever the architecture.
    const SIGSET_WORDS: usize = 1024 / u64::BITS as usize;

    /// Linux's `struct sigaction`, as declared in `<bits/sigaction.h>`.
    ///
    /// Field ORDER is the part that matters and the part that varies between
    /// kernels: on Linux (glibc and musl alike, `x86_64` and `aarch64`) it is
    /// handler, mask, flags, restorer. Some other platforms put flags first —
    /// which is one more reason this module is Linux-gated rather than
    /// hopefully-portable.
    #[repr(C)]
    struct SigAction {
        sa_handler: Option<extern "C" fn(c_int)>,
        sa_mask: [u64; SIGSET_WORDS],
        sa_flags: c_int,
        /// Filled in by glibc's `sigaction` wrapper. We must pass null.
        sa_restorer: Option<extern "C" fn()>,
    }

    // Pinned to what `offsetof` reports on Linux — see the same guard in
    // `super::tty` for why a hand-rolled layout gets a compile-time check.
    const _: () = {
        assert!(size_of::<SigAction>() == 152);
        assert!(size_of::<[u64; SIGSET_WORDS]>() == 128);
        assert!(std::mem::offset_of!(SigAction, sa_handler) == 0);
        assert!(std::mem::offset_of!(SigAction, sa_mask) == 8);
        assert!(std::mem::offset_of!(SigAction, sa_flags) == 136);
        assert!(std::mem::offset_of!(SigAction, sa_restorer) == 144);
    };

    unsafe extern "C" {
        fn sigaction(signum: c_int, act: *const SigAction, oldact: *mut SigAction) -> c_int;
    }

    /// The handler itself. Everything it is allowed to do, it does.
    ///
    /// A signal handler may only call async-signal-safe functions: it can
    /// interrupt the program mid-`malloc`, so allocating (or printing, or
    /// locking) here can deadlock the process. One relaxed-ordering-or-stronger
    /// atomic store is safe, and is all this needs.
    extern "C" fn on_sigint(_signum: c_int) {
        INTERRUPTED.store(true, Ordering::SeqCst);
    }

    pub fn install() -> io::Result<()> {
        let action = SigAction {
            sa_handler: Some(on_sigint),
            // All zeroes is the empty mask: block nothing extra while the
            // handler runs. There is nothing to protect — the handler touches
            // exactly one atomic.
            sa_mask: [0; SIGSET_WORDS],
            // Deliberately NOT `SA_RESTART`. Without it, a blocking `read` on
            // stdin returns `EINTR` when the signal lands, which is how the
            // shell notices Ctrl-C while it is waiting for piped input. With
            // it, the read would silently resume and the interrupt would go
            // unseen until the next line arrived — possibly never.
            sa_flags: 0,
            sa_restorer: None,
        };
        // SAFETY: `action` is a fully-initialized `SigAction` matching the
        // platform layout, alive for the duration of the call, which
        // `sigaction` only reads. `on_sigint` is an `extern "C"` fn with the
        // signature the kernel calls, and is async-signal-safe. A null
        // `oldact` is explicitly permitted and means "do not report the
        // previous action".
        if unsafe { sigaction(SIGINT, &action, ptr::null_mut()) } != 0 {
            return Err(io::Error::last_os_error());
        }
        Ok(())
    }
}

#[cfg(not(target_os = "linux"))]
mod imp {
    use std::io;

    pub fn install() -> io::Result<()> {
        Err(io::Error::new(
            io::ErrorKind::Unsupported,
            "SIGINT handling is implemented for Linux only",
        ))
    }
}

/// Install the `SIGINT` handler.
///
/// Idempotent in effect — installing twice just re-registers the same handler.
///
/// On failure the caller keeps working with the default disposition (Ctrl-C
/// kills the process), which is worse but not broken: the WAL still holds
/// every acked write.
pub fn install() -> io::Result<()> {
    imp::install()
}

#[cfg(test)]
mod tests {
    use super::*;

    /// One test, not three: the flag is process-global, so separate tests
    /// would race each other under the default parallel test runner.
    #[test]
    fn the_flag_is_consumed_exactly_once() {
        clear();
        assert!(!take(), "nothing pending to start with");

        INTERRUPTED.store(true, Ordering::SeqCst);
        assert!(take(), "the pending interrupt is reported");
        assert!(
            !take(),
            "and only once — two consumers must not both see one Ctrl-C"
        );

        INTERRUPTED.store(true, Ordering::SeqCst);
        clear();
        assert!(!take(), "clear discards a pending interrupt");
    }
}
