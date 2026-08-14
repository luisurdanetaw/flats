//! Terminal mode control via `termios`.
//!
//! A line editor needs the terminal to stop being helpful. In its default
//! *canonical* mode the kernel buffers a whole line, draws the characters
//! itself, and only hands the program bytes once Enter is pressed — which
//! means an arrow key is never seen by the program at all, it is just three
//! bytes appended to a buffer the kernel is editing on our behalf (badly, with
//! no history and no cursor movement).
//!
//! [`RawMode`] turns that off for as long as it is alive, and turns it back on
//! when it drops.

use std::io;

#[cfg(target_os = "linux")]
use std::ffi::{c_int, c_uchar, c_uint};
#[cfg(target_os = "linux")]
use std::mem::MaybeUninit;

// ---------------------------------------------------------------------------
// FFI
// ---------------------------------------------------------------------------

/// Size of `termios.c_cc` on Linux.
#[cfg(target_os = "linux")]
const NCCS: usize = 32;

/// Linux's `struct termios`, as declared in `<bits/termios.h>`.
///
/// The layout is fixed ABI: both glibc and musl lay this out identically on
/// every architecture Flats targets (`x86_64`, `aarch64`). Getting it wrong
/// would be undefined behavior, so it is written out field for field rather
/// than approximated — and `tcgetattr` only ever fills a value we then hand
/// straight back to `tcsetattr`, so no field is interpreted except the four
/// flag words and two `c_cc` slots named below.
#[cfg(target_os = "linux")]
#[repr(C)]
#[derive(Clone, Copy)]
struct Termios {
    c_iflag: c_uint,
    c_oflag: c_uint,
    c_cflag: c_uint,
    c_lflag: c_uint,
    c_line: c_uchar,
    c_cc: [c_uchar; NCCS],
    c_ispeed: c_uint,
    c_ospeed: c_uint,
}

// The layout above is load-bearing and unverifiable at runtime: a wrong offset
// does not fail, it silently reads and writes the wrong bytes of a struct libc
// owns. These pin it to the values `offsetof` reports on Linux, so a port that
// changes the shape fails to COMPILE rather than corrupting terminal state.
#[cfg(target_os = "linux")]
const _: () = {
    assert!(size_of::<Termios>() == 60);
    assert!(std::mem::offset_of!(Termios, c_iflag) == 0);
    assert!(std::mem::offset_of!(Termios, c_oflag) == 4);
    assert!(std::mem::offset_of!(Termios, c_cflag) == 8);
    assert!(std::mem::offset_of!(Termios, c_lflag) == 12);
    assert!(std::mem::offset_of!(Termios, c_line) == 16);
    assert!(std::mem::offset_of!(Termios, c_cc) == 17);
    assert!(std::mem::offset_of!(Termios, c_ispeed) == 52);
    assert!(std::mem::offset_of!(Termios, c_ospeed) == 56);
};

#[cfg(target_os = "linux")]
unsafe extern "C" {
    fn tcgetattr(fd: c_int, termios_p: *mut Termios) -> c_int;
    fn tcsetattr(fd: c_int, optional_actions: c_int, termios_p: *const Termios) -> c_int;
}

// Flag bits and `c_cc` indices, from `<bits/termios.h>`. Spelled in octal the
// way the header does, so they can be diffed against it by eye.
#[cfg(target_os = "linux")]
mod flags {
    /// Input: map CR to NL. Off, so Enter arrives as the `\r` it really is.
    pub const ICRNL: u32 = 0o0000400;
    /// Input: Ctrl-S / Ctrl-Q flow control. Off, so those are ordinary keys.
    pub const IXON: u32 = 0o0002000;
    /// Local: generate SIGINT/SIGQUIT/SIGTSTP from keystrokes.
    pub const ISIG: u32 = 0o0000001;
    /// Local: line-at-a-time input. The one that matters most.
    pub const ICANON: u32 = 0o0000002;
    /// Local: echo input characters.
    pub const ECHO: u32 = 0o0000010;

    /// `c_cc` index: read timeout in deciseconds.
    pub const VTIME: usize = 5;
    /// `c_cc` index: minimum bytes for a completed read.
    pub const VMIN: usize = 6;

    /// `tcsetattr`: apply after queued output drains, keeping queued INPUT.
    pub const TCSADRAIN: i32 = 1;
}

// ---------------------------------------------------------------------------
// RawMode
// ---------------------------------------------------------------------------

/// Puts a terminal in raw mode, and restores the previous mode on drop.
///
/// Restoring on `Drop` rather than at the end of a function is the whole
/// design: a terminal left in raw mode outlives the process and leaves the
/// user with a shell that does not echo. Every early return, `?`, and panic
/// unwind therefore has to restore it, and `Drop` is the only construct that
/// covers all three.
///
/// # Example
///
/// ```no_run
/// # use flats::platform::tty::RawMode;
/// let guard = RawMode::enable(0)?; // 0 = stdin
/// // ... read keystrokes one byte at a time ...
/// drop(guard); // terminal is back to normal
/// # Ok::<(), std::io::Error>(())
/// ```
pub struct RawMode(Inner);

#[cfg(target_os = "linux")]
struct Inner {
    fd: c_int,
    /// The mode to put back. Captured before any modification.
    saved: Termios,
}

/// Never constructed off Linux — `enable` always fails there.
#[cfg(not(target_os = "linux"))]
#[allow(dead_code)]
struct Inner;

impl RawMode {
    /// Switch `fd` to raw mode.
    ///
    /// Fails if `fd` is not a terminal — which is the normal case for piped
    /// input, and why the caller must treat this as an ordinary, expected
    /// error rather than a bug.
    #[cfg(target_os = "linux")]
    pub fn enable(fd: i32) -> io::Result<Self> {
        use flags::*;

        let mut saved = MaybeUninit::<Termios>::uninit();
        // SAFETY: `saved` is a live, correctly-aligned, writable allocation of
        // exactly `Termios` size, which is what `tcgetattr` expects to fill.
        // The call writes the whole struct or returns non-zero, so the
        // `assume_init` below is only reached once it is fully initialized.
        if unsafe { tcgetattr(fd, saved.as_mut_ptr()) } != 0 {
            return Err(io::Error::last_os_error());
        }
        // SAFETY: `tcgetattr` returned 0, so it initialized every field.
        let saved = unsafe { saved.assume_init() };

        let mut raw = saved;
        // ISIG off is what makes Ctrl-C readable as the byte 0x03 instead of a
        // signal: while the editor is running, Ctrl-C should abandon the line
        // being typed, not tear down the shell. (Ctrl-C during a QUERY is a
        // different story — the terminal is back in cooked mode by then, and
        // `signal::install_sigint` catches it.)
        raw.c_lflag &= !(ISIG | ICANON | ECHO);
        raw.c_iflag &= !(ICRNL | IXON);
        // OPOST is deliberately LEFT ON. Canonical raw mode clears it, which
        // stops `\n` from also returning the carriage — and would force every
        // `writeln!` in the shell, including ones that never touch the editor,
        // to spell `\r\n`. Output post-processing costs us nothing here; the
        // editor already emits an explicit `\r` when it needs column zero.
        raw.c_cc[VMIN] = 1; // block until at least one byte is available
        raw.c_cc[VTIME] = 0; // ... with no timeout

        // TCSADRAIN, not the more usual TCSAFLUSH: flushing would DISCARD
        // input already queued, and raw mode is entered once per line read.
        // Anything typed (or pasted) while the previous statement was running
        // is sitting in that queue, and dropping it would silently eat pasted
        // scripts.
        //
        // SAFETY: `&raw` points to a fully-initialized `Termios` that lives
        // for the duration of the call, and `tcsetattr` only reads from it.
        if unsafe { tcsetattr(fd, TCSADRAIN, &raw) } != 0 {
            return Err(io::Error::last_os_error());
        }
        Ok(RawMode(Inner { fd, saved }))
    }

    /// Switch `fd` to raw mode.
    ///
    /// Always fails on non-Linux targets: v1 is Linux-only (CLAUDE.md §3), and
    /// the shell falls back to line-at-a-time input when this errors.
    #[cfg(not(target_os = "linux"))]
    pub fn enable(_fd: i32) -> io::Result<Self> {
        Err(io::Error::new(
            io::ErrorKind::Unsupported,
            "raw terminal mode is implemented for Linux only",
        ))
    }
}

impl Drop for RawMode {
    fn drop(&mut self) {
        #[cfg(target_os = "linux")]
        {
            // Failure is unreportable and unactionable here — and the common
            // cause is the terminal already being gone, in which case there is
            // nothing left to restore.
            //
            // SAFETY: `self.0.saved` is a fully-initialized `Termios` captured
            // from this same `fd` in `enable`, and `tcsetattr` only reads it.
            unsafe { tcsetattr(self.0.fd, flags::TCSADRAIN, &self.0.saved) };
        }
    }
}
