//! Hand-rolled platform bindings.
//!
//! This is one of the two `unsafe` quarantines (CLAUDE.md §5); the other is
//! [`crate::simd`]. Nothing here has any database logic in it — it exists so
//! that the rest of the crate can talk to the operating system without a
//! `libc` dependency (CLAUDE.md §2), and so that every `extern "C"` block in
//! the project sits in one auditable place.
//!
//! # What lives here
//!
//! - [`tty`] — terminal mode control (`termios`), for the shell's line editor.
//! - [`signal`] — a `SIGINT` handler that turns Ctrl-C into a flag the program
//!   can poll instead of a process death.
//!
//! # Platform support
//!
//! v1 is Linux-only (CLAUDE.md §3), and the struct layouts below are Linux's.
//! Rather than fail to compile elsewhere, every entry point has a
//! non-Linux stub that reports [`io::ErrorKind::Unsupported`]. Callers must
//! already handle failure — a terminal can refuse raw mode for a dozen
//! ordinary reasons — so degrading is free, and it keeps `cargo check` honest
//! for anyone developing on a Mac.

pub mod signal;
pub mod tty;
