//! Diagnostics: the few things the engine reports that have no caller to
//! return an error to — a store rebuilt from the log on open, a salvage, a
//! background checkpoint or compaction that failed.
//!
//! By default each goes to stderr as one `flats: …` line. Install a hook to
//! send them to your own logger instead:
//!
//! ```
//! flats::diagnostics::set_log_hook(|level, message| {
//!     // e.g. log::log!(..), tracing::event!(..), or a metrics counter
//!     let _ = (level, message);
//! });
//! # flats::diagnostics::reset_log_hook();
//! ```
//!
//! The hook is process-wide and is called on engine threads (the WAL commit
//! thread, the flusher, the compactor): keep it fast, and don't call back
//! into the database from it.

use std::sync::RwLock;

/// How serious a diagnostic is.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Level {
    /// Something unexpected that the engine handled (a store rebuilt from the
    /// log, a salvage, background work recovering).
    Warn,
    /// Background work failing (checkpoints, compactions): the database still
    /// works, but something needs attention — a full disk, say.
    Error,
}

type Hook = Box<dyn Fn(Level, &str) + Send + Sync>;

static HOOK: RwLock<Option<Hook>> = RwLock::new(None);

/// Send every diagnostic to `hook` instead of stderr. Replaces any hook set
/// before.
pub fn set_log_hook(hook: impl Fn(Level, &str) + Send + Sync + 'static) {
    *HOOK.write().unwrap_or_else(|e| e.into_inner()) = Some(Box::new(hook));
}

/// Back to the default: one `flats: …` line on stderr per diagnostic.
pub fn reset_log_hook() {
    *HOOK.write().unwrap_or_else(|e| e.into_inner()) = None;
}

pub(crate) fn emit(level: Level, message: &str) {
    match &*HOOK.read().unwrap_or_else(|e| e.into_inner()) {
        Some(hook) => hook(level, message),
        None => eprintln!("flats: {message}"),
    }
}

pub(crate) fn warn(message: &str) {
    emit(Level::Warn, message);
}

pub(crate) fn error(message: &str) {
    emit(Level::Error, message);
}
