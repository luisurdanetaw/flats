// `wal::wal` trips clippy::module_inception. Allowed rather than renamed:
// the path is public and imported by tests/benches; revisit when the public
// surface is locked down.
#[allow(clippy::module_inception)]
pub mod wal;
