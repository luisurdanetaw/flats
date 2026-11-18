//! A downstream crate that reaches into an internal module. It must FAIL to
//! compile against `flats` with default features: the write-ahead log's
//! record types are not part of the supported API.

fn main() {
    let _ = std::mem::size_of::<flats::wal::wal::Install>();
}
