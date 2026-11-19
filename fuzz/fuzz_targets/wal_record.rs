//! Decoding behind every framing check: any bytes must decode or be refused,
//! never panic, abort or allocate without bound.
#![no_main]
use libfuzzer_sys::fuzz_target;

fuzz_target!(|data: &[u8]| {
    let _ = flats::wal::wal::fuzz_decode_record(data);
});
