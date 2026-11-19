//! The lexer and parser: any text must parse or be refused, never panic,
//! overflow the stack or allocate without bound.
#![no_main]
use libfuzzer_sys::fuzz_target;

fuzz_target!(|data: &[u8]| {
    if let Ok(sql) = std::str::from_utf8(data) {
        let _ = flats::sql::parser::parse(sql);
    }
});
