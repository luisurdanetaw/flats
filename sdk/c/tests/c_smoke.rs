//! `include/flats.h` against the real library, from C.
//!
//! The Rust tests in `abi.rs` call the functions through Rust's view of their
//! signatures; this one compiles `smoke.c` with the system C compiler against
//! the HEADER and links the built shared library, so a header that drifts
//! from the implementation fails here.
//!
//! The library is built into a scratch target directory by a nested cargo
//! invocation (the outer one holds the main target's lock while tests run).

use std::path::PathBuf;
use std::process::Command;

#[test]
fn a_c_program_builds_against_the_header_and_runs() {
    let crate_dir = PathBuf::from(env!("CARGO_MANIFEST_DIR"));
    let scratch = PathBuf::from(env!("CARGO_TARGET_TMPDIR")).join("c-smoke");
    let target = scratch.join("target");
    std::fs::create_dir_all(&scratch).unwrap();

    let built = Command::new(env!("CARGO"))
        .args([
            "build",
            "--offline",
            "--quiet",
            "-p",
            "flats-c",
            "--manifest-path",
        ])
        .arg(crate_dir.join("Cargo.toml"))
        .env("CARGO_TARGET_DIR", &target)
        .env_remove("RUSTFLAGS")
        .env_remove("CARGO_ENCODED_RUSTFLAGS")
        .status()
        .expect("run cargo");
    assert!(built.success(), "building the shared library failed");
    let lib_dir = target.join("debug");
    assert!(
        lib_dir.join("libflats_c.so").exists(),
        "no libflats_c.so in {lib_dir:?}"
    );

    let exe = scratch.join("smoke");
    let cc = std::env::var("CC").unwrap_or_else(|_| "cc".to_string());
    let compiled = Command::new(&cc)
        .args(["-std=c99", "-Wall", "-Wextra", "-Werror", "-o"])
        .arg(&exe)
        .arg(crate_dir.join("tests/smoke.c"))
        .arg("-I")
        .arg(crate_dir.join("include"))
        .arg("-L")
        .arg(&lib_dir)
        .arg(format!("-Wl,-rpath,{}", lib_dir.display()))
        .arg("-lflats_c")
        .output()
        .expect("run the C compiler");
    assert!(
        compiled.status.success(),
        "smoke.c did not compile against flats.h:\n{}",
        String::from_utf8_lossy(&compiled.stderr)
    );

    let dir = tempfile::tempdir().unwrap();
    let backup = dir.path().join("backup");
    let ran = Command::new(&exe)
        .arg(dir.path().join("db"))
        .arg(&backup)
        .output()
        .expect("run smoke");
    let stdout = String::from_utf8_lossy(&ran.stdout);
    assert!(
        ran.status.success(),
        "smoke failed:\n{stdout}\n{}",
        String::from_utf8_lossy(&ran.stderr)
    );
    assert!(stdout.contains("ok every row is there"), "{stdout}");
}
