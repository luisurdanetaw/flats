//! The public surface, checked from OUTSIDE the crate.
//!
//! The crate's own tests are built with the `internals` feature (see the
//! dev-dependency in Cargo.toml), so nothing in this test binary can tell
//! whether a path is public to a real caller. A downstream crate can, so each
//! check builds one: a tiny fixture crate, written to a scratch directory,
//! depending on this package with DEFAULT features, and `cargo check`ed.
//!
//! * `uses_root`         — everything a caller needs, by crate-root paths. Must compile.
//! * `reaches_internals` — `flats::wal::…`. Must NOT compile.
//!
//! Builds offline against this package's own lockfile, so it needs nothing the
//! main build did not already fetch. The fixtures share one target directory,
//! so only the first check pays for compiling the dependency tree.

use std::path::{Path, PathBuf};
use std::process::Command;
use std::sync::Mutex;

/// Both fixtures share a target dir; serialize them so neither waits on the
/// other's build lock with a confusing "Blocking" message in its output.
static CARGO: Mutex<()> = Mutex::new(());

fn fixture(name: &str) -> (bool, String) {
    let _guard = CARGO.lock().unwrap_or_else(|e| e.into_inner());
    let manifest_dir = Path::new(env!("CARGO_MANIFEST_DIR"));
    let scratch = PathBuf::from(env!("CARGO_TARGET_TMPDIR")).join("api-fixtures");
    let crate_dir = scratch.join(name);
    std::fs::create_dir_all(crate_dir.join("src")).unwrap();
    std::fs::write(
        crate_dir.join("Cargo.toml"),
        format!(
            "[package]\nname = \"{name}\"\nversion = \"0.0.0\"\nedition = \"2024\"\n\
             [dependencies]\nflats = {{ path = {:?} }}\n[workspace]\n",
            manifest_dir
        ),
    )
    .unwrap();
    std::fs::copy(
        manifest_dir
            .join("tests/api_fixtures")
            .join(format!("{name}.rs")),
        crate_dir.join("src/main.rs"),
    )
    .unwrap();
    // Same versions the main build resolved, so `--offline` finds them all.
    std::fs::copy(
        manifest_dir.join("Cargo.lock"),
        crate_dir.join("Cargo.lock"),
    )
    .unwrap();

    let out = Command::new(env!("CARGO"))
        .args(["check", "--offline", "--quiet", "--manifest-path"])
        .arg(crate_dir.join("Cargo.toml"))
        .env("CARGO_TARGET_DIR", scratch.join("target"))
        // Inherited from the test harness's own cargo run; a fixture must be
        // built as a plain downstream crate, not under those settings.
        .env_remove("RUSTFLAGS")
        .env_remove("CARGO_ENCODED_RUSTFLAGS")
        .output()
        .expect("run cargo");
    (
        out.status.success(),
        String::from_utf8_lossy(&out.stderr).into_owned(),
    )
}

#[test]
fn a_downstream_crate_using_only_root_paths_compiles() {
    let (ok, stderr) = fixture("uses_root");
    assert!(
        ok,
        "the root API is missing something a caller needs:\n{stderr}"
    );
}

#[test]
fn a_downstream_crate_cannot_reach_the_wal() {
    let (ok, stderr) = fixture("reaches_internals");
    assert!(!ok, "flats::wal is reachable from outside the crate");
    assert!(
        stderr.contains("wal") && (stderr.contains("private") || stderr.contains("could not find")),
        "it failed, but not because the module is hidden:\n{stderr}"
    );
}
