# Versioning

Flats follows [Semantic Versioning](https://semver.org). Before 1.0, a minor
bump (0.x → 0.y) may break things and a patch bump (0.x.y → 0.x.z) never
does. From 1.0 on, only a major bump breaks anything below.

## What is covered

- **The Rust API.** Covered: everything documented at the crate root, meaning
  the re-exports in `src/lib.rs` and the `diagnostics` module. Structs marked
  `#[non_exhaustive]` (`Stats`, `CollectionStats`) may gain fields in a minor
  version. Not covered: the `internals` feature. It exists for this crate's
  own tests, and anything behind it may change in any release.
- **V-SQL.** A statement accepted by one version is accepted, with the same
  meaning, by every later version in the same major. New statements and
  clauses may be added in a minor. A statement that was refused may start to
  work, but never the reverse.
- **The C ABI (`flats.h`).** Function names, signatures, status codes and
  `FLATS_TYPE_*` values are stable. New functions may be added in a minor.
  A library is a drop-in replacement for any earlier one with the same
  major.
- **The SDKs.** They are versioned with the crate. Methods, their arguments
  and the error classes they raise follow the same rules as the Rust API.

## On-disk formats

- **Reading old files.** A version opens every database written by the
  versions before it. The WAL reads back to `WAL_MIN_VERSION`, and the
  catalog reads v3 and later. It rewrites files in the current format as it
  goes: logs on truncation, the catalog on its next write.
  `tests/upgrade.rs` proves this against a database the previous build
  actually wrote, and a format change adds a new fixture (see
  `tests/fixtures/README.md`).

  The metadata and tuple snapshots and the vector index header read only
  their current version today. A bump to one of them must add a read path
  for the old version in the same change; without one, an old file would
  be refused as damaged.
- **Formats from the future.** A file newer than the build is refused by
  name ("unsupported version"), never guessed at.
- **Downgrading** is not supported. Once a newer version has rewritten a
  file, an older build refuses it. Back up before upgrading if you might
  roll back: `backup()` writes a copy the current version can open.
- **Bumping a version.** Every file type carries a version number:
  - the WAL;
  - the catalog;
  - the metadata and tuple snapshots;
  - the vector index header.

  A change to what a file holds bumps that number (CLAUDE.md §6), and the
  changelog's *Formats* section records it.

## Releasing

1. Move `CHANGELOG.md`'s *Unreleased* section under the new version number.
2. Bump the version in `Cargo.toml`, `sdk/c/Cargo.toml`,
   `sdk/node/Cargo.toml`, `sdk/node/package.json` and `sdk/python/setup.py`.
3. **Python:** `sdk/python/build_wheel.sh`, then `auditwheel repair` on a
   manylinux image.
4. **Node:** `sdk/node/scripts/prebuild.sh` on each platform, then
   `npm pack`.
5. **Rust:** `cargo publish -p flats`. This needs a `license` in
   `Cargo.toml`.
