#!/usr/bin/env bash
# Coverage-guided fuzzing with libFuzzer, AddressSanitizer on.
#
#   fuzz/run.sh                     # every target, 60 s each
#   fuzz/run.sh sql_parse 600       # one target, 10 minutes
#   FUZZ_JOBS=4 fuzz/run.sh all 3600
#
# Builds with the same flags `cargo fuzz` uses (sanitizer coverage, compare
# tracing, ASan), so neither this machine nor CI needs the cargo-fuzz plugin.
# Needs nightly. The corpus grows in fuzz/corpus/<target>; a crash lands in
# fuzz/artifacts/<target>/ — replay it with `fuzz/run.sh <target> 0 <file>`.
set -euo pipefail
cd "$(dirname "$0")"
TARGETS=(sql_parse sql_execute wal_record metadata_snapshot tuple_snapshot catalog)
which=${1:-all}
secs=${2:-60}
host=$(rustc +nightly -vV | sed -n 's/^host: //p')

export RUSTFLAGS="-Cpasses=sancov-module -Cllvm-args=-sanitizer-coverage-level=4 \
 -Cllvm-args=-sanitizer-coverage-inline-8bit-counters -Cllvm-args=-sanitizer-coverage-pc-table \
 -Cllvm-args=-sanitizer-coverage-trace-compares --cfg fuzzing -Cdebug-assertions \
 -Zsanitizer=address"
cargo +nightly build --release --target "$host" --bins
cargo +nightly run --release --target "$host" --bin make_seeds -- corpus >/dev/null

bins=target/$host/release
if [ "$which" = all ]; then set -- "${TARGETS[@]}"; else set -- "$which"; fi
status=0
for t in "$@"; do
  mkdir -p "corpus/$t" "artifacts/$t"
  if [ -n "${3:-}" ]; then
    "$bins/$t" "$3"; exit
  fi
  echo "== $t ($secs s)"
  # -rss_limit: an allocation the input can steer without bound is a bug too.
  "$bins/$t" "corpus/$t" -artifact_prefix="artifacts/$t/" -max_total_time="$secs" \
    -rss_limit_mb=2048 -max_len=65536 -jobs="${FUZZ_JOBS:-1}" -workers="${FUZZ_JOBS:-1}" \
    -print_final_stats=1 2>&1 | grep -E "^#|Done|stat::|ERROR|SUMMARY|panicked|crash-|leak" | tail -12 || status=1
  if ls "artifacts/$t"/* >/dev/null 2>&1; then
    echo "!! $t found inputs that crash: $(ls artifacts/$t)"; status=1
  fi
done
exit $status
