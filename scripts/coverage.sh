#!/usr/bin/env bash
# Line and branch coverage of the engine (src/) under the WHOLE test suite —
# unit tests, integration tests, and the randomized harnesses (differential,
# crash_kill with its child processes, failpoints).
#
# Uses rustc's own instrumentation (`-C instrument-coverage`) and the
# llvm-profdata / llvm-cov that ship in the `llvm-tools` rustup component —
# no cargo plugin to install.
#
#   scripts/coverage.sh                 # summary per file, total at the end
#   scripts/coverage.sh --html          # also target/coverage/html/index.html
#   COVERAGE_MIN=80 scripts/coverage.sh # fail if line coverage is below 80%
#
# Needs a toolchain with llvm-tools (`rustup component add llvm-tools`), and
# nightly for branch coverage (`-Z coverage-options=branch`); on stable it
# reports lines and regions only.
set -euo pipefail
cd "$(dirname "$0")/.."

tools() { echo "$(rustc --print sysroot)/lib/rustlib/$(rustc -vV | sed -n 's/^host: //p')/bin"; }
bin=$(tools)
if [ ! -x "$bin/llvm-profdata" ] && rustup run nightly rustc --version >/dev/null 2>&1; then
  export RUSTUP_TOOLCHAIN=nightly   # the active toolchain lacks llvm-tools; nightly may have them
  bin=$(tools)
fi
[ -x "$bin/llvm-profdata" ] || { echo "llvm-tools missing: rustup component add llvm-tools" >&2; exit 2; }
echo "toolchain: $(rustc --version)"

out=target/coverage
if [ "${COVERAGE_REPORT_ONLY:-}" != 1 ]; then rm -rf "$out" && mkdir -p "$out/profraw"; fi
export CARGO_TARGET_DIR=target/coverage-build
export LLVM_PROFILE_FILE="$PWD/$out/profraw/%p-%m.profraw"
flags="-C instrument-coverage"
if rustc --version | grep -q nightly; then
  flags="$flags -Z coverage-options=branch"
fi
export RUSTFLAGS="$flags"

# Keep the randomized harnesses short: coverage, not depth.
export FLATS_SIM_SEEDS=${FLATS_SIM_SEEDS:-12} FLATS_CRASH_ITERS=${FLATS_CRASH_ITERS:-8} \
       FLATS_FP_SEEDS=${FLATS_FP_SEEDS:-12} FLATS_FUZZ_ITERS=${FLATS_FUZZ_ITERS:-300}

# Timing assertions measure the instrumentation, not the code: skip them.
[ "${COVERAGE_REPORT_ONLY:-}" = 1 ] || cargo test --release --features failpoints --no-fail-fast -- --skip parallel_search_scales_with_threads \
  2>&1 | grep -E "^test result|FAILED|panicked" || true

"$bin/llvm-profdata" merge -sparse "$out"/profraw/*.profraw -o "$out/merged.profdata"

# Every test binary the run used (the instrumented code lives in each). Ask
# cargo rather than guess a directory layout: it has changed between releases.
# (llvm-cov takes the first one positionally, the rest as -object.)
objects=()
while IFS= read -r f; do
  if [ ${#objects[@]} -eq 0 ]; then objects+=("$f"); else objects+=(-object "$f"); fi
done < <(cargo test --release --features failpoints --no-run --message-format=json 2>/dev/null \
  | python3 -c 'import json,sys
for line in sys.stdin:
    try: m = json.loads(line)
    except ValueError: continue
    if m.get("reason") == "compiler-artifact" and m.get("executable"): print(m["executable"])')
[ ${#objects[@]} -gt 0 ] || { echo "no test binaries found" >&2; exit 2; }

ignore='(/\.cargo/registry/|/rustc/|/library/|^tests/|/tests/|benches/|sdk/|src/main\.rs|src/editor\.rs|src/platform/tty\.rs)'
"$bin/llvm-cov" report --instr-profile="$out/merged.profdata" "${objects[@]}" \
  --ignore-filename-regex="$ignore" > "$out/report.txt"
sed 's#^.*/src/#src/#' "$out/report.txt"

if [ "${1:-}" = "--html" ]; then
  "$bin/llvm-cov" show --format=html --output-dir="$out/html" --show-branches=count \
    --instr-profile="$out/merged.profdata" "${objects[@]}" --ignore-filename-regex="$ignore"
  echo "HTML: $out/html/index.html"
fi

# The TOTAL row: Regions Missed Cover Functions Missed Executed Lines Missed Cover ...
lines=$(awk '/^TOTAL/ {print $10}' "$out/report.txt" | tr -d '%')
echo "line coverage: ${lines}%"
if [ -n "${COVERAGE_MIN:-}" ] && awk -v l="$lines" -v m="$COVERAGE_MIN" 'BEGIN{exit !(l < m)}'; then
  echo "below the ${COVERAGE_MIN}% floor" >&2
  exit 1
fi
