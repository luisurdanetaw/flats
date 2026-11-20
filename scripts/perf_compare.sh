#!/usr/bin/env bash
# Compare this checkout's performance with another commit's, on THIS machine:
#
#   scripts/perf_compare.sh <base-ref> [rounds]     # e.g. origin/main 5
#
# Builds tests/perf_probe.rs for both (the probe is copied into the base
# checkout if it predates it), runs them interleaved `rounds` times (default
# 5) so drift on a shared machine hits both alike, and compares the medians.
# Fails if any measurement is more than PERF_THRESHOLD (default 30) percent
# slower than the base. Absolute numbers are not the point; only the ratio is.
set -euo pipefail
base_ref=${1:?usage: perf_compare.sh <base-ref> [rounds]}
rounds=${2:-5}
threshold=${PERF_THRESHOLD:-30}
root=$(git rev-parse --show-toplevel)
work=$(mktemp -d)
cleanup() { git -C "$root" worktree remove --force "$work/base" >/dev/null 2>&1 || true; rm -rf "$work"; }
trap cleanup EXIT

git -C "$root" worktree add --detach "$work/base" "$base_ref" >/dev/null
[ -f "$work/base/tests/perf_probe.rs" ] || cp "$root/tests/perf_probe.rs" "$work/base/tests/"

probe() { # dir target-dir -> path of the built test binary
  (cd "$1" && CARGO_TARGET_DIR="$2" cargo test --release --test perf_probe --no-run --message-format=json 2>/dev/null) \
    | python3 -c 'import json,sys
for l in sys.stdin:
    try: m = json.loads(l)
    except ValueError: continue
    if m.get("executable") and "perf_probe" in m["executable"]: print(m["executable"])' | tail -1
}
head_bin=$(probe "$root" "$root/target")
base_bin=$(probe "$work/base" "$work/target") || true
if [ -z "$base_bin" ]; then
  echo "perf_compare: the probe does not build against $base_ref (its API differs); skipping"
  exit 0
fi

for i in $(seq "$rounds"); do
  "$base_bin" --ignored --nocapture 2>/dev/null | grep '^PERF' | sed 's/^/base /' >> "$work/runs"
  "$head_bin" --ignored --nocapture 2>/dev/null | grep '^PERF' | sed 's/^/head /' >> "$work/runs"
done

python3 - "$work/runs" "$threshold" "$base_ref" <<'PY'
import statistics, sys
runs, threshold, base_ref = sys.argv[1], float(sys.argv[2]), sys.argv[3]
data = {}
for line in open(runs):
    side, _, name, value = line.split()
    data.setdefault(name, {}).setdefault(side, []).append(float(value))
worst = 0.0
print(f"{'measurement':<18} {'base ms':>10} {'head ms':>10} {'change':>8}")
for name, sides in data.items():
    b, h = statistics.median(sides["base"]), statistics.median(sides["head"])
    change = (h - b) / b * 100
    worst = max(worst, change)
    flag = "  <-- slower" if change > threshold else ""
    print(f"{name:<18} {b:>10.2f} {h:>10.2f} {change:>+7.1f}%{flag}")
if worst > threshold:
    print(f"\nperf_compare: a measurement is more than {threshold:.0f}% slower than {base_ref}")
    sys.exit(1)
print(f"\nperf_compare: no measurement more than {threshold:.0f}% slower than {base_ref}")
PY
