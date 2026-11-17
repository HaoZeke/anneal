#!/usr/bin/env bash
# Compile the ISA fixtures and check the auditor's pass and fail cases.
# ISA_AUDIT_ARCH=x86_64 runs the x86 fixtures.
# ISA_AUDIT_ARCH=aarch64 runs the SVE fixtures.
# Unset, the host architecture is used, and a cross compiler is used for
# the other architecture when one is on PATH.
# ISA_AUDIT_OUT, when set, receives the object that must pass the audit.
set -euo pipefail

ROOT="$(cd "$(dirname "$0")/../../.." && pwd)"
AUDIT=(python3 "$ROOT/ci/isa-audit/isa_audit.py")
TESTS="$ROOT/ci/isa-audit/tests"
HOST="$(uname -m)"
ARCH="${ISA_AUDIT_ARCH:-$HOST}"

python3 -m pip install -q -r "$ROOT/ci/isa-audit/requirements.txt"

work="$(mktemp -d)"
trap 'rm -rf "$work"' EXIT

fail() {
  echo "FAIL: $*" >&2
  exit 1
}

expect_ok() {
  local out
  if ! out="$("${AUDIT[@]}" "$@" 2>&1)"; then
    printf '%s\n' "$out"
    fail "expected a clean audit: $*"
  fi
  printf '%s\n' "$out"
}

expect_fail() {
  local needle="$1"
  shift
  local out code
  set +e
  out="$("${AUDIT[@]}" "$@" 2>&1)"
  code=$?
  set -e
  printf '%s\n' "$out"
  if [[ "$code" -eq 0 ]]; then
    fail "expected a violation: $*"
  fi
  if ! grep -q -- "$needle" <<<"$out"; then
    fail "violation text missing [$needle]"
  fi
}

run_x86() {
  local cc="${CC_X86:-cc}"
  if ! command -v "$cc" >/dev/null 2>&1; then
    echo "x86_64 fixture not exercised: no C compiler"
    return
  fi
  echo "== x86_64 fixtures ($cc) =="
  # The fixtures assume the x86-64 baseline outside their dispatched
  # kernels. A distribution compiler whose default -march is higher
  # (Ubuntu 26.04's gcc emits VEX scalar moves in plain -O2 code) would
  # fail the clean cases for the wrong reason, so pin the baseline.
  local base=(-march=x86-64)
  "$cc" -c -O2 "${base[@]}" "$TESTS/ok.c" -o "$work/ok.o"
  "$cc" -c -O3 "${base[@]}" -mavx2 "$TESTS/bad_global.c" -o "$work/bad_global.o"
  "$cc" -c -O2 "${base[@]}" "$TESTS/bad_unlisted.c" -o "$work/bad_unlisted.o"
  "$cc" -c -O2 "${base[@]}" -mpopcnt "$TESTS/v2_popcnt.c" -o "$work/v2.o"
  "$cc" -shared -fPIC -O2 "${base[@]}" "$TESTS/ok_dyn.c" -o "$work/ok.so"

  expect_ok "$work/ok.o" --allow "$TESTS/allow-ok.toml" --json "$work/ok.json"
  python3 - "$work/ok.json" <<'PY'
import json, sys
report = json.load(open(sys.argv[1]))
counts = report["allowed_feature_counts"]
assert report["violations"] == [], report["violations"]
assert counts.get("avx512f", 0) > 0, counts
assert counts.get("avx", 0) > 0, counts
print("json ok:", counts)
PY

  expect_fail "hot_loop" "$work/bad_global.o" --allow "$TESTS/allow-ok.toml"
  expect_fail "avx2" "$work/bad_global.o" --allow "$TESTS/allow-ok.toml"
  expect_fail "secret_kernel" "$work/bad_unlisted.o" --allow "$TESTS/allow-ok.toml"
  expect_fail "avx2" "$work/bad_unlisted.o" --allow "$TESTS/allow-ok.toml"

  cp "$work/ok.o" "$work/ok.stripped.o"
  strip --strip-all "$work/ok.stripped.o"
  expect_fail "no symbols" "$work/ok.stripped.o" --allow "$TESTS/allow-ok.toml"

  expect_ok "$work/ok.so" --allow "$TESTS/allow-ok.toml"

  cp "$work/ok.o" "$work/libopenblas.so.0"
  expect_ok "$work/libopenblas.so.0" --allow "$TESTS/allow-vendored.toml"
  cp "$work/ok.o" "$work/librgpot.so"
  expect_fail "kern_avx512" "$work/librgpot.so" --allow "$TESTS/allow-vendored.toml"

  expect_fail "popcnt" "$work/v2.o" --allow "$TESTS/allow-ok.toml" --baseline x86-64
  expect_ok "$work/v2.o" --allow "$TESTS/allow-ok.toml" --baseline x86-64-v2

  "$cc" -c -O2 "${base[@]}" "$TESTS/ctz.c" -o "$work/ctz.o"
  "$cc" -c -O2 "${base[@]}" -mbmi "$TESTS/andn.c" -o "$work/andn.o"
  expect_ok "$work/ctz.o" --allow "$TESTS/allow-ok.toml"
  expect_fail "andn_bits" "$work/andn.o" --allow "$TESTS/allow-ok.toml"
  expect_fail "bmi1" "$work/andn.o" --allow "$TESTS/allow-ok.toml"

  # Dispatched clones pass with no allowlist entry; look-alikes do not.
  # -fno-ipa-icf keeps GCC from folding the identical bad bodies into one.
  local icf=()
  if "$cc" -fno-ipa-icf -x c -c /dev/null -o "$work/icf.o" 2>/dev/null; then
    icf=(-fno-ipa-icf)
  fi
  # The baseline pin does not reach libgcc.a, which keeps its toolchain's
  # default (Ubuntu 26.04's amd64v3 GCC has BMI2 in __cpu_indicator_init),
  # so the fixtures link without it and import __cpu_model, which the
  # proof reads through its GOT slot.
  local dso=(-shared -fPIC -nodefaultlibs)
  "$cc" "${dso[@]}" -O2 "${base[@]}" "$TESTS/dispatch_ok.c" -o "$work/dispatch_ok.so"
  "$cc" "${dso[@]}" -O2 "${base[@]}" "${icf[@]}" "$TESTS/dispatch_bad.c" -o "$work/dispatch_bad.so"
  expect_ok "$work/dispatch_ok.so" --allow "$TESTS/allow-dispatch.toml" --json "$work/dispatch.json"
  python3 - "$work/dispatch.json" <<'PY'
import json, sys
report = json.load(open(sys.argv[1]))
names = {item["name"]: item for item in report["dispatched"]}
assert report["violations"] == [], report["violations"]
assert names["clone_avx2"]["via"] == ["dispatch_entry"], names
assert names["clone_tail"]["via"] == ["clone_avx2"], names
assert names["probed_avx2"]["via"] == ["probed_entry"], names
print("dispatch ok:", sorted(names))
PY
  for name in unguarded_avx2 wrong_guard_avx2 inverted_guard_avx2 escaped_avx2 \
    partial_guard_avx2 exported_avx2 lying_probe_avx2 weak_probe_avx2; do
    expect_fail "$name" "$work/dispatch_bad.so" --allow "$TESTS/allow-dispatch.toml"
  done
  # At -O0 GCC calls the probes and its target_clones resolver uses jle.
  # Clang -O0 keeps the probe result in a stack slot, which the proof does
  # not follow, so its clones stay violations.
  if "$cc" --version | grep -q "Free Software Foundation"; then
    "$cc" "${dso[@]}" -O0 "${base[@]}" "$TESTS/dispatch_ok.c" -o "$work/dispatch_ok0.so"
    "$cc" "${dso[@]}" -O0 "${base[@]}" "${icf[@]}" "$TESTS/dispatch_bad.c" -o "$work/dispatch_bad0.so"
    expect_ok "$work/dispatch_ok0.so" --allow "$TESTS/allow-dispatch.toml"
    for name in unguarded_avx2 wrong_guard_avx2 inverted_guard_avx2 escaped_avx2 \
      partial_guard_avx2 exported_avx2 lying_probe_avx2 weak_probe_avx2; do
      expect_fail "$name" "$work/dispatch_bad0.so" --allow "$TESTS/allow-dispatch.toml"
    done
  fi
  for name in clone_scalar dispatch_entry; do
    python3 "$TESTS/plant.py" "$work/dispatch_ok.so" "$work/planted.so" "$name"
    expect_fail "$name 0x[0-9a-f]* vzeroupper avx" "$work/planted.so" \
      --allow "$TESTS/allow-dispatch.toml"
  done
  python3 "$TESTS/plant.py" "$work/dispatch_ok.so" "$work/planted.so" clone_avx2
  expect_ok "$work/planted.so" --allow "$TESTS/allow-dispatch.toml"

  rm -f "$work/ok.whl"
  python3 - "$work" <<'PY'
import sys, zipfile
from pathlib import Path
root = Path(sys.argv[1])
with zipfile.ZipFile(root / "ok.whl", "w") as zf:
    zf.write(root / "ok.o", "rgpot/ok.o")
PY
  expect_ok "$work/ok.whl" --allow "$TESTS/allow-ok.toml"

  if [[ -n "${ISA_AUDIT_OUT:-}" ]]; then
    mkdir -p "$ISA_AUDIT_OUT"
    cp "$work/ok.o" "$ISA_AUDIT_OUT/ok.o"
  fi
}

run_arm() {
  local cc="${CC_ARM:-}"
  if [[ "$HOST" == "aarch64" || "$HOST" == "arm64" ]]; then
    cc="${cc:-cc}"
  elif command -v aarch64-linux-gnu-gcc >/dev/null 2>&1; then
    cc="${cc:-aarch64-linux-gnu-gcc}"
  fi
  if [[ -z "$cc" ]] || ! command -v "$cc" >/dev/null 2>&1; then
    echo "aarch64 fixture not exercised: no aarch64 C compiler"
    return
  fi
  echo "== aarch64 fixtures ($cc) =="
  "$cc" -c -O2 -march=armv8-a "$TESTS/ok_sve.c" -o "$work/ok_sve.o"
  "$cc" -c -O2 -march=armv8-a "$TESTS/bad_sve.c" -o "$work/bad_sve.o"
  expect_ok "$work/ok_sve.o" --allow "$TESTS/allow-sve.toml" --baseline armv8-a \
    --json "$work/sve.json"
  python3 - "$work/sve.json" <<'PY'
import json, sys
report = json.load(open(sys.argv[1]))
assert report["violations"] == [], report["violations"]
assert report["allowed_feature_counts"].get("sve", 0) > 0, report
print("json ok:", report["allowed_feature_counts"])
PY
  expect_fail "other_sve" "$work/bad_sve.o" --allow "$TESTS/allow-sve.toml" --baseline armv8-a
  expect_fail "sve" "$work/bad_sve.o" --allow "$TESTS/allow-sve.toml" --baseline armv8-a

  "$cc" -shared -fPIC -O2 -moutline-atomics -static-libgcc \
    "$TESTS/outline.c" -o "$work/outline.so"
  outline_ok=0
  if ! nm "$work/outline.so" | grep -q '__aarch64_ldadd'; then
    echo "aarch64 outline-atomics fixture not exercised: linked so has no __aarch64_ldadd"
  else
    for prefix in __aarch64_cas __aarch64_ldadd __aarch64_swp __aarch64_ldclr __aarch64_ldeor __aarch64_ldset; do
      nm "$work/outline.so" | grep -q "$prefix" || fail "linked so missing $prefix"
    done
    expect_ok "$work/outline.so" --allow "$TESTS/allow-none.toml" --baseline armv8-a \
      --json "$work/outline.json"
    python3 - "$work/outline.json" <<'PY'
import json, sys
report = json.load(open(sys.argv[1]))
assert report["violations"] == [], report["violations"]
assert report["allowed_feature_counts"].get("lse", 0) > 0, report
print("outline atomics ok:", report["allowed_feature_counts"])
PY
    outline_ok=1
  fi
  "$cc" -c -O2 -march=armv8.1-a -mno-outline-atomics \
    "$TESTS/inline_lse.c" -o "$work/inline_lse.o"
  expect_fail "bump_inline" "$work/inline_lse.o" --allow "$TESTS/allow-none.toml" --baseline armv8-a
  expect_fail "lse" "$work/inline_lse.o" --allow "$TESTS/allow-none.toml" --baseline armv8-a

  if [[ -n "${ISA_AUDIT_OUT:-}" ]]; then
    mkdir -p "$ISA_AUDIT_OUT"
    cp "$work/ok_sve.o" "$ISA_AUDIT_OUT/ok_sve.o"
    if [[ "$outline_ok" -eq 1 ]]; then
      cp "$work/outline.so" "$ISA_AUDIT_OUT/outline.so"
    fi
  fi
}

case "$ARCH" in
  x86_64|amd64)
    run_x86
    if [[ "${ISA_AUDIT_ARCH:-}" == "x86_64" ]]; then
      echo "aarch64 fixture not exercised on this runner"
    elif command -v aarch64-linux-gnu-gcc >/dev/null 2>&1; then
      run_arm
    else
      echo "aarch64 fixture not exercised: no cross compiler"
    fi
    ;;
  aarch64|arm64)
    run_arm
    echo "x86_64 fixture not exercised on this runner"
    ;;
  *)
    fail "unsupported architecture $ARCH"
    ;;
esac

python3 -m unittest discover -s "$TESTS" -p 'test_*.py' -v
echo "self-test passed"
