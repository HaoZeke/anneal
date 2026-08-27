#!/usr/bin/env bash
# Generate one immutable hard-LJ development calibration pool on an Elja compute node.
set -euo pipefail

if [[ -z ${SLURM_JOB_ID:-} ]]; then
  echo "elja_jcc_lj_calibration.sh must run under Slurm" >&2
  exit 1
fi

N=${1:?LJ point count}
BASE_SEED=${2:?base search seed}
SIGMA=${3:-0.01}
case "$N" in
  38|75|98|102|104) ;;
  *) echo "unsupported calibration size: $N" >&2; exit 2 ;;
esac

ANNEAL_ROOT=${LJ_ROOT:-$HOME/anneal-build}
REPRO_ROOT=${ANNEAL_REPRO_ROOT:-$HOME/anneal_repro}
PAIR_COUNT=${JCC_CALIBRATION_PAIRS:-200}
PAIR_DIR=$REPRO_ROOT/development/jcc/census_pairs
SIGNATURE_DIR=$REPRO_ROOT/development/jcc/signatures
PAIR_OUTPUT=$PAIR_DIR/lj${N}.jsonl
SIGNATURE_OUTPUT=$SIGNATURE_DIR/lj${N}.json
PARTIAL_PAIR=$PAIR_DIR/.lj${N}.${SLURM_JOB_ID}.jsonl
PARTIAL_SIGNATURE=$SIGNATURE_DIR/.lj${N}.${SLURM_JOB_ID}.json
BINARY=$ANNEAL_ROOT/target/release/examples/lj_census_calibration
PYTHON=${JCC_PYTHON:-$HOME/rgpot/.pixi/envs/xtbbld/bin/python}

[[ -x $BINARY ]] || { echo "missing calibration executable: $BINARY" >&2; exit 1; }
[[ -x $PYTHON ]] || { echo "missing calibration Python: $PYTHON" >&2; exit 1; }
[[ ! -e $PAIR_OUTPUT ]] || { echo "immutable output exists: $PAIR_OUTPUT" >&2; exit 1; }
[[ ! -e $SIGNATURE_OUTPUT ]] || { echo "immutable output exists: $SIGNATURE_OUTPUT" >&2; exit 1; }
[[ ! -e $PARTIAL_PAIR && ! -e $PARTIAL_SIGNATURE ]] || {
  echo "partial output already exists for job $SLURM_JOB_ID" >&2
  exit 1
}

mkdir -p "$PAIR_DIR" "$SIGNATURE_DIR"
export IRA_LIB_DIR=${IRA_LIB_DIR:-$HOME/ira/lib}
GCCLIB=${GCCLIB:-/opt/ohpc/pub/compiler/gcc/12.4.0/lib64}
export LD_LIBRARY_PATH="${IRA_LIB_DIR}:${GCCLIB}:${LD_LIBRARY_PATH:-}"

"$BINARY" \
  "$N" "$PAIR_COUNT" "$BASE_SEED" "$SIGMA" \
  "$PARTIAL_PAIR" "$PARTIAL_SIGNATURE"

"$PYTHON" -m json.tool "$PARTIAL_SIGNATURE" >/dev/null
"$PYTHON" - "$PARTIAL_PAIR" "$PAIR_COUNT" <<'PY'
import json
import sys

path, expected_text = sys.argv[1:]
expected = int(expected_text)
with open(path, encoding="utf-8") as stream:
    rows = [json.loads(line) for line in stream if line.strip()]
if len(rows) != expected:
    raise SystemExit(f"expected {expected} pairs, found {len(rows)}")
if len({row["pair_id"] for row in rows}) != expected:
    raise SystemExit("duplicate calibration pair identifiers")
PY

mv "$PARTIAL_PAIR" "$PAIR_OUTPUT"
mv "$PARTIAL_SIGNATURE" "$SIGNATURE_OUTPUT"
sha256sum "$PAIR_OUTPUT" "$SIGNATURE_OUTPUT"
echo "CALIBRATION_TASK_OK lj${N} pairs=${PAIR_COUNT} job=${SLURM_JOB_ID}"
