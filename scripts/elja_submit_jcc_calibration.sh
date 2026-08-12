#!/usr/bin/env bash
# Submit four independent hard-LJ calibration tasks and their validator.
set -euo pipefail

ROOT=${LJ_ROOT:-$HOME/anneal-build}
REPRO_ROOT=${ANNEAL_REPRO_ROOT:-$HOME/anneal_repro}
CALIBRATOR=$ROOT/scripts/elja_jcc_lj_calibration.sh
FINALIZER=$ROOT/scripts/elja_jcc_finalize_calibration.sh
LOG_DIR=${JCC_CALIBRATION_LOG_DIR:-$REPRO_ROOT/results_jcc/calibration/logs}

[[ -x $CALIBRATOR ]] || { echo "missing calibrator: $CALIBRATOR" >&2; exit 1; }
[[ -x $FINALIZER ]] || { echo "missing finalizer: $FINALIZER" >&2; exit 1; }
mkdir -p "$LOG_DIR"

declare -A ENERGY=(
  [75]=-397.492331
  [98]=-543.665361
  [102]=-569.363652
  [104]=-582.086642
)
declare -A BASE_SEED=(
  [75]=7500000
  [98]=9800000
  [102]=10200000
  [104]=10400000
)

job_ids=()
for n in 75 98 102 104; do
  job_id=$(sbatch \
    --parsable \
    --job-name="jcc-cal-lj${n}" \
    --partition="${ELJA_PARTITION:-s-normal}" \
    --account="${ELJA_ACCOUNT:-chem-ui}" \
    --time="${JCC_CALIBRATION_TIME:-04:00:00}" \
    --cpus-per-task=1 \
    --mem="${JCC_CALIBRATION_MEM:-8G}" \
    --output="$LOG_DIR/lj${n}-%j.out" \
    --export="ALL,ANNEAL_REPRO_ROOT=${REPRO_ROOT}" \
    "$CALIBRATOR" "$n" "${ENERGY[$n]}" "${BASE_SEED[$n]}" "${JCC_CALIBRATION_SIGMA:-0.01}")
  job_ids+=("${job_id%%;*}")
  echo "submitted lj${n}: ${job_id}"
done

dependency=$(IFS=:; echo "${job_ids[*]}")
final_job=$(sbatch \
  --parsable \
  --dependency="afterok:${dependency}" \
  --job-name=jcc-cal-finalize \
  --partition="${ELJA_PARTITION:-s-normal}" \
  --account="${ELJA_ACCOUNT:-chem-ui}" \
  --time=00:15:00 \
  --cpus-per-task=1 \
  --mem=2G \
  --output="$LOG_DIR/finalize-%j.out" \
  --export="ALL,ANNEAL_REPRO_ROOT=${REPRO_ROOT}" \
  "$FINALIZER")
echo "submitted calibration finalizer: ${final_job}"
