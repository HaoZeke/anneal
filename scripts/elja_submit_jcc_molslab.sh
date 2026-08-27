#!/usr/bin/env bash
# Submit paired shared/control molecule and slab ensemble arrays on Elja.
set -euo pipefail

STAGE=${1:?development, qualification, or production}
ROOT=${LJ_ROOT:-$HOME/anneal-build}
SOURCE_ROOT=${JCC_SOURCE_ROOT:-$ROOT}
RUNNER=$ROOT/scripts/elja_jcc_molslab_ensemble.sh
OUT_ROOT=${MOLSLAB_OUT:-$HOME/ljwork/jcc}
CAMPAIGN=${JCC_CAMPAIGN:-jcc-2026-${STAGE}}

case $STAGE in
  development)
    ENSEMBLES=${JCC_ENSEMBLES:-4}
    SEED_OFFSET_BASE=${SEED_OFFSET_BASE:-0}
    ;;
  qualification)
    ENSEMBLES=${JCC_ENSEMBLES:-12}
    SEED_OFFSET_BASE=${SEED_OFFSET_BASE:-100000}
    ;;
  production)
    ENSEMBLES=${JCC_ENSEMBLES:-24}
    SEED_OFFSET_BASE=${SEED_OFFSET_BASE:-200000}
    ;;
  *)
    echo "stage must be development, qualification, or production" >&2
    exit 2
    ;;
esac

if [[ ! -x $RUNNER ]]; then
  echo "missing molecule/slab ensemble runner: $RUNNER" >&2
  exit 1
fi
SOURCE_COMMIT_FILE=$ROOT/SOURCE_COMMIT
if [[ ! -s $SOURCE_COMMIT_FILE || ! -s $ROOT/MOLSLAB_BUILD_SHA256SUMS ]]; then
  echo "missing source record or molecular build artifact seal below $ROOT" >&2
  exit 1
fi
IFS= read -r SOURCE_COMMIT <"$SOURCE_COMMIT_FILE"
HEAD=$(git -C "$SOURCE_ROOT" rev-parse HEAD)
if [[ $SOURCE_COMMIT != "$HEAD" ]]; then
  echo "SOURCE_COMMIT=$SOURCE_COMMIT does not match HEAD=$HEAD" >&2
  exit 2
fi
if ! (cd "$SOURCE_ROOT" && git diff --quiet HEAD --); then
  echo "tracked source differs from HEAD=$HEAD" >&2
  (cd "$SOURCE_ROOT" && git status --short >&2)
  exit 2
fi
(cd "$ROOT" && sha256sum -c MOLSLAB_BUILD_SHA256SUMS)
mkdir -p "$OUT_ROOT/submissions"
last=$((ENSEMBLES - 1))
systems=(h2o2:2500 h2o4:2500 h2o6:4000 cuh2:2500)
for specification in "${systems[@]}"; do
  IFS=: read -r system budget <<<"$specification"
  for soap_mode in flexible off; do
    for arm in shared control; do
      log="$OUT_ROOT/submissions/${CAMPAIGN}-${system}-${soap_mode}-${arm}-%A_%a.out"
      sbatch \
        --parsable \
        --job-name="jcc-${system}-${soap_mode}-${arm}" \
        --array="0-${last}" \
        --partition="${ELJA_PARTITION:-s-normal}" \
        --account="${ELJA_ACCOUNT:-chem-ui}" \
        --time="${ELJA_MOLSLAB_TIME:-12:00:00}" \
        --cpus-per-task="${ELJA_CPUS_PER_TASK:-8}" \
        --mem="${ELJA_MOLSLAB_MEM:-16G}" \
        --output="$log" \
        --export="ALL,MOLSLAB_CAMPAIGN=${CAMPAIGN},SEED_OFFSET_BASE=${SEED_OFFSET_BASE}" \
        "$RUNNER" "$system" "$budget" "$arm" slurm-array "$soap_mode"
    done
  done
done
