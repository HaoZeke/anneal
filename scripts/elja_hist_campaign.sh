#!/usr/bin/env bash
# Submit one ensemble campaign to HyperQueue on Elja from an arm table.
#
# Usage: elja_hist_campaign.sh <campaign-dir> <n> <arms.tsv> [seeds=48]
#
# arms.tsv has one arm per line: <name> <budget> <cpus> [KEY=VALUE ...]
# Blank lines and lines starting with # are skipped. Every arm runs every
# seed 0..seeds-1 as one task, through hist_one.sh, on the executable in
# LJ_BIN (default the staging build). The campaign directory receives
# EXE_SHA, SOURCE_NOTE (from $SOURCE_NOTE, or the executable's build log
# reference), the arm table, and one <name>_<seed>.out per task, which
# scripts/hist_summarize.py reads.
set -euo pipefail
DIR=${1:?campaign dir}
N=${2:?n}
ARMS=${3:?arms.tsv}
SEEDS=${4:-48}
BIN=${LJ_BIN:-$HOME/anneal-psym/target/release/examples/lj_cluster_search}
MECH=${MECHANISMS:-thompson,rscreen,psymnew}
mkdir -p "$DIR"
cp "$ARMS" "$DIR/arms.tsv"
cp "$(dirname "$0")/elja_hist_one.sh" "$DIR/hist_one.sh"
chmod +x "$DIR/hist_one.sh"
sha256sum "$BIN" | tee "$DIR/EXE_SHA"
{
  echo "campaign $(basename "$DIR") n=$N seeds=$SEEDS mechanisms=$MECH"
  echo "submitted $(date -u +%FT%TZ)"
  echo "${SOURCE_NOTE:-source: see git log of the staging tree}"
} > "$DIR/SOURCE_NOTE"
last=$((SEEDS - 1))
while read -r name budget cpus rest; do
  [[ -z "$name" || "$name" == \#* ]] && continue
  envs=()
  for kv in $rest; do envs+=(--env "$kv"); done
  hq submit --name "$name" --array "0-$last" --cpus "$cpus" --time-limit "${TIME_LIMIT:-5h}" \
    --env "LJ_BIN=$BIN" "${envs[@]}" \
    --stdout "$DIR/${name}_%{TASK_ID}.out" --stderr "$DIR/${name}_%{TASK_ID}.err" \
    bash "$DIR/hist_one.sh" "$N" "$budget" "$MECH" | tail -1 | tee -a "$DIR/SOURCE_NOTE"
done < "$ARMS"
