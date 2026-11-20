#!/usr/bin/env bash
# Copyright (c) anneal developers
# Full 192-seed water-hexamer campaign in the pinned anneal_repro workflow.
# ANNEAL_REPRO_ROOT names the anneal_repro checkout; by default it is the
# sibling directory anneal_repro next to this repository.
set -euo pipefail

HERE=$(cd "$(dirname "$0")" && pwd)
ROOT=$(cd "$HERE/.." && pwd)
REPRO=${ANNEAL_REPRO_ROOT:-$ROOT/../anneal_repro}
REVISION=56160f2d100fc2c9f7365f4ee6d1d86653c8b31f
TREE=149899211dbec73aa88877cb7af0ebdf9b99f39e
CORES=${WATER_CREST_CORES:-8}

[[ -d $REPRO/.git ]] || {
  echo "missing anneal_repro checkout: $REPRO (set ANNEAL_REPRO_ROOT)" >&2
  exit 2
}
REPRO=$(cd "$REPRO" && pwd)
[[ $(git -C "$REPRO" rev-parse HEAD) == "$REVISION" ]] || {
  echo "anneal_repro must be at $REVISION" >&2
  exit 2
}
[[ $(git -C "$REPRO" rev-parse 'HEAD^{tree}') == "$TREE" ]] || {
  echo "anneal_repro tree does not match $TREE" >&2
  exit 2
}
[[ -z $(git -C "$REPRO" status --porcelain --untracked-files=no) ]] || {
  echo "anneal_repro tracked files are dirty" >&2
  exit 2
}
command -v pixi >/dev/null || { echo "pixi is required" >&2; exit 2; }

python3 "$ROOT/scripts/session_metadata.py" \
  --output "$ROOT/paper/jctc/data/water_v45b/sessions.jsonl" \
  --campaign water-crest-fair-n192 --root "$ROOT" --anneal-repro "$REPRO" -- \
  env "ANNEAL_REPRO_ROOT=$REPRO" \
  snakemake -s workflow/Snakefile water_crest_fair --cores "$CORES" --force

cd "$REPRO"
exec pixi run --locked --frozen snakemake water_crest_fair --cores "$CORES" --force
