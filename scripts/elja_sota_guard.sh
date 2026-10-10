#!/usr/bin/env bash
# Submit the SOTA guard campaign on Elja: the arms that carry the method's
# numbers (LJ75 one orbit chain of 4e6: 31/48 on 2026-09-07; LJ98 two orbit
# chains of 2e6: 33/48). Run after any change to the hop loop, the moves,
# or the symmetrisation, on a build of the staging tree, and compare with
#   python3 hist_summarize.py <campaign-dir>
# A count below 25/48 on LJ75 or 27/48 on LJ98 (about two standard
# deviations below the recorded rates) is a regression to investigate.
#
# Usage: elja_sota_guard.sh <campaign-root> [seeds=48]
set -euo pipefail
ROOT=${1:?campaign root}
SEEDS=${2:-48}
HERE=$(cd "$(dirname "$0")" && pwd)
export MECHANISMS=thompson,rscreen,orbit
export SOURCE_NOTE="sota guard: orbit chains; pass LJ75 >= 25/$SEEDS, LJ98 >= 27/$SEEDS"
bash "$HERE/elja_hist_campaign.sh" "$ROOT/guard75" 75 "$HERE/sota_guard/lj75.tsv" "$SEEDS"
bash "$HERE/elja_hist_campaign.sh" "$ROOT/guard98" 98 "$HERE/sota_guard/lj98.tsv" "$SEEDS"
