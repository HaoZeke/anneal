#!/usr/bin/env bash
set -euo pipefail
anneal_script_dir=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
anneal_plan=$(bash "$anneal_script_dir/terra_orbit_channels.sh" plan)
printf '%s\n' "$anneal_plan" | awk -F '\t' '
  function fail(message) { print message > "/dev/stderr"; bad = 1 }
  {
    if (NF != 8 || $1 != NR - 1) fail("task indices must be contiguous")
    if ($2 != 38 && $2 != 75 && $2 != 98) fail("unexpected instance")
    if ($3 != ($2 == 38 ? 400000 : 4000000)) fail("wrong aggregate budget")
    if ($4 < 0 || $4 >= ($2 == 38 ? 72 : 48)) fail("seed outside protocol")
    key = $2 SUBSEP $4 SUBSEP $5
    if (seen[key]++) fail("duplicate instance/seed/arm")
    if ($5 != "single" && $5 != "independent" && $5 != "bias" && $5 != "bias_ring")
      fail("unexpected arm")
    if ($6 != ($5 == "single" ? 1 : 4)) fail("wrong replica count")
    if ($7 != ($5 == "bias" || $5 == "bias_ring" ? 1 : 0)) fail("wrong bias toggle")
    if ($8 != ($5 == "bias_ring" ? "ring" : "none")) fail("wrong gossip topology")
    counts[$2 SUBSEP $5]++
  }
  END {
    if (NR != 672) fail("the paper protocol requires 672 tasks")
    split("38 75 98", instances, " ")
    split("single independent bias bias_ring", arms, " ")
    for (n in instances) for (a in arms)
      if (counts[instances[n] SUBSEP arms[a]] != (instances[n] == 38 ? 72 : 48))
        fail("an arm does not cover every seed")
    exit bad
  }'
printf 'orbit channel plan: 672 unique budget-matched task records\n'
