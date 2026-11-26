# NH3 octamer matched budget

Seeds 2000-2011 each have one row in `results.csv`. No new search was run.
Seeds 2000 and 2001 come from the capped rerun. The other rows come from
shards `nh3x8-s2002`, `nh3x8-s2004`, `nh3x8-s2005`, `nh3x8-s2006`,
`nh3x8-s2008`, and `nh3x8-s2010`. Those shards carry one shared `--seeds`
patch; this history keeps that patch.

## Seed 2005

`nh3x8-s2004` names seeds 2004 and 2005 in its subject. Its patch adds only
the 2004 row. The shard log starts seed 2005 and then exits 137, with no
results row. `nh3x8-s2005` reran seed 2005 to completion. The row kept here
is that rerun: hit, calls_to_hit 332675, stop target, best energy
-35.470770942940 Eh, wall 5239.392020 s.

## What 332675 counts

A hit row has calls_to_hit 332675, the campaign budget, and stop_reason
target. The harness reads `energy_calls` from `run.txt`
(`bench/octamer-matched/nh3x8/run_campaign.py:156`). When the final
`objective_best_eV` is at or below the target it stores that count and sets
stop_reason to target (`run_campaign.py:161-163`). A miss that spent the
budget is stored as 332676 (`run_campaign.py:164-166`).

`energy_calls` is `ledger.spent()` (`examples/gfn2_mol_h2h.rs:540` and
`examples/gfn2_mol_h2h.rs:545`). The example also writes `search_calls` as
that total minus `polish_calls` (`examples/gfn2_mol_h2h.rs:558-559`).

`search_from` holds a tail out of the same budget before the hop
(`src/methods/cluster_search.rs:1618-1620`). `molecular_answer_reserve`
sets the size of that tail (`src/methods/cluster_search.rs:1672-1679`).
After the hop, `finish_molecular_answers` spends the tail on the recorded
seats (`src/methods/cluster_search.rs:1683` and
`src/methods/cluster_search.rs:1723`). `Ledger::charge` refuses a call only
when the budget is spent or a caller stop is raised
(`src/methods/cluster_hopping.rs:880-886`). The target is a hop stop:
`cfg.target_energy` is installed with `Ledger::stop_below`
(`src/methods/cluster_hopping.rs:5474-5480`), and the hop returns when the
loose incumbent reaches it.

On every saved `run.txt` for seeds 2002-2011 the split is search_calls
331075 and polish_calls 1600. Those add to 332675. `polished_from_eV`, the
lowest loose seat, is above the target on every one of those seeds. The
reported `objective_best_eV` is the energy after the polish. A hit is a seed
whose polished energy is at or below the target, and the stored count is
still the full ledger spend. The (H2O)8 record measured the same final
polish after the hop and stored the polish evaluation that first crossed,
because that example writes `first_hit_call`. This harness copies
`energy_calls`, so a hit stores 332675 and is labelled target. Seeds 2000
and 2001 have no `run.txt` in this tree. Their rows were written by the
same harness fields.

## CREST

`bench/nh3-octamer/results.csv` has CREST seeds 17000-17005. A CREST hit is
that file's `hit` column: the run found the reference within tolerance.
`summarize.py` reports the two-sided Fisher exact test on the hit counts.

The CREST `calls` column is the full-run total. A comparison of anneal
`calls_to_hit` with those totals needs CREST first-hit counts, which are
not recorded for NH3. This record does not compute that comparison.

## Summary

`summarize.py` writes `REPORT.md` and `summary.json`. Hits use a Wilson 95%
interval. The calls_to_hit median uses a miss stored as 332676, then 10000
bootstrap resamples with seed 20261123 and linear quantiles.
