# NH3 octamer matched-budget protocol

## Existing comparison

The published octamer rows are in `bench/nh3-octamer/results.csv`, with the
budget ladder in `bench/nh3-octamer/ladder.csv` and the system description in
`bench/nh3-octamer/README.md`. The six CREST rows use seeds 17000--17005 and
record 243868, 221579, 332675, 241500, 134305, and 196621 calls. This campaign
does not rerun CREST.

Both arms use GFN2-xTB 6.7.1. The recorded scoring command is
`xtb --gfn 2 --opt vtight --acc 0.01 --iterations 500`
(`scripts/mol_crest_h2h.py:122`). The anneal evaluator is
`RgpotObjective::xtb` (`examples/gfn2_mol_h2h.rs:508`) with accuracy 0.01 and
500 electronic iterations. The historical hit definition is within
1e-6 Eh of -35.470769324714 Eh (`bench/nh3-octamer/README.md:3-7`).

## Anneal arm

The run uses seeds 2000--2011, serially, one process at a time, pinned with
`taskset -c 0`. Every seed receives 332675 charged calls, the maximum of the
six recorded per-seed CREST call counts. The executable is built with feature
`rgpot-ex` and the GFN2-xTB 6.7.1 engine revision recorded by the source tree.
Thread-count environment variables are fixed at one.

The configuration is `Config::recommended_molecular` with energy scale 1 eV,
as selected by the paper executable at `examples/gfn2_mol_h2h.rs:443`.
The harness changes only `target_energy`, setting it to
`(-35.470769324714 + 0.000001) * 27.21138505` eV. The hop returns when the
loose incumbent reaches that threshold. Otherwise the hop spends the calls
it can see, and the final polish spends the reserved tail of the same budget.

`run_campaign.py` creates a temporary example from the paper executable,
adds the target, builds it, and removes the temporary source before running.
It makes no library-source change. Seeds are written to `results.csv`
atomically after each process exits. A hit records `energy_calls` as
`calls_to_hit`. On a finished hit that total is the budget, 332675, because
the crossing is inside the final polish and the polish spends the reserved
tail. A miss records 332676 as `calls_to_hit`. Each row also records the stop
reason, best energy, wall time, binary digest, feature set, and source
revision. `README.md` names the lines.

## Analysis

The anneal seeds and CREST seeds are independent samples. CREST hit status and
calls come directly from `bench/nh3-octamer/results.csv`. Hits are reported
with two-sided Wilson 95% intervals. Calls-to-hit are reported with a
10000-resample percentile bootstrap interval for the median, using random seed
20261123 and linear quantiles. Misses are represented as 332676.

Hit counts are compared with a two-sided Fisher exact test. Anneal and CREST
call counts are not compared. The CREST `calls` column is a full-run total,
and CREST first-hit counts are not recorded for NH3. The anneal median uses
the miss value 332676.
