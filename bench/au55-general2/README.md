# plateau_heat against 24ed9551

The `plateau_heat` commits rebased onto develop 24ed9551, gated again in
one session, and the M80 rho=14 loss that kept them from landing on by
default. The 98dcfb31 gate is `bench/au55-general`. Between 98dcfb31 and
24ed9551 nothing under `src/` or in the h2h examples changed, and the
24ed9551 build repeats every 98dcfb31 base row's hit, calls and best.

## Arms

- `base`: 24ed9551, preset.
- `heat`: 6d2c70df, the doubling rule (heat x2 per call plateau up to x4, a
  new lowest minimum back to x1), on by default there.
- `records`: 65ab1c5e, the records-per-call rule, run with
  `H2H_PLATEAU_HEAT=1`. Off by default at that revision.

`scripts/au55_general/enter_bench.sh` builds the three revisions and runs
`run_gate.py` in `sudo unshare --uts` with hostname bench, one process on
`taskset -c 0`, nothing else running. Au55 seeds 2000-2023; M80 rho=14 and
sc108-55 seeds 2000-2035; LJ38, LJ55, LJ75, LJ98, LJ104, LJ150 and M80
rho=6 seeds 2000-2011; the example budgets. `rows.csv` carries
code_revision and host; `summary.csv` holds Fisher's exact test on hits and
the tie-corrected two-sided Mann-Whitney test on calls (misses at the
budget), with type-7 medians of calls and of the time inside the potential
(`pot`) and outside it (`out`).

## Result

Calls are medians; pot/out are median seconds inside and outside the
potential, one process on core 0.

| system | arm | n | base hits | arm hits | Fisher p | base calls | arm calls | MW p | base pot/out s | arm pot/out s |
|---|---|---|---|---|---|---|---|---|---|---|
| au55 | heat | 24 | 11 | 12 | 1 | 1000000 | 960380 | 0.681 | 26.1/8.7 | 25.3/7.7 |
| au55 | records | 24 | 11 | 13 | 0.773 | 1000000 | 659704 | 0.179 | 26.1/8.7 | 17.5/5.5 |
| lj38 | heat | 12 | 12 | 12 | 1 | 10616 | 10616 | 0.977 | 0.0/0.0 | 0.0/0.0 |
| lj38 | records | 12 | 12 | 12 | 1 | 10616 | 10616 | 1 | 0.0/0.0 | 0.0/0.0 |
| lj55 | heat | 12 | 12 | 12 | 1 | 12242 | 14084 | 0.908 | 0.1/0.1 | 0.1/0.1 |
| lj55 | records | 12 | 12 | 12 | 1 | 12242 | 12242 | 0.931 | 0.1/0.1 | 0.1/0.1 |
| lj75 | heat | 12 | 12 | 12 | 1 | 1409230 | 673598 | 0.069 | 11.7/11.5 | 5.6/8.2 |
| lj75 | records | 12 | 12 | 12 | 1 | 1409230 | 929421 | 0.403 | 11.7/11.5 | 7.7/8.3 |
| lj98 | heat | 12 | 10 | 12 | 0.478 | 1511030 | 365050 | 0.00165 | 21.5/15.9 | 5.3/3.7 |
| lj98 | records | 12 | 10 | 12 | 0.478 | 1511030 | 465270 | 0.0166 | 21.5/15.9 | 6.6/4.7 |
| lj104 | heat | 12 | 9 | 10 | 1 | 2328723 | 1012850 | 0.469 | 37.4/28.4 | 16.3/11.8 |
| lj104 | records | 12 | 9 | 9 | 1 | 2328723 | 2887942 | 0.663 | 37.4/28.4 | 46.2/33.5 |
| lj150 | heat | 12 | 5 | 11 | 0.0272 | 4000000 | 1440728 | 0.0105 | 131.8/86.3 | 47.8/33.5 |
| lj150 | records | 12 | 5 | 9 | 0.214 | 4000000 | 1669942 | 0.077 | 131.8/86.3 | 55.3/34.3 |
| m80r6 | heat | 12 | 2 | 3 | 1 | 4000000 | 4000000 | 0.597 | 104.1/52.3 | 104.7/51.8 |
| m80r6 | records | 12 | 2 | 2 | 1 | 4000000 | 4000000 | 1 | 104.1/52.3 | 104.5/53.6 |
| m80r14 | heat | 36 | 32 | 28 | 0.343 | 1713961 | 2128370 | 0.2 | 44.9/16.2 | 55.9/23.0 |
| m80r14 | records | 36 | 32 | 30 | 0.735 | 1713961 | 1988140 | 0.217 | 44.9/16.2 | 52.1/19.4 |
| sc108-55 | heat | 36 | 17 | 15 | 0.813 | 1000000 | 1000000 | 0.766 | 66.5/5.3 | 67.2/5.4 |
| sc108-55 | records | 36 | 17 | 17 | 1 | 1000000 | 1000000 | 0.917 | 66.5/5.3 | 67.1/5.6 |

## Why heat hurt M80 rho=14

`diagnosis-6d2c70df.csv` summarises hop traces of the doubling rule
(`screen-knobs.patch`, `AU_HEAT_TRACE`). On the M80 rho=14 seeds it lost
(2004, 2010) the walk sat at x4 for 99.5% of 48k hops, from about 30k calls
to the 4M budget. Its median energy sat 6.7-7.1 above the incumbent, it
accepted 0.54-0.56 of hops, and over 3.9M calls the x4 hops found one new
lowest minimum. No record meant no reset, so the heat never came down.
The cold walk of seed 2004 sat 2.0 above the incumbent, accepted 0.39, found
seven records after 50k calls and hit at 286k. On LJ75, LJ98 and LJ150 the
x4 walk accepts 0.71-0.79 and sits 3.5-5.7 above the incumbent; the win
there is one hot record in a new funnel followed by cold descent.

## Rules screened

Parallel screens on cores 1-6 (`screen-rows.csv`; calls are deterministic,
times are not), against the 24ed9551 base:

| rule | M80 rho=14 (n=36) | LJ150 | LJ98 |
|---|---|---|---|
| base | 32 | 5 | 10 |
| doubling | 28 | 11 | 12 |
| sawtooth, back to the incumbent at the cap (`return`) | 22 (Fisher 0.013) | | |
| records per call, back to the incumbent on cooling (`bandit_ret`) | 17 (Fisher 0.0003) | 6 | 9 |
| records per call (`bandit`, committed) | 30 | 9 | 12 |

Moving the chain back to the incumbent loses on every system screened: it
throws away the position a hot walk found, which is the LJ win, and on M80
it restarts the walk in the funnel it had already exhausted. The committed
rule counts, from the first plateau, the new lowest minima and the calls
spent at each of x1, x2 and x4, and on each plateau takes the factor with
the most records per plateau window, with one record and one window of
prior per factor; ties keep the factor, or take the coldest. A record keeps
the factor and the position. It has no constant that names a potential, a
packing or a size.

## Decision

`plateau_heat` stays opt-in (default false, 65ab1c5e). With the records
rule it no longer loses significantly anywhere, and on M80 rho=14 it is
two hits behind base instead of four, but a pooled deficit holds at n=36
(30/36 against 32/36, Fisher p=0.73). The losing seeds differ from the
doubling rule's (2007, 2018, 2025, 2026, 2028 against 2003, 2004, 2006,
2009, 2010, 2012, 2032), and four of the five are base hits after 750k
calls, so the M80 rho=14 outcome is close to a per-trajectory coin there.
That still fails "loses nowhere", so the preset does not change. LJ104
keeps its hits (9/12) with a higher median call count, 2.89M against
2.33M (Mann-Whitney p=0.66). `H2H_PLATEAU_HEAT=1`, or
`Config::plateau_heat = true`, turns it on.

## Pins

With the flag off (the default at 65ab1c5e), `scripts/covalent_h2h/pins.sh`
prints sha1 `a22093e707faa91d683813fa7cc507706c5265e6` and `h2h_metal run
au55` seeds 2000-2002, clock columns removed, sha1
`b22e212181eff57e9412f780c40e85208b355d07`, the values of 24ed9551 and
98dcfb31.

## Checks

On the delivered head: `cargo +1.88.0 fmt --all -- --check`, `cargo
+1.88.0 clippy --all-targets --features rgpot-ex -- -D warnings`, `cargo
+1.88.0 test --release`, the ChemFit tests (1005 passed, 4 skipped) and
`examples/chemfit_anneal.py` (best loss 2.347665788572983) on a wheel built
from it. `pytest pytest/` fails four tests that fail the same way on
24ed9551: three paper audit tests that need the v0.12.0 tag and revision
9149d1b0, absent from this clone, and the h2h coverage test, which wants
`agox_lj75` and `occat_h2h` classified in `bench/h2h.json`.
