# Paper headline panel on the 0.12.0 release build

Release tip `07f89cadbd9ccd8386e92408a8ff0e8153b180e3` (0.12.0, the develop
tip that becomes main), built with Rust 1.88.0, `--release --locked`.
`snakemake -s workflow/Snakefile paper_0120`: seeds 2000-2011, the paper
budgets, one process on core 0 (`taskset -c 0`, one thread, every other
process moved to cores 1-7), `sudo unshare --uts` with hostname `bench`,
nothing else running. Lennard-Jones runs through `h2h_gmin_proof`, Au55
through `h2h_metal` and Morse through `h2h_morse`, the examples of the
head-to-head rules. A miss is entered at the budget. Medians are
Hyndman-Fan type 7. Inside is `pot_s`, the time inside the potential;
outside is the rest of the wall.

- `rows.csv`: one row per run, appended, with `code_revision` and `host`.
- `summary.csv`, `compare.csv`: derived from `rows.csv` by
  `scripts/paper_0120/run_panel.py --summarize`.
- `wall-check.csv`: a same-session wall check, described below.
- `pins-07f89cad.txt`, `au55-pins-07f89cad.txt`: the pins on this build.

The develop side of each comparison is the newest develop revision in
`bench/` with rows for that system at seeds 2000-2011 and the same budget.
Fisher is the two-sided exact test on hits. Mann-Whitney is two-sided, the
normal approximation with the tie and continuity corrections, on calls to
the minimum. LJ55, LJ102 and LJ103 have only a summary on develop
(`bench/lj-n12-500aaf65`), so they get Fisher and no Mann-Whitney. "Same
seeds" counts seeds with the same hit, the same calls and, where the
develop row stores it, the same best energy to 1e-8.

| system | budget | 0.12.0 hits | develop hits | develop rows | Fisher p | median calls 0.12.0 | median calls develop | MW p | same seeds | 0.12.0 wall / inside / outside (s) |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| LJ38 | 4e5 | 12/12 | 12/12 | `65557416` perf-quench9 | 1 | 10,616 | 11,489 | 0.47 | 1 | 0.080 / 0.031 / 0.051 |
| LJ55 | 1e6 | 12/12 | 12/12 | `500aaf65` lj-n12 summary | 1 | 12,242 | 15,954 | - | - | 0.151 / 0.066 / 0.088 |
| LJ75 | 4e6 | 12/12 | 12/12 | `65557416` perf-quench9 | 1 | 1,409,230 | 1,409,230 | 1 | 12 | 28.69 / 13.73 / 13.76 |
| LJ75, `h2h_seed` | 4e6 | 12/12 | 12/12 | `7192e9f8` g3 develop arm | 1 | 31,014 | 31,014 | 1 | 12 | 0.670 / 0.264 / 0.406 |
| LJ98 | 4e6 | 10/12 | 10/12 | `65557416` perf-quench9 | 1 | 1,511,031 | 1,511,031 | 1 | 12 | 44.77 / 25.57 / 19.20 |
| LJ102 | 4e6 | 9/12 | 6/12 | `500aaf65` lj-n12 summary | 0.40 | 894,801 | 3,584,560 | - | - | 29.47 / 16.49 / 12.98 |
| LJ103 | 4e6 | 8/12 | 6/12 | `500aaf65` lj-n12 summary | 0.68 | 2,293,133 | 3,360,783 | - | - | 72.27 / 42.53 / 29.74 |
| LJ104 | 4e6 | 9/12 | 5/12 | `bb486bfd` regress-bb486bfd | 0.21 | 2,328,723 | 4,000,000 | 0.11 | 0 | 72.19 / 41.44 / 30.74 |
| LJ150 | 4e6 | 5/12 | 3/12 | `bb486bfd` regress-bb486bfd | 0.67 | 4,000,000 | 4,000,000 | 0.54 | 0 | 239.0 / 135.2 / 102.1 |
| Au55 | 1e6 | 6/12 | 6/12 | `7192e9f8` g3 priors-on develop arm | 1 | 852,736 | 852,736 | 1 | 12 | 31.03 / 22.95 / 8.05 |
| M80 ρ=6 | 4e6 | 2/12 | 2/12 | `7192e9f8` g3 priors-on develop arm | 1 | 4,000,000 | 4,000,000 | 1 | 12 | 159.0 / 105.6 / 53.2 |
| M80 ρ=14 | 4e6 | 9/12 | 9/12 | `76ff29a6` morse-panel | 1 | 1,338,933 | 1,338,933 | 1.0 | 12 | 48.75 / 37.14 / 11.61 |

## Where 0.12.0 moved

No system moved significantly: every Fisher p is at least 0.21 and every
Mann-Whitney p above 0.1.

- Au55, M80 ρ=6, LJ75 (both drivers) and LJ98 did not move. Every seed
  repeats the newest develop row: same hit, same calls, same best where
  stored. M80 ρ=6 is still 2/12.
- LJ38 moved in bookkeeping only. A rebuild of `65557416` on this host
  repeats its stored calls on all twelve seeds, and its `charged` and best
  energies equal 0.12.0's, so the walk is the same. Calls to the minimum are
  lower on 11 seeds (median 10,616 against 11,489) because `425a37ce` (fix:
  score every ledger improvement at its record) stamps a hit found by an
  orbit completion at the call where the incumbent was stored. `git bisect`
  over `65557416..07f89cad` on seed 2000 names that commit.
- LJ104, LJ150 and M80 ρ=14 walk differently from their develop rows, which
  are older (`bb486bfd`, `76ff29a6`): no seed matches. LJ104 hits 9/12
  against 5/12 (p = 0.21, Mann-Whitney 0.11), LJ150 5/12 against 3/12
  (p = 0.67), and M80 ρ=14 9/12 against 10/12 (p = 1) with a lower median
  call count (p = 0.34). The three M80 ρ=14 misses end at -340.802563, 0.0088
  above 80G. Its median time inside the potential per charged call is
  27.2 µs against 59.1 µs on the morse-panel rows, and outside 9.9 µs
  against 15.4 µs.
- LJ102 9/12 against 6/12 (p = 0.40), LJ103 8/12 against 6/12 (p = 0.68),
  and LJ55 12/12 on both. Develop has only medians for these three.

## Wall time

Walls are from one session and drift with the host. On the seeds whose
walk did not change, the panel's LJ75 and LJ98 walls are 8-22 % above the
stored develop walls, with inside and outside growing together. Au55 is
-1 % to +21 %, and M80 ρ=6, later in the session, is within 4 %. After
the panel, `scripts/paper_0120/wall_check.py` ran `65557416` and 0.12.0
interleaved on LJ75 seeds 2000-2005 and LJ98 seeds 2000-2002
(`wall-check.csv`): with the same calls, 0.12.0 takes 0.95-1.02 of
`65557416`'s wall, and `65557416` is within 3 % of its stored walls. The
panel's wall differences on unchanged walks are the session, not 0.12.0.

## LJ103

The release `h2h_gmin_proof` has no LJ103 entry. LJ103 runs on a second
build of `07f89cad` with `lj103-table.patch`, which adds the Cambridge
energy -575.766131 and the 4e6 budget and nothing else; its rows have
driver `h2h_gmin_proof+lj103`. The `500aaf65` LJ103 summary was taken the
same way.

## Pins

On this build `scripts/covalent_h2h/pins.sh` prints sha1
`a22093e707faa91d683813fa7cc507706c5265e6`, the value recorded for develop
`7192e9f8` in `bench/generic-alloy/g3-pins.txt` and for `7d8fa6ea` in
`bench/openconf-h2h/conf-fix4`. `h2h_metal run au55` on seeds 2000-2002,
clock columns removed, prints sha1
`1213f9cdf92a2dabfc8a51c83522086f8a9d7cdc`; the template-era value recorded
in `bench/openconf-h2h/conf-fix4` was
`b22e212181eff57e9412f780c40e85208b355d07`.
`tests/lj_trajectory_pin.rs` passes. The committed pins files,
`bench/generic-alloy/g2-pins.txt` and `g3-pins.txt`, are unchanged.

## asv

`asv run --python=same --bench Smoke` passes in `bench/paper-0120`,
`bench/gmin-h2h`, `bench/morse-gmin` and `bench/metal-cluster-h2h`. The
`paper_0120` smoke checks that each example's own budget equals
`PAPER_BUDGETS` for every panel system but LJ103, which the release
example lacks.
