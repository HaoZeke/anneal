# Structural priors ablation

`ANNEAL_STRUCTURAL_PRIORS=0` turns off every path that builds a structure from a known packing or chooses a move by classifying one (`Config::structural_priors`), including every SOAP proposal through `prior_soap`, so the priors-off arm draws no fivefold or icosahedral step. Priors on is the unset default. Each cell below is hits out of twelve, the median calls to hit with a miss counted as the full budget, the median wall, and the median seconds inside and outside the potential. Medians are Hyndman–Fan type 7. Fisher is the two-sided Fisher exact test of the hit counts; MW is the exact two-sided Mann–Whitney rank-sum test of the calls to hit, misses at the full budget, ties at midranks.

Seeds are 2000–2011, serial, `taskset -c 0`, hostname `bench` under `sudo unshare --uts`, working directory `/work`, at the harness budgets: 4e5 at 38 atoms, 1e6 at 55, 4e6 at 75 and above, and 2e6 for TIP4P. The Gupta alloys take the budgets of `bench/au55-scale/gate.csv`. `scripts/priors_ablation/enter_bench.sh` runs the anneal arms and `enter_gmin.sh` runs GMIN. `summarize.py` writes `summary.csv` and `table.md` from `rows.csv` and `gmin-rows.csv`.

Each binary scores its own hit within 0.001 of its reference: `h2h_gmin_proof` and `h2h_morse` against the Cambridge tables (M80 at ρ = 6 is 80B, −378.333471), `h2h_seed` for TIP4P (−116.5904186160, −305.5183221676 and −628.3722525077 kJ/mol), and `h2h_metal` for Au55, Cu6Au32 and Cu27Au28 against its table. Cu19Au19, Cu10Au45, Cu30Au8, Cu12Au26 and Pt12Cu11 take the `reference` column of `gate.csv` through `H2H_REFERENCE`, since `h2h_metal` has no table entry for most of them.

## Revisions

`rows.csv` and `gmin-rows.csv` are only ever appended. `code_revision` is the develop commit the series ran on, and for GMIN rows the commit whose harness and starts GMIN ran with. `host` is the first twelve hex digits of `/etc/machine-id`, since the hostname is always `bench`. Every row so far ran on host `dd8ee1b8e2c7`, an eight-vCPU Intel Xeon with AVX2, FMA and AVX-512. The 360 rows at `500aaf65` ran from one build of this series on develop `500aaf65`. The 480 rows at `f2a628c0` ran from one build of this series on develop `f2a628c0`, and they cover both arms of every anneal system: Au55, the seven alloys of `gate.csv`, M55 and M80 at ρ = 6 and 14, LJ38 to LJ150 and the three TIP4P sizes. The 240 rows at `7192e9f8` and `8d286bf2` are the same-session comparison of develop and this series below, at seeds 2000–2011 on LJ38, LJ75, LJ98, LJ104, LJ150, Au55, M80 at ρ = 3 and 6, TIP4P15 and M55 at ρ = 14, default arm only. The 900 rows at `8e50ff2e`, `946e05ce` and `77ca61dc` are the same-session comparison of develop `8e50ff2e` and this series on it below, on all 21 systems at seeds 2000–2011 and on LJ104, M55 at ρ = 14 and M80 at ρ = 3 and 6 at 2012–2023 as well, default arm only. `summarize.py --seeds 2000-2011` scores each (system, arm, seed) from its last row, so every priors-on cell is at `77ca61dc`, the series' last code commit on `8e50ff2e`, and every priors-off cell is at `f2a628c0`. `summary.csv` lists the revision and host of each cell. The off-against-on hits compare. The Lennard-Jones and Morse seconds do not (see below), and neither do the alloy calls to hit (see Reading).

The Lennard-Jones and TIP4P systems were to keep their `500aaf65` rows, on the premise that develop's physical-sphere change leaves them bit-identical. `scripts/pins.sh` disproved that premise. Develop `f2a628c0` and this series on it give identical pins, so the series does not move the default walk. Between the `500aaf65` build and `f2a628c0`, however, all sixteen Lennard-Jones and TIP4P `h2h_seed` traces differ. A bisect of the LJ13 seed-2000 trace over develop puts the change at `bb486bfd` (communicating chains, #59), which rewrites the pair-potential gradient as `f64::mul_add` and changes the cluster-search proposal path. Those systems were therefore rerun at `f2a628c0` too.

Develop `2e94773a` and `f2a628c0` give identical pins, all 91 of them and all sixteen Lennard-Jones and TIP4P `h2h_seed` traces, and this series on `2e94773a` gives the same pins as develop. The `f2a628c0` rows therefore stand at `2e94773a`, and none were rerun. Develop `65557416` moves 60 of the 91 pins against `2e94773a`, among them nine Lennard-Jones and three Morse `h2h_seed` traces, so the rows do not describe `65557416`; this series on `65557416` still gives develop's pins.

The same `bb486bfd` made the pair potential about four times as costly per Lennard-Jones call and twice as costly per Morse call, because the `fma` behind `f64::mul_add` is an out-of-line, runtime-dispatched call on the default x86-64 target. Develop `2e94773a` takes Lennard-Jones and Morse from the `rgpot-pair` C kernels, which dispatch at run time to an AVX2 and FMA build with inline `vfmadd`. Only their scalar fallback, on a CPU without FMA, still calls `fma` out of line. The median microseconds per call (inside seconds over charged calls) over seeds 2000–2002, default arm, 2e5 calls, serial on `taskset -c 0` on the same host:

| system | `500aaf65` | `f2a628c0` | `2e94773a` |
|---|---|---|---|
| lj75 | 10.79 | 42.02 | 8.48 |
| lj98 | 18.45 | 71.53 | 14.57 |
| m80r6 | 29.67 | 58.58 | 26.46 |

The Gupta and TIP4P per-call costs did not change at `bb486bfd`. The Lennard-Jones and Morse wall and inside seconds of the `f2a628c0` rows are therefore not comparable with the `500aaf65` rows, with GMIN or with `2e94773a`; the hits and calls to hit are.

## GMIN

GMIN ran in the `500aaf65` session on every system except the five alloys the `gate.csv` panel added: Cu19Au19, Cu10Au45, Cu30Au8, Cu12Au26 and Pt12Cu11. `scripts/priors_ablation/run_gmin.py` drives the Darby `BGUPTAT` build only for Au55, Cu6Au32 and Cu27Au28, so those five GMIN cells read `no row`. Lennard-Jones, Morse and TIP4P use `scripts/gmin_h2h/build_gmin.sh` at revision `aac1548a` with the Morse timer; Au55 and the Cu–Au alloys use the Darby `BGUPTAT` driver from `scripts/metal_h2h/build_gmin_darby.sh` at the same revision. Both are gfortran 13.3.0 with Ubuntu reference BLAS and LAPACK. Each GMIN run starts from the anneal binary's start for the same seed, and TIP4P from the `.start.gmin` the priors-on arm wrote. Those starts are byte-identical at `f2a628c0`, so the GMIN rows still compare like with like. The GMIN build does not time `NEWTIP`, so the TIP4P GMIN inside and outside cells are blank.

| system | GMIN | anneal priors on | anneal priors off | off vs on: Fisher / MW | off vs GMIN: Fisher / MW |
|---|---|---|---|---|---|
| lj38 | 7/12 · 389399 · 3.220 s · 2.370 / 0.846 | 12/12 · 11104 · 0.074 s · 0.027 / 0.047 | 12/12 · 95593 · 1.257 s · 1.032 / 0.225 | 1.000 / 0.012 | 0.037 / 0.003 |
| lj75 | 0/12 · 4000000 · 106.017 s · 91.370 / 14.652 | 12/12 · 1882984 · 35.291 s · 16.198 / 19.093 | 4/12 · 4000000 · 199.967 s · 166.628 / 33.409 | 0.001 / 0.004 | 0.093 / 0.093 |
| lj98 | 0/12 · 4000000 · 166.207 s · 147.851 / 18.271 | 11/12 · 1492086 · 38.226 s · 21.978 / 16.248 | 6/12 · 3719081 · 318.209 s · 265.150 / 51.725 | 0.069 / 0.019 | 0.014 / 0.014 |
| lj104 | 6/12 · 3696118 · 173.296 s · 155.391 / 17.864 | 6/12 · 3576014 · 102.830 s · 58.061 / 44.500 | 10/12 · 890230 · 87.587 s · 71.118 / 16.469 | 0.193 / 0.025 | 0.193 / 0.029 |
| lj150 | 9/12 · 3129518 · 295.154 s · 274.112 / 21.042 | 5/12 · 4000000 · 217.197 s · 133.563 / 83.420 | 8/12 · 907352 · 183.997 s · 151.184 / 32.814 | 0.414 / 0.217 | 1.000 / 0.437 |
| m55r6 | 12/12 · 37048 · 0.661 s · 0.562 / 0.099 | 12/12 · 234 · 0.007 s · 0.007 / 0.001 | 12/12 · 27694 · 0.927 s · 0.786 / 0.145 | 1.000 / 0.000 | 1.000 / 0.843 |
| m80r3 | no row | 12/12 · 337584 · 21.633 s · 10.180 / 11.454 | no row | – | – |
| m80r6 | 12/12 · 268934 · 9.567 s · 8.586 / 0.980 | 12/12 · 353708 · 16.699 s · 11.069 / 5.631 | 12/12 · 39608 · 3.119 s · 2.714 / 0.405 | 1.000 / 0.143 | 1.000 / 0.045 |
| m55r14 | 0/12 · 1000000 · 17.551 s · 15.178 / 2.348 | 11/12 · 527136 · 10.717 s · 7.602 / 3.115 | 10/12 · 453332 · 17.708 s · 14.190 / 3.362 | 1.000 / 0.977 | 0.000 / 0.000 |
| m80r14 | 0/12 · 4000000 · 141.093 s · 127.961 / 13.067 | 8/12 · 2388292 · 183.048 s · 141.148 / 38.956 | 6/12 · 3759634 · 275.806 s · 221.851 / 53.955 | 0.680 / 0.242 | 0.014 / 0.014 |
| au55 | 0/12 · 1000000 · 58.476 s · 55.628 / 2.854 | 6/12 · 965246 · 31.824 s · 25.228 / 7.608 | 3/12 · 1000000 · 40.772 s · 30.761 / 10.015 | 0.400 / 0.351 | 0.217 / 0.217 |
| cu6au32 | 0/12 · 400000 · 12.056 s · 11.172 / 0.884 | 12/12 · 400000 · 6.532 s · 6.056 / 0.475 | 0/12 · 400000 · 7.245 s · 6.087 / 1.175 | 0.000 / 1.000 | 1.000 / 1.000 |
| cu27au28 | 0/12 · 1000000 · 59.720 s · 56.836 / 2.886 | 12/12 · 1000000 · 33.305 s · 31.542 / 1.760 | 1/12 · 1000000 · 36.178 s · 31.705 / 4.450 | 0.000 / 1.000 | 1.000 / 1.000 |
| cu19au19 | no row | 12/12 · 400000 · 6.493 s · 6.026 / 0.466 | 0/12 · 400000 · 7.092 s · 6.054 / 0.994 | 0.000 / 1.000 | – |
| cu10au45 | no row | 12/12 · 1000000 · 33.342 s · 31.556 / 1.793 | 2/12 · 1000000 · 37.574 s · 31.935 / 5.642 | 0.000 / 0.478 | – |
| cu30au8 | no row | 12/12 · 400000 · 6.521 s · 6.053 / 0.472 | 5/12 · 400000 · 7.137 s · 6.069 / 1.056 | 0.005 / 0.037 | – |
| cu12au26 | no row | 0/12 · 400000 · 6.549 s · 6.074 / 0.476 | 0/12 · 400000 · 7.170 s · 6.112 / 1.092 | 1.000 / 1.000 | – |
| pt12cu11 | no row | 12/12 · 200000 · 1.288 s · 1.112 / 0.173 | 1/12 · 200000 · 1.438 s · 1.121 / 0.316 | 0.000 / 1.000 | – |
| tip4p4 | 12/12 · 210 · 0.001 s · – / – | 12/12 · 227 · 0.001 s · 0.000 / 0.000 | 12/12 · 227 · 0.000 s · 0.000 / 0.000 | 1.000 / 1.000 | 1.000 / 0.551 |
| tip4p8 | 8/12 · 240163 · 1.623 s · – / – | 12/12 · 6339 · 0.020 s · 0.012 / 0.009 | 12/12 · 6339 · 0.020 s · 0.012 / 0.008 | 1.000 / 1.000 | 0.093 / 0.000 |
| tip4p15 | 0/12 · 2000000 · 41.094 s · – / – | 7/12 · 1481620 · 13.456 s · 9.439 / 3.932 | 7/12 · 1481620 · 10.583 s · 7.872 / 2.710 | 1.000 / 1.000 | 0.005 / 0.005 |

## Reading

Priors off loses hits on every alloy the walk solves with priors on. Cu6Au32 and Cu19Au19 fall from 12/12 to 0/12 (Fisher p = 1.1e-6 each), Cu27Au28 and Pt12Cu11 to 1/12 (p = 1.0e-5), Cu10Au45 to 2/12 (p = 6.7e-5) and Cu30Au8 to 5/12 (p = 0.0046). Cu12Au26 is 0/12 in both arms, as it is in `gate.csv` at the same budget. Au55 is 6/12 with priors on at `77ca61dc` and 3/12 with priors off at `f2a628c0` (Fisher p = 0.40, MW p = 0.35); at `f2a628c0` priors on was 8/12, and at `500aaf65` the same seeds gave 5/12 and 0/12.

On Lennard-Jones, priors off loses hits on LJ75 (12/12 to 4/12, Fisher p = 0.0013, MW p = 0.0037), needs more calls on LJ98 (11/12 to 6/12, MW p = 0.019) and about nine times the calls on LJ38 (MW p = 0.012). On LJ104 it needs fewer (10/12 against 6/12, MW p = 0.025), and on LJ150 it hits more (8/12 against 5/12, MW p = 0.22). On M55 at ρ = 6 both arms hit every seed, but priors off needs far more calls (MW p = 1.1e-6). At ρ = 6, M80 hits all twelve seeds in both arms, priors off in fewer calls (MW p = 0.14); at `f2a628c0` priors on was 9/12. No other Lennard-Jones or Morse system separates the arms at 0.05. The TIP4P arms are identical, since no structural prior acts on rigid water.

On the alloys at `f2a628c0` the hitting energy is not among the walk's recorded improvements, so the ledger never stops below the reference and `h2h_metal` reports the full charge as the calls to hit. Develop `7192e9f8` hands the core cuts' best to the walk, which then stops at the reference, so the priors-on calls to hit at `77ca61dc` are the call that reached it. The priors-off rows are still censored at the full charge, so the alloy Mann–Whitney values are not a comparison of the arms, and Fisher carries it.

## Develop `7192e9f8` against this series

Develop `7192e9f8` and this series on it (`8d286bf2`, its last code commit) ran in one session on host `dd8ee1b8e2c7`, system by system and develop first, under `enter_bench.sh` with `ABL_SRC` naming each checkout, default arm, seeds 2000–2011 at the budgets above. `compare.py` writes the table. The series was then rebased onto develop `98dcfb31`, where `d8e9bbbb` is its last code commit; the 91 pins of `scripts/covalent_h2h/pins.sh` are byte-identical at each commit of the series on both bases, and these rows stay at the revisions they ran on. Each cell is hits out of twelve, the median calls to hit, the median wall, and the median seconds inside / outside the potential. Fisher is two-sided; MW is the exact two-sided Mann–Whitney test of the calls to hit, conditioned on the observed ties at midranks, so a block of misses at the full budget is one tie.

| system | `7192e9f8` | `8d286bf2` | Fisher / MW |
|---|---|---|---|
| lj38 | 12/12 · 10616 · 0.069 s · 0.027 / 0.044 | 12/12 · 11104 · 0.074 s · 0.027 / 0.047 | 1.000 / 0.966 |
| lj75 | 12/12 · 1409230 · 24.848 s · 11.835 / 11.902 | 12/12 · 1882984 · 35.291 s · 16.198 / 19.093 | 1.000 / 0.291 |
| lj98 | 10/12 · 1511030 · 39.032 s · 22.100 / 16.932 | 11/12 · 1492086 · 38.226 s · 21.978 / 16.248 | 1.000 / 0.681 |
| lj104 | 9/12 · 2328723 · 67.493 s · 37.822 / 29.671 | 6/12 · 3576014 · 102.830 s · 58.061 / 44.500 | 0.400 / 0.145 |
| lj150 | 5/12 · 4000000 · 226.003 s · 133.436 / 92.386 | 5/12 · 4000000 · 217.197 s · 133.563 / 83.420 | 1.000 / 0.859 |
| m80r3 | 12/12 · 683002 · 25.778 s · 17.678 / 8.100 | 12/12 · 337584 · 21.633 s · 10.180 / 11.454 | 1.000 / 0.178 |
| m80r6 | 1/12 · 4000000 · 180.748 s · 116.922 / 63.887 | 12/12 · 353708 · 16.699 s · 11.069 / 5.631 | 0.000 / 0.000 |
| m55r14 | 4/12 · 1000000 · 20.017 s · 13.949 / 5.832 | 11/12 · 527136 · 10.717 s · 7.602 / 3.115 | 0.009 / 0.010 |
| au55 | 6/12 · 852736 · 30.593 s · 22.550 / 7.790 | 6/12 · 965246 · 31.824 s · 25.228 / 7.608 | 1.000 / 0.637 |
| tip4p15 | 7/12 · 1481620 · 13.002 s · 9.525 / 3.477 | 7/12 · 1481620 · 13.456 s · 9.439 / 3.932 | 1.000 / 1.000 |

M80 at ρ = 6 goes from 1/12 to 12/12 (Fisher p = 9.6e-6) and M55 at ρ = 14 from 4/12 to 11/12 (p = 0.0094, MW p = 0.010). The icosahedral reseed now waits for an incumbent as old in calls as the search that found it, and the contact test reads each potential at its own pair minimum, so a descending Morse funnel is no longer discarded five hops after its last improvement. LJ98 gains one hit; LJ38, LJ75, LJ150, Au55, M80 at ρ = 3 and TIP4P15 hit as often.

LJ104 is the one system that loses more than one hit, 9/12 to 6/12 (Fisher p = 0.40, MW p = 0.15). The loss enters with the reseed-clock fix, which resets the reseed clock only on a lowest of the search itself, so reseeds fire on other hops and the seeds draw other funnels. In the screen at develop `92bc5805` with the same series, seeds 2012–2023 gave develop 8/12 and the series 9/12, so over 24 seeds it is 17/24 against 15/24. LJ150 at 2012–2023 gave 7/12 against 5/12 there, 12/24 against 10/24 pooled.

Per call, the potential costs the same at both revisions within 4% on every system except M80, where the median inside microseconds per charged call are 25.9 and 31.1 at ρ = 3 and 29.2 and 31.3 at ρ = 6. Outside the potential, M80 at ρ = 3 costs 34.1 µs per call against 12.1. On its seed 2002 at 2e5 calls, the pair-minimum contact test alone cuts the walk from about 4400 hops to about 940, consistent with icosahedral reseeds, each a fresh quench, firing more often, and the plateau gate brings it back to about 4200 at 4.0 s outside against develop's 2.9 s. M80 at ρ = 3 still reaches its hit in half the calls (MW p = 0.18) and a lower median wall, 21.6 s against 25.8 s. LJ75 hits every seed in both but at a higher median call count (MW p = 0.29), hence its longer wall.

## Develop `8e50ff2e` against this series

Develop `8e50ff2e` (`fix/audit-speed` on develop `24ed9551`) and this series rebased onto it ran in one session on host `dd8ee1b8e2c7` under hostname `bench`, serial on core 0, system by system and develop first. `946e05ce` is the rebased series before the assignment change and `77ca61dc` the series' last code commit; `77ca61dc` ran after the other two. Default arm, seeds 2000–2011 on every system, and 2012–2023 too on LJ104, M55 at ρ = 14 and M80 at ρ = 3 and 6. `compare.py --base 8e50ff2e --head 77ca61dc` writes the table, with the cells, Fisher and tie-conditioned Mann–Whitney tests of the section above.

| system | `8e50ff2e` | `77ca61dc` | Fisher / MW |
|---|---|---|---|
| lj38 | 12/12 · 10616 · 0.069 s · 0.027 / 0.044 | 12/12 · 11104 · 0.067 s · 0.028 / 0.040 | 1.000 / 0.966 |
| lj75 | 12/12 · 1409230 · 25.977 s · 12.623 / 12.422 | 12/12 · 1882984 · 31.370 s · 16.235 / 15.135 | 1.000 / 0.291 |
| lj98 | 10/12 · 1511030 · 38.158 s · 21.908 / 16.250 | 11/12 · 1492086 · 36.507 s · 21.502 / 15.005 | 1.000 / 0.681 |
| lj104 | 18/24 · 2016311 · 58.005 s · 32.643 / 25.362 | 15/24 · 2992364 · 80.966 s · 48.377 / 32.589 | 0.534 / 0.177 |
| lj150 | 5/12 · 4000000 · 225.015 s · 134.922 / 91.026 | 5/12 · 4000000 · 214.680 s · 133.542 / 80.861 | 1.000 / 0.859 |
| m55r6 | 12/12 · 602 · 0.010 s · 0.009 / 0.001 | 12/12 · 602 · 0.009 s · 0.008 / 0.001 | 1.000 / 1.000 |
| m80r3 | 24/24 · 423162 · 16.564 s · 11.541 / 5.023 | 24/24 · 483304 · 19.352 s · 12.857 / 5.920 | 1.000 / 0.911 |
| m80r6 | 4/24 · 4000000 · 159.862 s · 105.601 / 54.295 | 24/24 · 670938 · 25.848 s · 17.691 / 8.157 | 0.000 / 0.000 |
| m55r14 | 14/24 · 940364 · 17.010 s · 11.931 / 4.795 | 21/24 · 412180 · 7.489 s · 5.487 / 1.965 | 0.049 / 0.008 |
| m80r14 | 9/12 · 1359786 · 47.922 s · 35.745 / 12.176 | 4/12 · 4000000 · 145.516 s · 105.742 / 39.634 | 0.100 / 0.008 |
| au55 | 6/12 · 852736 · 30.422 s · 22.520 / 7.858 | 6/12 · 965246 · 31.695 s · 25.344 / 7.387 | 1.000 / 0.637 |
| cu6au32 | 12/12 · 202 · 0.094 s · 0.086 / 0.008 | 12/12 · 202 · 0.094 s · 0.086 / 0.008 | 1.000 / 1.000 |
| cu27au28 | 12/12 · 65933 · 5.922 s · 5.553 / 0.369 | 12/12 · 65933 · 5.877 s · 5.516 / 0.360 | 1.000 / 1.000 |
| cu19au19 | 12/12 · 7706 · 0.167 s · 0.154 / 0.013 | 12/12 · 7706 · 0.167 s · 0.154 / 0.013 | 1.000 / 1.000 |
| cu10au45 | 12/12 · 8296 · 1.359 s · 1.278 / 0.082 | 12/12 · 8296 · 1.369 s · 1.288 / 0.081 | 1.000 / 1.000 |
| cu30au8 | 12/12 · 9224 · 0.159 s · 0.146 / 0.013 | 12/12 · 9224 · 0.159 s · 0.146 / 0.013 | 1.000 / 1.000 |
| cu12au26 | 0/12 · 400000 · 7.596 s · 5.172 / 2.419 | 0/12 · 400000 · 7.524 s · 5.144 / 2.385 | 1.000 / 1.000 |
| pt12cu11 | 12/12 · 31040 · 0.203 s · 0.165 / 0.038 | 12/12 · 31040 · 0.203 s · 0.165 / 0.037 | 1.000 / 0.749 |
| tip4p4 | 12/12 · 227 · 0.000 s · 0.000 / 0.000 | 12/12 · 227 · 0.000 s · 0.000 / 0.000 | 1.000 / 1.000 |
| tip4p8 | 12/12 · 6339 · 0.020 s · 0.012 / 0.008 | 12/12 · 6339 · 0.020 s · 0.012 / 0.008 | 1.000 / 1.000 |
| tip4p15 | 7/12 · 1481620 · 10.632 s · 7.841 / 2.791 | 7/12 · 1481620 · 10.483 s · 7.789 / 2.694 | 1.000 / 1.000 |

`946e05ce` and `77ca61dc` agree on every row: hit, calls to hit, charged calls and best energy, on all 300 seeds. M80 at ρ = 6 goes from 4/24 to 24/24 (Fisher p = 1.3e-9) and M55 at ρ = 14 from 14/24 to 21/24 (Fisher p = 0.049, MW p = 0.008). The alloys and TIP4P walk as develop does. LJ38, LJ75, LJ98, LJ150, M55 and M80 at ρ = 3 and Au55 hit as often.

### Microseconds per call

Inside and outside seconds summed over seeds 2000–2011, over their charged calls, at `8e50ff2e` / `946e05ce` / `77ca61dc`. The Gupta alloys and TIP4P agree inside the potential within 4% at all three, and outside within 7%.

| system | inside | outside |
|---|---|---|
| lj38 | 2.27 / 2.31 / 2.25 | 3.46 / 3.86 / 3.00 |
| lj75 | 8.89 / 8.52 / 8.64 | 9.05 / 9.13 / 7.96 |
| lj98 | 14.68 / 14.60 / 14.55 | 10.37 / 10.09 / 9.93 |
| lj104 | 16.28 / 16.45 / 16.19 | 12.53 / 12.34 / 11.93 |
| lj150 | 34.04 / 34.41 / 33.54 | 25.28 / 25.26 / 24.18 |
| m55r6 | 14.16 / 12.81 / 12.77 | 1.98 / 1.73 / 1.72 |
| m80r3 | 27.77 / 27.47 / 26.56 | 12.62 / 26.49 / 13.27 |
| m80r6 | 26.39 / 26.46 / 26.23 | 13.77 / 13.19 / 12.19 |
| m55r14 | 12.54 / 12.54 / 12.53 | 5.08 / 4.75 / 4.55 |
| m80r14 | 26.26 / 26.44 / 26.53 | 9.87 / 10.50 / 10.27 |
| au55 | 26.40 / 26.55 / 26.44 | 8.98 / 8.57 / 8.50 |

M55 at ρ = 6 hits in about 600 calls, so its per-call figures rest on a few milliseconds. The seeds 2012–2023 are left out: there `77ca61dc` costs 5–15% more per call inside the potential than at 2000–2011 on LJ104, M55 at ρ = 14 and M80 at ρ = 3, with the same pair kernel, while the screens below ran on cores 4–6. The guest's eight vCPUs have no SMT siblings, so a screen shares only the L3 cache and memory bandwidth with core 0, and screens ran through most of the session. The check suite's builds also ran on cores 4–7 during both arms' M80 at ρ = 3 and 6 extensions and `77ca61dc`'s LJ38 and LJ75 rows.

M80 at ρ = 3 cost 26.49 µs per call outside the potential at `946e05ce`, against 12.62 at develop; in the `7192e9f8` session the median was 34.1 against 12.1. Once the contact test reads the Morse pair minimum, the icosahedral reseed fires there, and on a plateau every reseed projects its fresh cluster onto the nearest D5h structure. A projection solves about 120 assignments of 80 atoms, six inertia orientations by the twenty operations, and each was a Kuhn–Munkres walk that restarts its search after every dual step. On seed 2001 at 2e5 calls, 137 projections took 6.1 of the 7.3 s outside the potential, against 2.4 s on develop. The projections are not dead weight: 12 of the 1911 on seeds 2000–2011 set a new lowest, two of them the hit (seeds 2005 and 2009), while on LJ75 none of 1218 did. `77ca61dc` keeps them and keeps the walk's permutation. Below 98 atoms an O(n³) shortest-path scan supplies the permutation when its duals prove that every other assignment costs more above the optimum than the walk's tolerance allows it to fall short, and the walk still runs on a near tie. A fifth-turned orientation poses each operation's problem as the unturned orientation posed the conjugate operation, and an inverse operation the transposed problem, so those reuse a proved permutation after checking its duals on the new costs. Outside the potential M80 at ρ = 3 now costs 13.27 µs per call. Every other system costs no more there than on develop, except M80 at ρ = 14 (4% more) and Cu6Au32 (6%, of 1.2 µs). A `VERIFY_KM` build that also ran the walk on every certified solve, on twelve systems at seeds 2000–2005 and 4e5 calls, found no permutation that differed. The 91 pins are `a98cf10e…` at both `946e05ce` and `77ca61dc`.

### Screens

Hits are a function of the seed alone, so screens of the same builds, three at a time on cores 4–6 outside the `bench` namespace, attribute the differences. They are not in `rows.csv`. Seeds 2000–2011 below are the bench rows; the rest are screens.

| system | seeds | `8e50ff2e` | series | Fisher / MW |
|---|---|---|---|---|
| lj104 | 2000–2071 | 49/72 | 49/72 | 1.000 / 0.821 |
| lj150 | 2000–2047 | 26/48 | 28/48 | 0.837 / 0.624 |
| m80r14 | 2000–2071 | 50/72 | 43/72 | 0.296 / 0.225 |

LJ104's three-hit loss at 2000–2023 is a draw of seeds: at 2024–2071 develop hits 31/48 and the series 34/48. The `7192e9f8` section put that loss on the reseed-clock fix, `7759ac1d`. On `8e50ff2e` that commit alone hits 15/24, and the series without it 14/24; over 2000–2071 the series without it hits 47/72. LJ150 at 2012–2023 gave develop 7/12 and the series 5/12, and at 2024–2047, 14/24 and 18/24.

M80 at ρ = 14 loses 9/12 to 4/12 at 2000–2011, ties 15/24 at 2012–2035 and is 26/36 against 24/36 at 2036–2071. No change of the series carries it alone. At 2000–2011 the reseed-clock fix alone hits 8/12, the pair-minimum contact test alone 6/12, the series without the reseed-clock fix 6/12, the series without the pair minimum in `h2h_morse` 6/12, and the whole series 4/12. Each of those parts is also a win elsewhere:

- Without the reseed-clock fix (`Config::reseed_clock_lowest`, off), M80 at ρ = 6 hits 21/24 in place of 24/24 and M55 at ρ = 14 20/24 in place of 21/24, at seeds 2000–2023.
- Without the pair minimum in `h2h_morse`, M80 at ρ = 6 hits 21/24, M55 at ρ = 14 17/24 and M80 at ρ = 3 23/24, at the same seeds.
- Without the plateau gate, M80 at ρ = 6 is develop's 4/24.

So all three stay in the default, and M80 at ρ = 14 is the open regression, Fisher p = 0.30 and MW p = 0.23 over 72 seeds. Six of its eight misses at 2000–2011 end on −340.80256253, 0.0088 above the global minimum −340.81137066.
