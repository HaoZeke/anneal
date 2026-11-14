# Structural priors ablation

`ANNEAL_STRUCTURAL_PRIORS=0` turns off every path that builds a structure from a known packing or chooses a move by classifying one (`Config::structural_priors`), including every SOAP proposal through `prior_soap`, so the priors-off arm draws no fivefold or icosahedral step. Priors on is the unset default. Each cell below is hits out of twelve, the median calls to hit with a miss counted as the full budget, the median wall, and the median seconds inside and outside the potential. Medians are Hyndman–Fan type 7. Fisher is the two-sided Fisher exact test of the hit counts; MW is the exact two-sided Mann–Whitney rank-sum test of the calls to hit, misses at the full budget, ties at midranks.

Seeds are 2000–2011, serial, `taskset -c 0`, hostname `bench` under `sudo unshare --uts`, working directory `/work`, at the harness budgets: 4e5 at 38 atoms, 1e6 at 55, 4e6 at 75 and above, and 2e6 for TIP4P. The Gupta alloys take the budgets of `bench/au55-scale/gate.csv`. `scripts/priors_ablation/enter_bench.sh` runs the anneal arms and `enter_gmin.sh` runs GMIN. `summarize.py` writes `summary.csv` and `table.md` from `rows.csv` and `gmin-rows.csv`.

Each binary scores its own hit within 0.001 of its reference: `h2h_gmin_proof` and `h2h_morse` against the Cambridge tables (M80 at ρ = 6 is 80B, −378.333471), `h2h_seed` for TIP4P (−116.5904186160, −305.5183221676 and −628.3722525077 kJ/mol), and `h2h_metal` for Au55, Cu6Au32 and Cu27Au28 against its table. Cu19Au19, Cu10Au45, Cu30Au8, Cu12Au26 and Pt12Cu11 take the `reference` column of `gate.csv` through `H2H_REFERENCE`, since `h2h_metal` has no table entry for most of them.

## Revisions

`rows.csv` and `gmin-rows.csv` are only ever appended. `code_revision` is the develop commit the series ran on, and for GMIN rows the commit whose harness and starts GMIN ran with. `host` is the first twelve hex digits of `/etc/machine-id`, since the hostname is always `bench`. Every row so far ran on host `dd8ee1b8e2c7`, an eight-vCPU Intel Xeon with AVX2, FMA and AVX-512. The 360 rows at `500aaf65` ran from one build of this series on develop `500aaf65`. The 480 rows at `f2a628c0` ran from one build of this series on develop `f2a628c0`, and they cover both arms of every anneal system: Au55, the seven alloys of `gate.csv`, M55 and M80 at ρ = 6 and 14, LJ38 to LJ150 and the three TIP4P sizes. `summarize.py` scores each (system, arm, seed) from its last row, so every cell below is at `f2a628c0`, and `summary.csv` lists the revisions and hosts of each arm.

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
| lj38 | 7/12 · 389399 · 3.220 s · 2.370 / 0.846 | 12/12 · 13470 · 0.201 s · 0.150 / 0.053 | 12/12 · 95593 · 1.257 s · 1.032 / 0.225 | 1.000 / 0.052 | 0.037 / 0.003 |
| lj75 | 0/12 · 4000000 · 106.017 s · 91.370 / 14.652 | 10/12 · 1501636 · 83.311 s · 62.631 / 21.513 | 4/12 · 4000000 · 199.967 s · 166.628 / 33.409 | 0.036 / 0.011 | 0.093 / 0.093 |
| lj98 | 0/12 · 4000000 · 166.207 s · 147.851 / 18.271 | 10/12 · 1785816 · 162.864 s · 127.527 / 35.336 | 6/12 · 3719081 · 318.209 s · 265.150 / 51.725 | 0.193 / 0.062 | 0.014 / 0.014 |
| lj104 | 6/12 · 3696118 · 173.296 s · 155.391 / 17.864 | 8/12 · 2492736 · 253.790 s · 200.343 / 54.040 | 10/12 · 890230 · 87.587 s · 71.118 / 16.469 | 0.640 / 0.256 | 0.193 / 0.029 |
| lj150 | 9/12 · 3129518 · 295.154 s · 274.112 / 21.042 | 3/12 · 4000000 · 831.539 s · 671.438 / 164.805 | 8/12 · 907352 · 183.997 s · 151.184 / 32.814 | 0.100 / 0.054 | 1.000 / 0.437 |
| m55r6 | 12/12 · 37048 · 0.661 s · 0.562 / 0.099 | 12/12 · 234 · 0.007 s · 0.007 / 0.001 | 12/12 · 27694 · 0.927 s · 0.786 / 0.145 | 1.000 / 0.000 | 1.000 / 0.843 |
| m80r6 | 12/12 · 268934 · 9.567 s · 8.586 / 0.980 | 9/12 · 798376 · 69.319 s · 50.578 / 18.795 | 12/12 · 39608 · 3.119 s · 2.714 / 0.405 | 0.217 / 0.143 | 1.000 / 0.045 |
| m55r14 | 0/12 · 1000000 · 17.551 s · 15.178 / 2.348 | 9/12 · 348446 · 14.879 s · 11.667 / 3.212 | 10/12 · 453332 · 17.708 s · 14.190 / 3.362 | 1.000 / 0.887 | 0.000 / 0.000 |
| m80r14 | 0/12 · 4000000 · 141.093 s · 127.961 / 13.067 | 8/12 · 2388292 · 183.048 s · 141.148 / 38.956 | 6/12 · 3759634 · 275.806 s · 221.851 / 53.955 | 0.680 / 0.242 | 0.014 / 0.014 |
| au55 | 0/12 · 1000000 · 58.476 s · 55.628 / 2.854 | 8/12 · 287364 · 11.357 s · 8.958 / 1.999 | 3/12 · 1000000 · 40.772 s · 30.761 / 10.015 | 0.100 / 0.034 | 0.217 / 0.217 |
| cu6au32 | 0/12 · 400000 · 12.056 s · 11.172 / 0.884 | 12/12 · 400000 · 6.532 s · 6.056 / 0.475 | 0/12 · 400000 · 7.245 s · 6.087 / 1.175 | 0.000 / 1.000 | 1.000 / 1.000 |
| cu27au28 | 0/12 · 1000000 · 59.720 s · 56.836 / 2.886 | 12/12 · 1000000 · 33.305 s · 31.542 / 1.760 | 1/12 · 1000000 · 36.178 s · 31.705 / 4.450 | 0.000 / 1.000 | 1.000 / 1.000 |
| cu19au19 | no row | 12/12 · 400000 · 6.493 s · 6.026 / 0.466 | 0/12 · 400000 · 7.092 s · 6.054 / 0.994 | 0.000 / 1.000 | – |
| cu10au45 | no row | 12/12 · 1000000 · 33.342 s · 31.556 / 1.793 | 2/12 · 1000000 · 37.574 s · 31.935 / 5.642 | 0.000 / 0.478 | – |
| cu30au8 | no row | 12/12 · 400000 · 6.521 s · 6.053 / 0.472 | 5/12 · 400000 · 7.137 s · 6.069 / 1.056 | 0.005 / 0.037 | – |
| cu12au26 | no row | 0/12 · 400000 · 6.549 s · 6.074 / 0.476 | 0/12 · 400000 · 7.170 s · 6.112 / 1.092 | 1.000 / 1.000 | – |
| pt12cu11 | no row | 12/12 · 200000 · 1.288 s · 1.112 / 0.173 | 1/12 · 200000 · 1.438 s · 1.121 / 0.316 | 0.000 / 1.000 | – |
| tip4p4 | 12/12 · 210 · 0.001 s · – / – | 12/12 · 227 · 0.001 s · 0.000 / 0.000 | 12/12 · 227 · 0.000 s · 0.000 / 0.000 | 1.000 / 1.000 | 1.000 / 0.551 |
| tip4p8 | 8/12 · 240163 · 1.623 s · – / – | 12/12 · 6339 · 0.020 s · 0.012 / 0.009 | 12/12 · 6339 · 0.020 s · 0.012 / 0.008 | 1.000 / 1.000 | 0.093 / 0.000 |
| tip4p15 | 0/12 · 2000000 · 41.094 s · – / – | 7/12 · 1481620 · 10.533 s · 7.834 / 2.699 | 7/12 · 1481620 · 10.583 s · 7.872 / 2.710 | 1.000 / 1.000 | 0.005 / 0.005 |

## Reading

Priors off loses hits on every alloy the walk solves with priors on. Cu6Au32 and Cu19Au19 fall from 12/12 to 0/12 (Fisher p = 1.1e-6 each), Cu27Au28 and Pt12Cu11 to 1/12 (p = 1.0e-5), Cu10Au45 to 2/12 (p = 6.7e-5) and Cu30Au8 to 5/12 (p = 0.0046). Cu12Au26 is 0/12 in both arms, as it is in `gate.csv` at the same budget. Au55 is 8/12 with priors on and 3/12 with priors off (Fisher p = 0.10, MW p = 0.034). At `500aaf65` the same seeds gave 5/12 and 0/12.

On Lennard-Jones, priors off loses hits on LJ75 (10/12 to 4/12, Fisher p = 0.036, MW p = 0.011) and needs about seven times the calls on LJ38 (MW p = 0.052). On LJ150 it hits more (8/12 against 3/12, MW p = 0.054). On M55 at ρ = 6 both arms hit every seed, but priors off needs far more calls (MW p = 1.1e-6). No other Lennard-Jones or Morse system separates the arms at 0.05. The TIP4P arms are identical, since no structural prior acts on rigid water.

On the alloys the hitting energy is not among the walk's recorded improvements, so the ledger never stops below the reference and `h2h_metal` reports the full charge as the calls to hit. The Mann–Whitney test sees only ties there, and Fisher carries the comparison.
