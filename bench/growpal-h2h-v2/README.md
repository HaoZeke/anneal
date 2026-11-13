# GrowPAL head-to-head, built from source

Anneal `621761ec` (`Config::recommended`, unchanged) against GrowPAL 0.1.6 and aegon 1.3.15, one `sudo unshare --uts` session with hostname `bench` on one Xeon VM, every timed process on core 0, systems run in order and the (seed, arm) pairs shuffled inside each with schedule seed 20261113. The driver is `scripts/growpal_h2h_v2/enter_bench.sh`; the Snakemake rule is `growpal_h2h_v2` and the asv suite is `benchmarks/growpal_h2h_v2.py`.

## Setup

- GrowPAL 0.1.6 is built from the PyPI sdist, sha256 `1625495ce899eddd7f6c8f449a4272f463efc5d7c8c8b1d9c2d40114ab195077`. The LJ10 and LJ9 seed files come from GitHub `isaacgcampos/GrowPAL` commit `73952fa3a33ca9a1b1618d79bd154fd1480df518`.
- aegon 1.3.15 is built from the PyPI sdist, sha256 `9d26a388d97ef7c75c3b613c6dc38660a3fdccda4bf714481ea6886d86ac881f`. Both sdists are built into wheels by `install_growpal.sh`; the other dependencies are pinned in `scripts/growpal_h2h_v2/constraints.txt` (ASE 3.26.0, scipy 1.18.1, numba 0.68.1, Python 3.12.3).
- Lennard-Jones: the production loop of GrowPAL's `examples/LJ/x_run_growpal.py` with aegon's Lennard-Jones and L-BFGS-B (`gtol` 1e-6, `maxiter` 1000), growth from LJ10 (Mo, `dtol` 1.2), five reoptimisation rounds, USR deduplication (0.99, 0.1), the `5 N^(2/3)` window, Ward selection of `round(1.8 N)` per cluster, and the 0.01 early stop against aegon's LJ table. One worker.
- Morse M38 at rho = 6: aegon's generic ASE route with ASE's `MorsePotential` (`epsilon` 1, `rho0` 6, `r0` 3, smooth cutoff from 1000 `r0`) and aegon's ASE worker defaults (ASE LBFGS, `fmax` 0.01, 500 steps). No per-size early stop.
- Anneal uses its own Lennard-Jones and Morse pair sums.
- Gupta Au55 is not run: GrowPAL has no Gupta potential.

## Counting and timing

One call is one energy-and-gradient evaluation: aegon's `_obj_and_grad`, the energy the early stop reads, or one `MorsePotential.calculate`. The cap stops the optimiser inside a quench. The budgets are `paper_budget` in `h2h_gmin_proof` and `h2h_morse`: 400 000 at 38 atoms, 1 000 000 at 55, 4 000 000 at 75 and 98. A hit is a quenched energy at or below the Cambridge minimum plus 0.001; a miss is stored as the full budget. `wall_s` is each code's search clock after imports and JIT warmup; `pot_s` is the part inside the potential and `outside_s` the rest; `process_s` is spawn to exit. Medians and IQRs are Hyndman–Fan type 7. The ratio is the GrowPAL median wall over the anneal median wall. `identity.csv` holds both codes' energy of one anneal start per system.

## Result

GrowPAL hit 0 of 30 runs. At the budget it had grown to 18 atoms on LJ38, 20 on LJ55, 28 on LJ75 and LJ98, and 17 on M38. Anneal hit 29 of 30; LJ98 seed 2004 ended at -543.642957.

| System | Arm | Hits | Budget | Median calls | Median calls to hit | Median wall (s) | Inside (s) | Outside (s) | Inside µs/call | Outside µs/call | Ratio | Reached |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| LJ38 | anneal | 6/6 | 400000 | 10189.5 | 10189.5 | 0.086 | 0.033 | 0.054 | 3.12 | 4.43 | 144.9 | 38 |
| LJ38 | growpal | 0/6 | 400000 | 400000 | — | 12.496 | 0.858 | 11.642 | 2.15 | 29.10 | 144.9 | 18 |
| LJ55 | anneal | 6/6 | 1000000 | 11246.0 | 11246.0 | 0.174 | 0.075 | 0.099 | 6.86 | 9.16 | 162.0 | 55 |
| LJ55 | growpal | 0/6 | 1000000 | 1000000 | — | 28.157 | 2.440 | 25.676 | 2.44 | 25.68 | 162.0 | 20 |
| LJ75 | anneal | 6/6 | 4000000 | 979290.5 | 979290.5 | 28.547 | 12.011 | 16.031 | 11.90 | 15.89 | 3.71 | 75 |
| LJ75 | growpal | 0/6 | 4000000 | 4000000 | — | 105.887 | 12.117 | 93.770 | 3.03 | 23.44 | 3.71 | 28 |
| LJ98 | anneal | 5/6 | 4000000 | 1387168.5 | 1044686.0 | 59.956 | 26.248 | 33.708 | 19.17 | 21.40 | 1.61 | 98 |
| LJ98 | growpal | 0/6 | 4000000 | 4000000 | — | 96.514 | 10.943 | 85.588 | 2.74 | 21.40 | 1.61 | 28 |
| M38 rho 6 | anneal | 6/6 | 400000 | 297.5 | 297.5 | 0.0028 | 0.0022 | 0.0005 | 7.58 | 1.82 | 141683 | 38 |
| M38 rho 6 | growpal | 0/6 | 400000 | 400000 | — | 393.312 | 198.690 | 194.622 | 496.72 | 486.55 | 141683 | 17 |

## Uncapped GrowPAL

`uncapped.csv` (rule `growpal_h2h_v2_uncapped`): GrowPAL with a 5e8-call ceiling, seed 2000, anneal tree `2b46484d`, in its own `unshare` session after the campaign.

| System | Hit | Calls to hit | Times the paper budget | Wall (s) | Inside (s) | Outside (s) | Inside µs/call | Outside µs/call |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| LJ38 | yes | 21038447 | 52.6 | 560.6 | 81.3 | 479.3 | 3.87 | 22.78 |
| LJ55 | yes | 104093769 | 104.1 | 3409.8 | 614.6 | 2795.2 | 5.90 | 26.85 |

Per-seed rows are `seeds.csv` and `rows/<system>/<arm>-<seed>.row`; summaries are `results.csv`. All three are appended, never rewritten. No anneal source changed.
