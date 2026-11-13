# GrowPAL head-to-head, built from source

Anneal `621761ec` (`Config::recommended`, unchanged) against GrowPAL 0.1.6 and aegon 1.3.15, one `sudo unshare --uts` session with hostname `bench` on one Xeon VM, every timed process on core 0, systems run in order and the (seed, arm) pairs shuffled inside each with schedule seed 20261113. The driver is `scripts/growpal_h2h_v2/enter_bench.sh`; the Snakemake rule is `growpal_h2h_v2` and the asv suite is `benchmarks/growpal_h2h_v2.py`.

## What GrowPAL is here

- GrowPAL 0.1.6 is the newest release. It exists only as the PyPI sdist, sha256 `1625495ce899eddd7f6c8f449a4272f463efc5d7c8c8b1d9c2d40114ab195077`. GitHub `isaacgcampos/GrowPAL` main is `73952fa3a33ca9a1b1618d79bd154fd1480df518`, a 0.1.4 tree whose `sort_by_energy(..., 1)` and example `sort_by_energy(..., opt=1)` raise on aegon 1.3.15. On the one-species path the two trees differ only in that call and in moving `focused_expansion` into aegon. The published LJ10 and LJ9 seed files are read from that commit.
- aegon 1.3.15 is built from its sdist, sha256 `9d26a388d97ef7c75c3b613c6dc38660a3fdccda4bf714481ea6886d86ac881f`; its homepage repository is not public. Both sdists are built into wheels in `install_growpal.sh`. The other dependencies are pinned in `scripts/growpal_h2h_v2/constraints.txt` (ASE 3.26.0, scipy 1.18.1, numba 0.68.1, Python 3.12.3).
- The loop is the LJ production loop of `examples/LJ/x_run_growpal.py`: interstitial growth from LJ10 (Mo, `dtol` 1.2), aegon L-BFGS-B (`gtol` 1e-6, `maxiter` 1000), five reoptimisation rounds, USR deduplication (0.99, 0.1), the `5 N^(2/3)` window, Ward selection of `round(1.8 N)` per cluster, and the 0.01 early stop against aegon's LJ table. One core and one worker, including the k-NN graph.
- Morse M38 at rho = 6 goes through aegon's documented generic ASE strategy with ASE's `MorsePotential` (`epsilon` 1, `rho0` 6, `r0` 3 so the Mo radii and the insertion expansion see their Lennard-Jones scale, smooth cutoff moved to 1000 `r0`). Relaxation is aegon's ASE worker with its defaults (ASE LBFGS, `fmax` 0.01, 500 steps). The LJ10 seeds are relaxed on the Morse surface first and charged. aegon has no Morse table, so there is no per-size early stop.
- Gupta Au55 is not run. Neither GrowPAL nor aegon has a Gupta potential, and ASE has no Gupta calculator to route through.

## Counting and timing

One call is one energy-and-gradient evaluation: aegon's `_obj_and_grad`, the one energy the early stop reads, or one `MorsePotential.calculate`. The cap stops their optimiser inside a quench (`growpal_arm.py check`). The budgets are `paper_budget` in `h2h_gmin_proof` and `h2h_morse`: 400 000 at 38 atoms, 1 000 000 at 55, 4 000 000 at 75 and 98. A hit is a quenched energy at or below the Cambridge minimum plus 0.001; a miss is stored as the full budget. `wall_s` is each code's search clock after imports and JIT warmup; `pot_s` is the part inside the potential and `outside_s` the rest; `process_s` is spawn to exit. The warmup grows the published LJ9 seeds once before the clock: without it the first growth step compiles molsympy in the worker and cost 30 s of the first LJ38 seed. Medians and IQRs are Hyndman–Fan type 7. The ratio is the GrowPAL median wall over the anneal median wall.

`identity.csv` evaluates one anneal start per system in both codes. Anneal equals the full pair sum. aegon's Lennard-Jones drops pairs past 18 Å (6.73 sigma) without a shift, which moves the spread LJ75 and LJ98 starts by 1e-3 and 7e-3; the Cambridge LJ38–LJ98 minima span at most 5.2 sigma and score identically, and a GrowPAL candidate hit is rescored on the full sum. ASE's Morse equals the Cambridge form to 1e-14.

## Result

GrowPAL took no hit on any system. Under every budget the cap arrived while it was still growing, at 18 atoms (LJ38), 20 (LJ55), 28 (LJ75, LJ98), and 17 (M38), so no target-sized structure was quenched. Every intermediate size it finished matched aegon's table inside the 0.01 test. Its update draws no random numbers; the six seeds are replicates of one deterministic search and differ only in wall time. Anneal hit LJ98 on five seeds; seed 2004 ended at -543.642957. On Morse, anneal's preset quenches one Mackay cap before the walk when the second neighbour is bound, and that cap is the M38 minimum at rho = 6. The Morse inside time is ASE's Python calculator, so call counts, not wall, are the comparison there.

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

`uncapped.csv` is a diagnostic beside the comparison, not part of it (rule `growpal_h2h_v2_uncapped`). The same deterministic GrowPAL loop ran once per size with a 5e8 safety ceiling, seed 2000, anneal tree `2b46484d`, in its own `unshare` session after the campaign.

| System | Hit | Calls to hit | Times the paper budget | Wall (s) | Inside µs/call | Outside µs/call |
| --- | --- | --- | --- | --- | --- | --- |
| LJ38 | yes | 21038447 | 52.6 | 560.6 | 3.87 | 22.78 |
| LJ55 | yes | 104093769 | 104.1 | 3409.8 | 5.90 | 26.85 |

Per-seed rows are `seeds.csv` and `rows/<system>/<arm>-<seed>.row`; summaries are `results.csv`. All three are appended, never rewritten. No anneal source changed.
