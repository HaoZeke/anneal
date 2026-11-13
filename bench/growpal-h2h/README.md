# GrowPAL head-to-head

Anneal `63991087` against GrowPAL 0.1.6 and aegon 1.3.15, one session on this Xeon. The driver is `scripts/growpal_h2h/enter_bench.sh`: `sudo unshare --uts`, hostname `bench`, working directory `/work`, `taskset -c 0`. LJ38 runs before LJ75, then LJ98, then LJ104. Inside a size the six seeds and two arms are shuffled with schedule seed 20261112. The budget is `paper_budget` in `h2h_gmin_proof`: 400_000 evaluations at 38 atoms and 4_000_000 at 75, 98, and 104. A hit is a quenched energy at or below the Cambridge minimum plus 0.001. A miss is stored as the full budget. Wall, evaluation, and per-evaluation medians are Hyndman–Fan type 7. The ratio is the GrowPAL median wall divided by the anneal median wall. Inside microseconds are potential time divided by charged evaluations. Outside microseconds are the rest of the wall on the same denominator.

GrowPAL is the LJ production loop documented with the 0.1.6 release (`examples/LJ/x_run_growpal.py`). It grows the published `examples/LJ/LJ010.xyz` seeds by interstitial insertion (`dtol` 1.2, element Mo), relaxes with L-BFGS-B (`gtol` 1e-6, `maxiter` 1000), repeats that relaxation five times, drops USR duplicates at tolerances 0.99 and 0.1, cuts the pool to `5 N^(2/3)` above the best energy, and keeps `round(N * 1.8)` structures from each Ward cluster. The cluster count is the number of special-lineage structures, at least one. Their notes set `nproc` to `cpu_count - 2` and say to lower it on a shared machine. This session uses one core and one worker, including the k-NN graph inside Ward clustering. Checkpoint xyz files are not written. The published script has no evaluation ceiling; the driver stops L-BFGS-B when the charged count reaches the paper budget, so the cap is enforced. PyPI 0.1.6 is the release that was installed. GitHub `main` is `73952fa3a33ca9a1b1618d79bd154fd1480df518`, an older tree.

Every completed GrowPAL size matched the aegon Lennard-Jones reference, which agrees with the Cambridge values used here, inside their 0.01 early-stop test. The cap arrived during growth to 18 atoms on the 400_000 budget and during growth to 28 atoms on the 4_000_000 budget. No target-sized structure was quenched, so the best energy at the target size is blank and every GrowPAL row is a miss at the full budget. The published update does not draw random numbers. The six seeds are replicates of that search; the seed is stored and applied to the process RNGs. Wall times differ by a fraction of a second.

GrowPAL did not take more hits, or an equal hit count at a lower median wall, on any of the four sizes. The anneal preset was not changed.

| Size | Arm | Hits | Median evals | Median wall (s) | Ratio | Inside µs | Outside µs |
| --- | --- | --- | --- | --- | --- | --- | --- |
| LJ38 | anneal | 6/6 | 10189.500000 | 0.076948 | 203.177315 | 2.8059 | 3.8221 |
| LJ38 | growpal | 0/6 | 400000.000000 | 15.634088 | 203.177315 | 2.0112 | 37.0715 |
| LJ75 | anneal | 4/6 | 3089589.000000 | 79.580870 | 1.264287 | 10.6626 | 14.6616 |
| LJ75 | growpal | 0/6 | 4000000.000000 | 100.613077 | 1.264287 | 2.6869 | 22.4663 |
| LJ98 | anneal | 5/6 | 1239833.000000 | 53.180870 | 1.912565 | 18.1658 | 24.5659 |
| LJ98 | growpal | 0/6 | 4000000.000000 | 101.711882 | 1.912565 | 2.7020 | 22.7261 |
| LJ104 | anneal | 6/6 | 1812258.000000 | 85.165086 | 1.179673 | 20.4706 | 25.9449 |
| LJ104 | growpal | 0/6 | 4000000.000000 | 100.466945 | 1.179673 | 2.6810 | 22.4322 |

Per-seed rows are `seeds.csv` and `rows/<system>/<arm>-<seed>.row`.
