# GMIN wall comparison, v5

Anneal `845d501e` against the same serial GMIN binary as v4, in one session. The driver is `scripts/gmin_h2h/enter_bench.sh`: `sudo unshare --uts`, hostname `bench`, working directory `/work`, `taskset -c 0`. LJ98 runs before LJ104. Inside a size the six seeds and two arms are shuffled with schedule seed 20261111. The budget is `PAPER_BUDGETS`, 4_000_000 force calls. A hit is a quenched energy within 0.001 of the Cambridge minimum (LJ98 −543.665361, LJ104 −582.086642). Wall medians are Hyndman–Fan type 7. The ratio is the GMIN median wall divided by the anneal median wall.

This commit drops the dual seeding. A same-session A/B at budget 300_000, seeds 2000–2005, interleaved on core 0, put the outside-potential median at 45.85 µs (scan) against 47.97 µs (seeded) for LJ98, and 47.37 µs against 51.09 µs for LJ104. A fivefold axis is assigned only when the nearest-image residual can still clear the cut, and a repeated residual on the same coordinates is reused. The LJ98 trajectory hash is unchanged.

GMIN is Wales revision `aac1548a8aec13b53ad12a8d6904625081685d2e`, built without OpenMP, MPI, or ALIGN. Its row has no potential timer. `pot_s` is the time inside `value_and_gradient`. Inside and outside microseconds are that split divided by the charged evaluations. The medians below are type 7 across the six seeds.

| Size | Arm | Hits | Median wall (s) | Ratio | v4 ratio | Earlier ratio | Inside µs | Outside µs |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| LJ98 | anneal | 4/6 | 151.503 | 1.114 | 1.032 | 1.129 | 18.15 | 46.66 |
| LJ98 | gmin | 0/6 | 168.737 | 1.114 | 1.032 | 1.129 |  |  |
| LJ104 | anneal | 3/6 | 251.227 | 0.744 | 0.702 | 0.770 | 20.49 | 53.86 |
| LJ104 | gmin | 2/6 | 186.964 | 0.744 | 0.702 | 0.770 |  |  |

Hit counts match v4. LJ98 is close to the earlier 1.129. LJ104 improved on v4 and is still behind GMIN.

A `cpu-clock` profile of this binary, LJ104 seed 2000, budget 200_000, core 0, no longer shows `prepare_duals` or the seeded matrix fill. The potential is 30.2% of samples. Outside it, the dense scan remains the large share: `scan_avx2` 26.7% and `successive_shortest_paths` 14.5%. Symmetry detection is 4.9% and orbit completion 2.2%.
