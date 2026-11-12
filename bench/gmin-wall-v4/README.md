# GMIN wall comparison, v4

Anneal `bfd50659` against the existing serial GMIN binary, in one session. The binary that produced these numbers was `1dc6292a`. Its Rust sources match `bfd50659`; develop only added paper and results files around them. The driver is `scripts/gmin_h2h/enter_bench.sh`: `sudo unshare --uts`, hostname `bench`, working directory `/work`, `taskset -c 0`. LJ98 runs before LJ104. Inside a size the six seeds and two arms are shuffled with schedule seed 20261111. The budget is `PAPER_BUDGETS`, 4_000_000 force calls. A hit is a quenched energy within 0.001 of the Cambridge minimum (LJ98 −543.665361, LJ104 −582.086642). Wall medians are Hyndman–Fan type 7. The ratio is the GMIN median wall divided by the anneal median wall.

GMIN is Wales revision `aac1548a8aec13b53ad12a8d6904625081685d2e`, built without OpenMP, MPI, or ALIGN. Its row has no potential timer, so only anneal has an inside/outside split. `pot_s` is the time inside `value_and_gradient`. Inside and outside microseconds are that split divided by the charged evaluations. The medians below are type 7 across the six seeds.

| Size | Arm | Hits | Median wall (s) | Ratio GMIN/anneal | Earlier ratio | Inside µs | Outside µs |
| --- | --- | --- | --- | --- | --- | --- | --- |
| LJ98 | anneal | 4/6 | 161.409 | 1.032 | 1.129 | 17.99 | 51.50 |
| LJ98 | gmin | 0/6 | 166.584 | 1.032 | 1.129 |  |  |
| LJ104 | anneal | 3/6 | 265.068 | 0.702 | 0.770 | 20.28 | 58.29 |
| LJ104 | gmin | 2/6 | 186.079 | 0.702 | 0.770 |  |  |

The earlier ratios are the previous same-session panel, after the scan change and before this dual seeding. Hit counts are unchanged: anneal 4/6 and 3/6, GMIN 0/6 and 2/6. Anneal's median wall is slower on both sizes, and LJ104 remains behind GMIN.

A `cpu-clock` profile of the same release binary, LJ104 seed 2000, budget 300_000, still on core 0, puts 26.1% of samples in `PairPotential::value_and_gradient` (the run's `pot_s` was 6.10 s of 23.29 s). Outside the potential, the dense assignment is the large share: `scan_avx2` 15.9%, the squared-distance matrix fill in `minimum_cost_squared_distance_warm` 15.4%, and `prepare_duals` 14.5%. The unseeded fallback scan is 1.3%. Symmetry detection is 5.0%, orbit completion 1.9%.
