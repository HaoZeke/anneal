# GMIN wall comparison, v7

Anneal `325ebb15` against the same serial GMIN binary as v4, v5, and v6, in one session. The driver is `scripts/gmin_h2h/enter_bench.sh`: `sudo unshare --uts`, hostname `bench`, working directory `/work`, `taskset -c 0`. LJ98 runs before LJ104. Inside a size the six seeds and two arms are shuffled with schedule seed 20261111. The budget is `PAPER_BUDGETS`, 4_000_000 force calls. A hit is a quenched energy within 0.001 of the Cambridge minimum (LJ98 −543.665361, LJ104 −582.086642). Wall medians are Hyndman–Fan type 7. The ratio is the GMIN median wall divided by the anneal median wall.

A `cpu-clock` profile of the previous binary, LJ104 seed 2000, budget 150_000, core 0, put the potential at 36.9% of samples. Outside it, the dense assignment is still the large share: `scan_avx2` 19.6% and `successive_shortest_paths` 15.0%. Symmetry detection is 5.3%, a small sort 2.3%, and orbit completion 2.2%. SOAP descriptors, neighbour lists, the quench line search, allocation, and the ledger are each well under 2%.

This commit subtracts the Dijkstra delta from unused reduced costs four columns at a time, and scans nearest-image distances on centred coordinates held in contiguous columns. The recorded permutation is unchanged, including ties, and the LJ98 accepted-transition hash is unchanged. The evaluations to a hit match v6 on every seed.

GMIN is Wales revision `aac1548a8aec13b53ad12a8d6904625081685d2e`, built without OpenMP, MPI, or ALIGN. Its row has no potential timer. `pot_s` is the time inside `value_and_gradient`. Inside and outside microseconds are that split divided by the charged evaluations. The medians below are type 7 across the six seeds.

| Size | Arm | Hits | Median wall (s) | Median time to solution (s) | Ratio | v6 ratio | Earlier ratio | Inside µs | Outside µs |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| LJ98 | anneal | 4/6 | 133.061 | 133.061 | 1.283 | 1.274 | 1.129 | 18.25 | 38.71 |
| LJ98 | gmin | 0/6 | 170.722 | 170.722 | 1.283 | 1.274 | 1.129 |  |  |
| LJ104 | anneal | 3/6 | 221.714 | 221.714 | 0.862 | 0.871 | 0.770 | 20.64 | 45.09 |
| LJ104 | gmin | 2/6 | 191.076 | 191.076 | 0.862 | 0.871 | 0.770 |  |  |

The harness stops at the first hit, so a hit's `wall_s` is the time to solution and a miss is censored at the budget wall. `tts_s` in the CSV is that value. Its type-7 median is the median wall. A miss never reaches the target. When both hit, the smaller time wins.

| Session | Size | Anneal first | GMIN first | Neither |
| --- | --- | --- | --- | --- |
| v4 | LJ98 | 4/6 | 0/6 | 2/6 |
| v4 | LJ104 | 2/6 | 2/6 | 2/6 |
| v5 | LJ98 | 4/6 | 0/6 | 2/6 |
| v5 | LJ104 | 2/6 | 2/6 | 2/6 |
| v6 | LJ98 | 4/6 | 0/6 | 2/6 |
| v6 | LJ104 | 2/6 | 2/6 | 2/6 |
| v7 | LJ98 | 4/6 | 0/6 | 2/6 |
| v7 | LJ104 | 2/6 | 2/6 | 2/6 |

The pairing is the same as v4 through v6. At LJ98 anneal is first on every seed it hits, and GMIN hits none. At LJ104 each arm is first on two seeds: anneal on 2001 and 2002, GMIN on 2003 and on 2004, where both hit and GMIN's 69 s beats anneal's 129 s. Seeds 2000 and 2005 are misses for both.

GMIN's per-evaluation time, type 7 of `us_per_eval` on the six seeds, is the same binary across sessions. It rose through v6 and came back down in this one. The wall ratio moves with it.

| Session | LJ98 µs/eval | LJ104 µs/eval |
| --- | --- | --- |
| v4 | 41.65 | 47.09 |
| v5 | 42.18 | 47.45 |
| v6 | 46.46 | 51.28 |
| v7 | 42.68 | 47.88 |

Hit counts match v6. The LJ104 outside-potential median is 45.09 µs, from 48.23 µs in v6. The same-session ratio is 0.862, so LJ104 is still behind GMIN. Anneal's own median wall fell from 232.258 s in v6 to 221.714 s here, and from 145.857 s to 133.061 s at LJ98. GMIN's median wall in this session was 191.076 s at LJ104 and 170.722 s at LJ98, against 202.241 s and 185.832 s in v6.
