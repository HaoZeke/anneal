# GMIN wall comparison, v6

Anneal `b8fda7d4` against the same serial GMIN binary as v4 and v5, in one session. The driver is `scripts/gmin_h2h/enter_bench.sh`: `sudo unshare --uts`, hostname `bench`, working directory `/work`, `taskset -c 0`. LJ98 runs before LJ104. Inside a size the six seeds and two arms are shuffled with schedule seed 20261111. The budget is `PAPER_BUDGETS`, 4_000_000 force calls. A hit is a quenched energy within 0.001 of the Cambridge minimum (LJ98 −543.665361, LJ104 −582.086642). Wall medians are Hyndman–Fan type 7. The ratio is the GMIN median wall divided by the anneal median wall.

A count at LJ104, seed 2000, budget 200_000, recorded 26_990 dense assignments across 1_648 hops and 200_000 charged evaluations: 0.135 per evaluation and 16.4 per hop. D_5h was 18_924 of them (70 percent), from 19 operations on each of six orientations. C_5 was 3_984 (15 percent), C_i 996 (4 percent), and SOAP fivefold alignment 3_086 (11 percent). Symmetry detection does not call the dense assignment. All six orientations were assigned in full, and D_5h won the group choice on 148 of 166 projections.

This commit reuses an operation's permutation for its inverse when the dual has no alternating cycle within a reduced-cost slack of 1e-8. That held for about seven scans in eight. D_5h is solved first, so the C_5 turns are taken from that result. The LJ98 accepted-transition hash is unchanged.

GMIN is Wales revision `aac1548a8aec13b53ad12a8d6904625081685d2e`, built without OpenMP, MPI, or ALIGN. Its row has no potential timer. `pot_s` is the time inside `value_and_gradient`. Inside and outside microseconds are that split divided by the charged evaluations. The medians below are type 7 across the six seeds.

| Size | Arm | Hits | Median wall (s) | Median time to solution (s) | Ratio | v5 ratio | Earlier ratio | Inside µs | Outside µs |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| LJ98 | anneal | 4/6 | 145.857 | 145.857 | 1.274 | 1.114 | 1.129 | 19.28 | 44.39 |
| LJ98 | gmin | 0/6 | 185.832 | 185.832 | 1.274 | 1.114 | 1.129 |  |  |
| LJ104 | anneal | 3/6 | 232.258 | 232.258 | 0.871 | 0.744 | 0.770 | 20.75 | 48.23 |
| LJ104 | gmin | 2/6 | 202.241 | 202.241 | 0.871 | 0.744 | 0.770 |  |  |

The harness stops at the first hit, so a hit's `wall_s` is the time to solution and a miss is censored at the budget wall. `tts_s` in the CSV is that value. Its type-7 median is therefore the median wall. What the wall median does not say is which arm reaches the target first on the same seed. A miss never reaches it. When both hit, the smaller time wins.

| Session | Size | Anneal first | GMIN first | Neither |
| --- | --- | --- | --- | --- |
| v4 | LJ98 | 4/6 | 0/6 | 2/6 |
| v4 | LJ104 | 2/6 | 2/6 | 2/6 |
| v5 | LJ98 | 4/6 | 0/6 | 2/6 |
| v5 | LJ104 | 2/6 | 2/6 | 2/6 |
| v6 | LJ98 | 4/6 | 0/6 | 2/6 |
| v6 | LJ104 | 2/6 | 2/6 | 2/6 |

At LJ98 anneal is first on every seed it hits, and GMIN hits none. At LJ104 each arm is first on two seeds: anneal on 2001 and 2002, GMIN on 2003 and on 2004, where both hit and GMIN's 71 s beats anneal's 150 s. Seeds 2000 and 2005 are misses for both. The same pairing holds in v4 and v5.

GMIN's per-evaluation time, type 7 of `us_per_eval` on the six seeds, drifted up across the three sessions. That is the same binary. The wall ratio moves with it.

| Session | LJ98 µs/eval | LJ104 µs/eval |
| --- | --- | --- |
| v4 | 41.65 | 47.09 |
| v5 | 42.18 | 47.45 |
| v6 | 46.46 | 51.28 |

Hit counts match v5. The LJ104 outside-potential median is 48.23 µs, from 53.86 µs in v5. The same-session ratio is 0.871, so LJ104 is still behind GMIN. Anneal's own median wall fell from 251.227 s in v5 to 232.258 s here. GMIN's median wall in this session was 202.241 s, against 186.964 s in v5 and 186.079 s in v4.
