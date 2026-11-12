# GMIN wall comparison, v6

Anneal `b8fda7d4` against the same serial GMIN binary as v4 and v5, in one session. The driver is `scripts/gmin_h2h/enter_bench.sh`: `sudo unshare --uts`, hostname `bench`, working directory `/work`, `taskset -c 0`. LJ98 runs before LJ104. Inside a size the six seeds and two arms are shuffled with schedule seed 20261111. The budget is `PAPER_BUDGETS`, 4_000_000 force calls. A hit is a quenched energy within 0.001 of the Cambridge minimum (LJ98 −543.665361, LJ104 −582.086642). Wall medians are Hyndman–Fan type 7. The ratio is the GMIN median wall divided by the anneal median wall.

A count at LJ104, seed 2000, budget 200_000, recorded 26_990 dense assignments across 1_648 hops and 200_000 charged evaluations: 0.135 per evaluation and 16.4 per hop. D_5h was 18_924 of them (70 percent), from 19 operations on each of six orientations. C_5 was 3_984 (15 percent), C_i 996 (4 percent), and SOAP fivefold alignment 3_086 (11 percent). Symmetry detection does not call the dense assignment. All six orientations were assigned in full, and D_5h won the group choice on 148 of 166 projections.

This commit reuses an operation's permutation for its inverse when the dual has no alternating cycle within a reduced-cost slack of 1e-8. That held for about seven scans in eight. D_5h is solved first, so the C_5 turns are taken from that result. The LJ98 accepted-transition hash is unchanged.

GMIN is Wales revision `aac1548a8aec13b53ad12a8d6904625081685d2e`, built without OpenMP, MPI, or ALIGN. Its row has no potential timer. `pot_s` is the time inside `value_and_gradient`. Inside and outside microseconds are that split divided by the charged evaluations. The medians below are type 7 across the six seeds.

| Size | Arm | Hits | Median wall (s) | Ratio | v5 ratio | Earlier ratio | Inside µs | Outside µs |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| LJ98 | anneal | 4/6 | 145.857 | 1.274 | 1.114 | 1.129 | 19.28 | 44.39 |
| LJ98 | gmin | 0/6 | 185.832 | 1.274 | 1.114 | 1.129 |  |  |
| LJ104 | anneal | 3/6 | 232.258 | 0.871 | 0.744 | 0.770 | 20.75 | 48.23 |
| LJ104 | gmin | 2/6 | 202.241 | 0.871 | 0.744 | 0.770 |  |  |

Hit counts match v5. The LJ104 outside-potential median is 48.23 µs, from 53.86 µs in v5. The same-session ratio is 0.871, so LJ104 is still behind GMIN. The anneal median fell from 251.227 s to 232.258 s, and this session's GMIN median was 202.241 s.
