# LJ150: anneal develop against GMIN

anneal `c2de652dca0bc75e2b0cb4afffac91fd656404e4` (develop) against GMIN
`aac1548a8aec13b53ad12a8d6904625081685d2e`, seeds 2000-2011, 4,000,000
potential calls each, the `PAPER_BUDGETS` cap. One session:
`snakemake -s workflow/Snakefile gmin_h2h_lj150`, which builds the anneal
revision in a detached worktree (Rust 1.88.0, `--release --locked`) and runs
`scripts/gmin_h2h/run_campaign.py` serially on core 0 under
`sudo unshare --uts` with hostname `bench`. Each arm gets the same start per
seed. Before it, `asv run` of `bench/gmin-h2h` (`time_seed` at size 150,
seed 2000, then in `GminHeadToHead`, now `GminHeadToHeadLJ150`) repeated
seed 2000 on both arms: anneal 1,672,453 calls, GMIN 3,562,012, the calls
in `rows.csv`.

- `rows.csv`: one row per run, appended as each run finished, with
  `code_revision` and `host`. `pot_s` is the time inside the potential and
  `outside_s` the rest of the wall.
- `summary.csv`: hits and median calls per arm, from the runs.
- `stats.csv`: Fisher's two-sided exact test on hits; the two-sided
  Mann-Whitney test (normal approximation, tie and continuity corrections)
  on calls to the minimum, on inside time and on outside time. A miss is
  entered at the budget. Medians are Hyndman-Fan type 7.

| | anneal `c2de652d` | GMIN `aac1548a` | test |
| --- | --- | --- | --- |
| hits | 5/12 | 9/12 | Fisher p = 0.214 |
| median calls | 4,000,000 | 3,129,518.5 | Mann-Whitney U = 95.5, p = 0.168 |
| median inside (s) | 148.1 | 318.5 | p = 0.214 |
| median outside (s) | 100.5 | 24.1 | p = 0.00090 |

develop still loses LJ150 to GMIN, 5/12 against 9/12. The gap in hits is
not significant at n = 12. anneal's calls repeat `bench/paper-0120` on every
seed (hit seeds 2000, 2002, 2006, 2007, 2009); GMIN's repeat
`bench/lj-large` on seed 2000.

Per seed: result, calls to the minimum, best energy, wall / inside /
outside in seconds. The putative global minimum is -893.310258.

| seed | anneal | best | wall / inside / outside | GMIN | best | wall / inside / outside |
| --- | --- | --- | --- | --- | --- | --- |
| 2000 | hit 1,672,453 | -893.310258 | 102.9 / 61.2 / 41.7 | hit 3,562,012 | -893.310187 | 374.7 / 347.8 / 26.9 |
| 2001 | miss | -884.640664 | 252.9 / 155.1 / 97.8 | miss | -884.731156 | 461.7 / 426.7 / 35.0 |
| 2002 | hit 2,805,381 | -893.310258 | 168.9 / 107.9 / 61.0 | hit 623,314 | -893.310229 | 64.9 / 60.3 / 4.6 |
| 2003 | miss | -885.282282 | 279.2 / 150.1 / 129.2 | hit 3,741,909 | -893.310156 | 412.8 / 383.5 / 29.4 |
| 2004 | miss | -883.175074 | 265.1 / 147.7 / 117.4 | miss | -883.175037 | 565.3 / 519.9 / 45.4 |
| 2005 | miss | -884.636203 | 254.1 / 150.9 / 103.2 | hit 395,583 | -893.310206 | 43.4 / 40.4 / 3.0 |
| 2006 | hit 10,134 | -893.310258 | 0.6 / 0.4 / 0.3 | hit 3,192,903 | -893.310192 | 371.0 / 345.0 / 26.0 |
| 2007 | hit 3,446,339 | -893.310256 | 221.7 / 124.4 / 97.3 | hit 54,271 | -893.310169 | 6.3 / 5.9 / 0.4 |
| 2008 | miss | -883.496893 | 264.1 / 150.2 / 113.9 | hit 211,353 | -893.310189 | 21.9 / 20.4 / 1.5 |
| 2009 | hit 1,307,484 | -893.310258 | 88.3 / 49.5 / 38.8 | hit 1,274,153 | -893.310218 | 155.7 / 144.8 / 10.9 |
| 2010 | miss | -883.496893 | 267.2 / 148.4 / 118.8 | hit 3,066,134 | -893.310210 | 314.4 / 292.1 / 22.2 |
| 2011 | miss | -883.110588 | 317.8 / 166.1 / 151.7 | miss | -885.282265 | 481.6 / 447.5 / 34.0 |

The times are not from a quiet machine. Core 0 ran only this session, but
cores 2-7 ran other single-threaded searches for the whole session, so the
shared cache and memory bus were loaded. GMIN's seed 2000 took 374.7 s here
against 333 s in `bench/lj-large`. Calls and hits do not depend on load;
the times do, for both arms.

The anneal misses stop improving early. The lowest minimum the walker
accepts on seeds 2001, 2003, 2004 and 2005 (-884.64, -882.63, -883.18,
-884.64) is reached by 142,315, 219,056, 32,263 and 16,128 calls, and the
walker accepts nothing lower in the rest of the budget.

## With `branch_on_stall`

`snakemake -s workflow/Snakefile gmin_h2h_lj150_branch` repeats the panel
with `Config::branch_on_stall` on in the anneal arm (`H2H_BRANCH=1`) and
GMIN rerun in the same session, on a machine running nothing else. The
anneal build is tree `bfeae2bdd4dc758705194d8d6c28def393416b8a`, the commit
that records `bench/branch-gate`; it differs from that gate's branch tree
only in the `bench/branch-gate` files. Its `code_revision` is a local commit
with that tree. Rows are in `branch-rows.csv`, then `branch-summary.csv` and
`branch-stats.csv`, computed as above. GMIN's calls and energies repeat the
first session on every seed.

| | anneal, `branch_on_stall` | GMIN `aac1548a` | test |
| --- | --- | --- | --- |
| hits | 11/12 | 9/12 | Fisher p = 0.590 |
| median calls | 843,181.5 | 3,129,518.5 | Mann-Whitney U = 51.5, p = 0.247 |
| median wall (s) | 52.0 | 314.7 | |
| median inside (s) | 29.8 | 292.1 | p = 0.014 |
| median outside (s) | 22.2 | 22.6 | p = 0.260 |

With the flag on anneal hits more LJ150 seeds than GMIN, 11/12 against
9/12, and takes a tenth of the time inside the potential at the median;
the difference in hits is not significant at n = 12. The anneal arm's
outside time is about 0.7 of its inside time, against 0.08 for GMIN.
Seed 2004 is missed by both arms; GMIN also misses 2001 and 2011.
develop (`c2de652d`, flag off) is 5/12 on the same seeds. The flag is off
by default because it loses a seed on four other systems in
`bench/branch-gate`.

| seed | anneal | best | wall / inside / outside | GMIN | best | wall / inside / outside |
| --- | --- | --- | --- | --- | --- | --- |
| 2000 | hit 1,502,835 | -893.310258 | 88.4 / 52.0 / 36.5 | hit 3,562,012 | -893.310187 | 358.7 / 332.9 / 25.8 |
| 2001 | hit 798,646 | -893.310258 | 48.8 / 27.8 / 21.0 | miss | -884.731156 | 396.9 / 368.4 / 28.5 |
| 2002 | hit 688,014 | -893.310258 | 39.7 / 23.4 / 16.3 | hit 623,314 | -893.310229 | 61.9 / 57.5 / 4.4 |
| 2003 | hit 404,537 | -893.310258 | 23.4 / 14.2 / 9.2 | hit 3,741,909 | -893.310156 | 362.0 / 336.0 / 26.0 |
| 2004 | miss | -885.215293 | 242.5 / 137.4 / 105.1 | miss | -883.175037 | 410.0 / 381.1 / 28.9 |
| 2005 | hit 3,080,181 | -893.310258 | 175.4 / 104.5 / 70.9 | hit 395,583 | -893.310206 | 40.0 / 37.1 / 2.9 |
| 2006 | hit 10,134 | -893.310258 | 0.6 / 0.3 / 0.3 | hit 3,192,903 | -893.310192 | 313.8 / 291.3 / 22.5 |
| 2007 | hit 534,363 | -893.310258 | 31.2 / 18.2 / 13.0 | hit 54,271 | -893.310169 | 5.3 / 5.0 / 0.4 |
| 2008 | hit 887,717 | -893.310258 | 55.3 / 31.8 / 23.4 | hit 211,353 | -893.310189 | 21.2 / 19.7 / 1.5 |
| 2009 | hit 2,549,074 | -893.310258 | 154.9 / 89.5 / 65.4 | hit 1,274,153 | -893.310218 | 128.0 / 118.8 / 9.2 |
| 2010 | hit 345,852 | -893.310258 | 21.4 / 12.1 / 9.3 | hit 3,066,134 | -893.310210 | 315.6 / 293.0 / 22.6 |
| 2011 | hit 2,028,271 | -893.310258 | 128.8 / 72.9 / 55.9 | miss | -885.282265 | 385.5 / 357.3 / 28.2 |
