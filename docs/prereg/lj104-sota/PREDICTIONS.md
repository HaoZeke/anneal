Copyright (c) anneal developers

# LJ104 held-out comparison

This note registers the LJ104 comparison before any scored run.
The seeds in that run are 5000-5029.
Those seeds stay unrun in this commit.
The runner beside this note refuses them.
Seeds 5000 through 5029 are registered for this held-out comparison.
Their scored receipt comes later.
This note does not run them.

## Codes

The candidate is anneal.
It runs arm B v2's preset together with arm D v3.
The arm D search is the faithful dynamic-lattice search.
Arm B v2 is the GMIN-style symmetrisation and the surface moves.
The exact preset id is UNFILLED.
It is filled only from arm B's serial angular on/off grid.
That later commit is made before any run of seeds 5000-5029.

The arm D v3 tip sha is UNFILLED.
It is filled in that same later commit, still before any of those seeds.

GMIN_BEST is the patched GMIN binary.
Its sha256 begins b8eeddfa6d18a697.
That binary prints h2h_row.
On seeds 3000-3029 that binary scored 19 of 30.
This commit does not remeasure that block.
The prediction below carries the count across.

The full sha256 of that binary follows.
b8eeddfa6d18a697a76d0714c0a6b356c98c1d40022bd639ac55581e119baf95.
The executable is not in this tree.
That digest is the one recorded for the binary that prints h2h_row.
The recorded digest begins with this prefix.
The shared keyword block is gmin_best.keywords.
Its sha256 follows.
62618de90c53478bbfe724efe527dd1fa67cd6386da0cf00cb0cc8431bd7037d.
A scored data file is that block, then a blank line.
RANSEED is set to the seed.

GMIN counts one unit per entry to POTENTIAL.
In source/potential.f90, line 221 reads the potential-call comment.
The comment is "Count the number of potential calls."
Line 222 is NPCALL=NPCALL+1.
Those are the old-file lines of the h2h patch hunk.
That hunk starts at line 220.
The L-BFGS quench calls POTENTIAL once with the gradient requested.
That call is at source/mymylbfgs.F90 line 287.
Line 287 is the old-file line of the hunk that starts at line 285.
One entry is one value-and-gradient evaluation.
That is the same unit as src/methods/lattice_search.rs line 779.
No conversion sits between the two counters.

Plain DLS is the in-tree dynamic lattice search.
It is the dls reader in src/methods/lattice_search.rs.
It is a separate column from arm D v3.

## Seeds

The held-out seeds are 5000 through 5029, thirty seeds.
They were not used to tune the candidate, GMIN_BEST, or plain DLS.
paper/seeds.toml lines 65-71 register 5000-5063.
That registration is a different comparison, soap on against off.
It uses an icosahedral 38-mer and 12 Lennard-Jones sites.
That block is not a tuning set for the three codes here.

## Predicted hit rates

A hit is the LJ104 rule below.
Each prediction is a count out of these thirty seeds.

GMIN_BEST is predicted to hit 19 of 30.
The binary is the one already scored at 19 of 30 on seeds 3000-3029.
The held-out prediction is that same count.

The candidate is predicted to hit 19 of 30.
No combined run of the unfilled preset and the unfilled tip exists.
There is no pilot count to scale.
The registered prediction is parity with the GMIN_BEST carry-forward.
Parity is reported as parity.
The number stays as it is when the preset id and the tip sha are filled.

Plain DLS is predicted to hit 20 of 30.
That count is Papers' recompute of lattice v2 on seeds 3000-3029.
The lowest-index count is 20 of 30.
The uniform count is 22 of 30.
The GMIN count on those seeds is 19 of 30.
The prediction takes the lowest-index count.
Those seeds are 3000-3029.
They are different seeds from 5000-5029.
This prediction is not a measurement on the held-out block.

## Win rule

The decision between the candidate and GMIN_BEST is one-sided.
The test is an exact McNemar test.
The test uses the paired outcomes of the same thirty seeds.
A seed is discordant when exactly one of the two codes hits.
n is the number of discordant seeds.
b is the number of those seeds that only the candidate hits.
Under the null, b is binomial with n trials and probability 1/2.
The p-value is the probability of b or more.
Alpha is 0.05, which is the fraction 1/20.
The candidate wins when that exact tail is at most 1/20.
The integer form used by the script follows.
20 times the upper tail, counted in successes, is at most 2 to the n.

The threshold in hits is the smallest such b at the observed n.
mcnemar_threshold.py prints that minimum for every discordant count.
The counts run from 0 through 30.
The print includes the exact tail and the tail to eight decimal places.
mcnemar_threshold.tsv is the script's stdout and is the table of record.
The decimal column is a rendering.
The decision uses the exact tail.

Below 5 discordant seeds, no split wins.
At 5 discordant seeds the candidate wins only with b = 5.
The exact tail is 1/32, which is 0.03125.
At 30 discordant seeds the minimum is b = 20.
The exact tail is 26504551/536870912, which renders as 0.04936857.

```
discordant  candidate_only_min  p_exact                 p_decimal
0           none
1           none
2           none
3           none
4           none
5           5                   1/32                    0.03125000
6           6                   1/64                    0.01562500
7           7                   1/128                   0.00781250
8           7                   9/256                   0.03515625
9           8                   5/256                   0.01953125
10          9                   11/1024                 0.01074219
11          9                   67/2048                 0.03271484
12          10                  79/4096                 0.01928711
13          10                  189/4096                0.04614258
14          11                  235/8192                0.02868652
15          12                  9/512                   0.01757812
16          12                  2517/65536              0.03840637
17          13                  1607/65536              0.02452087
18          13                  1577/32768              0.04812622
19          14                  2083/65536              0.03178406
20          15                  5425/262144             0.02069473
21          15                  5135/131072             0.03917694
22          16                  13757/524288            0.02623940
23          16                  763/16384               0.04656982
24          17                  536155/16777216         0.03195733
25          18                  363103/16777216         0.02164263
26          18                  2533987/67108864        0.03775935
27          19                  3505699/134217728       0.02611949
28          19                  11698223/268435456      0.04357928
29          20                  8244773/268435456       0.03071417
30          20                  26504551/536870912      0.04936857
```

A tie is reported as a tie.
Parity, the case b = n - b, is equal hit counts.
It is reported as parity.
Neither is reported as a win.
A candidate lead short of the table minimum is reported as that lead.
The report includes its p-value, and it is not a win.
The test awards a win only to the candidate.
It awards that win only at the threshold above.
The point predictions are not a substitute for the paired test.

## No-regression panel

The panel is LJ38, LJ55, LJ75, and LJ98, with n = 6 seeds at each size.
The six seeds are 6000, 6001, 6002, 6003, 6004, and 6005.
They were never used for tuning.
They sit outside 5000-5029.
They also sit outside the ranges named in paper/seeds.toml.
Those ranges are 0-31, 2000-2247, and 3000-3029.
They also include 3100-3199, 4000-4047, and 5000-5063.
This paragraph names them before any run of the panel.

The floor at each size is the hit count of the current anneal default.
The count is taken on these same six seeds, after `lj_requench`.
The default search is `Config::recommended`.
That is the `recommended` arm of `examples/h2h_seed.rs`.
Each size runs at its PAPER_BUDGETS entry.
LJ38 line 16 is 400000.
LJ55 line 17 is 1000000.
LJ75 line 18 is 4000000.
LJ98 line 19 is 4000000.
The search may stop at E_known + 0.001.
That stop is not a hit.
A panel hit is the re-quenched energy at or below E_known + 1e-6.
The four E_known values are the Cambridge Cluster Database minima.
They are stored in `examples/h2h_arm3.rs` lines 46-49.
They are also listed in `e_known.tsv`.
A count below its floor is a mechanical FAIL.

LJ38 floor: 6/6.
LJ55 floor: 6/6.
LJ75 floor: 6/6.
LJ98 floor: 6/6.

The four lines above are the re-quench floors from requench.tsv.
They were written by requench_panel.py.
A hit is re-quenched E <= E_known + 1e-6.
Per-seed raw energy and re-quenched energy are in requench.md.
The hit flag and the re-quench calls are in that same note.
floors.md keeps the earlier 0.001 window count and is not the floor.
A count below its floor is a mechanical FAIL.

## Matched compute

The budget is in energy calls.
It comes from PAPER_BUDGETS in scripts/elja_submit_jcc_lj.sh.
The array is lines 15-22.
LJ104 is line 21, `[104]=4000000`.
The candidate, GMIN_BEST, and plain DLS each receive the same budget.
That budget is 4000000 energy calls.

The panel uses the same array.
LJ38 is line 16, 400000.
LJ55 is line 17, 1000000.
LJ75 is line 18, 4000000.
LJ98 is line 19, 4000000.

Every energy call is charged.
That includes each per-generation atom energy call.
One call is one unit of the budget.
The in-tree ledger prices a partial evaluation as a fraction of a call.
`Ledger::charge_frac` in src/methods/cluster_hopping.rs is at line 720.
It accumulates that fraction.
A residue under one call stays uncharged.
`Lattice::search` in src/methods/lattice_search.rs is at line 1388.
Line 1388 pays `1.0 + (m * n) / pairs` for one greedy pass.
This comparison charges each of those calls as a whole call.

Wall time is recorded for a strictly serial run, one process at a time.
The runner takes an exclusive lock.
It refuses to start while another scored run holds that lock.
The receipt is written at the start of the process.
It is written again at the end.
The receipt holds the kernel boot_id and the uptime reading.
It also holds the lscpu listing.
The start receipt and the end receipt are both kept.
A run whose two boot_id values differ is not one boot.
That run is not used.

## Hit rule

Every code in this comparison, GMIN_BEST included, uses one re-quench.
Every code uses that same hit rule.
The re-quench is `lj_requench`.
`lj_requench` is `WarmLbfgs::minimize` in `src/methods/warm_lbfgs.rs`.
`lj_requench` lives in `src/methods/lattice_search.rs`.
`lj_requench_calls` lives in that same file.
The solver is `WarmLbfgs::default` and the iteration cap is 5000.
The calls that minimization returns are the re-quench charge.
They are recorded apart from the search calls.
A hit is re-quenched E <= E_known + 1e-6.
E_known values are the Cambridge Cluster Database minima.
The list is `e_known.tsv`.
LJ38, LJ55, LJ75, and LJ98 are the lines in `examples/h2h_arm3.rs`.
Those lines are 46, 47, 48, and 49.
LJ104 is -582.086642 at `examples/lj_cluster_search.rs` line 1295.
Its hit line is E <= -582.086641.
The test re-quenches the known LJ104 minimum to the hit line.
The test name follows.
`lj_requench_brings_the_known_lj104_minimum_to_the_hit_line`.
The fixture is `tests/fixtures/lj104_gm.xyz`.
The test requires that hit line.

## Scored rows

The runner is lj104_sota_runner.py.
Every scored row cites the git blob id of that file.
The row also cites HEAD and the tree of HEAD.
The columns are runner_blob, head, and tree.
A row that omits any of the three is not a scored row.
test_lj104_sota_runner.py checks the concurrent-start refusal.
A second scored start is refused.
The test also checks that a written row carries those three fields.
