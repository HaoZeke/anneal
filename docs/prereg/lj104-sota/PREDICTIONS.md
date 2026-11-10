Copyright © anneal developers

# LJ104 held-out comparison

We register the LJ104 comparison before any scored run.
We leave seeds 5000-5029 unrun in this note.
The runner beside this note refuses those seeds.
Seeds 5000 through 5029 are registered for this held-out comparison.
Their scored receipt comes later.
We do not run those seeds here.

## Codes

The candidate is anneal.
It runs arm B v2's preset together with arm D v3.
Dynamic lattice search (DLS) is the faithful arm D search.
The basin-hopping program (GMIN) supplies the symmetrisation.
GMIN also supplies the surface moves in arm B v2.
The exact preset id is UNFILLED.
We fill that id only from arm B's serial angular on/off grid.
That later commit is made before any run of seeds 5000-5029.

The arm D v3 tip sha is UNFILLED.
We fill that sha in the same later commit.
That fill still comes before any of those seeds.

GMIN_BEST is the patched GMIN binary.
Its sha256 begins b8eeddfa6d18a697.
That binary prints h2h_row.
On seeds 3000-3029 that binary scored 19 of 30.
We leave that block as already scored.
The prediction below carries the count across.

The recorded sha256 string follows.
b8eeddfa6d18a697a76d0714c0a6b356c98c1d40022bd639ac55581e119baf95.
The executable is absent from this tree.
The digest is unverified.
We do not treat that string as a rebuilt match.
The source archive sha256 follows.
3f31e4a4cb234aa602470440fa1996b789725e537d47b20eb3101a0749525b70.
The patch sha256 follows.
e9e8395c210375ac33c7f174c85726b1ccbeed4bbfd9d2e130e6cbb4a253eb35.
The GMIN source has no pinned upstream commit.
`GNU Fortran 13.3.0` produced the baseline with the recorded digest.
A local rebuild is recorded apart from that baseline.
We do not claim the local digest equals the recorded string.
The shared keyword block is gmin_best.keywords.
Its sha256 follows.
62618de90c53478bbfe724efe527dd1fa67cd6386da0cf00cb0cc8431bd7037d.
A scored data file is that block, then a blank line.
RANSEED is set to the seed.

## Evidence rows

We keep every file and number in the rows below.
The rows are the file, the number, and the record.

| file | n | record |
| --- | --- | --- |
| source/potential.f90 | 221 | call-count comment |
| source/potential.f90 | 222 | NPCALL=NPCALL+1 |
| source/potential.f90 | 220 | hunk start |
| source/mymylbfgs.F90 | 287 | POTENTIAL, gradient on |
| source/mymylbfgs.F90 | 285 | hunk start |
| src/methods/lattice_search.rs | 779 | value-and-gradient unit |
| src/methods/cluster_hopping.rs | 720 | Ledger::charge_frac |
| src/methods/lattice_search.rs | 1388 | 1.0 + (m * n) / pairs |
| examples/h2h_arm3.rs | 46 | LJ38 -173.928427 |
| examples/h2h_arm3.rs | 47 | LJ55 -279.248470 |
| examples/h2h_arm3.rs | 48 | LJ75 -397.492331 |
| examples/h2h_arm3.rs | 49 | LJ98 -543.665361 |
| examples/lj_cluster_search.rs | 1295 | LJ104 -582.086642 |
| scripts/elja_submit_jcc_lj.sh | 15 | PAPER_BUDGETS start |
| scripts/elja_submit_jcc_lj.sh | 16 | LJ38 400000 |
| scripts/elja_submit_jcc_lj.sh | 17 | LJ55 1000000 |
| scripts/elja_submit_jcc_lj.sh | 18 | LJ75 4000000 |
| scripts/elja_submit_jcc_lj.sh | 19 | LJ98 4000000 |
| scripts/elja_submit_jcc_lj.sh | 21 | LJ104 4000000 |
| scripts/elja_submit_jcc_lj.sh | 22 | PAPER_BUDGETS end |
| paper/seeds.toml | 65 | 5000-5063 start |
| paper/seeds.toml | 71 | 5000-5063 end |

GMIN counts one unit per entry to POTENTIAL.
The comment text is "Count the number of potential calls."
One entry is one value-and-gradient evaluation.
That unit matches the lattice search record in the rows above.
No conversion sits between the two counters.

Plain DLS is the in-tree reader in src/methods/lattice_search.rs.
It is a separate column from arm D v3.

## Seeds

The held-out seeds are 5000 through 5029, thirty seeds.
We did not use them to tune the candidate.
We did not use them to tune GMIN_BEST.
We did not use them to tune plain DLS.
paper/seeds.toml registers 5000-5063 for a different comparison.
The evidence rows name that registration.
That comparison is soap on against off.
It uses an icosahedral 38-mer and 12 Lennard-Jones sites.
That block is a different comparison from the three codes here.

## Predicted hit rates

A hit is the LJ104 rule below.
Each prediction is a count out of these thirty seeds.

GMIN_BEST is predicted to hit 19 of 30.
The binary is the one already scored at 19 of 30 on seeds 3000-3029.
The held-out prediction is that same count.

The candidate is predicted to hit 19 of 30.
No combined run of the unfilled preset and the unfilled tip exists.
There is no pilot count to scale.
We register parity with the GMIN_BEST carry-forward.
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
This prediction is a carry from those other seeds.

## Win rule

We compare the candidate and GMIN_BEST with one test.
The test is a one-sided exact McNemar test.
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
The report includes its p-value, and the lead stays a lead.
We award a win only to the candidate.
We award that win only at the threshold above.
The point predictions stay separate from the paired test.

## No-regression panel

We score the panel at LJ38 with n = 6.
We score LJ55 with the same n.
We score LJ75 with the same n.
We score LJ98 with the same n.
The six seeds are 6000, 6001, 6002, 6003, 6004, and 6005.
They were never used for tuning.
They sit outside 5000-5029.
They also sit outside the ranges named in paper/seeds.toml.
Those ranges are 0-31, 2000-2247, and 3000-3029.
They also include 3100-3199, 4000-4047, and 5000-5063.
We name the seeds before any run of the panel.

The floor at each size is the hit count of the current anneal default.
We take the count on these same six seeds, after `lj_requench`.
The default search is `Config::recommended`.
That is the `recommended` arm of `examples/h2h_seed.rs`.
Each size runs at its PAPER_BUDGETS entry.
The evidence rows list 400000, 1000000, and 4000000.
The search may stop at E_known + 0.001.
That stop is a search stop.
A panel hit is the re-quenched energy at or below E_known + 1e-6.
The four E_known values are the Cambridge Cluster Database minima.
The evidence rows list those four energies.
A count below its floor is a mechanical FAIL.

LJ38 floor: 6/6.
LJ55 floor: 6/6.
LJ75 floor: 6/6.
LJ98 floor: 6/6.

The four lines above are the re-quench floors from requench.tsv.
requench_panel.py writes them.
A hit is re-quenched E <= E_known + 1e-6.
Per-seed raw energy and re-quenched energy are in requench.md.
The hit flag and the re-quench calls are in that same note.
floors.md keeps the earlier 0.001 window count.
The window count is a separate tally from the floor.
A count below its floor is a mechanical FAIL.

## Matched compute

The budget is in energy calls.
It comes from PAPER_BUDGETS in scripts/elja_submit_jcc_lj.sh.
The evidence rows name the array bounds and each size budget.
LJ104 receives 4000000 energy calls.
The array entry is `[104]=4000000`.
The candidate, GMIN_BEST, and plain DLS each receive that budget.

The panel uses the same array.
LJ38 receives 400000 energy calls.
LJ55 receives 1000000 energy calls.
LJ75 receives 4000000 energy calls.
LJ98 receives 4000000 energy calls.

Every energy call is charged.
That includes each per-generation atom energy call.
One call is one unit of the budget.
The in-tree ledger prices a partial evaluation in `Ledger::charge_frac`.
The evidence rows name that routine.
It accumulates the partial payment.
A residue under one call stays uncharged.
One greedy pass pays `1.0 + (m * n) / pairs`.
The evidence rows name `Lattice::search` for that payment.
We charge each of those calls as a whole call in this comparison.

Wall time is recorded for a strictly serial run, one process at a time.
The runner takes an exclusive lock.
It refuses to start while another scored run keeps that lock.
The receipt is written at the start of the process.
It is written again at the end.
The receipt records the kernel boot_id and the uptime reading.
The receipt also records the lscpu listing.
The start receipt and the end receipt are both kept.
A run whose two boot_id values differ is a different boot.
We leave that run unused.

## Hit rule

We score every code in this comparison, GMIN_BEST included, one way.
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
The evidence rows list the LJ38, LJ55, LJ75, and LJ98 energies.
LJ104 is -582.086642.
The evidence rows name `examples/lj_cluster_search.rs` for that value.
Its hit line is E <= -582.086641.
The test re-quenches the known LJ104 minimum to the hit line.
The test name follows.
`lj_requench_brings_the_known_lj104_minimum_to_the_hit_line`.
The fixture is `tests/fixtures/lj104_gm.xyz`.
The test requires that hit line.

## Scored rows

The runner is lj104_sota_runner.py.
Every scored row cites the git blob id of that file.
The git commit identifier (HEAD) is the second citation.
The row also cites the tree of that commit.
The columns are runner_blob, head, and tree.
A row that omits any of the three is outside the scored set.
test_lj104_sota_runner.py checks the concurrent-start refusal.
A second scored start is refused.
The test also checks that a written row carries those three fields.
