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
Copyright (c) anneal developers

Floors from the raw logs of the default panel.
A hit is lowest <= reference + 0.001.
runner_blob d9575213d875f6a43fd9ea7aa17dac63208acc3d.
head cc0587fa24b9bd72c4e0399b942bd3a5c3c8cff5.
tree b8f8a9a73a839235bf3d34a17ef9d27a5afed9fb.
boot_id b3fc2d5f-65ed-490e-ba1a-5f1ba891ad20.
wall_sum_seconds 272.291000.
span_sum_seconds 272.499802.
session_uptime_seconds 272.510000.

LJ38 floor: 6/6.
LJ38 seed 6000 hit energy -173.9274484071 calls 3158 wall_seconds 0.040.
LJ38 seed 6001 hit energy -173.9275632824 calls 25651.
wall_seconds 0.434.
LJ38 seed 6002 hit energy -173.9279904767 calls 1314 wall_seconds 0.009.
LJ38 seed 6003 hit energy -173.9280191945 calls 399 wall_seconds 0.002.
LJ38 seed 6004 hit energy -173.9282706652 calls 24072.
wall_seconds 0.438.
LJ38 seed 6005 hit energy -173.9283968304 calls 43871.
wall_seconds 0.796.

LJ55 floor: 6/6.
LJ55 seed 6000 hit energy -279.2475359835 calls 7219 wall_seconds 0.244.
LJ55 seed 6001 hit energy -279.2475972189 calls 15360.
wall_seconds 0.468.
LJ55 seed 6002 hit energy -279.2478492651 calls 25536.
wall_seconds 0.535.
LJ55 seed 6003 hit energy -279.2479422126 calls 7316 wall_seconds 0.151.
LJ55 seed 6004 hit energy -279.2481782004 calls 11580.
wall_seconds 0.366.
LJ55 seed 6005 hit energy -279.2480728953 calls 13770.
wall_seconds 0.526.

LJ75 floor: 6/6.
LJ75 seed 6000 hit energy -397.4917169053 calls 85509.
wall_seconds 6.286.
LJ75 seed 6001 hit energy -397.4916116384 calls 47598.
wall_seconds 2.763.
LJ75 seed 6002 hit energy -397.4916902116 calls 4822 wall_seconds 0.089.
LJ75 seed 6003 hit energy -397.4916758791 calls 14511.
wall_seconds 0.618.
LJ75 seed 6004 hit energy -397.4916003635 calls 48418.
wall_seconds 2.677.
LJ75 seed 6005 hit energy -397.4914148662 calls 39590.
wall_seconds 2.906.

LJ98 floor: 6/6.
LJ98 seed 6000 hit energy -543.6648665531 calls 228802.
wall_seconds 20.052.
LJ98 seed 6001 hit energy -543.6645044718 calls 707953.
wall_seconds 67.271.
LJ98 seed 6002 hit energy -543.6647250813 calls 233766.
wall_seconds 22.132.
LJ98 seed 6003 hit energy -543.6648616953 calls 554532.
wall_seconds 48.458.
LJ98 seed 6004 hit energy -543.6645439642 calls 53643.
wall_seconds 3.525.
LJ98 seed 6005 hit energy -543.6646655690 calls 1032010.
wall_seconds 91.505.
Copyright (c) anneal developers

Re-quenched panel.
A hit is requenched_e <= E_known + 1e-6.
requench_calls are the WarmLbfgs::minimize evaluations.
They are not search calls.
raw_e is the lowest line of the untouched search log.

LJ38 floor: 6/6.
E_known -173.928427.

seed raw_e requenched_e hit requench_calls.
6000 -173.9274484071 -173.9284265906289022 hit 35.
6001 -173.9275632824 -173.9284265906287033 hit 77.
6002 -173.9279904767 -173.9284265906287033 hit 23.
6003 -173.9280191945 -173.9284265906284759 hit 65.
6004 -173.9282706652 -173.9284265906287317 hit 99.
6005 -173.9283968304 -173.9284265906284190 hit 22.

LJ55 floor: 6/6.
E_known -279.248470.

seed raw_e requenched_e hit requench_calls.
6000 -279.2475359835 -279.2484704630212491 hit 42.
6001 -279.2475972189 -279.2484704630215333 hit 118.
6002 -279.2478492651 -279.2484704630214196 hit 82.
6003 -279.2479422126 -279.2484704630213059 hit 109.
6004 -279.2481782004 -279.2484704630215333 hit 33.
6005 -279.2480728953 -279.2484704630219312 hit 110.

LJ75 floor: 6/6.
E_known -397.492331.

seed raw_e requenched_e hit requench_calls.
6000 -397.4917169053 -397.4923309828757851 hit 164.
6001 -397.4916116384 -397.4923309828761830 hit 165.
6002 -397.4916902116 -397.4923309828768083 hit 160.
6003 -397.4916758791 -397.4923309828760125 hit 115.
6004 -397.4916003635 -397.4923309828762399 hit 110.
6005 -397.4914148662 -397.4923309828763536 hit 248.

LJ98 floor: 6/6.
E_known -543.665361.

seed raw_e requenched_e hit requench_calls.
6000 -543.6648665531 -543.6653607708954041 hit 163.
6001 -543.6645044718 -543.6653607708961999 hit 122.
6002 -543.6647250813 -543.6653607708951768 hit 160.
6003 -543.6648616953 -543.6653607708943809 hit 119.
6004 -543.6645439642 -543.6653607708940399 hit 157.
6005 -543.6646655690 -543.6653607708948357 hit 112.
Copyright (c) anneal developers

This replay is one git commit-tree pass.
There is no sleep.
GIT_AUTHOR_DATE is not set.
GIT_COMMITTER_DATE is not set.
The seconds are the ones git writes.
A commit may share its parent's second.
None is earlier than its parent.
Shared seconds are real.

These dates replace the archived author dates.
The archived history stays reachable.
refs/archive/prereg/lj104-sota-v1.3 follows.
835b90741211f2fd00b5b9ea68cde4cc877b31af.
refs/archive/prereg/lj104-sota-v1.3.1 follows.
6cde8551b2791cc33b1060311dd334a04d3d1042.

The clock zone is America/Chicago.
The offset git writes is -0600.
Author and committer are Rohit Goswami.
The email is rgoswami@ieee.org.
The commits are unsigned.

Each new commit uses the archived commit's tree.
The subject starts with lj104-prereg.
The receipt subject is at most 52 columns.
The other subjects are at most 72 columns.
replay-map.tsv is the old-to-new map.
TREE_EQUAL.txt records that each pair of trees is the same object.
replay_commit_tree.py is the pass.
replay-commit-tree.log is the raw log of that pass.

The predictions are archived blobs.
The McNemar table and the seed sets are archived blobs.
The thresholds are archived blobs too.
This replay does not edit them.
Copyright (c) anneal developers

This stat is against the v1.2 tree.
That tree follows.
4b168122d30bc00afa0e27a26c67863ffc7b312c.
This note is left out of the count.

Path docs/prereg/lj104-sota/PREDICTIONS.md.
211 insertions and 35 deletions.
Path docs/prereg/lj104-sota/REPLAY.txt.
37 insertions and 0 deletions.
Path docs/prereg/lj104-sota/SCOPE.md.
223 insertions and 0 deletions.
Path docs/prereg/lj104-sota/TREE_EQUAL.txt.
82 insertions and 0 deletions.
Path docs/prereg/lj104-sota/floors.md.
51 insertions and 32 deletions.
Path docs/prereg/lj104-sota/gmin_best.keywords.
12 insertions and 0 deletions.
Path docs/prereg/lj104-sota/prose/proseguard-prose-human.txt.
6 insertions and 0 deletions.
Path docs/prereg/lj104-sota/prose/proseguard-rohit-paper.txt.
6 insertions and 0 deletions.
Path docs/prereg/lj104-sota/prose/snapper-fmt.txt.
6 insertions and 0 deletions.
Path docs/prereg/lj104-sota/prose/snapper.txt.
6 insertions and 0 deletions.
Path docs/prereg/lj104-sota/replay-commit-tree.log.
327 insertions and 0 deletions.
Path docs/prereg/lj104-sota/replay-map.tsv.
16 insertions and 0 deletions.
Path docs/prereg/lj104-sota/replay_commit_tree.py.
279 insertions and 0 deletions.
Path docs/prereg/lj104-sota/requench.md.
47 insertions and 41 deletions.
Path docs/prereg/lj104-sota/requench_panel.py.
0 insertions and 1 deletions.
Path docs/prereg/lj104-sota/scope_paths.py.
96 insertions and 0 deletions.
Copyright (c) anneal developers

Each replayed commit uses the archived commit's tree.
equal is yes when the two tree ids are the same object.
Shared seconds in the clock are real.
These dates replace the archived author dates.

The old commit is 4082f82f75c2a79f0e0a007856c21a1a0422e58f.
The new commit is b519432afb9cd316687ae11088601b4799816175.
The tree is 31925e50d57ee6e1cd4d733adf5bad02eadf6a13.
equal yes.

The old commit is 33532452f0d768255aeed36eb7d7fdfc99530716.
The new commit is 88efd07f80b6476c6cfb31b711fbbfca8aceee45.
The tree is 53722e6405b930b174415d60c23c0717b33ca7e1.
equal yes.

The old commit is 44b64369a209221e599a27f0808e5a0d63e64044.
The new commit is 005bd6a340d1396f436adf90d770de958278f020.
The tree is c1b79198488fa3234bf3cf50d2c22d9ce710d4cd.
equal yes.

The old commit is 218defbd87ddf38db9015c9799f9f7d600a3c369.
The new commit is fa8c75a6b66c36fdbb02a6c5cea06185f96782ed.
The tree is 7bef65dc8612e9e0454c6c565422d1a5e8d20a81.
equal yes.

The old commit is 4391ceb7c8b24ce3c38cb5ba371db8d75d835277.
The new commit is 5a4c00306a9fbf1cd3b6de6781b19ae78f7fa9d7.
The tree is 927a0cc7b3db6525f8bb7f4255d6ce9fa03ce08c.
equal yes.

The old commit is 75e2a125287f9ea76055d20b4a4254ef2a6b9a53.
The new commit is 24583692a005eec3527cd20a4cc1e22ecc1d1e2c.
The tree is 034b4ae9d4b7f4a56a1e29d68b885a712558998d.
equal yes.

The old commit is b488fef50c97e23183c349311537267eb02f300a.
The new commit is 8ad608bb6f2a717b2e860f694a3e7c39567af76e.
The tree is b8f8a9a73a839235bf3d34a17ef9d27a5afed9fb.
equal yes.

The old commit is 4c9b32008ff0f4e3851a89aedf036b47f48f9fb2.
The new commit is b185455047ab8978e97d887a3d290860929ff29f.
The tree is beac7bacf893c5af2cde198237aed8448e1a44a7.
equal yes.

The old commit is e8426dc301a6ec81ac4eca129496fcadb625770f.
The new commit is f13ea87479e75a408e1cc8167a1b77ab7a055bfa.
The tree is bebd8a1e882787bb9c7741e763b6160155d85411.
equal yes.

The old commit is 66bf4acac19dcdccf9e1e5989e03e863b5704106.
The new commit is adf7d96c4a861b66b1e019d93da51e7248464dab.
The tree is ef739091ff6e6e8dd49f159314a9af41f550ca46.
equal yes.

The old commit is 26240c31db7b8de658081396b4407b1efb5903c3.
The new commit is 43e517387af2812644fa517182372826af6fb42e.
The tree is d882caf066b85850023036d51c8c8edfc221bcfe.
equal yes.

The old commit is bb5c667a5a6774e7d5073437f78bf778a394f580.
The new commit is adaea980d3e5e59823b76880db9e189b3bdd849b.
The tree is 4b168122d30bc00afa0e27a26c67863ffc7b312c.
equal yes.

The old commit is 2a3e8f2d8115156cbb2db88785124d69aafb4f95.
The new commit is c39f2e58deca7be3f7e0e1cf4bd59056c09a50fb.
The tree is 279a3721f5203bf624fc5efd41b478b3a1b5e136.
equal yes.

The old commit is 0bfcb3727f7b78aa0ac221278164d5c97f1188a4.
The new commit is 9556c0b115063606a7b4ba3bfdf28a83f21b3dca.
The tree is b5d20591f6ce08f7878cfee614b022127779df7d.
equal yes.

The old commit is 6cde8551b2791cc33b1060311dd334a04d3d1042.
The new commit is 846c4d6ec4c7178b690f08a145175a4b881cc697.
The tree is f300067d17a971b7466825aaa54bbc4fd324064a.
equal yes.

Copyright (c) anneal developers

The only ref this record publishes is refs/heads/prereg/lj104-sota-v1.
Archive refs are never pushed.
This commit does not push.
Copyright (c) anneal developers

Scope of the twelve commits after 91b15a02 through 5fb7368d.
Subjects and paths are taken from git.
An added path is a diff-tree addition.
A modified path is a diff-tree modification.

Commit fdc321918537e44ea452b887ee4d818289e95884.
Subject follows.
Register the LJ104 held-out predictions, the one-sided McNemar.
thresholds, and the unfilled six-seed floors..
Adds 3 paths.
Path docs/prereg/lj104-sota/PREDICTIONS.md.
Path docs/prereg/lj104-sota/mcnemar_threshold.py.
Path docs/prereg/lj104-sota/mcnemar_threshold.tsv.

Commit 8004d790392e4f8614cc0e9e64a1aead7d0f3ce7.
Subject follows.
Add the serial LJ104 runner and tests that it refuses a second scored.
start, writes its blob, HEAD, and tree, and leaves seeds 5000-5029.
unrun..
Adds 2 paths.
Path docs/prereg/lj104-sota/lj104_sota_runner.py.
Path docs/prereg/lj104-sota/test_lj104_sota_runner.py.

Commit c9bac09f0d3328f35e767d4c5059eebe737f1729.
Subject follows.
Name the no-regression panel seeds 6000-6005 and record that they were.
not used for tuning..
Adds no path.
Modifies 1 path.
Path docs/prereg/lj104-sota/PREDICTIONS.md.

Commit 45139a21e18d0c911c7c60149d62e4c87e5428e0.
Subject follows.
Bind lj_requench to WarmLbfgs::minimize and test it on the known LJ104.
minimum..
Adds no path.
Modifies 2 paths.
Path docs/prereg/lj104-sota/PREDICTIONS.md.
Path src/methods/lattice_search.rs.

Commit 54818da1e54c6dc4e3bfce7a8c6a129f2af289d4.
Subject follows.
Record the plain DLS prediction of 20 of 30 from the lattice v2.
recompute on seeds 3000-3029..
Adds no path.
Modifies 1 path.
Path docs/prereg/lj104-sota/PREDICTIONS.md.

Commit d77d8eb557e4288c9eeee1ecd553e4de2525599a.
Subject follows.
Define the default panel as recommended at the paper budgets, and teach.
the serial runner to execute those 24 seeds one at a time..
Adds no path.
Modifies 3 paths.
Path docs/prereg/lj104-sota/PREDICTIONS.md.
Path docs/prereg/lj104-sota/lj104_sota_runner.py.
Path docs/prereg/lj104-sota/test_lj104_sota_runner.py.

Commit cc0587fa24b9bd72c4e0399b942bd3a5c3c8cff5.
Subject follows.
Restore the LJ104 occupation test attribute that the requench test.
displaced..
Adds no path.
Modifies 1 path.
Path src/methods/lattice_search.rs.

Commit 380ceedd7fcd3db9bd0184fd987216822de684d3.
Subject follows.
Commit the untouched default-panel logs, geometries, traces, spans,.
scored rows, and the start and end receipts..
Adds 100 paths.
Path docs/prereg/lj104-sota/panel/01-lj38-6000.span.
Path docs/prereg/lj104-sota/panel/01-lj38-6000.stderr.
Path docs/prereg/lj104-sota/panel/01-lj38-6000.trace.csv.
Path docs/prereg/lj104-sota/panel/01-lj38-6000.xyz.
Path docs/prereg/lj104-sota/panel/02-lj38-6001.span.
Path docs/prereg/lj104-sota/panel/02-lj38-6001.stderr.
Path docs/prereg/lj104-sota/panel/02-lj38-6001.trace.csv.
Path docs/prereg/lj104-sota/panel/02-lj38-6001.xyz.
Path docs/prereg/lj104-sota/panel/03-lj38-6002.span.
Path docs/prereg/lj104-sota/panel/03-lj38-6002.stderr.
Path docs/prereg/lj104-sota/panel/03-lj38-6002.trace.csv.
Path docs/prereg/lj104-sota/panel/03-lj38-6002.xyz.
Path docs/prereg/lj104-sota/panel/04-lj38-6003.span.
Path docs/prereg/lj104-sota/panel/04-lj38-6003.stderr.
Path docs/prereg/lj104-sota/panel/04-lj38-6003.trace.csv.
Path docs/prereg/lj104-sota/panel/04-lj38-6003.xyz.
Path docs/prereg/lj104-sota/panel/05-lj38-6004.span.
Path docs/prereg/lj104-sota/panel/05-lj38-6004.stderr.
Path docs/prereg/lj104-sota/panel/05-lj38-6004.trace.csv.
Path docs/prereg/lj104-sota/panel/05-lj38-6004.xyz.
Path docs/prereg/lj104-sota/panel/06-lj38-6005.span.
Path docs/prereg/lj104-sota/panel/06-lj38-6005.stderr.
Path docs/prereg/lj104-sota/panel/06-lj38-6005.trace.csv.
Path docs/prereg/lj104-sota/panel/06-lj38-6005.xyz.
Path docs/prereg/lj104-sota/panel/07-lj55-6000.span.
Path docs/prereg/lj104-sota/panel/07-lj55-6000.stderr.
Path docs/prereg/lj104-sota/panel/07-lj55-6000.trace.csv.
Path docs/prereg/lj104-sota/panel/07-lj55-6000.xyz.
Path docs/prereg/lj104-sota/panel/08-lj55-6001.span.
Path docs/prereg/lj104-sota/panel/08-lj55-6001.stderr.
Path docs/prereg/lj104-sota/panel/08-lj55-6001.trace.csv.
Path docs/prereg/lj104-sota/panel/08-lj55-6001.xyz.
Path docs/prereg/lj104-sota/panel/09-lj55-6002.span.
Path docs/prereg/lj104-sota/panel/09-lj55-6002.stderr.
Path docs/prereg/lj104-sota/panel/09-lj55-6002.trace.csv.
Path docs/prereg/lj104-sota/panel/09-lj55-6002.xyz.
Path docs/prereg/lj104-sota/panel/10-lj55-6003.span.
Path docs/prereg/lj104-sota/panel/10-lj55-6003.stderr.
Path docs/prereg/lj104-sota/panel/10-lj55-6003.trace.csv.
Path docs/prereg/lj104-sota/panel/10-lj55-6003.xyz.
Path docs/prereg/lj104-sota/panel/11-lj55-6004.span.
Path docs/prereg/lj104-sota/panel/11-lj55-6004.stderr.
Path docs/prereg/lj104-sota/panel/11-lj55-6004.trace.csv.
Path docs/prereg/lj104-sota/panel/11-lj55-6004.xyz.
Path docs/prereg/lj104-sota/panel/12-lj55-6005.span.
Path docs/prereg/lj104-sota/panel/12-lj55-6005.stderr.
Path docs/prereg/lj104-sota/panel/12-lj55-6005.trace.csv.
Path docs/prereg/lj104-sota/panel/12-lj55-6005.xyz.
Path docs/prereg/lj104-sota/panel/13-lj75-6000.span.
Path docs/prereg/lj104-sota/panel/13-lj75-6000.stderr.
Path docs/prereg/lj104-sota/panel/13-lj75-6000.trace.csv.
Path docs/prereg/lj104-sota/panel/13-lj75-6000.xyz.
Path docs/prereg/lj104-sota/panel/14-lj75-6001.span.
Path docs/prereg/lj104-sota/panel/14-lj75-6001.stderr.
Path docs/prereg/lj104-sota/panel/14-lj75-6001.trace.csv.
Path docs/prereg/lj104-sota/panel/14-lj75-6001.xyz.
Path docs/prereg/lj104-sota/panel/15-lj75-6002.span.
Path docs/prereg/lj104-sota/panel/15-lj75-6002.stderr.
Path docs/prereg/lj104-sota/panel/15-lj75-6002.trace.csv.
Path docs/prereg/lj104-sota/panel/15-lj75-6002.xyz.
Path docs/prereg/lj104-sota/panel/16-lj75-6003.span.
Path docs/prereg/lj104-sota/panel/16-lj75-6003.stderr.
Path docs/prereg/lj104-sota/panel/16-lj75-6003.trace.csv.
Path docs/prereg/lj104-sota/panel/16-lj75-6003.xyz.
Path docs/prereg/lj104-sota/panel/17-lj75-6004.span.
Path docs/prereg/lj104-sota/panel/17-lj75-6004.stderr.
Path docs/prereg/lj104-sota/panel/17-lj75-6004.trace.csv.
Path docs/prereg/lj104-sota/panel/17-lj75-6004.xyz.
Path docs/prereg/lj104-sota/panel/18-lj75-6005.span.
Path docs/prereg/lj104-sota/panel/18-lj75-6005.stderr.
Path docs/prereg/lj104-sota/panel/18-lj75-6005.trace.csv.
Path docs/prereg/lj104-sota/panel/18-lj75-6005.xyz.
Path docs/prereg/lj104-sota/panel/19-lj98-6000.span.
Path docs/prereg/lj104-sota/panel/19-lj98-6000.stderr.
Path docs/prereg/lj104-sota/panel/19-lj98-6000.trace.csv.
Path docs/prereg/lj104-sota/panel/19-lj98-6000.xyz.
Path docs/prereg/lj104-sota/panel/20-lj98-6001.span.
Path docs/prereg/lj104-sota/panel/20-lj98-6001.stderr.
Path docs/prereg/lj104-sota/panel/20-lj98-6001.trace.csv.
Path docs/prereg/lj104-sota/panel/20-lj98-6001.xyz.
Path docs/prereg/lj104-sota/panel/21-lj98-6002.span.
Path docs/prereg/lj104-sota/panel/21-lj98-6002.stderr.
Path docs/prereg/lj104-sota/panel/21-lj98-6002.trace.csv.
Path docs/prereg/lj104-sota/panel/21-lj98-6002.xyz.
Path docs/prereg/lj104-sota/panel/22-lj98-6003.span.
Path docs/prereg/lj104-sota/panel/22-lj98-6003.stderr.
Path docs/prereg/lj104-sota/panel/22-lj98-6003.trace.csv.
Path docs/prereg/lj104-sota/panel/22-lj98-6003.xyz.
Path docs/prereg/lj104-sota/panel/23-lj98-6004.span.
Path docs/prereg/lj104-sota/panel/23-lj98-6004.stderr.
Path docs/prereg/lj104-sota/panel/23-lj98-6004.trace.csv.
Path docs/prereg/lj104-sota/panel/23-lj98-6004.xyz.
Path docs/prereg/lj104-sota/panel/24-lj98-6005.span.
Path docs/prereg/lj104-sota/panel/24-lj98-6005.stderr.
Path docs/prereg/lj104-sota/panel/24-lj98-6005.trace.csv.
Path docs/prereg/lj104-sota/panel/24-lj98-6005.xyz.
Path docs/prereg/lj104-sota/panel/end.receipt.
Path docs/prereg/lj104-sota/panel/rows.tsv.
Path docs/prereg/lj104-sota/panel/session.txt.
Path docs/prereg/lj104-sota/panel/start.receipt.

Commit 3c1069acdc90306feda50f770489d892bd8ade85.
Subject follows.
Add the untouched default-panel stdout logs that the ignore rule had.
left out..
Adds 24 paths.
Path docs/prereg/lj104-sota/panel/01-lj38-6000.log.
Path docs/prereg/lj104-sota/panel/02-lj38-6001.log.
Path docs/prereg/lj104-sota/panel/03-lj38-6002.log.
Path docs/prereg/lj104-sota/panel/04-lj38-6003.log.
Path docs/prereg/lj104-sota/panel/05-lj38-6004.log.
Path docs/prereg/lj104-sota/panel/06-lj38-6005.log.
Path docs/prereg/lj104-sota/panel/07-lj55-6000.log.
Path docs/prereg/lj104-sota/panel/08-lj55-6001.log.
Path docs/prereg/lj104-sota/panel/09-lj55-6002.log.
Path docs/prereg/lj104-sota/panel/10-lj55-6003.log.
Path docs/prereg/lj104-sota/panel/11-lj55-6004.log.
Path docs/prereg/lj104-sota/panel/12-lj55-6005.log.
Path docs/prereg/lj104-sota/panel/13-lj75-6000.log.
Path docs/prereg/lj104-sota/panel/14-lj75-6001.log.
Path docs/prereg/lj104-sota/panel/15-lj75-6002.log.
Path docs/prereg/lj104-sota/panel/16-lj75-6003.log.
Path docs/prereg/lj104-sota/panel/17-lj75-6004.log.
Path docs/prereg/lj104-sota/panel/18-lj75-6005.log.
Path docs/prereg/lj104-sota/panel/19-lj98-6000.log.
Path docs/prereg/lj104-sota/panel/20-lj98-6001.log.
Path docs/prereg/lj104-sota/panel/21-lj98-6002.log.
Path docs/prereg/lj104-sota/panel/22-lj98-6003.log.
Path docs/prereg/lj104-sota/panel/23-lj98-6004.log.
Path docs/prereg/lj104-sota/panel/24-lj98-6005.log.

Commit 9c072e5faa7f28d815b917c3d330f91190480829.
Subject follows.
Write the four default-panel floors from the raw logs, with the.
per-seed outcomes beside them..
Adds 2 paths.
Path docs/prereg/lj104-sota/floors.md.
Path docs/prereg/lj104-sota/write_floors.py.
Modifies 1 path.
Path docs/prereg/lj104-sota/PREDICTIONS.md.

Commit 6a813084d36962c1d61154eeafd1ea377d150390.
Subject follows.
Score every code, GMIN included, by lj_requench at E_known plus 1e-6,.
and count those calls apart from the search..
Adds 3 paths.
Path docs/prereg/lj104-sota/e_known.tsv.
Path docs/prereg/lj104-sota/requench_panel.py.
Path examples/lj_requench_xyz.rs.
Modifies 2 paths.
Path docs/prereg/lj104-sota/PREDICTIONS.md.
Path src/methods/lattice_search.rs.

Commit 5fb7368d5ea15cdf517c8bc48e2f56286ccb8b33.
Subject follows.
Re-quench the 24 saved structures and set the floors from those.
energies and their separate call counts..
Adds 2 paths.
Path docs/prereg/lj104-sota/requench.md.
Path docs/prereg/lj104-sota/requench.tsv.
Modifies 2 paths.
Path docs/prereg/lj104-sota/PREDICTIONS.md.
Path docs/prereg/lj104-sota/requench_panel.py.
