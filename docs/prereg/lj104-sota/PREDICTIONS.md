Copyright (c) anneal developers

# LJ104 held-out comparison

This note registers the LJ104 comparison before any scored run of seeds 5000-5029. Those seeds stay unrun in this commit. The runner beside this note refuses them.

## Codes

The candidate is anneal, running arm B v2's preset together with arm D v3's faithful dynamic-lattice search. Arm B v2 is the GMIN-style symmetrisation and the surface moves. The exact preset id is UNFILLED. It is filled only from arm B's serial angular on/off grid, in a later commit, and that commit is made before any run of seeds 5000-5029.

The arm D v3 tip sha is UNFILLED. It is filled in that same later commit, still before any of those seeds.

GMIN_BEST is the patched GMIN binary whose sha256 begins b8eeddfa6d18a697 and which prints h2h_row. On seeds 3000-3029 that binary scored 19 of 30. This commit does not remeasure that block. The prediction below carries the count across.

Plain DLS is the in-tree dynamic lattice search, the dls reader in src/methods/lattice_search.rs. It is a separate column from arm D v3.

## Seeds

The held-out seeds are 5000 through 5029, thirty seeds. They were not used to tune the candidate, GMIN_BEST, or plain DLS. paper/seeds.toml lines 65-71 register 5000-5063 for a different comparison, soap on against off, on an icosahedral 38-mer and on 12 Lennard-Jones sites. That block is not a tuning set for the three codes here.

## Predicted hit rates

A hit is the LJ104 rule below. Each prediction is a count out of these thirty seeds.

GMIN_BEST is predicted to hit 19 of 30. The binary is the one already scored at 19 of 30 on seeds 3000-3029, and the held-out prediction is that same count.

The candidate is predicted to hit 19 of 30. No combined run of the unfilled preset and the unfilled tip exists, so there is no pilot count to scale. The registered prediction is parity with the GMIN_BEST carry-forward. Parity is reported as parity. The number stays as it is when the preset id and the tip sha are filled.

Plain DLS has no committed LJ104 hit count at the paper budget under the charging rule in this note. Its predicted hit rate is UNFILLED. Seeds 5000-5029 do not fill it.

## Win rule

The decision between the candidate and GMIN_BEST is a one-sided exact McNemar test on the paired outcomes of the same thirty seeds. A seed is discordant when exactly one of the two codes hits. n is the number of discordant seeds. b is the number of those seeds that only the candidate hits. Under the null, b is binomial with n trials and probability 1/2. The p-value is the probability of b or more. Alpha is 0.05, which is the fraction 1/20. The candidate wins when that exact tail is at most 1/20. The integer form used by the script is: 20 times the upper tail, counted in successes, is at most 2 to the n.

The threshold in hits is the smallest such b at the observed n. mcnemar_threshold.py prints that minimum for every discordant count from 0 through 30, with the exact tail and the tail to eight decimal places. mcnemar_threshold.tsv is the script's stdout and is the table of record. The decimal column is a rendering. The decision uses the exact tail.

Below 5 discordant seeds, no split wins. At 5 discordant seeds the candidate wins only with b = 5, and the exact tail is 1/32, which is 0.03125. At 30 discordant seeds the minimum is b = 20, and the exact tail is 26504551/536870912, which renders as 0.04936857.

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

A tie is reported as a tie. Parity, the case b = n - b, is the same event as equal hit counts, and it is reported as parity. Neither is reported as a win. A candidate lead whose b is short of the minimum in the table is reported as that lead, with its p-value, and it is not a win. The test awards a win only to the candidate, and only at the threshold above. The point predictions are not a substitute for the paired test.

## No-regression panel

The panel is LJ38, LJ55, LJ75, and LJ98, with n = 6 seeds at each size. The six integers are outside 5000-5029. No committed file names the current default's six-seed panel, so the six integers are UNFILLED.

The floor at each size is the current anneal default hit count on that panel. A count below its floor is a mechanical FAIL.

LJ38 floor: UNFILLED.
LJ55 floor: UNFILLED.
LJ75 floor: UNFILLED.
LJ98 floor: UNFILLED.

No committed file states a six-seed default hit count for these sizes. Counts that are committed use other sample sizes, and this note does not convert them into floors. Those records are src/methods/cluster_hopping/preset.rs lines 20-24, docs/orgmode/architecture.org lines 191-193, and scripts/elja_sota_guard.sh lines 3-8. While a floor is UNFILLED, that size cannot be passed.

## Matched compute

The budget is in energy calls, from PAPER_BUDGETS in scripts/elja_submit_jcc_lj.sh. The array is lines 15-22. LJ104 is line 21, `[104]=4000000`. The candidate, GMIN_BEST, and plain DLS each receive 4000000 energy calls.

The panel uses the same array. LJ38 is line 16, 400000. LJ55 is line 17, 1000000. LJ75 is line 18, 4000000. LJ98 is line 19, 4000000.

Every energy call is charged, including each per-generation atom energy call. One call is one unit of the budget. The in-tree ledger prices a partial evaluation as a fraction of a call: `Ledger::charge_frac` in src/methods/cluster_hopping.rs at line 720 accumulates that fraction and leaves a residue under one call uncharged. `Lattice::search` in src/methods/lattice_search.rs at line 1363 pays `1.0 + (m * n) / pairs` for one greedy pass. This comparison charges each of those calls as a whole call.

Wall time is recorded for a strictly serial run, one process at a time. The runner takes an exclusive lock and refuses to start while another scored run holds it. At the start of the process, and again at the end, the receipt holds the kernel boot_id, the uptime reading, and the lscpu listing. The start receipt and the end receipt are both kept. A run whose two boot_id values differ is not one boot, and it is not used.

## Hit rule

Every final structure from the candidate, from GMIN_BEST, and from plain DLS is re-quenched with the same lj_requench. A hit means E <= -582.086641. No symbol named lj_requench is committed in this tree. The binding of that name to an implementation is UNFILLED. The inequality is the hit rule.

## Scored rows

The runner is lj104_sota_runner.py. Every scored row cites the git blob id of that file, HEAD, and the tree of HEAD. The columns are runner_blob, head, and tree. A row that omits any of the three is not a scored row. test_lj104_sota_runner.py checks that a second concurrent start is refused, and that a written row carries those three fields.
