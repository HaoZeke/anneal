# template-free2 gate, shard a

The rule below is fixed before any search in this shard.

## Arms

- A and A' are two separate clean release builds of develop
  `61fa67cf380e46876b569da45e7cc091647a775b`, library tree
  `cff796c507a8fa1d0bfb28888c3570b61988055e`.
- B is that commit plus the one-patch morphology-escape series.
  Its library tree is `14b8458d118153d7a1179325058f8d09aea2d65d`.
- The three drivers (`h2h_morse`, `h2h_sc`, `h2h_gmin_proof`) are
  built from the same example source. Only the linked library differs.
- Release build, `rustc 1.88.0`, `cargo build --release --locked`.

## Cells

Paper budgets: 4e5 at 38 atoms, 1e6 at 55, 4e6 at 75 and above.
References are the Cambridge values already compiled into the drivers.
No cell was invented.

| Cell | Driver | Budget | Reference | Where the definition lives |
| --- | --- | ---: | ---: | --- |
| m38r6 | h2h_morse | 400000 | -157.477108 | `examples/h2h_morse.rs`, `scripts/morse_gmin/run_panel.py` |
| m38r14 | h2h_morse | 400000 | -144.321054 | same, and the land4 gate harness |
| m55r6 | h2h_morse | 1000000 | -250.286609 | same, and the land4 gate harness |
| m55r14 | h2h_morse | 1000000 | -220.646208 | same, and the land4 gate harness |
| m80r6 | h2h_morse | 4000000 | -378.333471 | same, `scripts/paper_0120/run_panel.py`, land4 |
| m80r14 | h2h_morse | 4000000 | -340.811371 | same, and the land4 gate harness |
| sc126-38 | h2h_sc | 400000 | -35419.9804 | `examples/h2h_sc.rs`, `scripts/metal_h2h/run_sc.py` |
| sc126-55 | h2h_sc | 1000000 | -52884.6806 | same |
| lj38 | h2h_gmin_proof | 400000 | -173.928427 | `examples/h2h_gmin_proof.rs`, `scripts/paper_0120/run_panel.py` |
| lj55 | h2h_gmin_proof | 1000000 | -279.248470 | same, and the land4 gate harness |
| lj75 | h2h_gmin_proof | 4000000 | -397.492331 | `examples/h2h_gmin_proof.rs`, `scripts/paper_0120/run_panel.py` |
| lj98 | h2h_gmin_proof | 4000000 | -543.665361 | same |
| lj102 | h2h_gmin_proof | 4000000 | -569.363652 | same |
| lj103 | h2h_gmin_proof | 4000000 | -575.766131 | same |
| lj104 | h2h_gmin_proof | 4000000 | -582.086642 | same |
| lj150 | h2h_gmin_proof | 4000000 | -893.310258 | same |

No cell is missing. Every reference, budget, and driver is in the
land4 gate harness or in the committed drivers and paper panels above.

## Seeds

Every cell uses seeds 2000-2011 (n = 12).

That is the twelve-seed list in `scripts/paper_0120/run_panel.py`,
`scripts/morse_gmin/run_panel.py`, `scripts/metal_h2h/run_sc_n12.py`,
and the manuscript's GMIN and ASE tables. `scripts/morse_h2h/run_campaign.py`
and `scripts/metal_h2h/run_sc.py` also contain a six-seed list,
2000-2005. Those are a different panel. The claims under this gate
are counted out of 12, so the twelve-seed list is the one that applies.
No cell in this shard has a paper twelve-seed list other than 2000-2011.

## Protocol

- One runner, one session. Arms of a (cell, seed) pair run back to
  back, then the next seed. No other search shares the core.
- Order: for offset k = seed - 2000, rotate (A, A', B) left by
  (k // 2) mod 3, then reverse that order when k is odd. Each arm
  is in each slot 4 times. Odd seeds are the reverse of the previous
  even seed.
- `taskset -c 0`. Every listed thread variable is 1. No warm-up.
  `--quick` is refused. No `H2H_*` variable is set.
- Hit: finite best <= reference + 0.001. A non-finite best, a non-zero
  exit, or a timeout is a miss. The row keeps `calls_to_hit` at the
  budget. The tests below re-rank a miss as +infinity.
- `machine_id_sha256` is the SHA-256 of the platform machine id.
  `session` is a 16-hex digest minted at the first row and reused
  if the run is resumed. `sequence` is the execution index.
- Morphology activations: after the search returns, and after wall
  time and potential time are already stored, each driver sums
  `proposal_counts` for move types `twin` and `observed` and prints
  `morph_activations N` only when `MORPH_ACTIVATIONS` is set. The
  search does not read that variable. Unset, the added branch is
  skipped and the trajectory is the driver's ordinary trajectory.
  The same driver source is compiled into A, A', and B.

## Fail rule

B fails a cell if any of these fire. Otherwise the cell passes.
The shard fails if any cell fails.

(a) Hits are lower and the exact one-sided Fisher p-value is < 0.05.
The p-value is the hypergeometric probability of a B hit count less
than or equal to the observed count, with both margins fixed. A hit
drop whose p-value is not below 0.05 is reported and does not fail
the cell.

(b) Calls are worse on all three of the following. A miss is ranked
+infinity; a hit keeps `calls_to_hit`. Medians are Hyndman-Fan type 7.
+infinity/+infinity has ratio 1. A finite median over +infinity has
ratio 0. +infinity over a finite median has ratio +infinity.

- one-sided Mann-Whitney p < 0.05 for B larger than A, using the
  tie-corrected normal approximation and the continuity correction;
- the median ratio B/A is > 1;
- that ratio is strictly above the A/A' band. The band's upper edge
  is the maximum of 1 and the median ratio A'/A.

(c) Median per-row `outside_s/charged`, B/A, is > 1.10, while A'/A
is inside [0.95, 1.05]. Rows with `charged` = 0 are omitted from
this median. The median is Hyndman-Fan type 7.

## Author claims, unchecked until the rows exist

These are recorded here so the check is not chosen after the runs.
`summarize` writes the verdict into `summary.csv`.

- m38r14 hits go from 4/12 on A to 12/12 on B.
- The sc126-38 calls median gets smaller on B than on A, with misses
  ranked +infinity as in (b).
- m80r6 hits go from 7/12 on A to 9/12 on B.
- Nothing loses: no cell has fewer hits on B than on A, and no cell
  fails the rule above.

## Results

Shard verdict: FAIL. `m80r14` and `lj98` fail clause (c). No cell
fails (a) or (b). The only hit drop is `m55r14`, 9/12 to 8/12, with
one-sided Fisher p = 0.5.

A and A' reproduced the same hit bit and the same calls on every
seed, so the calls ratio A'/A is 1 on every cell. The outside A'/A
column is the timing band of two clean builds.

Median calls rank a miss as +infinity. Fisher p is the one-sided
probability that B has this few hits or fewer. Mann-Whitney p is
the one-sided probability that B's calls are larger. Activations
are the sum of `twin` and `observed` proposals on B.

| Cell | Hits A/A'/B | Fisher p | Median calls A/A'/B | MW p | Outside B/A | Outside A'/A | B activations | Verdict |
| --- | --- | ---: | --- | ---: | ---: | ---: | ---: | --- |
| m38r6 | 12/12/12 | 1 | 49438.5/49438.5/49438.5 | 0.5575 | 0.9649 | 0.9922 | 195 | PASS |
| m38r14 | 4/4/12 | 1 | inf/inf/138184 | 0.9873 | 0.9550 | 1.0190 | 3045 | PASS |
| m55r6 | 12/12/12 | 1 | 22596/22596/22596 | 0.5 | 1.0046 | 0.9887 | 501 | PASS |
| m55r14 | 9/9/8 | 0.5 | 451561.5/451561.5/506548 | 0.4534 | 1.0307 | 0.9592 | 16369 | PASS |
| m80r6 | 12/12/12 | 1 | 62677/62677/62677 | 0.5689 | 1.0016 | 0.9913 | 11490 | PASS |
| m80r14 | 6/6/11 | 0.9977 | inf/inf/2486485 | 0.9521 | 1.2373 | 0.9995 | 76867 | FAIL |
| sc126-38 | 12/12/12 | 1 | 113504.5/113504.5/58095 | 0.9080 | 1.0212 | 0.9947 | 2502 | PASS |
| sc126-55 | 12/12/12 | 1 | 18379.5/18379.5/18379.5 | 0.4769 | 0.9568 | 1.0697 | 292 | PASS |
| lj38 | 12/12/12 | 1 | 127839.5/127839.5/126456 | 0.7184 | 0.9886 | 0.9945 | 1917 | PASS |
| lj55 | 12/12/12 | 1 | 5722/5722/5722 | 0.5115 | 0.9910 | 1.0880 | 0 | PASS |
| lj75 | 5/5/7 | 0.8898 | inf/inf/3596765 | 0.6671 | 1.0714 | 0.9785 | 61865 | PASS |
| lj98 | 7/7/11 | 0.9931 | 3166811.5/3166811.5/2558250.5 | 0.8315 | 1.2027 | 1.0188 | 40305 | FAIL |
| lj102 | 6/6/7 | 0.7932 | inf/inf/2551083.5 | 0.7760 | 1.0968 | 1.0055 | 37587 | PASS |
| lj103 | 6/6/10 | 0.9864 | inf/inf/880801 | 0.8971 | 1.0373 | 0.9850 | 24716 | PASS |
| lj104 | 5/5/8 | 0.9502 | inf/inf/3118832.5 | 0.6534 | 1.0861 | 1.0009 | 44819 | PASS |
| lj150 | 8/8/9 | 0.8146 | 1129381.5/1129381.5/230913.5 | 0.6590 | 1.0316 | 0.9817 | 20028 | PASS |

`m80r14` outside B/A is 1.237 with A'/A at 0.9995. `lj98` outside
B/A is 1.203 with A'/A at 1.019. Both sit above 1.10 while the
clean-build band is inside [0.95, 1.05]. `lj102` reaches 1.097 and
stays under the line.

Claims, checked against these rows:

- m38r14 from 4/12 to 12/12: confirmed (A 4/12, A' 4/12, B 12/12).
- sc126-38 median gets better: confirmed (113504.5 to 58095).
- m80r6 from 7 to 9: not confirmed. This tree hits 12/12 on A and
  on B, so the stated 7/12 baseline is not what ran.
- Nothing loses: not confirmed. `m55r14` drops from 9/12 to 8/12
  (Fisher p = 0.5, so clause (a) does not fire) and `m80r14` and
  `lj98` fail clause (c).
