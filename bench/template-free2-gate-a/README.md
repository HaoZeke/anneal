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
