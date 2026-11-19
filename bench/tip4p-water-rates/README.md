Copyright (c) anneal developers

TIP4P success rates for (H2O)10, 15 and 20 on the paper preset, seeds 0-5,
written by `scripts/tip4p_rates/enter_bench.sh`. The runner builds
`h2h_seed` from the checked-out tree, runs every seed serially under
`unshare --uts` and refuses a log whose hostname is not bench. A hit is a
best energy within 0.001 kcal/mol (0.004184 kJ/mol) of the Wales-Hodges
reference.

- `rates.csv`: the rows of b47d442a, after rigid TIP4P left the atomic
  stall controls. 15 molecules run at 5e5 calls and 20 at 2e6.
- `rates-a24ba760.csv`: the rows of a24ba760, every size at 5e5 calls,
  with the stall controls still on. b47d442a overwrote them in
  `rates.csv`; this file is that blob unchanged. The TIP4P subsection of
  `paper/jctc/manuscript.tex` reports these rows, and its cross-field
  section and conclusions compare them with `rates.csv`.

Neither file has a `code_revision` or `host` column. The hostname is bench
by the runner's check; the rows name no revision for the tree that the runner built.
