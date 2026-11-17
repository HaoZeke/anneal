# Gate: `branch_on_stall` against develop

`Config::branch_on_stall` (the branch build run with `H2H_BRANCH=1`) against
develop `c2de652dca0bc75e2b0cb4afffac91fd656404e4` on the paper panel, seeds
2000-2011, at the paper budgets: 1,000,000 calls for Au55 and 4,000,000 for
the rest. One session: `snakemake -s workflow/Snakefile branch_gate`, which
builds both revisions in detached worktrees (Rust 1.88.0, `--release
--locked`; LJ103 from a second build of each with
`bench/paper-0120/lj103-table.patch`) and runs `scripts/branch_gate/run_gate.py`
serially on core 0 under `sudo unshare --uts` with hostname `bench`, develop
then branch for each system and seed. The branch rows were made from the
tree `aa8b6b8514654e249465add1e85b021547e1ac70` (`code_tree`), whose
`code_revision` is a local commit; the tree is the one `git am` of this
series gives at that commit.

- `rows.csv`: one row per run, appended as each run finished, with
  `code_revision`, `code_tree`, `host` and `cpu`. `pot_s` is the time inside
  the potential and `outside_s` the rest of the wall.
- `compare.csv`: per system, Fisher's two-sided exact test on hits; the
  two-sided Mann-Whitney test (normal approximation, tie and continuity
  corrections) on calls to the minimum, a miss entered at the budget, and on
  inside and outside time; `same_seeds` counts seeds where both arms gave
  the same hit, calls and energy. Medians are Hyndman-Fan type 7.

| system | branch hits | develop hits | Fisher p | median calls branch | median calls develop | Mann-Whitney p |
| --- | --- | --- | --- | --- | --- | --- |
| LJ38 | 12/12 | 12/12 | 1 | 16,341 | 10,615.5 | 0.015 |
| LJ55 | 12/12 | 12/12 | 1 | 13,582 | 12,242 | 1 |
| LJ75 | 11/12 | 12/12 | 1 | 1,333,849.5 | 1,409,229.5 | 0.89 |
| LJ98 | 12/12 | 10/12 | 0.48 | 1,220,314 | 1,511,030.5 | 0.26 |
| LJ102 | 12/12 | 9/12 | 0.22 | 1,044,477.5 | 894,801 | 0.93 |
| LJ103 | 9/12 | 8/12 | 1 | 2,219,931 | 2,293,133 | 0.98 |
| LJ104 | 8/12 | 9/12 | 1 | 2,420,208.5 | 2,328,723 | 0.64 |
| LJ150 | 11/12 | 5/12 | 0.027 | 843,181.5 | 4,000,000 | 0.0088 |
| Au55 | 5/12 | 6/12 | 1 | 1,000,000 | 852,736 | 0.36 |
| M80 rho 6 | 3/12 | 2/12 | 1 | 4,000,000 | 4,000,000 | 0.60 |
| M80 rho 14 | 8/12 | 9/12 | 1 | 2,368,625 | 1,338,932.5 | 0.50 |

Totals: branch 103/132, develop 94/132. Time inside the potential, median
seconds, branch against develop: LJ150 31.3 against 138.3 (p = 0.010), Au55
33.1 against 23.5 (p = 0.040), LJ38 0.039 against 0.027 (p = 0.014); no
other system differs at p < 0.05. Outside time follows the same pattern
(LJ150 23.3 against 99.5 s, p = 0.0086).

Seeds that change: the branch gains LJ98 2000 and 2001, LJ102 2001, 2006
and 2010, LJ103 2001, LJ104 2010, LJ150 2001, 2003, 2005, 2008, 2010 and
2011, and M80 rho 6 2000; it loses LJ75 2003, LJ104 2006 and 2009, Au55 2010
and M80 rho 14 2009. LJ150 2004 is missed by both. Every lost seed is one
develop hits late: 705,472 of 1,000,000 calls on Au55, 2,832,701 to
3,826,871 of 4,000,000 elsewhere. The walk never sees a branch, so it
reaches each minimum later than develop by the calls the branches spent
before it, and here that pushes those hits past the budget. LJ38 hits on every seed in both arms but takes more calls
with the branch on. It loses on four systems, so `branch_on_stall` stays
off by default and is an opt-in record.

The trigger, the stop rules and the choice to defer a branch's best until it
beats the walk were picked while looking at these seeds (2000-2011) on
these systems, so the LJ150 margin here is not a held-out estimate.
