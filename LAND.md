# LAND: anneal PR 171 restack gate

- Target: HaoZeke/anneal PR #171
- Lease: 5fe4bbbc72f1727141e474840da4fe453a316f60 (tree ffce2c61)
- Base A: develop 9b65fda3972fd8c014bce886a3ea5c05956d64af (tree a49a66d1)
- Gated B: tree f172df01104c53c74daf6c74f63bf08c45d19b06 (c961b86f,
  run as squash 5a8fc333 on A)
- Gate tree: 8e0d526b28f12d5a6146e3db73f5034c12ade0a2 (4 commits on A)
- Builder: told to Inky by message
- Measurer: told to Inky by message

## Why f172df01 covers the lease

c961b86f..5fe4bbbc is two commits.  The only change outside
`mod tests` is in `candidate_from_basis` (src/lattice.rs): the cap
becomes `frames.min(max_sites)` for every n, where it used to apply
only when C(n, 4) >= n.  That condition fails only for n <= 4.  Every
gated cell has n >= 6, so the lease runs the same production code as
the gated tree.

## Verdict: PASS

Box recompute from rows.csv (288 rows, all exit 0).  Ratios are
seed-paired geometric means with 10000 bootstrap draws at seed
20261123.  A real regression needs the B/A lower bound above 1 and
above the A2/A upper bound.

| cell | hits A/A2/B/B2 | calls B/A | outside B/A | outside A2/A | RSS B/A |
| --- | --- | ---: | --- | --- | --- |
| m38r14d | 8/8/12/12 | 1.000 | 0.968 (0.947, 0.987) | 1.022 (0.979, 1.069) | 1.000 |
| lj38 | 12/12/12/12 * | 1.001 | 0.988 (0.966, 1.006) | 0.981 (0.951, 1.006) | 1.000 |
| lj75 | 5/5/5/5 | 1.000 | 1.020 (0.975, 1.079) | 0.988 (0.954, 1.023) | 1.001 (1.000, 1.001) |
| lj98 | 7/7/7/7 | 1.002 | 1.013 (1.003, 1.024) | 1.003 (0.992, 1.017) | 1.002 (1.001, 1.003) |
| tip4p6 | 24/24/24/24 | 1.000 | 0.900 (0.817, 0.977) | 0.946 (0.830, 1.081) | 1.000 |

All B/A outside upper bounds are at most 1.10 and at most
max(1.10, A2/A upper).  No RSS lower bound exceeds 1.10.

Notes:

- m38r14d: the 8 to 12 hit gain is the closed-quench answer fix.
  Calls and `seen.lowest` are identical per seed.  On seeds 2001,
  2004, 2008 and 2010, A returned an `out.best` above its own lowest.
  The kit's calls B/A of 0.379 only reflects misses scored as
  budget + 1.  B and B2 have `best_is_lowest` on 12/12 stop runs.
- lj98: the B/A lower bound of 1.0025 is above 1 but below the A2/A
  upper bound of 1.017, so it is noise.  Per call it is 1.011
  (0.999, 1.023).
- * lj38: the committed summary shows 0 hits in every arm because
  its stop level of -173.928427 sits 4e-7 below the global minimum
  that the driver reaches (-173.92842659).  The driver logs report
  a hit in all 48 runs, and evals-to-hit B/A is 1.0005.  The cell is
  valid for per-call time.  Its hit column needs a stop-level fix
  in cells.py before any later gate.
