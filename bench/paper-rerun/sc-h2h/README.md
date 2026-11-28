# Sutton-Chen head-to-head rerun

Anneal against GMIN on the printed cells: reduced Sutton-Chen 12-6, 10-8 and 9-6
at 38, 55 and 80 atoms, seeds 2000-2011, at `PAPER_BUDGETS` in
`scripts/metal_h2h/run_sc.py`. One session, serial, ABBA by seed, cpu 0.
Code revision `61fa67cf`. GMIN rows are pin `9cbec1a4`.

A hit is best <= reference + 0.001 (`scripts/target_tol.py`, `DEFAULT_HIT_TOL`).
Fisher is two-sided exact on hits. Mann-Whitney is two-sided on calls with the
tie correction and the continuity correction, and a miss ranked worse than a hit
at the full budget. The calls median is type 7 of the stored call counts.

The GMIN data file is `scripts/metal_h2h/run_sc.py` `data_text`: the container
left to GMIN, the Doye-Wales temperature, and the step 0.40 sigma in units of a.

Earlier numbers whose revision is not reachable were removed. They remain on
`origin/fix/sc-shelf-waypoint` and `origin/record/bench-sc-h2h-n12`.

| System | Arm | Code revision | Hits | Calls median | Fisher p | Mann-Whitney p calls |
|---|---|---|---|---|---|---|
| sc126-38 | anneal | 61fa67cf | 11/12 | 50658.5 | — | — |
| sc126-38 | gmin | 9cbec1a4 | 12/12 | 42469.5 |  |  |
| sc126-55 | anneal | 61fa67cf | 12/12 | 25529 | — | — |
| sc126-55 | gmin | 9cbec1a4 | 12/12 | 43404.5 |  |  |
| sc126-80 | anneal | 61fa67cf | 4/12 | 1000000 | — | — |
| sc126-80 | gmin | 9cbec1a4 | 0/12 | 1000000 |  |  |
| sc108-38 | anneal | 61fa67cf | 9/12 | 270247.5 | — | — |
| sc108-38 | gmin | 9cbec1a4 | 3/12 | 400000 |  |  |
| sc108-55 | anneal | 61fa67cf | 7/12 | 707416.5 | — | — |
| sc108-55 | gmin | 9cbec1a4 | 1/12 | 1000000 |  |  |
| sc108-80 | anneal | 61fa67cf | 0/12 | 1000000 | — | — |
| sc108-80 | gmin | 9cbec1a4 | 0/12 | 1000000 |  |  |
| sc96-38 | anneal | 61fa67cf | 12/12 | 20236 | — | — |
| sc96-38 | gmin | 9cbec1a4 | 12/12 | 37616.5 |  |  |
| sc96-55 | anneal | 61fa67cf | 12/12 | 28040 | — | — |
| sc96-55 | gmin | 9cbec1a4 | 12/12 | 28021 |  |  |
| sc96-80 | anneal | 61fa67cf | 6/12 | 993354.5 | — | — |
| sc96-80 | gmin | 9cbec1a4 | 1/12 | 1000000 |  |  |

Rows: 216.

The rows file is blob `8e6d73ea528b6c68ecf7be18b49d15254c42b83a`.
Harness blob `ac6b3c85b4eb435c4787a8a7425ab7e6858f9b34` writes the GMIN pin
`9cbec1a446352bc9ef064bd2b1a5767194655919` into `code_revision` on GMIN rows.
Commit `c2afa54b00421482de4e67c70ab8a02462bf4509` relabelled those cells to
the anneal revision `61fa67cf380e46876b569da45e7cc091647a775b` after the run.
The GMIN binary was pin `9cbec1a446352bc9ef064bd2b1a5767194655919`.
