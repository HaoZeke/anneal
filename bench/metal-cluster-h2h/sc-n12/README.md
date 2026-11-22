# Sutton-Chen, twelve seeds

Anneal against GMIN on reduced Sutton-Chen 12-6, 10-8 and 9-6 at N=38, 55
and 80, seeds 2000-2011. The printed counts are the develop rerun in
`bench/paper-rerun/sc-h2h`, code revision `61fa67cf`. GMIN rows are pin `9cbec1a4`. One session, serial,
ABBA by seed, cpu 0. A hit is within 0.001 of the reference. Fisher is the
two-sided exact test on hits. Mann-Whitney is two-sided on calls, with the
tie correction and the continuity correction. A miss counts the full budget.
Calls are type-7 medians.

Wall medians and the earlier harness rows were removed because their
revisions are not reachable. Those rows remain on
`origin/fix/sc-shelf-waypoint` and `origin/record/bench-sc-h2h-n12`.

| System | Arm | Code revision | Hits | Calls median | Fisher p | Mann-Whitney p calls |
|---|---|---|---|---|---|---|
| sc126-38 | anneal | 61fa67cf | 11/12 | 50658.5 | 1 | 0.930987 |
| sc126-38 | gmin | 9cbec1a4 | 12/12 | 42469.5 |  |  |
| sc126-55 | anneal | 61fa67cf | 12/12 | 25529 | 1 | 0.260236 |
| sc126-55 | gmin | 9cbec1a4 | 12/12 | 43404.5 |  |  |
| sc126-80 | anneal | 61fa67cf | 4/12 | 1000000 | 0.0931677 | 0.0366874 |
| sc126-80 | gmin | 9cbec1a4 | 0/12 | 1000000 |  |  |
| sc108-38 | anneal | 61fa67cf | 9/12 | 270247.5 | 0.0391257 | 0.00956056 |
| sc108-38 | gmin | 9cbec1a4 | 3/12 | 400000 |  |  |
| sc108-55 | anneal | 61fa67cf | 7/12 | 707416.5 | 0.0271907 | 0.0109168 |
| sc108-55 | gmin | 9cbec1a4 | 1/12 | 1000000 |  |  |
| sc108-80 | anneal | 61fa67cf | 0/12 | 1000000 | 1 | 1 |
| sc108-80 | gmin | 9cbec1a4 | 0/12 | 1000000 |  |  |
| sc96-38 | anneal | 61fa67cf | 12/12 | 20236 | 1 | 0.665006 |
| sc96-38 | gmin | 9cbec1a4 | 12/12 | 37616.5 |  |  |
| sc96-55 | anneal | 61fa67cf | 12/12 | 28040 | 1 | 0.97697 |
| sc96-55 | gmin | 9cbec1a4 | 12/12 | 28021 |  |  |
| sc96-80 | anneal | 61fa67cf | 6/12 | 993354.5 | 0.0686499 | 0.0283629 |
| sc96-80 | gmin | 9cbec1a4 | 1/12 | 1000000 |  |  |

The rows file is blob `8e6d73ea528b6c68ecf7be18b49d15254c42b83a`.
Harness blob `ac6b3c85b4eb435c4787a8a7425ab7e6858f9b34` writes the GMIN pin
`9cbec1a446352bc9ef064bd2b1a5767194655919` into `code_revision` on GMIN rows.
Commit `c2afa54b00421482de4e67c70ab8a02462bf4509` relabelled those cells to
the anneal revision `61fa67cf380e46876b569da45e7cc091647a775b` after the run.
The GMIN binary was pin `9cbec1a446352bc9ef064bd2b1a5767194655919`.
