# Sutton-Chen head-to-head rerun

Anneal against GMIN on the printed cells: reduced Sutton-Chen 12-6, 10-8 and 9-6
at 38, 55 and 80 atoms, seeds 2000-2011, at `PAPER_BUDGETS` in
`scripts/metal_h2h/run_sc.py`. One session, serial, ABBA by seed, cpu 0.

A hit is best <= reference + 0.001 (`scripts/target_tol.py`, `DEFAULT_HIT_TOL`).
Fisher is two-sided exact on hits. Mann-Whitney is two-sided on calls with the
tie correction and the continuity correction, and a miss ranked worse than a hit
at the full budget. The calls median is type 7 of the stored call counts.

The GMIN data file is `scripts/metal_h2h/run_sc.py` `data_text`: the container
left to GMIN, the Doye-Wales temperature, and the step 0.40 sigma in units of a.

| System | Arm | Hits | Calls median | Fisher p | Mann-Whitney p calls |
|---|---|---|---|---|---|
| sc126-38 | anneal | 11/12 | 50658.500000 | 1 | 0.930987 |
| sc126-38 | gmin | 12/12 | 42469.500000 |  |  |
| sc126-55 | anneal | 12/12 | 25529.000000 | 1 | 0.260236 |
| sc126-55 | gmin | 12/12 | 43404.500000 |  |  |
| sc126-80 | anneal | 4/12 | 1000000.000000 | 0.0931677 | 0.0366874 |
| sc126-80 | gmin | 0/12 | 1000000.000000 |  |  |
| sc108-38 | anneal | 9/12 | 270247.500000 | 0.0391257 | 0.00956056 |
| sc108-38 | gmin | 3/12 | 400000.000000 |  |  |
| sc108-55 | anneal | 7/12 | 707416.500000 | 0.0271907 | 0.0109168 |
| sc108-55 | gmin | 1/12 | 1000000.000000 |  |  |
| sc108-80 | anneal | 0/12 | 1000000.000000 | 1 | 1 |
| sc108-80 | gmin | 0/12 | 1000000.000000 |  |  |
| sc96-38 | anneal | 12/12 | 20236.000000 | 1 | 0.665006 |
| sc96-38 | gmin | 12/12 | 37616.500000 |  |  |
| sc96-55 | anneal | 12/12 | 28040.000000 | 1 | 0.97697 |
| sc96-55 | gmin | 12/12 | 28021.000000 |  |  |
| sc96-80 | anneal | 6/12 | 993354.500000 | 0.0686499 | 0.0283629 |
| sc96-80 | gmin | 1/12 | 1000000.000000 |  |  |

Printed counts are the cells in `bench/metal-cluster-h2h/sc-n12/README.md`.
Two GMIN rows are printed for each system: the earlier data file and the corrected file.

| System | Arm | Printed harness | Printed hits | Printed calls | New hits | New calls |
|---|---|---|---|---|---|---|
| sc126-38 | anneal | e3ed7dad | 12/12 | 10344.5 | 11/12 | 50658.5 |
| sc126-38 | gmin | e3ed7dad | 1/12 | 400000 | 12/12 | 42469.5 |
| sc126-38 | gmin | f9d1ac05 | 12/12 | 48872.5 | 12/12 | 42469.5 |
| sc108-38 | anneal | e3ed7dad | 12/12 | 11193 | 9/12 | 270247.5 |
| sc108-38 | gmin | e3ed7dad | 0/12 | 400000 | 3/12 | 400000 |
| sc108-38 | gmin | f9d1ac05 | 0/12 | 400000 | 3/12 | 400000 |
| sc96-38 | anneal | e3ed7dad | 12/12 | 9177 | 12/12 | 20236 |
| sc96-38 | gmin | e3ed7dad | 0/12 | 400000 | 12/12 | 37616.5 |
| sc96-38 | gmin | 9149d1b0 | 12/12 | 44974 | 12/12 | 37616.5 |
| sc126-55 | anneal | e3ed7dad | 12/12 | 65397.5 | 12/12 | 25529 |
| sc126-55 | gmin | e3ed7dad | 0/12 | 1000000 | 12/12 | 43404.5 |
| sc126-55 | gmin | a50f4b02 | 12/12 | 30565.5 | 12/12 | 43404.5 |
| sc108-55 | anneal | e3ed7dad | 5/12 | 1000000 | 7/12 | 707416.5 |
| sc108-55 | gmin | e3ed7dad | 0/12 | 1000000 | 1/12 | 1000000 |
| sc108-55 | gmin | a50f4b02 | 2/12 | 1000000 | 1/12 | 1000000 |
| sc96-55 | anneal | e3ed7dad | 12/12 | 17280 | 12/12 | 28040 |
| sc96-55 | gmin | e3ed7dad | 0/12 | 1000000 | 12/12 | 28021 |
| sc96-55 | gmin | a50f4b02 | 12/12 | 32261 | 12/12 | 28021 |
| sc126-80 | anneal | e3ed7dad | 5/12 | 1000000 | 4/12 | 1000000 |
| sc126-80 | gmin | e3ed7dad | 0/12 | 1000000 | 0/12 | 1000000 |
| sc126-80 | gmin | a50f4b02 | 0/12 | 1000000 | 0/12 | 1000000 |
| sc108-80 | anneal | e3ed7dad | 4/12 | 1000000 | 0/12 | 1000000 |
| sc108-80 | gmin | e3ed7dad | 0/12 | 1000000 | 0/12 | 1000000 |
| sc108-80 | gmin | a50f4b02 | 0/12 | 1000000 | 0/12 | 1000000 |
| sc96-80 | anneal | e3ed7dad | 7/12 | 773712.5 | 6/12 | 993354.5 |
| sc96-80 | gmin | e3ed7dad | 0/12 | 1000000 | 1/12 | 1000000 |
| sc96-80 | gmin | a50f4b02 | 0/12 | 1000000 | 1/12 | 1000000 |

Rows: 216.
