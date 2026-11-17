# Template-removal rerun deltas (append-only rows)

All reruns use seeds 2000--2011 (`n=12`) on `host=bench`, with appended rows keyed by `code_revision` and `harness_revision`.

## GMIN panel deltas

| System | Old (template-era) | New (template-free) |
| --- | --- | --- |
| M55, rho=6 | anneal 12/12 vs GMIN 12/12, Fisher p=1.0 | anneal 12/12 vs GMIN 12/12, Fisher p=1.0 |
| M55, rho=14 | anneal 7/12 vs GMIN 0/12, Fisher p=0.00457666 | anneal 7/12 vs GMIN 0/12, Fisher p=0.00457666 |
| M80, rho=6 | anneal 2/12 vs GMIN 12/12, Fisher p=6.73038e-05 | anneal 2/12 vs GMIN 12/12, Fisher p=6.73038e-05 |
| M80, rho=14 | anneal 9/12 vs GMIN 0/12, Fisher p=0.000336519 | anneal 9/12 vs GMIN 0/12, Fisher p=0.000336519 |
| Au55 | anneal 12/12 vs GMIN 1/12, Fisher p=0.000698663 | anneal 4/12 vs GMIN 1/12, Fisher p=0.31677 |

Current gmin-panel summary medians for changed rows:

- Au55 wall/in/out (anneal): 38.043 / 28.083 / 9.966 s.
- M55 rho=6 wall/in/out (anneal): 1.742 / 1.196 / 0.547 s.
- M55 rho=14 wall/in/out (anneal): 12.814 / 9.196 / 3.618 s.
- M80 rho=6 wall/in/out (anneal): 161.389 / 107.846 / 53.778 s.
- M80 rho=14 wall/in/out (anneal): 49.325 / 37.783 / 11.542 s.

Significant clusters in `bench/gmin-panel/summary.csv` (Fisher p < 0.05):

- Old template-era panel: 7 (LJ38, LJ75, LJ98, Au55, M55 rho=14, M80 rho=6, M80 rho=14)
- New template-free panel: 6 (LJ38, LJ75, LJ98, M55 rho=14, M80 rho=6, M80 rho=14)

## OCCAT and ASE Au55 deltas

- OCCAT (`bench/occat-h2h`): appended template-free anneal Au55 rows are 4/12 against OCCAT 0/12 (Fisher p=1.0).
- ASE Au55 (`bench/ase-au55`): anneal `recommended` remains 0/12; ASE BH 0/12; ASE MH 0/12 (Fisher p=1.0 against each arm).
- ASE Au55 median best energies above reference: anneal +0.05083548 eV, ASE BH +1.19646891 eV, ASE MH +0.73689833 eV.

## Notes on unavailable legacy rows

- The current tree does not include `bench/abcluster-h2h`; no ABCluster rows were available here to append within this change set.
