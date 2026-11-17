Copyright (c) anneal developers

conf-fix10 follows conf-fix9 on 512c654a and targets the loss mechanism
found there: late develop hits were being displaced when `elite_return`
fired early in a run.

## Mechanism

`elite_return` now keeps its quiet-hop floor (`elite_patience`, default 40)
but fires only when the chain is also on a self-calibrated plateau above
its own best.

- A chain records accepted uphill hop sizes (up to 64).
- After 12 finite samples, it estimates a trigger height from its own
  uphill distribution (`max(2*median, median + q90)`).
- A return can fire only when `current_energy - best_energy` exceeds that
  learned height.

The trigger is general (no molecule-specific cuts), deterministic, and keeps
bit-identical behaviour to the old run until the first return.

## Development seeds (3000-3119, all 8 molecules)

`dev-rows.csv` compares develop (027c9c71) with this branch (`adaptive40`).

| arm | z34 | amprenavir | lopinavir | 915 | fluvastatin | xld | lorlatinib | rq3 | total |
|---|---|---|---|---|---|---|---|---|---|
| develop | 6 | 8 | 1 | 81 | 46 | 93 | 56 | 120 | 411 |
| adaptive40 | 11 | 8 | 6 | 84 | 49 | 93 | 56 | 120 | 427 |

Development deltas are concentrated where the floor mattered most
(z34 +5, lopinavir +5, 915 +3, fluvastatin +3), while the heavy churn seen
in conf-fix9's default-on variant is reduced.

## Gate (fresh seeds 2024-2047, all 8 molecules)

One serial bench session on host `bench`, core 0:

`snakemake -s workflow/Snakefile openconf_h2h_conf_fix10`

- branch: this tree, anneal arm with `--elite-return` on.
- develop: anneal 027c9c715564a065d02ae566d845c58e7e6da2a5.
- openconf: rowansci/openconf e32c5fa7b6e6aac944bda31c851be42e14340235.

| molecule | hits branch | hits develop | hits openconf | Fisher p vs develop | Fisher p vs openconf | n_3kcal branch | n_3kcal develop | n_3kcal openconf | ΔE_min branch | ΔE_min develop | ΔE_min openconf |
|---|---|---|---|---|---|---|---|---|---|---|---|
| rq3 | 24/24 | 24/24 | 21/24 | 1.0000 | 0.2340 | 45 | 45 | 22.5 | 0.00 | 0.00 | 0.00 |
| 915 | 21/24 | 19/24 | 9/24 | 0.7008 | 0.0008 | 4 | 5 | 3 | 0.00 | 0.00 | 0.01 |
| xld | 21/24 | 20/24 | 4/24 | 1.0000 | 0.0000 | 33 | 34.5 | 4 | 0.00 | 0.00 | 2.72 |
| fluvastatin | 10/24 | 13/24 | 4/24 | 0.5639 | 0.1107 | 38 | 36.5 | 7 | 0.13 | 0.03 | 0.98 |
| z34 | 2/24 | 2/24 | 0/24 | 1.0000 | 0.4894 | 22.5 | 27 | 1 | 0.53 | 0.53 | 2.00 |
| amprenavir | 6/24 | 1/24 | 0/24 | 0.0971 | 0.0219 | 7.5 | 1 | 0 | 0.15 | 2.29 | 4.69 |
| lopinavir | 1/24 | 3/24 | 0/24 | 0.6085 | 1.0000 | 0 | 0 | 0 | 4.73 | 3.76 | 8.95 |
| lorlatinib | 7/24 | 13/24 | 6/24 | 0.1425 | 1.0000 | 2.5 | 2 | 2 | 1.21 | 0.00 | 2.68 |

Branch totals: 92/192, develop 95/192, openconf 44/192.

Seed flips branch vs develop:

- 915: +2 / -0
- xld: +3 / -2
- fluvastatin: +5 / -8
- amprenavir: +6 / -1
- lopinavir: +1 / -3
- lorlatinib: +1 / -7
- z34, rq3: no change

| molecule | calls to hit branch | calls to hit develop | calls to hit openconf | Mann-Whitney p vs develop | Mann-Whitney p vs openconf |
|---|---|---|---|---|---|
| rq3 | 5883 | 5883 | 3098 | 0.6425 | 0.1362 |
| 915 | 15497 | 15497 | 2654 | 0.5972 | 0.0015 |
| xld | 21456 | 32000.5 | 20903.5 | 0.1833 | 0.6299 |
| fluvastatin | 13142 | 26423 | 8117.5 | 0.1534 | 0.1791 |
| z34 | 12867 | 12867 |  | 1.0000 |  |
| amprenavir | 40251.5 | 50576 |  | 0.4533 |  |
| lopinavir | 14946 | 73813 |  | 0.3711 |  |
| lorlatinib | 27619 | 41920 | 8002 | 0.1535 | 0.1004 |

Mann-Whitney uses SciPy's asymptotic method with continuity correction
(tie-corrected).

Median wall time and inside/outside potential remain close to develop:

| molecule | wall branch (s) | wall develop (s) | wall openconf (s) | branch inside (s) | branch outside (s) | develop inside (s) | develop outside (s) |
|---|---|---|---|---|---|---|---|
| rq3 | 3.84 | 3.85 | 3.77 | 2.22 | 1.62 | 2.22 | 1.62 |
| 915 | 1.70 | 1.70 | 2.01 | 0.96 | 0.74 | 0.96 | 0.74 |
| xld | 4.33 | 4.33 | 5.00 | 2.57 | 1.77 | 2.56 | 1.77 |
| fluvastatin | 4.01 | 4.07 | 4.87 | 2.37 | 1.65 | 2.41 | 1.65 |
| z34 | 5.03 | 5.01 | 5.92 | 2.96 | 2.07 | 2.95 | 2.06 |
| amprenavir | 5.05 | 5.08 | 7.61 | 2.98 | 2.08 | 3.02 | 2.06 |
| lopinavir | 9.86 | 9.90 | 20.79 | 5.89 | 3.98 | 5.90 | 3.98 |
| lorlatinib | 3.42 | 3.43 | 7.09 | 1.96 | 1.46 | 1.97 | 1.46 |

## Decision

The adaptive trigger still loses on three molecules (fluvastatin, lopinavir,
lorlatinib) in the gate, so this remains an opt-in record. `elite_return`
stays out of `recommended_molecular`.
