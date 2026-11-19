# SOAP general3 default-on gate

Overall verdict: **DEFAULT-ON REJECTED**.

B is acceptable as the default only when no system loses. A system loses if B has fewer hits, if its median budget-censored calls are higher with exact two-sided Mann–Whitney p < 0.05, or if its median outside-potential time is higher than A by more than the A/A relative noise band.

Calls and outside-potential p-values use the exact permutation distribution of the pooled midranks, including ties. Fisher p-values are exact and two-sided. Wall, potential, and outside-potential entries are medians in seconds.

## A/B results

| System | Hits A/B | Fisher p | Calls A/B | Calls p | Wall A/B | Pot A/B | Outside A/B | Outside p | Verdict |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---|
| lj38 | 12/12 / 12/12 | 1 | 6357 / 6357 | 0.90029643 | 2.4698055 / 2.492224 | 0.900662 / 0.903194 | 1.569374 / 1.590681 | 0.44283318 | TIE |
| lj75 | 12/12 / 12/12 | 1 | 44176 / 44634 | 0.83284027 | 75.792513 / 75.565787 | 33.849653 / 33.912844 | 42.214882 / 42.048898 | 0.75528483 | TIE |
| lj98 | 10/12 / 12/12 | 0.47826087 | 1321410 / 756300 | 0.31136739 | 130.08987 / 131.83112 | 57.835343 / 57.526047 | 72.290481 / 73.836624 | 0.2189208 | LOSES |
| lj150 | 9/12 / 10/12 | 1 | 649406 / 497304 | 0.99076459 | 278.82404 / 278.11061 | 134.197 / 134.15418 | 143.81829 / 144.05371 | 0.71253508 | WINS |
| m55r14 | 3/12 / 1/12 | 0.59006211 | 1000000 / 1000000 | 0.46583851 | 18.327799 / 18.524002 | 12.676886 / 12.773423 | 5.677532 / 5.7889245 | 0.24152379 | LOSES |
| m80r6 | 12/12 / 12/12 | 1 | 238328 / 165277.5 | 0.99077716 | 159.25419 / 158.42918 | 105.99014 / 106.4963 | 52.529908 / 51.029395 | 0.51372258 | WINS |
| m38r10 | 12/12 / 12/12 | 1 | 9111 / 9826.5 | 0.85590106 | 4.0461355 / 4.1002105 | 2.4258565 / 2.4255925 | 1.6299405 / 1.670911 | 0.47757526 | LOSES |
| tip4p8 | 12/12 / 12/12 | 1 | 6339 / 6339 | 1 | 5.915347 / 5.932898 | 3.7612105 / 3.7643065 | 2.139781 / 2.1702545 | 0.14316778 | TIE |
| tip4p15 | 7/12 / 7/12 | 1 | 1481619.5 / 1481619.5 | 1 | 14.565375 / 14.505091 | 10.611488 / 10.612721 | 3.9227455 / 3.911388 | 0.67065953 | TIE |
| tip4p20 | 6/12 / 6/12 | 1 | 1686249.5 / 1686249.5 | 1 | 22.986973 / 22.944489 | 17.683239 / 17.576394 | 5.388821 / 5.4340225 | 0.9323005 | TIE |
| tip4p21 | 0/12 / 0/12 | 1 | 2000000 / 2000000 | 1 | 25.337739 / 25.455646 | 19.297734 / 19.266664 | 6.129764 / 6.1576975 | 0.75528483 | TIE |
| tip4p25 | 1/12 / 1/12 | 1 | 2000000 / 2000000 | 1 | 34.654063 / 34.844741 | 26.400151 / 26.418267 | 8.234635 / 8.358694 | 0.14316778 | TIE |

## A/A outside-potential band

| System | Median A1 | Median A2 | Relative delta | Exact p |
|---|---:|---:|---:|---:|
| lj75 | 41.736227 | 42.065533 | 0.0078901477 | 0.84283599 |
| m80r6 | 51.89734 | 52.725697 | 0.015961473 | 0.6297255 |
| tip4p15 | 3.943373 | 3.9142765 | -0.0073785817 | 0.55116717 |

The A/A band is the largest absolute relative median delta: 0.015961473.

## Provenance

- A: commit `dd054749d4703923f4cf4c34f50589c574ee1276`, tree `f1705d6f02f7f257f50e831c16057740283efea5`.
- B: commit `a4bfc404e439efb09df34531d88cf82b7d420d18`, tree `e497d9777a725b0d9d732c6dadcf1f0e20aaa68f`.
- Seeds 2000–2011; full paper budgets; serial fresh processes pinned to core 0.
- References: `scripts/gmin_panel/run_panel.py` for LJ and M55/M80; `scripts/ase_nonlj/run_one.py` at commit `cf6833e1316cf8c7a1bb7879a477ef2b32e9d8f5` for M38 ρ=10; `scripts/tip4p_h2h/run_h2h.py` for TIP4P, including its pooled-session rule for TIP4P25.
- Losing systems: lj98, m55r14, m38r10.
