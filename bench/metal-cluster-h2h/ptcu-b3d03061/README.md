# Pt–Cu Gupta, anneal against GMIN

Sizes 13, 23 and 38 are the cluster sizes in Wu, Lin, He, Chen and Shi, *Nanoscale* **18**, 14001 (2026), doi:10.1039/d6nr01123c. The abstract names Pt11Cu12 and Pt19Cu19 and a Pt-rich and a Cu-rich case. This panel uses those two stoichiometries, Pt7Cu6, Pt12Cu11, Pt6Cu17, and the pure metals at 13 and 38.

The Gupta rows are Cleri and Rosato, *Phys. Rev. B* **48**, 22 (1993), Table I, as listed by Guedes-Sobrinho, Freire, da Silva and Galvão, *Eur. Phys. J. B* **91**, 170 (2018), Table 1, and by the Pele documentation of the same table.

| pair | A / eV | ξ / eV | p | q | r0 / Å |
|---|---:|---:|---:|---:|---:|
| Pt–Pt | 0.2975 | 2.695 | 10.612 | 4.004 | 2.775 |
| Cu–Cu | 0.0855 | 1.224 | 10.960 | 2.278 | 2.556 |
| Pt–Cu | 0.1915 | 1.9595 | 10.786 | 3.141 | 2.6655 |

The cross term is the arithmetic mean, equation (4) of Guedes-Sobrinho et al. at w = 0.5. The Cambridge Cluster Database Gupta pages list Zn, Cd, Pb, Na, Co and Ag. They do not list Pt–Cu.

Wu et al. is closed and does not publish an open table of Gupta total energies. The `reference_eV` column is the lowest energy either arm reached for that system in this session, not a number copied from the paper. A hit is within 0.001 eV of that energy. Every seed spends its budget.

`PAPER_BUDGETS` for 38 atoms is 400000, the LJ38 entry of `scripts/lj_board/boardlib.py`. 13 and 23 atoms are not in that map; they use 100000 and 200000. The LJ dict itself is unchanged.

Six seeds, 2000–2005. The (system, seed, arm) list is shuffled with schedule seed 20261113. One job at a time, `taskset -c 0`, `sudo unshare --uts`, hostname `bench`, working directory `/work`. `pot_per_eval_s` is potential time per charged call. `outside_per_eval_s` is the rest of the wall on the same denominator. The wall ratio is the GMIN median wall divided by the anneal median wall.

`h2h_metal` reads `H2H_BASIN_STRIDE` and otherwise takes 8. These rows were recorded at stride 1, which is what the driver sets. Lennard-Jones and TIP4P do not set the stride.

```text
snakemake -s workflow/Snakefile ptcu_h2h
```
