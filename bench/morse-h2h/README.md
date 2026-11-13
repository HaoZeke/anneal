# Morse head-to-head

Cambridge Cluster Database, Doye and Wales, Morse table (`rho = 3, 6, 10, 14`). A bold entry is the global minimum at that range. The potential is already in anneal: `PairKind::Morse` is `exp(rho(1-r))(exp(rho(1-r))-2)`, the same form as GMIN's `MORSE` keyword. No separate rgpot port.

| system | structure | rho | energy | budget |
| --- | --- | --- | --- | --- |
| m38r6 | 38E | 6 | -157.477108 | 400000 |
| m38r14 | 38D | 14 | -144.321054 | 400000 |
| m55r6 | 55B | 6 | -250.286609 | 1000000 |
| m55r14 | 55C | 14 | -220.646208 | 1000000 |
| m80r14 | 80G | 14 | -340.811371 | 4000000 |

Budgets follow the Lennard-Jones paper list: 4e5 at 38, 1e6 at 55, 4e6 at 75 and above. Six seeds, 2000-2005. Within a system the seed and arm pairs are shuffled with `Random(20261113)`. Serial, one core, hostname `bench`, working directory `/work`. A hit is within 0.001 of the table. `pot_s` is inside the potential. `outside_s` is the rest. Per-seed rows are `seeds.csv`. The type-7 medians are `results.csv`.

GMIN is the public Wales build, revision `aac1548a8aec13b53ad12a8d6904625081685d2e`. The data file is the LJ38 introduction plus `MORSE <rho>`.

These rows were measured with `h2h_morse` built at `fe9671c8`, this harness on develop `23815131`. The rebase onto `b6b6c94` leaves those harness files identical. The counts were kept as measured. The library at `b6b6c94` is later, so a rebuild there can move the evaluation counts. Eleven GMIN jobs had been written before `MORSEGH` carried the potential timer already used for Lennard-Jones: every m38r14 seed, and m55r14 seeds 2001-2005. Those jobs were repeated on the timed binary. Each remained a miss, and `pot_s` is that second pass.

| system | anneal | gmin | wall ratio |
| --- | --- | --- | --- |
| m38r14 | 6/6 | 0/6 | 16.868 |
| m55r14 | 4/6 | 0/6 | 1.343 |
| m80r14 | 2/6 | 0/6 | 0.721 |
| m38r6 | 6/6 | 6/6 | 0.922 |
| m55r6 | 6/6 | 6/6 | 0.131 |

The wall ratio is the GMIN median wall divided by the anneal median wall.

At rho = 14 anneal is ahead on every size. GMIN's closest energies are -141.698 (2.62 above 38D), -219.521 (1.13 above 55C), and -336.858 (3.95 above 80G).

The two M55 misses, seeds 2002 and 2005, both finish at -220.518533, the Cambridge 55D energy at rho = 14. 55D is a Cs packing whose listed range starts at rho 15.09. The four M80 misses, seeds 2000-2003, all finish at -340.802563, 0.0088 above 80G and outside the 0.001 window. Seeds 2004 and 2005 reach -340.811371. The same energy on four seeds is one repeated minimum. At rho = 6 both arms are 6/6. GMIN's median evaluations are 88057 at M38 and 23759 at M55, against 103679 and 129013.
