# Morse head-to-head

Cambridge Cluster Database, Doye and Wales, Morse table (`rho = 3, 6, 10, 14`). A bold entry is the global minimum at that range. The potential is already in anneal: `PairKind::Morse` is `exp(rho(1-r))(exp(rho(1-r))-2)`, the same form as GMIN's `MORSE` keyword. No separate rgpot port.

| system | structure | rho | energy | budget |
| --- | --- | --- | --- | --- |
| m38r6 | 38E | 6 | -157.477108 | 400000 |
| m38r14 | 38D | 14 | -144.321054 | 400000 |
| m55r6 | 55B | 6 | -250.286609 | 1000000 |
| m55r14 | 55C | 14 | -220.646208 | 1000000 |
| m80r14 | 80G | 14 | -340.811371 | 4000000 |

Budgets follow the Lennard-Jones paper list: 4e5 at 38, 1e6 at 55, 4e6 at 75 and above. Six seeds, 2000-2005. Within a system the seed and arm pairs are shuffled with `Random(20261113)`. Serial, one core, hostname `bench`, working directory `/work`. A hit is within 0.001 of the table. `pot_s` is inside the potential. `outside_s` is the rest.

GMIN is the public Wales build. The data file is the LJ38 introduction plus `MORSE <rho>`.
