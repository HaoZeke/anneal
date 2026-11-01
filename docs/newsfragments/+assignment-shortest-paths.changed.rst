Dense assignments of 98 points or more are solved by successive shortest paths.
The continuous-symmetry projection skips the matching for the identity operation.
Smaller assignments keep the Kuhn-Munkres solver, so searches below 98 atoms follow the same trajectories as before.
From 98 atoms the solver finds a matching of the same minimum cost but can pick another among equal optima.
Seeded trajectories from 98 atoms therefore change.
On one thread, a 2e5-call search of the 98-atom Lennard-Jones cluster takes 15.7 to 18.3 s in place of 34.7 to 39.7 s.
