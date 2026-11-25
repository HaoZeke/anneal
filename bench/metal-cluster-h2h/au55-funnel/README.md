# Single-species packing leave

`8e076b7f` against `7b574de5`. A species vector that repeats one element now takes the unlabelled packing-growth leave, the energy-lattice exchange, and the fivefold SOAP hop. Two distinct labels stay a homotop.

Six seeds, 2000–2005. The (system, seed, arm) list is shuffled with schedule seed 20261113. One job at a time, `sudo unshare --uts`, hostname `bench`. `pot_per_eval_s` is potential time per charged call. `outside_per_eval_s` is the rest of the wall on the same denominator.

Au38 and Au55 use `H2H_BASIN_STRIDE=8` and the Cambridge stop, −130.54001708 and −191.03085280. Au38 is 6/6 on both arms. Au55 is 0/6 on both. The head leaves the −190.980 decahedron on seeds 2003, 2004 and 2005. Seed 2003 reaches −191.00367602. The Cambridge crystallite is −191.03085280.

Pt13, Pt38 and Cu38 use stride 1, the setting of the Pt–Cu panel. Those systems have no published stop, so every seed spends its budget and `hit` is false. The quenched energy matches `8e076b7f` on every seed: Pt13 −64.91633912, Pt38 −199.94185387, Cu38 −108.96759055.

Lennard-Jones `zn38` seed 2000 and TIP4P `n=4` at 4000 calls match `8e076b7f` in calls, charged count, hops and energy.
