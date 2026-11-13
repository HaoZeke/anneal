# Pt–Cu homotop check, develop against the unrelaxed exchange

The mixed 23- and 38-atom rows of the Pt–Cu panel are 1/6. This control asks whether a miss is the wrong geometry or the wrong chemical ordering on that geometry.

Common-neighbour analysis of the saved minima, cutoff 1.25 times the median nearest-neighbour distance, species-blind:

| system | hit signature | nearest miss |
|---|---|---|
| pt12cu11 | 25 bonds of 555 | 10 of 555, plus 200 / 311 / 422 |
| pt11cu12 | 24 of 555, 84 bonds | 23 of 555, 83 bonds, pair-distance RMS 0.14 Å |
| pt6cu17 | 33 of 555 | 14 of 555, plus 422 |
| pt19cu19 | 60 bonds of 421, no 422 | 36 of 422 and 24 of 421 |

An exchange-only polish of each nearest miss, greedy and Metropolis, unrelaxed and then quenched, does not move the energy. The same polish on a label-scramble of the hit returns the hit energy when exchanges are not quenched, and falls short when each exchange is quenched. The misses are different geometries whose ordering is already a single-swap minimum. Quenching between exchanges is what walks a scrambled homotop off the sites it is labelling.

The search change orders a new minimum by unrelaxed unlike-pair exchange, then quenches that ordering once. It does not read a reference energy, and it does not run on a one-species, molecular, or rigid library.

Both arms below are seeds 2000–2005, basin stride 1, serial, `taskset -c 0`, `sudo unshare --uts` with hostname `bench`, working directory `/work`. The list was shuffled with schedule seed 20261114. A hit is within 0.001 eV of the panel reference in `ptcu-b3d03061/results.csv`. Develop reproduces those references. The ordering change does not raise the 1/6.

| system | panel reference / eV | develop | homotop | homotop best / eV |
|---|---:|---|---|---:|
| pt12cu11 | −96.03443577 | 1/6 | 1/6 | −96.03443577 |
| pt11cu12 | −94.01575979 | 1/6 | 0/6 | −93.84259053 |
| pt6cu17 | −81.89514983 | 1/6 | 1/6 | −81.90409805 |
| pt19cu19 | −161.50663450 | 1/6 | 0/6 | −161.50426353 |

On pt6cu17 one homotop seed is 0.009 eV below the panel reference. On pt19cu19 the best homotop seed is 0.002 eV above it. Paired with develop, pt12cu11 is lower on 5 of 6 seeds and the other three compositions split.
