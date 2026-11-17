# (H2O)12 at the CREST call cap

Base is develop `98dcfb3106f191baf7381fecd5624426831094fe`. That commit already contains the SCC rejection stop `8b4a2592`.

The earlier anneal runs are the fixed driver at `8b4a2592`, six seeds, each capped at that seed's CREST evaluation count. Seeds 2 and 5 of (H2O)12 finish at −1660.718036 eV. CREST on those starts is −1660.932675 eV and −1660.931043 eV. Seeds 0, 3 and 4 already reach −1660.934286 eV. Seed 1 sits with CREST at −1660.842645 eV.

## What the traces show

Seed 2 enters −1660.717475 eV at hop 94, 16619 calls, and the next 2113 hops move the best by 0.0003 eV. Seed 5 enters −1660.717201 eV at hop 38, 7516 calls, and the rest of the budget moves it by 0.0006 eV. Both polished answers are the same minimum, −1660.718036 eV.

Acceptance is high and improvements of the incumbent are rare. Seed 2 accepts 524 of 796 group shakes, 193 of 199 surface moves, 181 of 185 bursts, 350 of 385 cycle reversals and 411 of 642 rewires. Those arms improve the incumbent 9, 2, 4, 3 and 4 times. Seed 5 records 17 improvements in 1207 hops.

The recommended water arm pays the allocator for improvements and, after 32 hops without a new best, leaves from `StallArchive`. That archive keeps the eight lowest energies. Once the chain is in −1660.718 those seats are variants of that minimum: the same-minimum test is one part in 1e8, so a tenth of a meV is a different seat, and the higher basins from the way in are dropped. The temperature dial does not move either. A block whose acceptance sits within a quarter of one half leaves the temperature alone, and the dial is frozen when its ceiling equals the preset. (H2O)12 at about 46 eV per atom is under the 64 eV steep-well cut, so the ceiling is the preset. A molecular shake did not take an acceptance scale at all.

There is no temperature history in those traces because the temperature never changed. The arm table is the acceptance history. The restarts that did run stayed in the one funnel.

## The leave

`Config::plateau_population` is off unless a caller sets it. `H2H_PLATEAU=1` sets it on the water driver. The word `plateau` sets it on `lj_cluster_search`. The resolved JSON omits the flag while it is off, and `tests/lj_trajectory_pin.rs` still matches, so the published pin is the same search.

On, for every potential:

- The kick grows when a block of 40 hops accepts more than three quarters of its trials, and shrinks when it accepts less than a quarter. The factor stays between 0.25 and 4. A group shake's translation and its rotation take that factor. At factor 1 the draw is the preset draw.
- The temperature rises when a block accepts and never lowers the incumbent, and falls back toward the preset when a block does. The factor stays between 1 and 4.
- The population keeps eight structures that are far apart in their sorted pair distances, not the eight lowest energies. After `restart_patience` charged calls with no new incumbent the chain leaves for the farthest kept structure, and the incumbent-gap pull stays off for another patience of calls. The energy-keyed water leave does not also run.

It stays opt-in. On LJ38 the plain arm solves 5 of 12 seeds and the plateau arm solves 2 of 12 (Fisher exact p = 0.371). The tie-corrected Mann-Whitney test on the twelve energies has p = 0.105. LJ55 is twelve identical solves on both arms. LJ75 solves none on either arm (Mann-Whitney p = 0.565). That is not a significant loss, and it is not a reason to change the default. The pin stays bit-identical with the flag off.

## Water, same starts, same caps

Hostname `bench`, one thread, `H2H_PLATEAU=1`. Δ is the new energy minus the previous anneal energy, in meV. A negative Δ is lower.

| n | Seed | Calls | New eV | Previous eV | Δ (meV) | Leaves |
| --- | --- | --- | --- | --- | --- | --- |
| 12 | 0 | 187434 | −1660.934286 | −1660.934286 | 0.000 | 35 |
| 12 | 1 | 294713 | −1660.934286 | −1660.842645 | −91.642 | 57 |
| 12 | 2 | 263614 | −1660.934286 | −1660.718036 | −216.250 | 47 |
| 12 | 3 | 244543 | −1660.934286 | −1660.934286 | 0.000 | 46 |
| 12 | 4 | 291306 | −1660.934286 | −1660.934286 | 0.000 | 53 |
| 12 | 5 | 151471 | −1660.934286 | −1660.718036 | −216.250 | 26 |

All six (H2O)12 seeds finish on the minimum seeds 0, 3 and 4 already had. Seeds 2 and 5 are 216.250 meV lower than before. Seed 1 is 91.642 meV lower. Seeds 0, 3 and 4 are unchanged at the recorded precision. Seeds 6–11 were not given new CREST runs.

(H2O)6, six seeds, ends at −829.852228 eV on every seed. The previous energies agree to better than 0.001 meV. (H2O)8 ends at −1106.995547 eV on every seed, the same minimum as before.

The rows are `bench/water12/water.csv`. The Lennard-Jones seeds are `bench/water12/lj.csv`.
