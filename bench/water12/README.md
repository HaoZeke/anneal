# (H2O)12 at the CREST call cap

Base is develop `98dcfb3106f191baf7381fecd5624426831094fe`. That commit already contains the SCC rejection stop `8b4a2592`.

The recorded anneal runs are the fixed `8b4a2592` driver, six seeds, each capped at that seed's CREST evaluation count. Seeds 2 and 5 of (H2O)12 both finish at −1660.718036 eV. CREST on those starts is −1660.932675 eV and −1660.931043 eV. The other four seeds reach −1660.934286 eV or sit within 1.5 meV of CREST.

## What the traces show

Seed 2 enters −1660.717475 eV at hop 94, 16619 calls, and the next 2113 hops move the best by 0.0003 eV. Seed 5 enters −1660.717201 eV at hop 38, 7516 calls, and the remaining budget moves it by 0.0006 eV. Both polished answers are the same minimum, −1660.718036 eV.

Acceptance is high and improvements of the incumbent are rare. Seed 2 accepts 524 of 796 group shakes, 193 of 199 surface moves, 181 of 185 bursts, 350 of 385 cycle reversals and 411 of 642 rewires, and those arms improve the incumbent 9, 2, 4, 3 and 4 times. Seed 5 is the same pattern, 17 improvements in 1207 hops.

The recommended water arm pays the allocator for improvements and, after 32 hops without a new best, leaves from `StallArchive`. That archive keeps the eight lowest energies. After the chain is in −1660.718 those seats are variants of that minimum: the same-minimum test is one part in 1e8, so a tenth of a meV is a different seat, and the higher basins from the way in are dropped. The temperature dial does not move either. A block whose acceptance sits within a quarter of one half leaves the temperature alone, and the dial is frozen when its ceiling equals the preset. (H2O)12 at about 46 eV per atom is under the 64 eV steep-well cut, so the ceiling is the preset. The kick scale for a molecular shake was not the acceptance scale at all.

There is no temperature history in the trace because the temperature never changed. The arm table is the acceptance history. Restarts that did run stayed in the one funnel.

## The leave

`Config::plateau_population` is off unless a caller sets it (`H2H_PLATEAU=1` on the water driver, `plateau` on the Lennard-Jones example). The resolved JSON omits it while it is off, and `tests/lj_trajectory_pin.rs` still matches.

On, three things move, for every potential:

- The kick grows when a block of 40 hops accepts more than three quarters of its trials, and shrinks when it accepts less than a quarter. The factor stays between 0.25 and 4. A group shake's translation and its rotation take that factor. At factor 1 the draw is the preset draw.
- The temperature rises when a block accepts and never lowers the incumbent, and falls back toward the preset when a block does. The factor stays between 1 and 4.
- The population keeps eight structures that are far apart in their sorted pair distances, not the eight lowest energies. After `restart_patience` charged calls with no new incumbent the chain leaves for the farthest kept structure, and the incumbent-gap pull stays off for another patience of calls. The energy-keyed water leave does not also run.

Seed 5, same start, same 151471 calls, hostname `bench`, one thread: −1660.934286 eV. That is the minimum seeds 0, 3 and 4 already had, 216.250 meV below the old seed 5, and 3.243 meV below CREST. The kick and the temperature both ended at the cap of 4, after 26 leaves. The low basin is first recorded at 44507 calls.
