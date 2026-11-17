# Sutton-Chen wall with rgpot efd0750f built with OpenMP

rgpot efd0750f28d6bf55ac6282e4bf430227d547fb75 is the head of rgpot's `feat/sc-call-overhead` branch. Its last two commits run the team slices of the potential and of the neighbour walk on whatever threads the OpenMP runtime starts, and `../sc-wall2` ran efd0750f without OpenMP, so those commits were not exercised there. Here `build_rgpot_sc.sh` builds it with `-fopenmp` (`RGPOT_OPENMP=1`). rgpot's develop is still at 251aaade and anneal's `Cargo.lock` still pins 8efdd246, so the build is only recorded here and in the rows. `scripts/metal_h2h/enter_sc_omp.sh` runs `sc_wall.sh` on the six 55- and 80-atom sc-n12 systems, seeds 2000-2011 at 1000000 calls, with OMP_NUM_THREADS=1 on core 0 and OMP_NUM_THREADS=8 on cores 0-7, one job at a time in a UTS namespace with hostname `bench`, harness 12b980c6. The rows are in `seeds.csv` and `micro.csv`, with the OpenMP flag, the thread count and the cores. `bits.csv` holds `sc_omp_bits`, which compares the build with OpenMP and the build without on the seed-2000 start of each system and its 63 noisy copies. Harness 12b980c6 was a local commit, pushed as bc82a380 with the same patch; `../../revision-aliases.csv` maps it to its pushed commit.

## Calls and energies

Each row is matched by seed against `../sc-wall2/seeds.csv` in the hit, calls to the hit, calls charged, best and final energy, hops, escapes, restarts and paths.

| Threads | Cores | System | Seeds | Same as 14680ddb | Same as efd0750f without OpenMP |
|---|---|---|---|---|---|
| 1 | 0 | sc126-55 | 12 | 12 | 12 |
| 1 | 0 | sc108-55 | 12 | 12 | 12 |
| 1 | 0 | sc96-55 | 12 | 12 | 12 |
| 1 | 0 | sc126-80 | 12 | 12 | 12 |
| 1 | 0 | sc108-80 | 12 | 12 | 12 |
| 1 | 0 | sc96-80 | 12 | 12 | 12 |
| 8 | 0-7 | sc126-55 | 12 | 0 | 0 |
| 8 | 0-7 | sc108-55 | 12 | 0 | 0 |
| 8 | 0-7 | sc96-55 | 12 | 0 | 0 |
| 8 | 0-7 | sc126-80 | 12 | 0 | 0 |
| 8 | 0-7 | sc108-80 | 12 | 0 | 0 |
| 8 | 0-7 | sc96-80 | 12 | 0 | 0 |

With OMP_NUM_THREADS=1 on core 0, each of the 72 rows has the hit, calls to the hit, calls charged, best and final energy, hops, escapes, restarts and paths of its 14680ddb and efd0750f without OpenMP rows in `../sc-wall2/seeds.csv`. With OMP_NUM_THREADS=8 on cores 0-7, 72 of the 72 rows differ from their 14680ddb and efd0750f without OpenMP rows, on sc126-55, sc108-55, sc96-55, sc126-80, sc108-80 and sc96-80.

## Energy and force bits

| Threads | Cores | System | Geometries | Energy bits differ | Force bits differ | Max abs dE | Max abs dF | Max dE over E |
|---|---|---|---|---|---|---|---|---|
| 1 | 0 | sc126-55 | 64 | 0 | 0 | 0 | 0 | 0 |
| 1 | 0 | sc108-55 | 64 | 0 | 0 | 0 | 0 | 0 |
| 1 | 0 | sc96-55 | 64 | 0 | 0 | 0 | 0 | 0 |
| 1 | 0 | sc126-80 | 64 | 0 | 0 | 0 | 0 | 0 |
| 1 | 0 | sc108-80 | 64 | 0 | 0 | 0 | 0 | 0 |
| 1 | 0 | sc96-80 | 64 | 0 | 0 | 0 | 0 | 0 |
| 8 | 0-7 | sc126-55 | 64 | 0 | 0 | 0 | 0 | 0 |
| 8 | 0-7 | sc108-55 | 64 | 9 | 0 | 2.27e-13 | 0 | 1.18e-16 |
| 8 | 0-7 | sc96-55 | 64 | 7 | 0 | 4.55e-13 | 0 | 1.83e-16 |
| 8 | 0-7 | sc126-80 | 64 | 0 | 0 | 0 | 0 | 0 |
| 8 | 0-7 | sc108-80 | 64 | 8 | 0 | 4.55e-13 | 0 | 1.56e-16 |
| 8 | 0-7 | sc96-80 | 64 | 11 | 0 | 4.55e-13 | 0 | 1.18e-16 |

With OMP_NUM_THREADS=1 the build with OpenMP gives the energy and force bits of the build without on all 384 geometries. With OMP_NUM_THREADS=8 the energy bits differ on 35 of the 384 geometries and the force bits on 0; the largest energy difference is 4.55e-13, 1.83e-16 of the energy.

## Where the bits move

In rgpot efd0750f, `SuttonChenPot::compute` in `CppCore/rgpot/SuttonChen/SuttonChenPot.cc` takes its thread count from `nlist::threadsRequested()` (line 178), which in a build with OpenMP returns OMP_NUM_THREADS whatever the number of atoms (`CppCore/rgpot/nlist/Neighbours.cc`, lines 46-64). With one thread the density pass adds each pair energy to `epair` as the walk visits the pair (line 244). With more, `forCellTeam` (`CppCore/rgpot/nlist/Neighbours.hpp`, line 188) cuts the unbinned search into contiguous runs of atoms with `partition` (line 212), each slice adds the pair energies of its atoms into `thread_e[t]` (line 279), and lines 286-288 add those partial sums to `epair`. The pair terms are the same and only their grouping changes, so the last bits of the energy move. The unbinned walk takes each atom's partners in index order, so each slice adds the density (line 277) and the force (line 383) of its central atom in the order the serial pass adds them, with the sign of each pair term flipped exactly, and they keep their bits. An energy that differs in its last bit can change which of two minima a step keeps, so from there the search runs another way, as the rows with more than one thread show. rgpot's `docs/orgmode/reference/sutton_chen.org` states this drift: the energy, summed per run, can move in the last bits. It is not the dropped-slice bug that 6917dea3 and efd0750f fix.

## Wall against GMIN

Wall per call is the total over the twelve seeds divided by charged calls; inside is the time in the potential. GMIN's rows are the corrected-file reruns in `../sc-n12/seeds.csv`. Anneal without OpenMP is rgpot efd0750f in `../sc-wall2/seeds.csv`, an earlier session. The wall ratio is anneal's wall per call over GMIN's. Mann-Whitney, two-sided with the tie correction and the continuity correction, compares the per-seed wall per call of the two codes, and the code ahead is the one lower in more of the 144 seed pairs; n.s. marks p >= 0.05.

| Threads | Cores | System | GMIN harness | Anneal inside (µs/call) | Anneal outside (µs/call) | Anneal wall (µs/call) | Anneal without OpenMP (µs/call) | GMIN wall (µs/call) | Wall ratio | Mann-Whitney p | Wall ahead |
|---|---|---|---|---|---|---|---|---|---|---|---|
| 1 | 0 | sc126-55 | a50f4b02 | 13.71 | 4.67 | 18.38 | 18.11 | 18.84 | 0.976 | 0.175 | anneal, n.s. |
| 1 | 0 | sc108-55 | a50f4b02 | 14.77 | 5.94 | 20.71 | 19.92 | 21.25 | 0.975 | 0.0404 | anneal |
| 1 | 0 | sc96-55 | a50f4b02 | 14.80 | 4.82 | 19.63 | 18.99 | 20.14 | 0.974 | 0.0226 | anneal |
| 1 | 0 | sc126-80 | a50f4b02 | 28.38 | 11.37 | 39.76 | 39.71 | 38.01 | 1.046 | 0.0404 | GMIN |
| 1 | 0 | sc108-80 | a50f4b02 | 31.02 | 11.41 | 42.43 | 40.23 | 42.87 | 0.990 | 0.26 | anneal, n.s. |
| 1 | 0 | sc96-80 | a50f4b02 | 30.88 | 11.11 | 41.99 | 40.16 | 40.51 | 1.037 | 0.00165 | GMIN |
| 8 | 0-7 | sc126-55 | a50f4b02 | 63.46 | 5.65 | 69.11 | 18.11 | 18.84 | 3.668 | 3.66e-05 | GMIN |
| 8 | 0-7 | sc108-55 | a50f4b02 | 71.24 | 6.43 | 77.66 | 19.92 | 21.25 | 3.654 | 3.66e-05 | GMIN |
| 8 | 0-7 | sc96-55 | a50f4b02 | 72.11 | 6.06 | 78.17 | 18.99 | 20.14 | 3.881 | 3.66e-05 | GMIN |
| 8 | 0-7 | sc126-80 | a50f4b02 | 162.72 | 13.18 | 175.90 | 39.71 | 38.01 | 4.628 | 3.66e-05 | GMIN |
| 8 | 0-7 | sc108-80 | a50f4b02 | 167.06 | 13.38 | 180.44 | 40.23 | 42.87 | 4.209 | 3.66e-05 | GMIN |
| 8 | 0-7 | sc96-80 | a50f4b02 | 164.10 | 12.48 | 176.59 | 40.16 | 40.51 | 4.359 | 3.66e-05 | GMIN |

With OMP_NUM_THREADS=1, anneal beats GMIN on wall per call on sc108-55 and sc96-55, at Mann-Whitney p < 0.05. With OMP_NUM_THREADS=1, anneal's lead on wall per call on sc126-55 and sc108-80 is not significant. With OMP_NUM_THREADS=1, GMIN beats anneal on wall per call on sc126-80 and sc96-80, at Mann-Whitney p < 0.05. With OMP_NUM_THREADS=8, GMIN beats anneal on wall per call on sc126-55, sc108-55, sc96-55, sc126-80, sc108-80 and sc96-80, at Mann-Whitney p < 0.05.

With OMP_NUM_THREADS=8 a call spends 4.63 to 5.73 times as long inside the potential as with one thread. `compute` takes `threadsRequested()` as its team for any number of atoms, while the neighbour-list build asks `teamSize` (`CppCore/rgpot/nlist/NeighbourList.cc`, lines 482-494), which keeps fewer than 8192 atoms on the calling thread because, in its own comment, a parallel region costs more than the search below that size. On these 55- and 80-atom clusters each call of the potential pays for the team.

## sc_wall_micro

Medians in µs per call over three repeats, as in `../sc-wall/README.md`; dE and dF are against the plain loop, the largest of the repeats.

| Threads | Cores | System | Entry | Kept | Kept, streaming | Plain | max abs dE | max abs dF |
|---|---|---|---|---|---|---|---|---|
| 1 | 0 | sc126-55 | 13.47 | 13.47 | 13.47 | 9.55 | 0 | 0 |
| 1 | 0 | sc108-55 | 15.07 | 15.15 | 15.03 | 10.14 | 0 | 0 |
| 1 | 0 | sc96-55 | 15.04 | 14.85 | 14.87 | 10.22 | 0 | 0 |
| 1 | 0 | sc126-80 | 28.55 | 28.07 | 28.15 | 19.97 | 0 | 0 |
| 1 | 0 | sc108-80 | 31.88 | 31.39 | 31.32 | 21.10 | 0 | 0 |
| 1 | 0 | sc96-80 | 31.00 | 30.72 | 31.08 | 21.48 | 0 | 0 |
| 8 | 0-7 | sc126-55 | 85.99 | 75.49 | 75.62 | 9.52 | 0 | 0 |
| 8 | 0-7 | sc108-55 | 100.95 | 104.61 | 104.78 | 10.11 | 0 | 0 |
| 8 | 0-7 | sc96-55 | 87.41 | 86.43 | 86.58 | 10.20 | 0 | 0 |
| 8 | 0-7 | sc126-80 | 167.00 | 166.24 | 166.03 | 19.98 | 0 | 0 |
| 8 | 0-7 | sc108-80 | 175.20 | 161.10 | 152.26 | 21.08 | 0 | 0 |
| 8 | 0-7 | sc96-80 | 171.91 | 165.42 | 165.92 | 21.44 | 0 | 0 |
