# Cell-list EAM

`EAMPot` evaluates the embedded-atom energy and forces for aluminium,
palladium and gold. The interaction is short ranged: the widest cutoff in
the element table is the aluminium value, 5.555 Angstrom.

## Neighbour list

The list is the shared neighbour layer (`rgpot::nlist::Neighbours`): a
linked-cell search with a Verlet skin. The skin is one fifth of the widest
cutoff (`kEamSkinOverCutoff`). A streaming walk uses no skin. A later call
on the same object reuses a stored list while every atom has moved by at
most half of the skin and the cell, the atom count and the neighbour mode
are unchanged. The species are not part of that test: the list holds
candidates out to the widest cutoff plus the skin for every atom. The
minimum image is one half-box correction after the coordinates have been
wrapped into the cell.

The neighbour mode defaults to `Mode::Auto`, which stores the list when
its index fits in memory and otherwise streams. `setNeighbourMode`
selects a mode. With a stored list and one thread, the force loop visits
each atom's partners in increasing index order. That is the same
arithmetic, on the same pairs, as the direct double loop, and the energy
and forces match that loop bit for bit. A streaming evaluation walks the
cells twice, once for the density and once for the forces, and stores no
pair list. It evaluates the pair terms from cubic Hermite tables
(`CppCore/rgpot/EAM/StreamSpline.hpp`) instead of `std::exp`, so it is
not bit-identical to the direct loop.

The stored list is the neighbours inside the cutoff plus the skin. On a
bulk aluminium crystal that is a few dozen neighbours per atom, not one
slot per atom pair.

## Threads

`OMP_NUM_THREADS` is read on each call. When it is unset, empty, not a
number or below 1, the call stays on one thread. A build without OpenMP
always uses one thread.

With a stored list, a value above 1 splits the atoms into contiguous
slices, one per requested thread. Each slice writes only the forces of
its own atoms. The first pass computes the density, the pair potential
to higher-index partners and that atom's own embedding term. A second
pass, which starts once every slice has finished the first, adds the
neighbours' embedding terms and the pair forces from lower-index
partners, using the scales stored on the first pass. Partial energies
are added in slice order, so a fixed thread count repeats bit for bit.
The threads the runtime starts take every slice, also when there are
fewer of them, as under `OMP_THREAD_LIMIT` or inside another OpenMP
region.

A streaming evaluation splits the cells into slices, one per requested
thread. Each slice keeps its own density and force rows, and the rows
are added in slice order.

Several threads are not bit-identical to one thread. Energy and every
force component stay within a relative tolerance of `1e-9`
(`kEamThreadedTolerance`). A component whose one-thread magnitude is below
1 is compared on an absolute scale of the same size. `EAMCellVerletTest`
checks the one-thread bits, the skin rebuild, the list length, and the
threaded tolerance, all on a stored list.

## Timings

FCC aluminium, lattice constant 4.05 Angstrom, on one Intel Xeon socket
with 8 cores and 1 thread per core. g++ 13.3.0, `-O3 -DNDEBUG`. The new
binary is also built with `-fopenmp`. Each quoted time is a median of
seven warm calls that reuse the neighbour list.

Develop still walked every pair and allocated a neighbour table of
`N * (N + 1)` integers, which is about 8 GB at 32 000 atoms and about
93 GB at 108 000.

| atoms | develop (s) | this branch (s) | speedup |
| ---: | ---: | ---: | ---: |
| 108 | 0.000122 | 0.000112 | 1.09 |
| 500 | 0.001270 | 0.000547 | 2.32 |
| 2048 | 0.016273 | 0.002130 | 7.64 |
| 6912 | 0.170350 | 0.007174 | 23.7 |
| 32000 |  | 0.032605 |  |
| 108000 |  | 0.109318 |  |

The first call at 6912 atoms dropped from 2.181 s to 0.0418 s. That call
used to build the full pair table and then ignore it.

Strong scaling of the stored list, 108 000 atoms:

| threads | median (s) | speedup |
| ---: | ---: | ---: |
| 1 | 0.109018 | 1.00 |
| 2 | 0.073623 | 1.48 |
| 4 | 0.038092 | 2.86 |
| 8 | 0.020201 | 5.40 |

At 32 000 atoms the same list reaches 5.99x on 8 threads (0.005425 s
against 0.032498 s). Two threads are 1.65x there. The earlier separate
density and force passes were 1.20x at two threads on 108 000 atoms.
The cross-potential tables, both neighbour modes, and the weak-scaling
rows are in `docs/neighbour_layer.md`.

## Reproduce

`EAM.cpp` needs the neighbour layer, the minimum image and the exact
multiply-add sources at link time.

<!-- rgpot-doc: syntax compiles for several seconds and runs the full benchmark -->
```bash
g++ -std=c++20 -O3 -DNDEBUG -fopenmp -I CppCore -I CppCore/rgpot/EAM \
  -DEAM_STANDALONE \
  CppCore/tests/eam_cell_bench.cc CppCore/rgpot/EAM/EAM.cpp \
  CppCore/rgpot/nlist/Neighbours.cc CppCore/rgpot/nlist/MinimumImage.cpp \
  CppCore/rgpot/simd/exact_fma_hw.cc CppCore/rgpot/simd/exp.cc \
  -o eam_cell_bench
./eam_cell_bench
```

Meson builds the same program as the `eam_cell_bench` benchmark.
