# Shared neighbour layer

The public query is `rgpot::nlist::Query`, `Builder`, `Table` and `Walk`.
A query is either a pair shell or a cluster of four. `build` stores the
pairs. `Walk` streams them. Classical potentials still use
`rgpot::nlist::Neighbours` for a minimum-image force walk. Metatomic and
UMA are adapters: they ask this query for pairs and then copy into the
layout their model expects. The calls are in `docs/orgmode/tutorials/neighbour-list.org`.

## What was taken, and why

The stored table is a structure of arrays: central index, partner index,
three integer cell shifts, three vector components, one distance, and a
mask. That is the layout a SIMD kernel or a GPU fill can read without
gathering fields out of a struct. LAMMPS Kokkos already stores neighbours
as a Kokkos view of indices plus a per-atom count
([neigh_list_kokkos.h](https://raw.githubusercontent.com/lammps/lammps/develop/src/KOKKOS/neigh_list_kokkos.h)).
Vesin returns parallel arrays of pairs, shifts, distances and vectors, and
records which device owns them (`third_party/vesin/vesin.h`,
`VesinNeighborList`). Freud stores query index, point
index and a flat distance array
([NeighborList.h](https://raw.githubusercontent.com/glotzerlab/freud/main/freud/locality/NeighborList.h)).
An array of pair structs would fight those consumers.

Half and full lists follow vesin's rule, including the self-image test
that drops a negative shift of an atom with itself
(`third_party/vesin/vesin-single-build.cpp`, the `options.full` branch).
A half list is the default in LAMMPS
([neigh_request.h](https://raw.githubusercontent.com/lammps/lammps/develop/src/neigh_request.h)
sets `half` and can derive a half list from a full one) and in HOOMD
([NeighborList.h](https://raw.githubusercontent.com/glotzerlab/hoomd-blue/trunk/hoomd/md/NeighborList.h),
`half` stores `i < j`, `full` stores every neighbour). ASE's `bothways`
flag is the same choice
([neighborlist.py](https://gitlab.com/ase/ase/-/raw/master/ase/neighborlist.py)).
Cluster4 is the extra format. GROMACS does not store one pair per atom:
an i-cluster of atoms faces packed j-clusters, with a bitmask of which
lanes interact
([pairlist.h](https://gitlab.com/gromacs/gromacs/-/raw/main/src/gromacs/nbnxm/pairlist.h),
`nbnxm_cj_t` and `NBNXM_INTERACTION_MASK_ALL`). Cluster4 is that idea at
the smallest width that is still a mask: four consecutive central atoms,
one partner, one shift, and four bits. The vector and the distance belong
to the lowest set bit. A plain half list cannot feed that kernel without
a regroup.

Strict means distance strictly below the cutoff, which is vesin's
`distance2 < cutoff2` and torch-nl's `strict_nl` mask
([neighbor_list.py](https://raw.githubusercontent.com/felixmusil/torch_nl/main/torch_nl/neighbor_list.py)).
Loose means distance below cutoff + skin. The skin is the Verlet buffer.
HOOMD rebuilds when a particle moves more than half the buffer. ASE
rebuilds when the largest squared displacement exceeds `skin**2`, which
is twice that far and can miss a pair. The builder here uses half the
skin, the distance that keeps every pair that was inside the cutoff.

A stream does not allocate the pairs of the whole system. `Walk::pull`
gathers one central atom, or one cluster of four, at a time, writes at
most the length of the caller buffer, and the caller may stop. JAX-MD cannot do this: a jit-compiled list has a
static capacity, `neighbor_fn.allocate` sizes it, and `did_buffer_overflow`
tells the caller to allocate again
([partition.py](https://raw.githubusercontent.com/jax-md/jax-md/main/jax_md/partition.py)).
That is the right answer when the consumer is XLA. It is the wrong answer
when the consumer is a force kernel that can run during the walk. The C
function `rgpot_nlist_build` still reports the full count when the caller
buffer is short, so a fixed buffer can grow the way JAX-MD does, without
making that the only path. The public query has no automatic mode: the
caller picks `build`, `Builder` or `Walk`. The automatic choice between
storing and streaming is `Mode::Auto` in `rgpot::nlist::Neighbours`,
described below.

Cell shifts are integers along the three cell vectors, and the vector is
`r_j - r_i + S H`, which is the sentence in vesin's `VesinNeighborList`
comment. Partial periodicity leaves the shift on an open axis at zero.
The search bound is the cutoff divided by the cell height perpendicular
to each face, so a small triclinic cell returns every image inside the
sphere, not one minimum image. ArborX's half neighbour list is a
bounding-volume hierarchy query into Kokkos offsets and indices
([ArborX_NeighborList.hpp](https://raw.githubusercontent.com/kokkos/ArborX/master/src/spatial/detail/ArborX_NeighborList.hpp)).
cuNSearch keeps a spatial grid on the device and rebuilds it with
`update_point_sets`
([cuNSearch.h](https://raw.githubusercontent.com/InteractiveComputerGraphics/cuNSearch/master/include/cuNSearch.h)).
Neither is a cell list, and neither speaks triclinic shifts. This build
does not fill a GPU. `Table::device` is 0. The arrays are the contract a
later device fill would write.

`build` uses threads when `OMP_NUM_THREADS` is above 1 and there are at
least 8192 atoms, the rule every potential call follows (`teamSize` in
`CppCore/rgpot/Slices.hpp`). A pair list splits the cells into
contiguous ranges when the cell is unsheared and at least two search
radii across every periodic axis. A sheared or narrower cell may need
several images of a pair, so the pair list then splits the atoms and
each slice runs the one-thread walk; a cluster list always splits the
atoms. Every thread count stores the same pairs and shifts. The results
are concatenated in slice order, so a cluster list keeps the one-thread
order; a pair list may store its rows in a different order. Vesin merges
per-thread lists without promising an order. Below 512 atoms a pair list
is a direct loop over atom pairs, without cells.

Matscipy's `neighbour_list` returns whichever of pairs, distances and
vectors the caller names in a string
([neighbours.py](https://raw.githubusercontent.com/libAtoms/matscipy/master/matscipy/neighbours.py)).
`Query::fields` is that choice with a fixed set of arrays instead of a
format string.

Metatomic is not the layout of this table. Its labels are one row of
`(first_atom, second_atom, cell_shift_a, cell_shift_b, cell_shift_c)` and
its values are a tensor of shape `(pairs, 3, 1)`. `MetatomicPot` reads
`full_list` and `strict` off the model's request, calls `build`, wraps
the vector array with `torch::from_blob`, and copies only the five label
columns. UMA's edge list does the same and keeps its own sort.

## C ABI and Cap'n Proto

`rgpot_nlist.h` is version 4 (`rgpot_nlist_abi_version`). Every
caller-allocated struct starts with `bytes`. `rgpot_nlist_request_init`
fills a half pair list, periodic boundaries and every geometry column.
`rgpot_nlist_build` and `rgpot_nlist_walk_pull` write caller-owned
structure-of-arrays buffers. `rgpot_nlist_strerror` names a code. A
cluster with a half shell is `RGPOT_NLIST_SHAPE`.

The rgpot-local schema diff `CppCore/rgpot/rpc/local/neighbour-chunk.diff`
adds `NeighborRequest`, `NeighborList`, an optional `ForceInput.neighbors` at
ordinal 9, an optional `PotentialResult.neighborList` at ordinal 28, and
`Potential.neighbours` at ordinal 13 to the `Potentials.capnp` that rgpot
compiles and installs; the vendored copy is the pinned upstream file. Older fields are
unchanged. `NeighborRequest.mode = 2` asks for a stream: a chunk sets
`NeighborList.more` and `next`, and the client repeats the call with
`cursor` instead of receiving every pair in one message. `mode = 1` is
one stored chunk. potserv does not implement `Potential.neighbours` and
does not fill `PotentialResult.neighborList`.

<!-- rgpot-doc: run link=rgpot -->
```cpp
#include "rgpot/nlist/NeighbourList.hpp"

#include <cstddef>

int main() {
  const double positions[] = {0.0, 0.0, 0.0, 1.2, 0.0, 0.0, 0.0, 1.3, 0.0};
  const std::size_t n = 3;
  const double box[9] = {20, 0, 0, 0, 20, 0, 0, 0, 20};

  rgpot::nlist::Query query;
  query.cutoff = 5.0;
  query.skin = 0.5;
  query.shape = rgpot::nlist::Pairs{rgpot::nlist::Shell::Half};
  query.cut = rgpot::nlist::Cut::Loose;
  rgpot::nlist::Builder builder;
  builder.update(positions, n, box, query);
  rgpot::nlist::View pairs = builder.table().view();
  // pairs.first[k], pairs.second[k], pairs.shift[3 * k + axis]
  return pairs.first.size() == 3 ? 0 : 1;
}
```

EAM, EAMTable, Lennard-Jones, LJCluster, Sutton-Chen, Stillinger-Weber,
Tersoff, Morse and ZBL take candidate pairs from
`rgpot::nlist::Neighbours`.

The layer bins atoms into cells whose edge is at least the cutoff plus
the skin.
`Round` is the orthorhombic truncation used by the pair kernels. `Half`
is one correction about the half-box, which is what EAM uses after
wrapping coordinates into the cell. `General`, and any sheared cell,
calls `MinimumImage`.

`Mode::Stored` keeps a Verlet list and reuses it while every atom has moved by at most half the skin.
The list holds partner indices and each walk takes the image from the current coordinates, so the move is measured to the nearest periodic image of the reference position: an atom that the caller wraps back through a face of the cell has moved by its step, not by a cell vector, and keeps the list.
The bins are redone after such a wrap.
A change of the cell, of the atom count or of the spec rebuilds the list.
`Builder`, whose table stores the image shifts, still rebuilds when an atom wraps.
`Mode::Stream` walks the 27 neighbouring cells and stores no pair list.
The bins are reused only when the geometry is identical.
`Mode::Auto` stores when the candidate index array fits in `MemAvailable` and in a 32-bit offset, and streams otherwise.
`Mode::Stored` makes the same test and also falls back to a stream when the index does not fit.
When `MemAvailable` cannot be read, both modes store.
There is no cutoff on the atom count.

An infinite cutoff or skin has no search radius, in any mode.
The update bins nothing and stores nothing, and `allPairs()` reports it with `streaming()` set.
Each walk takes the pairs straight from the coordinates, `i` ascending and then `j` ascending, which is the order of the one bin a search wider than the system gives, and folds each pair with the image of the spec.
No move takes a pair out of such a search, so the update is reused for any coordinates while the atom count, the cell and the spec stay the same.
A team walk with no bins gives each slice a contiguous range of atoms; with more slices than atoms, some ranges are empty.
Sutton-Chen with `cutoff <= 0` searches this way.
The public `Query` still rejects a cutoff that is not finite.

A potential call on fewer than 8192 atoms (`kTeamMinAtoms`) runs on the
calling thread, whatever `OMP_NUM_THREADS` says, and keeps the serial
pair order. `teamSize` in `CppCore/rgpot/Slices.hpp` gives the slice
count of every call, and `teamsOpened` counts the regions a thread has
opened, so a test can see that a small call opened none.

From 8192 atoms up a result has the bits of one thread on any
`OMP_NUM_THREADS`. Each atom keeps the energy and strain of its pairs
with higher-index partners, and the atoms are added in index order. The
pair potentials (`nlist::sumPairs` in `CppCore/rgpot/nlist/PairSum.hpp`),
Sutton-Chen and EAM give each slice a contiguous range of atoms, and
`forRowTeam` walks each atom's whole row on the slice that owns it, so
the force on an atom has one writer and sums in the order of the
one-thread walk. In stored mode and in an unbounded search, a partner
below the atom takes the negated vector of the half walk from that
partner, so both visits of a pair agree even where two images tie. One
thread on a stored list or an unbounded search walks the half list. On a
stored list with at least `kTeamMinAtoms` atoms and at least 4096
candidate upper pairs (`kMinParallelHalfPairs`), `MorsePot` evaluates
each upper pair once in a two-phase OpenMP region (`forEachSlicePhase`),
batching `simd::exp_fixed_n` and partitioning atoms by pair count on
8-atom boundaries before reducing forces in partner index order; sparser
stored lists stay on one thread. `fastTiles` runs
only when `streaming()` holds, so `forCellTeam` on a stored list walks
`walkStored` instead of rebuilding a 27-cell tile stencil. A
stream walks whole rows on any thread count, so on one thread a two-body
potential evaluates each pair twice where a stored list evaluates it
once. Stillinger-Weber and Tersoff
(`CppCore/rgpot/CentreTerms.hpp`) sum the terms of each centre into one
force on the centre and on each partner, an energy and a strain, and add
the centres in index order. `forEachSlice`, with every other OpenMP
region, hands every slice to the threads the runtime starts, also when
there are fewer of them, as under `OMP_THREAD_LIMIT` or inside another
OpenMP region, so those teams give the same bits too.

Stillinger-Weber and Tersoff use this layer for `Auto` as well as
`Stored` and `Stream`. Bonds are gathered for one atom at a time, sorted
by neighbour index, then discarded. That order is not the old
`orderedPairs` walk, so one thread is not bit-identical to it. On a
distorted 216-atom diamond the largest absolute difference was
`3.4e-13` for Stillinger-Weber and `3.3e-16` for Tersoff. Morse and ZBL
use the same half-list. The old `PairListCache` and its private cell
walk are gone.

A streaming three-body call no longer reserves 32 bonds on every atom.
The extra memory is the bins, the coordinates, and one scratch row.

## Timings

One Intel Xeon socket, 8 cores, 1 thread per core. g++ 13.3.0,
`-O3 -DNDEBUG -fopenmp`. Each time is the median of seven warm calls
after two discarded calls. The peak resident set below is from a
separate process per potential, one million-atom streaming call:

| potential | atoms | peak RSS (kB) |
| --- | ---: | ---: |
| EAM | 1048576 | 139796 |
| Lennard-Jones | 1048576 | 115204 |
| Sutton-Chen | 1048576 | 131268 |
| Stillinger-Weber | 1000000 | 112596 |
| Tersoff | 1000000 | 119472 |

Lennard-Jones streaming applies the pair term inside the cell walk and
does not store a half-list. Sutton-Chen streaming keeps the per-atom
density between two cell walks and does not store partner indices. The
walk emits each unordered pair once.

Three stream layouts were timed in one process each, cutoff 3, FCC with
lattice `sqrt(2)`. The time is the median of seven calls after one
discarded call.

| layout | 108000 atoms (s) | RSS (kB) | 1048576 atoms (s) | RSS (kB) |
| --- | ---: | ---: | ---: | ---: |
| partner indices from the density walk | 0.985956 | 134360 | 9.392517 | 917680 |
| second walk, indices kept for 4096 atoms | 1.131455 | 21868 | 10.056180 | 135956 |
| second walk, density only | 1.024955 | 17320 | 9.365840 | 131268 |

The density-only walk matches the index store's time and stays with the
atom and cell arrays. The 4096-atom tile is a little slower. The index
store is the 918 MB resident set. In the serial stream table, the
Sutton-Chen entries at 108,000 and 1,048,576 atoms are this density-only
remeasurement. The smaller Sutton-Chen stream rows, and the
strong-scaling row for that stream, are the earlier index-store run.

### Serial, stored list

| atoms | EAM (s) | LJ (s) | Sutton-Chen (s) | atoms | SW (s) | Tersoff (s) |
| ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 108 | 0.000111 | 0.000038 | 0.000260 | 64 | 0.000030 | 0.000041 |
| 500 | 0.000545 | 0.000171 | 0.002856 | 216 | 0.000100 | 0.000135 |
| 2048 | 0.002130 | 0.000704 | 0.011370 | 1000 | 0.000446 | 0.000621 |
| 6912 | 0.007063 | 0.002394 | 0.037197 | 4096 | 0.001804 | 0.002553 |
| 32000 | 0.032551 | 0.011227 | 0.169652 | 13824 | 0.006158 | 0.008707 |
| 108000 | 0.110193 | 0.038031 | 0.561026 |  |  |  |

### Serial, streaming walk

| atoms | EAM (s) | LJ (s) | Sutton-Chen (s) | atoms | SW (s) | Tersoff (s) |
| ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 108 | 0.000227 | 0.000166 | 0.000302 | 64 | 0.000054 | 0.000097 |
| 500 | 0.003304 | 0.003621 | 0.005873 | 216 | 0.000260 | 0.000287 |
| 2048 | 0.013226 | 0.040806 | 0.085449 | 1000 | 0.001137 | 0.001294 |
| 6912 | 0.038887 | 0.106809 | 0.369985 | 4096 | 0.004749 | 0.005300 |
| 32000 | 0.161625 | 0.408769 | 1.577800 | 13824 | 0.015888 | 0.017356 |
| 108000 | 0.501919 | 1.443270 | 1.024955 |  |  |  |
| 1048576 | 4.919678 | 12.503108 | 9.365840 | 1000000 | 1.081986 | 1.141701 |

Lennard-Jones and Sutton-Chen at one million atoms are an FCC crystal of
64 repeats (1,048,576 atoms). Stillinger-Weber and Tersoff are a diamond
of 50 repeats (1,000,000 atoms).

### Strong scaling

Metals are 32,000 atoms. Diamond potentials are 13,824 atoms. Speedup is
against the one-thread time of that mode.

| potential | mode | 1 | 2 | 4 | 8 |
| --- | --- | ---: | ---: | ---: | ---: |
| EAM | stored | 0.032498 | 0.019711 (1.65x) | 0.010249 (3.17x) | 0.005425 (5.99x) |
| EAM | stream | 0.156389 | 0.081042 (1.93x) | 0.042261 (3.70x) | 0.023984 (6.52x) |
| LJ | stored | 0.011644 | 0.006217 (1.87x) | 0.003582 (3.25x) | 0.002460 (4.73x) |
| LJ | stream | 0.393600 | 0.202438 (1.94x) | 0.104098 (3.78x) | 0.053970 (7.29x) |
| Sutton-Chen | stored | 0.171792 | 0.115669 (1.49x) | 0.058216 (2.95x) | 0.029906 (5.74x) |
| Sutton-Chen | stream | 0.810416 | 0.479725 (1.69x) | 0.281400 (2.88x) | 0.165698 (4.89x) |
| SW | stored | 0.006136 | 0.003248 (1.89x) | 0.001723 (3.56x) | 0.001037 (5.92x) |
| SW | stream | 0.015693 | 0.008133 (1.93x) | 0.004148 (3.78x) | 0.002283 (6.87x) |
| Tersoff | stored | 0.008756 | 0.004402 (1.99x) | 0.005016 (1.75x) | 0.002909 (3.01x) |
| Tersoff | stream | 0.016700 | 0.008453 (1.98x) | 0.005964 (2.80x) | 0.003416 (4.89x) |

Stored EAM at 108,000 atoms: 0.107392 s, 0.074392 s (1.44x), 0.037288 s
(2.88x), 0.020593 s (5.22x). Streaming EAM at the same size: 0.501968 s,
0.260581 s (1.93x), 0.132694 s (3.78x), 0.068205 s (7.36x). Two threads
used to be slower than one because the stream recomputed the density
exponential in a second pass. The stream now makes two cell walks, one
for the density and one for the forces, with the pair terms read from
tabulated splines, and each slice keeps its own density and force rows.
Tersoff's parallel stress used to land on one shared array. It now uses
the private buffer. At 13,824 atoms the stored list reaches 3.01x on 8
threads and the stream 4.89x. At 1,000,000 atoms the stream median on 8
threads is 0.236637 s, against 1.141701 s on one thread (4.82x).

### Weak scaling

Metal cells are 6x6x6, 6x6x12, 6x12x12 and 12x12x12 conventional FCC
cells (864, 1,728, 3,456 and 6,912 atoms) on 1, 2, 4 and 8 threads.
Diamond repeats are 4, 5, 6 and 8 (512, 1,000, 1,728 and 4,096 atoms).

| potential | mode | 1 thread | 2 | 4 | 8 |
| --- | --- | ---: | ---: | ---: | ---: |
| EAM | stored | 0.000919 | 0.001159 | 0.001191 | 0.001426 |
| EAM | stream | 0.006408 | 0.012207 | 0.011236 | 0.010896 |
| LJ | stored | 0.000296 | 0.000392 | 0.000428 | 0.000481 |
| LJ | stream | 0.008052 | 0.010016 | 0.012411 | 0.015674 |
| Sutton-Chen | stored | 0.004784 | 0.006297 | 0.006353 | 0.006476 |
| Sutton-Chen | stream | 0.016872 | 0.023663 | 0.033095 | 0.050187 |
| SW | stored | 0.000231 | 0.000241 | 0.000221 | 0.000301 |
| SW | stream | 0.000748 | 0.000584 | 0.000528 | 0.001053 |
| Tersoff | stored | 0.000318 | 0.000521 | 0.000834 | 0.001916 |
| Tersoff | stream | 0.000672 | 0.000769 | 0.000833 | 0.002062 |

### Thread count

`thread_count_bench` times the force call of each potential on a free
cluster: the N sites of its lattice nearest one site, with Gaussian noise
of 0.02 on each coordinate, over 16 frames in turn. LJ, with a cutoff of
2.5, Morse, EAM, SW and Tersoff put the cluster in the middle of a cell
with vacuum on every side. The LJ cluster, ZBL and Sutton-Chen take no
cell, and Sutton-Chen has no cutoff, as a global optimisation calls it.
The neighbour mode is `Auto`, which stored the list at every size in the
first table. A time is the median of five runs of at least 0.2 s, in
microseconds per call. Both builds are the Meson release build of the
pixi `devbld` environment with GCC 14.3.0, on the eight cores above under
`taskset -c 0-7`. Before is 47c46f97.

| potential | atoms | 1 thread, before | 1 thread, after | 8 threads, before | 8 threads, after |
| --- | ---: | ---: | ---: | ---: | ---: |
| Sutton-Chen | 38 | 6.9 | 6.3 | 26.7 | 6.3 |
| Sutton-Chen | 55 | 14.1 | 13.0 | 41.9 | 13.0 |
| Sutton-Chen | 80 | 33.3 | 27.1 | 90.8 | 27.1 |
| Sutton-Chen | 500 | 1136 | 1021 | 3236 | 1017 |
| Sutton-Chen | 4000 | 85998 | 82619 | 309289 | 82530 |
| Sutton-Chen | 10000 | 547670 | 541027 | 2113282 | 156037 |
| LJ | 38 | 10.1 | 8.8 | 23.6 | 8.7 |
| LJ | 55 | 20.4 | 17.3 | 39.1 | 17.3 |
| LJ | 80 | 38.5 | 33.7 | 60.9 | 33.7 |
| LJ | 500 | 495 | 463 | 484 | 463 |
| LJ | 4000 | 5054 | 4651 | 6829 | 4651 |
| LJ | 10000 | 13139 | 12581 | 20444 | 3701 |
| LJ cluster | 38 | 4.5 | 4.0 | 5.4 | 4.0 |
| LJ cluster | 55 | 9.5 | 8.5 | 7.8 | 8.5 |
| LJ cluster | 80 | 18.0 | 16.2 | 9.9 | 16.2 |
| LJ cluster | 500 | 220 | 212 | 78.5 | 212 |
| LJ cluster | 4000 | 2301 | 2133 | 554 | 2127 |
| LJ cluster | 10000 | 6040 | 5528 | 1388 | 1639 |
| Morse | 38 | 21.1 | 20.0 | 26.8 | 19.9 |
| Morse | 55 | 42.7 | 40.2 | 51.8 | 40.2 |
| Morse | 80 | 82.0 | 80.3 | 94.4 | 77.8 |
| Morse | 500 | 1241 | 1192 | 1171 | 1190 |
| Morse | 4000 | 13814 | 13154 | 15035 | 13196 |
| Morse | 10000 | 36873 | 36139 | 41316 | 10886 |
| ZBL | 38 | 2.9 | 2.7 | 8.3 | 2.7 |
| ZBL | 55 | 3.9 | 3.7 | 11.0 | 3.7 |
| ZBL | 80 | 6.2 | 5.8 | 16.3 | 5.8 |
| ZBL | 500 | 43.3 | 40.8 | 236 | 40.9 |
| ZBL | 4000 | 386 | 366 | 10937 | 365 |
| ZBL | 10000 | 996 | 1001 | 66237 | 361 |
| EAM | 38 | 17.0 | 16.8 | 9.9 | 16.8 |
| EAM | 55 | 29.8 | 28.9 | 14.8 | 28.8 |
| EAM | 80 | 51.6 | 48.8 | 30.0 | 48.6 |
| EAM | 500 | 459 | 455 | 178 | 455 |
| EAM | 4000 | 4110 | 4042 | 1411 | 4041 |
| EAM | 10000 | 10625 | 10339 | 3439 | 3337 |
| SW | 38 | 11.2 | 11.1 | 6.7 | 11.1 |
| SW | 55 | 16.1 | 16.0 | 8.2 | 16.0 |
| SW | 80 | 25.8 | 25.7 | 10.8 | 25.7 |
| SW | 500 | 213 | 216 | 60.7 | 216 |
| SW | 4000 | 1997 | 2007 | 469 | 2006 |
| SW | 10000 | 5165 | 5375 | 1236 | 1115 |
| Tersoff | 38 | 16.5 | 16.0 | 10.0 | 16.0 |
| Tersoff | 55 | 22.8 | 22.2 | 11.7 | 22.2 |
| Tersoff | 80 | 37.6 | 36.6 | 16.9 | 36.6 |
| Tersoff | 500 | 272 | 266 | 118 | 266 |
| Tersoff | 4000 | 2420 | 2371 | 818 | 2369 |
| Tersoff | 10000 | 6174 | 6284 | 2032 | 1274 |

Below 8192 atoms every call now runs on the calling thread, with the
bits the old build gave on one thread, at most 1.2% slower (SW at 500
atoms) and up to 19% faster (Sutton-Chen at 80 atoms). The old build
started a team on any call. In this run, on 8 threads Sutton-Chen took
2.7 to 3.9 times its one-thread time at every size, the LJ and Morse
teams were slower than one thread at five of the six sizes, and the ZBL
team at all six. Three more runs of Sutton-Chen at 38 to 500 atoms put
the old 8-thread call at 2.7 to 6.3 times one thread, 2.7 to 4.4 times
at 55 and 80 atoms, while the new times stayed within 1.2% of the table.

EAM, the LJ cluster, SW and Tersoff split their atoms into slices, and
on 8 threads those teams beat one thread from 38 or 55 atoms up. Below
8192 atoms they now run on one thread, up to 4.3 times slower than their
old teams: SW at 4000 atoms takes 2006 µs per call, where its team took
469 µs. That is the cost of one threshold for every call. The sums do not
depend on it, so a lower threshold for these potentials would change
only their speed.

At 10000 atoms 8 threads are 2.8 to 4.9 times faster than one, and
faster than the old team for every potential but the LJ cluster, 18%
slower: its team now evaluates each pair once from each end, where its
old slices evaluated a pair once and added both forces to private
buffers. One thread costs at most 4% more than before, for SW, whose
centres now sum their terms before adding them.

On 8 threads the old build moved the energy on 10 to 16 of the 16 frames
for every potential but ZBL, by up to 2.06e-4 epsilon for the reduced
Sutton-Chen at 10000 atoms, and moved the forces of every potential but
Sutton-Chen and ZBL. The new build gives the bits of one thread on every
frame. At 10000 atoms one thread now adds the same per-atom sums as a
team, so its bits are not those of the old build.

At 100000 atoms:

| potential | 1 thread, before | 1 thread, after | 8 threads, before | 8 threads, after |
| --- | ---: | ---: | ---: | ---: |
| LJ | 144577 | 139135 | 192722 | 36995 |
| LJ cluster | 8773537 | 14933502 | 1872959 | 1909984 |
| Morse | 409610 | 402374 | 406363 | 109952 |
| EAM | 113326 | 112897 | 32373 | 32774 |
| SW | 55391 | 57159 | 12427 | 10341 |
| Tersoff | 64455 | 64937 | 24281 | 10983 |

But for the LJ cluster, one thread costs at most 3.2% more than before
(SW), and 8 threads are 3.4 to 5.9 times faster than one: up to 5.2
times faster than the old team, and 1.2% slower for EAM. The LJ cluster
takes no cell, so the layer bins a box a million wide, which falls back
to one bin. At this size its candidates do not fit an index, so both
builds stream that bin and test every pair on each call. ZBL takes no
cell either and was not timed here.

A stream walks each atom's whole row on any thread count, so one thread
has the bits of a team, and a two-body potential evaluates each pair
from both ends. At 100000 atoms that makes the LJ cluster 1.70 times
slower on one thread, and 2% slower on 8. In `Stream` mode at 10000
atoms:

| potential | 1 thread, before | 1 thread, after | 8 threads, before | 8 threads, after |
| --- | ---: | ---: | ---: | ---: |
| LJ | 30466 | 49931 | 18115 | 12004 |
| Morse | 32512 | 50444 | 42515 | 8325 |
| EAM | 36088 | 50280 | 12000 | 11514 |
| SW | 13560 | 13656 | 2168 | 2154 |
| Tersoff | 13873 | 13969 | 2563 | 2215 |

One thread on a stream is 1.39 to 1.64 times slower for EAM, Morse and
LJ, which used to walk half pairs. SW and Tersoff gather each centre's
bonds either way and stay within 1%. On 8 threads none is slower than
its old team.

## Reproduce

`benches/neighbour_api_bench.py` times the public query against vesin
0.6.2, ASE 3.29.0 and matscipy 1.3.1. FCC copper uses lattice 3.615 and
a strict cutoff of 5. A 30-cube cell is 108,000 atoms and a 64-cube
cell is 1,048,576. The triclinic cell is a 4×4×4 grid in

<!-- rgpot-doc: skip cell matrix copied from benches/neighbour_api_bench.py, not a program -->
```text
[[6.0, 0.0, 0.0], [1.2, 5.4, 0.0], [0.3, 0.7, 5.8]]
```

Skin is zero. Each library is a fresh process. The time is the median
of seven calls after two discarded calls, and the resident set is
`VmHWM`. vesin is `NeighborList.compute` with `n_threads` set and
`quantities="ijSDd"`. ASE's full list is `neighbor_list("ijSD")`. ASE's
half list is a new `NeighborList(..., bothways=False)` on every call.
matscipy's `neighbour_list("ijSD")` is a full list and has no half
switch. ASE and matscipy take no thread count; both thread columns were
still run.

On the triclinic cell the half list is 5,952 pairs and the full list is
11,904, in rgpot, vesin and ASE. At 108,000 atoms the half list is
2,268,000 pairs and the full list is 4,536,000. At 1,048,576 atoms the
half list is 22,020,096 pairs and the full list is 44,040,192.

| system | list | threads | rgpot (s) | vesin (s) | ASE (s) | matscipy (s) |
| --- | --- | ---: | ---: | ---: | ---: | ---: |
| triclinic | half | 1 | 0.000545 | 0.000414 | 0.005154 |  |
| triclinic | half | 8 | 0.000400 | 0.000415 | 0.005157 |  |
| triclinic | full | 1 | 0.001044 | 0.000641 | 0.035869 | 0.001769 |
| triclinic | full | 8 | 0.000625 | 0.000622 | 0.036025 | 0.001730 |
| 108000 | half | 1 |  | 0.145285 | 2.537847 |  |
| 108000 | half | 8 |  | 0.081414 | 2.509724 |  |
| 108000 | full | 1 |  | 0.235651 | 8.686973 | 0.634442 |
| 108000 | full | 8 |  | 0.158852 | 8.548236 | 0.637582 |
| 1048576 | half | 1 |  | 1.487588 | 25.746639 |  |
| 1048576 | half | 8 |  | 0.794806 | 25.911394 |  |
| 1048576 | full | 1 |  | 2.507937 |  | 6.392880 |
| 1048576 | full | 8 |  | 1.463148 |  | 6.618861 |

These rgpot timings are from an earlier version of the public query
that visited every atom pair; `build` now walks the cell list. One
half-list build at 108,000 atoms was still inside its first call when 180 seconds
elapsed, so the larger rgpot cells were not started. ASE's full list
at 1,048,576 atoms was killed by the kernel at 48,137,444 kB and
48,142,196 kB anonymous resident set, once per thread column, before
it printed a time.

Resident set, kB, for the runs that printed one:

| system | list | threads | rgpot | vesin | ASE | matscipy |
| --- | --- | ---: | ---: | ---: | ---: | ---: |
| triclinic | half | 1 | 5144 | 33408 | 70252 |  |
| triclinic | half | 8 | 6060 | 33460 | 70388 |  |
| triclinic | full | 1 | 6192 | 34220 | 89840 | 31760 |
| triclinic | full | 8 | 7356 | 34180 | 89980 | 31696 |
| 108000 | half | 1 |  | 316020 | 485528 |  |
| 108000 | half | 8 |  | 426784 | 485776 |  |
| 108000 | full | 1 |  | 590764 | 5947312 | 529248 |
| 108000 | full | 8 |  | 802128 | 5947412 | 529368 |
| 1048576 | half | 1 |  | 2786588 | 4144192 |  |
| 1048576 | half | 8 |  | 3524116 | 4144092 |  |
| 1048576 | full | 1 |  | 5452884 |  | 4872420 |
| 1048576 | full | 8 |  | 6641784 |  | 4872272 |

`classical_scaling_bench` prints the serial, strong and weak tables.
`classical_scaling_bench rss EAM` (and `LJ`, `SuttonChen`, `SW`,
`Tersoff`) prints the peak resident set of one million-atom streaming
call in that process. Meson builds the same program as the
`classical_scaling_bench` benchmark.

`thread_count_bench` prints the thread-count tables, a line per
potential, size and thread count, and Meson builds it as the
`thread_count_bench` benchmark. The old build is 47c46f97 with
`CppCore/tests/thread_count_bench.cc` and its Meson target added. The
first table is, on each build,

<!-- rgpot-doc: syntax needs a Meson build and runs for several minutes -->
```bash
for pot in sc lj ljcluster morse zbl eam sw tersoff; do
  taskset -c 0-7 build/CppCore/thread_count_bench --pots "$pot" \
    --sizes 38,55,80,500,4000,10000 --threads 1,8
done
```

`--sizes 100000` gives the second table, and `--mode stream --sizes
10000` the third. Each line also counts, against the first thread count,
the frames whose energy bits differ and the force components whose bits
differ, and hashes the bits of every energy and force, so two builds can
be compared.

`morse_omp_bench` (`CppCore/tests/morse_omp_bench.cc`, Meson benchmark
`morse_omp_bench`) runs a smoke check for `MorsePot` at 1, 2, 4, 8 and
16 threads on a 6×6×6 Cu vacancy (863 atoms, below `kTeamMinAtoms`), a
14×14×14 Cu vacancy (10,975 atoms, stored list) and the 10,000-atom Pt
cluster, checking bit-for-bit agreement against 1 thread. On the 8-core
Xeon host above (GCC 13.3.0, release build), evaluating each stored
upper pair once and batching `simd::exp_fixed_n` moves the 14×14×14 Cu
vacancy median from 10,795 / 11,317 / 5,771 µs at 1 / 2 / 4 threads to
5,591 / 3,459 / 1,789 µs (1.62× at 2 threads, 3.12× at 4 threads), the
10,000-atom Pt cluster from 36,007 / 45,698 / 23,290 µs to 25,026 /
17,672 / 9,185 µs (1.42× at 2 threads, 2.72× at 4 threads), and the
863-atom Cu vacancy from 805 µs to 457 µs.
