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
least 8192 atoms. A pair list splits the cells into contiguous ranges
when the cell is unsheared and at least two search radii across every
periodic axis. A sheared or narrower cell may need several images of a
pair, so the pair list then splits the atoms and each slice runs the
one-thread walk; a cluster list always splits the atoms. Every thread
count stores the same pairs and shifts. The results are concatenated in
slice order, so a cluster list keeps the one-thread order; a pair list
may store its rows in a different order. Vesin merges per-thread lists
without promising an order. Below 512 atoms a pair list is a direct loop
over atom pairs, without cells.

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

`OMP_NUM_THREADS` above 1 splits the work into that many slices. Most
two-body walks use `forCellTeam`, which gives each slice a contiguous
range of cells. Stored EAM, LJCluster and the three-body potentials
split atoms into contiguous slices. `forEachSlice`, in
`CppCore/rgpot/Slices.hpp` with every other OpenMP region, hands every
slice to the threads the runtime starts, also when there are fewer of
them, as under `OMP_THREAD_LIMIT` or inside another OpenMP region.
Private forces are added in slice order, so a fixed thread count repeats
bit for bit and stays within `1e-9` of one thread. One thread keeps the
serial pair order.

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
