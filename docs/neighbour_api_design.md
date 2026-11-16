<!-- rgpot-foreign: python/vesin/vesin/_neighbors.py -->
# Neighbour query, four passes

The public surface is `rgpot::nlist::Query`, `Builder` and `Walk`, the C
ABI in `CppCore/rgpot/nlist/rgpot_nlist.h`, and `NeighborRequest` /
`NeighborList`, which the rgpot-local schema diff
`CppCore/rgpot/rpc/local/neighbour-chunk.diff` adds to the
`Potentials.capnp` that rgpot compiles and installs. The classical
kernels keep `rgpot::nlist::Neighbours`. This log is the design of the
public surface. Each pass below was written against the API as it stood,
then the API moved. A code block inside a pass shows the API at that
pass; the current calls are in "What the final surface is".

## Survey

Read in November 2026. Vendored vesin is an older snapshot
(`third_party/vesin/vesin.h`); the notes below use the upstream files
unless they say otherwise.

### vesin

C header: [vesin/include/vesin.h](https://github.com/Luthaf/vesin/blob/master/vesin/include/vesin.h).
Python: [python/vesin/vesin/_neighbors.py](https://github.com/Luthaf/vesin/blob/master/python/vesin/vesin/_neighbors.py).
The repository root
([contents listing](https://github.com/Luthaf/vesin)) has `vesin`,
`python`, `fortran`, `vesin-jax` and `vesin-torch`. It has no Rust
crate. A Rust caller uses the C ABI, which is the same route this
library offers.

What it gets right:

- One options struct and one list struct. Pairs, shifts, distances and
  vectors are parallel arrays. The vector is documented as
  `r_j - r_i + S @ H`, and the box is one lattice vector per row.
- `full` is a bool, not a pile of flags. Upstream Python refuses
  `padded_to` unless `full_list` is true (`_neighbors.py`, the
  constructor). That combination cannot succeed by accident.
- Skin is explicit. A positive skin rebuilds when an atom moves more
  than `skin / 2`. Zero disables the cache. Negative asks the library
  to pick a skin. The comment says so.
- `vesin_neighbors` returns an `int32_t` and a `const char*` message
  the caller does not free. Failure is not a zero length list.
- `pre_allocated_outputs` is the caller-owned path. Otherwise vesin
  allocates and `vesin_free` releases. Ownership is one bool, not a
  hidden cache.
- `n_threads == 0` means the OpenMP default, and the comment says so.

What it gets wrong:

- `VesinOptions` is a C struct with no `sizeof` field. Adding a field
  changes the ABI with no way for an old caller to say how large a
  struct it passed.
- A zeroed `VesinOptions` has `cutoff == 0`, `full == false`, and every
  `return_*` flag false. The obvious C call returns a half list with no
  geometry. Python makes `cutoff` and `full_list` keyword-only, so the
  safe path is not the C path.
- `error_message` points at storage the header does not describe as
  thread-local or immortal. Callers cannot tell whether the next call
  overwrites it.
- There is no chunked walk. The whole list is materialised, or the
  caller guesses a capacity and the call fails.
- The vendored header in this tree has none of `padded_to`,
  `pre_allocated_outputs` or the negative-skin convention. Matching the
  vendored snapshot would freeze those gaps.

### JAX-MD

[jax_md/partition.py](https://github.com/jax-md/jax-md/blob/main/jax_md/partition.py),
`neighbor_list` at the function of that name and `NeighborListFormat`.

What it gets right:

- `Dense`, `Sparse` and `OrderedSparse` are an enum.
  `is_format_valid` rejects anything else instead of silently picking
  dense.
- Overflow is data: `did_buffer_overflow` and `PartitionError`. A jit
  update cannot throw, so the bit is the error channel, and the example
  in the docstring reallocates when it is set.
- `allocate` and `update` are different functions. The capacity is
  chosen at allocate time (`capacity_multiplier`, default 1.25) and the
  update cannot change a shape.
- The rebuild distance is `dr_threshold / 2`, the same half-skin rule
  as vesin. The docstring says the box is upper triangular with lattice
  vectors as columns, the transpose of ASE's cell. The convention is
  written next to the argument.

What it gets wrong:

- The default format is `Dense`, whose width is the busiest atom. A
  sparse consumer who forgets `format` pays that padding and then
  indexes a square matrix as if it were pairs.
- `mask_self` defaults to true, so the self pair is dropped, but a
  custom mask can put it back. The self pair is a policy argument, not
  a consequence of the format.
- `NeighborListFns.__call__` still accepts the old "allocate or update"
  signature and logs a deprecation. Two ways to do the same thing
  remain.
- There is no streaming walk. A list that does not fit the static
  buffer does not exist until the caller allocates again.
- The error code is a bitfield on the list. Ignoring
  `did_buffer_overflow` and reading `idx` is a successful-looking wrong
  answer.

### LAMMPS

[src/neigh_request.h](https://github.com/lammps/lammps/blob/develop/src/neigh_request.h)
and
[src/KOKKOS/neigh_list_kokkos.h](https://github.com/lammps/lammps/blob/develop/src/KOKKOS/neigh_list_kokkos.h).

What it gets right:

- The request is separate from the list. Half is the default (`half`
  starts set, `full` does not). `enable_full` is a method, so a caller
  who wants both directions asks for that.
- Kokkos stores a 2D view of neighbour indices plus `d_numneigh`.
  `get_neighbors(i)` returns a small view of one atom's partners. The
  device kernel does not gather an array of structs.
- A request can name a fixed cutoff (`set_cutoff_fixed`) instead of
  inheriting the maximum pair cutoff. Analysis and pair styles do not
  share a silent default.

What it gets wrong:

- The request is a long list of `int` flags (`half`, `full`, `ghost`,
  `newton`, `occasional`, `bond`, package bits). Nothing in the type
  stops `half` and `full` both being 1. `Neighbor` later morphs the
  flags (`halffull`, `copy`, `trim`). The state a user set and the
  state that runs are different structs sharing one object.
- Defaults live in the constructor, not in the header next to the
  fields. A reader of the struct cannot see them.
- There is no per-call error code. A bad request fails inside the
  neighbour build, often as a LAMMPS error string far from the flag
  that caused it.
- The list is per atom, not a pair table. A consumer that wants
  `(i, j, shift, vector)` has to reconstruct it. That is the right
  layout for a LAMMPS pair style and the wrong one for metatomic.

### ArborX

[src/spatial/detail/ArborX_NeighborList.hpp](https://github.com/kokkos/ArborX/blob/master/src/spatial/detail/ArborX_NeighborList.hpp).

What it gets right:

- Half and full are two functions, `findHalfNeighborList` and
  `findFullNeighborList`. There is no flag that can be both.
- The output is Kokkos views the caller owns: offsets of length
  `n + 1` and a flat index array. The library reallocates those views.
  CSR is a zero-copy layout for a kernel that walks one atom's
  neighbours.
- The execution space is an argument. Which thread (or which GPU) runs
  the query is not a global.

What it gets wrong:

- The experimental header has no shifts and no periodic box. The
  predicate is a sphere of radius `radius` in the coordinates it is
  given. A triclinic cell is the caller's problem.
- There is no status return. A view the execution space cannot see
  fails the `static_assert` on accessibility, which is a compile error
  for that case and nothing for a bad radius.
- Indices only. Distances and vectors are not part of the result, so
  every consumer recomputes them.
- No ABI story. It is a C++ header, and the views follow Kokkos.

### matscipy and ASE

matscipy
[matscipy/neighbours.py](https://github.com/libAtoms/matscipy/blob/master/matscipy/neighbours.py),
`neighbour_list`.
ASE
[ase/neighborlist.py](https://gitlab.com/ase/ase/-/blob/master/ase/neighborlist.py),
`neighbor_list` and `NewPrimitiveNeighborList`.

What they get right:

- The quantity string (`"ijSdD"`) asks for indices, shifts, distances
  and vectors, and unknown letters raise `ValueError`
  (`make_result` in matscipy). The caller does not allocate arrays they
  will not read.
- Both document `D = positions[j] - positions[i] + S.dot(cell)` next to
  the function. The cell is the ASE 3×3 matrix, rows as lattice
  vectors.
- matscipy returns triplets (`ijk`) from the same connectivity as
  pairs. A three-body term does not build a second list with a second
  convention.
- ASE's primitive list takes `bothways` as a bool. Half is the default
  on the class.

What they get wrong:

- ASE's function defaults `self_interaction` to false. The class
  `NewPrimitiveNeighborList` defaults it to true. The same word does
  not mean the same default.
- ASE rebuilds when the largest squared displacement exceeds
  `skin**2` (`update` compares against `self.skin**2`). That is a full
  skin, not half a skin. Two atoms can each move almost `skin` towards
  each other and a pair that belongs inside the cutoff is still
  missing. The skin default of 0.3 hides the comparison.
- The quantity string is untyped. `"ijS"` and `"jiS"` are both legal
  and easy to swap, and the result tuple changes length with the
  string. A C++ signature cannot express that.
- matscipy's default `pbc` is a nonperiodic box. Forgetting it on a
  crystal drops every image. ASE takes periodicity from the `Atoms`
  object, so the same omission does not happen there.
- Neither has a C ABI or a version field.

### freud

[freud/locality/NeighborQuery.h](https://github.com/glotzerlab/freud/blob/main/freud/locality/NeighborQuery.h),
[AABBQuery.h](https://github.com/glotzerlab/freud/blob/main/freud/locality/AABBQuery.h),
[NeighborList.h](https://github.com/glotzerlab/freud/blob/main/freud/locality/NeighborList.h),
[NeighborBond.h](https://github.com/glotzerlab/freud/blob/main/freud/locality/NeighborBond.h).

What it gets right:

- `NeighborQuery` is the object that holds points and the box.
  `query` returns an iterator. The caller can stop. `AABBQuery` is one
  acceleration structure behind that interface, and the header says how
  the tree and the periodic images work.
- `QueryArgs` is a single struct. `validateQueryArgs` infers ball
  versus nearest from which fields were set, then rejects a combination
  that still does not make sense. `QueryType::none` exists so a zeroed
  argument is not silently a ball query.
- A point set of length zero throws `std::invalid_argument`. A 2D box
  with a nonzero z throws. Failure is an exception with a sentence.
- `NeighborList` stores query index and point index in an `(n_bonds, 2)`
  array, plus flat distances and vectors. Bonds are also a small struct
  for the iterator. The stored list and the streamed bond are the same
  fields.
- The query borrows the points pointer. The constructor does not copy
  the coordinates. Lifetime is the caller's, and the type does not
  pretend otherwise.

What it gets wrong:

- `query` throws `std::domain_error` when any axis is nonperiodic
  ("Pair queries in a non-periodic box are not implemented"). Partial
  periodicity is a hard error, not a supported convention.
- `DEFAULT_R_MAX` is `-1`. A forgotten cutoff is a sentinel, and the
  validity check has to know that sentinel. A negative length that is
  not the sentinel is a different bug.
- `NeighborBond` is an array of structs on the iterator path and a
  structure of arrays on `NeighborList`. Callers learn both.
- The points pointer is bare. Nothing in the type records that the
  array must outlive the query. The documentation is the comment, and
  a use after free compiles.
- No C ABI. The Python layer is Cython over this C++.

### torch-nl

[torch_nl/neighbor_list.py](https://github.com/felixmusil/torch_nl/blob/main/torch_nl/neighbor_list.py).

What it gets right:

- The linked-cell candidate list and the strict cutoff are separate.
  `strict_nl` keeps pairs with `d2 < cutoff * cutoff`. The strict
  inequality is the code, not a comment.
- The return is three tensors: `mapping` of shape `(2, n_neighbors)`,
  a batch index, and integer cell shifts. Structure of arrays, and the
  mapping is `(center, neighbor)`.
- `self_interaction` defaults to false, and the docstring says so.
- Positions are documented as already wrapped into the cell.

What it gets wrong:

- The docstring says partial periodic boundaries are not supported, and
  the code still accepts a `pbc` tensor. A mixed `[true, true, false]`
  is a runtime surprise rather than a type that cannot say it.
- There is no skin and no error type. A bad cell fails inside a tensor
  kernel.
- Shifts are indices into the cell, and the vector is recomputed by
  the caller (`pos[j] - pos[i] - cell_shifts` in `strict_nl`, with the
  opposite sign from vesin's `r_j - r_i + S H` depending on how
  `compute_cell_shifts` is defined). Two libraries in this survey
  disagree on the sign unless the caller reads both implementations.
- No version field and no C ABI. TorchScript is the stability boundary.

### What this API should take

From that reading, the public query should:

- Use one layout description that cannot be a half list and a cluster
  at the same time (vesin's `padded_to` check, ArborX's two functions,
  JAX's enum check).
- Default the way the documented example does, in C++ and in C
  (vesin's Python is strict, its C zero-fill is not).
- Return a code and a stable message in C, and throw one exception
  type in C++ (vesin's status pointer, freud's `invalid_argument`).
  Do not stash the message in a global.
- Hand the caller buffers or an owning table of parallel arrays, and
  a walk that fills those arrays in chunks (JAX overflow plus freud's
  iterator). The C structs and the Cap'n Proto structs name the same
  arrays.
- Put `bytes` first on every C struct the caller allocates, so a later
  field can be added without a new major (the gap in `VesinOptions`).
- Document the cell as ASE's row vectors, the vector as
  `r_j - r_i + S H`, the strict test as `r < cutoff`, and the skin
  rebuild as half the skin. Say which calls may run on two threads.

## Baseline, before any pass

A half-list Lennard-Jones energy against the API at `2a195864` looks
like this. The fields that make it a half list are the defaults, and
the fields that make it periodic and stored are not, depending on the
language.

<!-- rgpot-doc: skip historical API at 2a195864; Format and Mode no longer exist -->
```cpp
rgpot::nlist::Query query;
query.cutoff = 2.5;
query.format = rgpot::nlist::Format::Half;
query.cut = rgpot::nlist::Cut::Strict;
query.mode = rgpot::nlist::Mode::Stored;
rgpot::nlist::Table table;
rgpot::nlist::build(table, positions, n, box, query);
for (std::size_t k = 0; k < table.size(); ++k) {
  const double r = table.distance[k];
  const double sr6 = std::pow(1.0 / r, 6.0);
  energy += 4.0 * (sr6 * sr6 - sr6);
}
```

The same request in C, zero-filled the way a C caller writes a struct:

<!-- rgpot-doc: skip historical C API at 2a195864; rgpot_nlist_build took a separate capacity then -->
```c
rgpot_nlist_request request;
memset(&request, 0, sizeof request);
request.abi = RGPOT_NLIST_ABI;
request.cutoff = 2.5;
size_t count = 0, needed = 0;
rgpot_nlist_build(positions, n, box, &request, NULL, 0, &count, &needed);
```

Counted by hand against `rgpot_nlist.h` and `NeighbourList.cc`:

- The C++ call is about fifteen lines before the energy loop and is
  the right physics if `Mode::Auto` does not refuse to store.
  `Mode::Auto` compares a combinatorial pair bound with
  `availableBytes()` and returns an empty table with `stored == false`
  (`build` in `NeighbourList.cc`). The loop above then reports zero
  energy. The call compiled.
- A zeroed C request has `periodic` all false, `want_shifts`,
  `want_vectors` and `want_distances` all false, and `mode` 0, which
  `query_from_c` reads as `Mode::Auto`. The C++ defaults are periodic,
  all three fields, and `Mode::Stored`. The same mental model is not
  the same call.
- `format` other than 1 or 2 becomes a half list. `cut` other than 0
  becomes loose. A typo succeeds.
- `rgpot_nlist_build` returns 1 only when `abi` mismatches. A null
  `positions` with `n > 0` is undefined. There is no message function.
- `rgpot_nlist_walk_pull` copies every chunk through temporary vectors
  into an array of `rgpot_nlist_pair`. Cap'n Proto and C++ are
  structures of arrays. C is an array of structs.
- `Walk::pull` returns 0 and leaves `done` false when `dest.first` is
  null (`NeighbourList.cc`). A caller who forgot a pointer loops.
- `NeighborRequest.mode` defaults to 1 (stored) in the schema, while C
  mode 0 is auto. A stream sets `more` and the client repeats the whole
  call, with no cursor, so a stateless server cannot choose the next
  chunk.
- Cluster4 is a third `Format`, so a half list and a cluster are
  mutually exclusive in the enum. The mask array is still present on a
  half list, filled with `0xff`, and a C caller who sets `format` to 2
  while leaving the rest of a half-list mental model gets every
  directed pair packed into groups of four. Nothing says that cluster
  means full.

The passes below close those gaps without changing the pair walk the
kernels use.

## Pass 1: the failure is a sentence, and the C default matches C++

The baseline C call returned 0 for a nonperiodic half list with no
vectors, because a zeroed struct is mode auto and every flag false.
`rgpot_nlist_build` returned 1 for an abi mismatch and nothing else.
`Walk::pull` spun when the destination pointers were null.

After this pass the C caller initialises the request, and a bad value
is a code with a fixed sentence:

<!-- rgpot-doc: skip historical pass-1 C API; rgpot_nlist_build took a separate capacity then -->
```c
rgpot_nlist_request request;
rgpot_nlist_request_init(&request);
request.cutoff = 2.5;
size_t count = 0, needed = 0;
int32_t rc = rgpot_nlist_build(positions, n, box, &request, NULL, 0,
                               &count, &needed);
if (rc != RGPOT_NLIST_OK) {
  fputs(rgpot_nlist_strerror(rc), stderr);
}
```

`rgpot_nlist_request_init` sets `bytes` to the compiled size, the abi,
a half list, a strict cutoff, periodic boundaries, shifts, vectors,
distances, and a stored list. A zeroed request fails with
`RGPOT_NLIST_ERR_ABI` instead of building the wrong crystal. `format` 9,
`cut` 3 and `mode` 7 return `RGPOT_NLIST_SHAPE` instead of becoming a
half list. A negative cutoff returns `RGPOT_NLIST_CUTOFF`. A short
buffer returns `RGPOT_NLIST_SHORT` and still writes what fit, with
`needed` set, which is the JAX overflow bit expressed as a code. A
size probe (`out` null and `cap` 0) stays `RGPOT_NLIST_OK`.

C++ throws `rgpot::nlist::Error`. `code()` is the same integer the C
ABI returns. `what()` is the sentence. `Builder::update` checks the
geometry before it clears a table, so a rejected update leaves the
previous pairs in place. `Walk::pull` throws when `cap > 0` and the
index pointers are null, and does not advance.

`bytes` is the first field. The library copies
`min(bytes, sizeof request)` and rejects a `bytes` that does not cover
`mode`. A newer caller that appends fields still works. The message
pointer from `rgpot_nlist_strerror` is a literal. There is no
thread-local last error.

What this pass leaves open: `Mode::Auto` can still return an empty
stored table, `Format` is still one enum that a reader can confuse
with a modifier, and the C result is still an array of
`rgpot_nlist_pair` copied out of the structure of arrays. The walk
open now returns a code (`rgpot_nlist_walk **` out-parameter) so a
rejected geometry is not only a null pointer.

## Pass 2: a half list and a cluster are different types

A three-body term wants every ordered pair around one atom. The pass-1
call could still ask for a cluster with the half-list default next to
it, and `Mode::Auto` could hand back an empty table:

<!-- rgpot-doc: skip historical pass-1 C++ API; Format and Mode no longer exist -->
```cpp
rgpot::nlist::Query query;
query.cutoff = 3.0;
query.format = rgpot::nlist::Format::Cluster4; // full interactions, easy to miss
query.mode = rgpot::nlist::Mode::Auto;         // may store nothing
```

`Format` was one enum, so the C++ type could not be half and cluster
at once, but the name read as a modifier on the default half list.
The C integer `format = 2` did not say "this is not a half list".
`build` treated `Mode::Stream` and a large `Mode::Auto` as success
with `stored == false`.

The shape is now a variant. Pairs carry a shell. A cluster does not:

<!-- rgpot-doc: fragment -->
```cpp
rgpot::nlist::Query query;
query.cutoff = 3.0;
query.shape = rgpot::nlist::Pairs{rgpot::nlist::Shell::Full};
rgpot::nlist::Table table;
rgpot::nlist::build(table, positions, n, box, query);
// or, for the masked kernel:
query.shape = rgpot::nlist::Cluster4{};
```

There is no `mode` on `Query`. `build` and `Builder` store every pair.
`Walk` is the stream. A cluster expands to the same interactions as a
full pair list; the mask uses the low four bits and never `0xff`.

In C the two fields are `shape` and `shell`. `shape == 1` with
`shell == 0` returns `RGPOT_NLIST_SHAPE`. The C ABI returns only the
code; `rgpot_nlist_strerror` gives the generic sentence for it. The
sentence "a cluster list stores every neighbour; set shell to 1 (full)"
is the `what()` of the C++ `Error`. The abi is 3
because `format` and `mode` left the struct. `bytes` still has to
cover the struct the caller compiled.

Metatomic sets `Pairs` from `full_list()`. UMA sets a full pair list.
Neither can ask the public query for a list that then turns out empty
because the machine looked short of memory.

## Pass 3: C speaks the same columns as C++ and Cap'n Proto

The C walk allocated six temporary vectors and copied them into
`rgpot_nlist_pair`. A Lennard-Jones loop then read `pairs[k].distance`
while metatomic read `table.vector[3 * k]`. The columns were the same
physics and a different type.

`rgpot_nlist_pair` is gone. The caller owns `rgpot_nlist_soa`, whose
first field is `bytes` and whose columns are `first`, `second`,
`shift`, `vector`, `distance` and `mask`. Those are the names on
`rgpot::nlist::View` and on `NeighborList` in the schema. `build` and
`walk_pull` write the caller's arrays. They do not allocate a second
copy.

<!-- rgpot-doc: fragment -->
```c
rgpot_nlist_soa soa;
memset(&soa, 0, sizeof soa);
soa.bytes = sizeof soa;
soa.cap = needed;
soa.first = first;
soa.second = second;
soa.distance = distance;
/* shifts and vectors omitted: the request must not ask for them */
int32_t rc = rgpot_nlist_build(positions, n, box, &request, &soa,
                               &count, &needed);
```

C++ `Walk::pull` takes `Buffers` of `std::span`. The length of `first`
is the capacity. A requested column that is shorter throws
`Error(Code::Short)` before the walk advances. `Table::view()` returns
`std::span` columns. The view dies at the next `update` of that table.

The abi is 4. A caller still holding an array of pairs does not link.
A size probe is a null `rgpot_nlist_soa *`, or a struct with `cap` 0,
and returns `RGPOT_NLIST_OK` with `needed` set. A positive `cap` that
is still short returns `RGPOT_NLIST_SHORT` after filling that many
pairs. A column the request asked for, whose pointer is null, returns
`RGPOT_NLIST_NULL` and writes nothing.

## Pass 4: the wire has a cursor, and the examples compile

A Cap'n Proto client that set `more` had to repeat the whole call, and
a stateless server could not know which chunk was next. `NeighborRequest`
now has `cursor` and `limit`. `NeighborList` returns `next`, `status`
and `message`. The client sends `next` back as `cursor` until `more` is
false. `format` stays one field, so a half cluster is still not a value
on the wire. Mode 0 is not a mode: mode 1 is one stored chunk and mode 2
is the stream that honours `cursor` and `limit`. A `limit` of 0 means
4096 pairs. potserv does not implement `Potential.neighbours`; only the
example below answers these messages.

`CppCore/examples/nlist_stream.cc` is that client and a server in two
processes on one socket. The server slices a stored list. The client
checks each chunk against a build of the same geometry. The other
examples are the calls from the passes, built and run as tests:
`nlist_lj.cc` (half-list Lennard-Jones, energy -1),
`nlist_threebody.cc` (a right angle from a full list),
`nlist_metatomic.cc` (the five label columns), `nlist_c.c` (no C++),
and `nlist_rust/main.rs` (the C ABI, hand-written).

Thread safety is in the headers. Distinct tables, builders and walks
may run at the same time. One object is not shared. `build` may start
an OpenMP team; inside another OpenMP region that team may get fewer
threads, and the list does not depend on how many.
`rgpot_nlist_strerror` returns a literal. A walk borrows the positions
until it is closed. A view dies at the next update of its table.

The tutorial is `docs/orgmode/tutorials/neighbour-list.org`.

## What the final surface is

C++: `Query` holds a cutoff, a skin, a `Shape` (`Pairs` with `Shell`,
or `Cluster4`), a `Cut`, periodicity and `Fields`. `build` fills a
`Table`. `Table::view` is spans. `Builder::update` rebuilds when an
atom moves more than half the skin. `Walk::pull` fills `Buffers`.
Failure throws `Error`, whose `code()` matches the C integer.

C: `rgpot_nlist_request_init`, `rgpot_nlist_build`,
`rgpot_nlist_build_systems`, `rgpot_nlist_walk_open`,
`rgpot_nlist_walk_pull`, `rgpot_nlist_walk_close`,
`rgpot_nlist_builder_open`, `rgpot_nlist_builder_update`,
`rgpot_nlist_builder_close`, `rgpot_nlist_fortran_open`,
`rgpot_nlist_fortran_build`, `rgpot_nlist_fortran_close`,
`rgpot_nlist_strerror`, `rgpot_nlist_abi_version`. `RGPOT_NLIST_ABI` is 4.
`bytes` leads the request and the buffer struct. Columns are `first`,
`second`, `shift`, `vector`, `distance`, `mask`.

Cap'n Proto: `NeighborRequest` and `NeighborList` use those columns.
`mode = 2` with `cursor`, `limit`, `next`, `status` and `message` is the
stream. The potserv server does not answer `Potential.neighbours`.

## The query walks the cell list

`build` used to visit every atom pair and every periodic image. A half
list of the 108000-atom copper block did not return within three
minutes. It now bins atoms with `rgpot::nlist::Neighbours` and walks
that cell list. A box at least two cutoffs wide, and not sheared, has
one image, so the folded cell vector is the pair. A narrower or
sheared cell still tests the shift stencil, but only for atoms that
share a neighbouring cell. `Walk` keeps one cell list for its whole
life.

`NeighbourListTest` rejects an all-pairs scan: the 108000-atom half
list must finish in under 30 seconds and contain 21 neighbours per atom.
A wide orthorhombic cell is checked against the same brute force as
the triclinic cell.

Times below are the median of seven calls after two discarded calls,
from `benches/neighbour_api_bench.py`. The resident set is VmHWM of a
fresh process. rgpot is the cell-list query. vesin 0.6.2, ASE 3.29 and
matscipy 1.3.1 ran in the same driver. ASE's full list of the million
atoms held about 30 GB and was stopped. matscipy has no half list.

| system | list | threads | rgpot s | rgpot MB | vesin s | vesin MB | ASE s | matscipy s |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| tri | half | 1 | 0.000267 | 4.5 | 0.000410 | 33.5 | 0.00532 |  |
| tri | half | 8 | 0.000259 | 4.4 | 0.000408 | 33.4 | 0.00513 |  |
| tri | full | 1 | 0.000469 | 4.7 | 0.000627 | 34.2 | 0.0369 | 0.00172 |
| tri | full | 8 | 0.000469 | 4.7 | 0.000642 | 34.1 | 0.0357 | 0.00174 |
| 108000 | half | 1 | 0.125 | 139 | 0.143 | 316 | 2.51 |  |
| 108000 | half | 8 | 0.0514 | 182 | 0.0794 | 428 | 2.46 |  |
| 108000 | full | 1 | 0.225 | 256 | 0.230 | 591 | 8.45 | 0.632 |
| 108000 | full | 8 | 0.0988 | 341 | 0.165 | 807 | 8.32 | 0.638 |
| 1048576 | half | 1 | 1.27 | 1317 | 1.39 | 2787 | 25.2 |  |
| 1048576 | half | 8 | 0.571 | 1634 | 0.751 | 3565 | 24.8 |  |
| 1048576 | full | 1 | 2.24 | 2457 | 2.39 | 5453 |  | 6.17 |
| 1048576 | full | 8 | 0.998 | 3192 | 1.64 | 6668 |  | 5.95 |

A system of fewer than 8192 atoms stays on the calling thread, so the
64-atom list costs the same with `OMP_NUM_THREADS=8` as with one
thread. Below 512 atoms a pair list is also a direct pair loop: a
sheared cell that small has only a few bins, and the cell list was
spending the call on setup.
Eight threads share one cell list and write into slices of one table.
The slices are then packed in place. The resident set is that table.
Every row above is faster than vesin and uses less memory.

## Pass 5: several systems are one call

Writing a batched consumer from the tutorial meant a loop of `build`,
then adding each system's atom count onto `first` and `second`, and
keeping a side array of system ids. Nothing stopped a pair from being
requested across the boundary except the caller's own bookkeeping.
The C caller had the same loop and no column for the system id.
Cap'n Proto had one `ForceInput` and no place to name a second system.

`Systems` is the concatenated positions, a count per system and a box
per system. `build` into a `Batch` returns one pair table whose
indices address that concatenation, plus `system[p]`. Systems do not
see each other's atoms. `rgpot_nlist_build_systems` is the same call.
`NeighborRequest.systems` and `NeighborList.system` carry it on the
wire. `CppCore/examples/nlist_batch.cc` builds two dimers and checks
that the second pair is indexed in the second system.

## Pass 6: the skin belongs to a builder, in C as well

A Verlet loop written from the tutorial used `Builder` in C++ and, in
C, called `rgpot_nlist_build` on every step. The C caller could not
ask whether the list was still inside half the skin. A strict cut
with a positive skin compiled and stored pairs out to the cutoff
only, so the skin did not protect a later move. That request now
throws `Code::Skin` (`RGPOT_NLIST_SKIN`), with the sentence "a
positive skin needs a loose cut; a strict cut stores no buffer".

`rgpot_nlist_builder_open`, `rgpot_nlist_builder_update` and
`rgpot_nlist_builder_close` keep the list. `update` writes `*reused`
when every atom is still within half the skin. `nlist_verlet.c`
moves one atom inside the skin, then past it.

## Pass 7: one walk for the potentials, and three lines of Python

Lennard-Jones, Morse, EAM, Sutton-Chen, Stillinger-Weber and Tersoff
were each calling `Neighbours::update` and `forDirected` with their
own `Spec`. Metatomic and UMA already called `build`. The six
potentials now call `updateCells`. Lennard-Jones, Morse, Sutton-Chen,
Stillinger-Weber and Tersoff walk it with `visitDirected`, `visitHalf`,
`forCellTeam` or `forAtom`. EAM walks it with `forCellTeam` when it
streams and keeps its own index list when it stores.
Those functions are the cell list `build` walks, so a kernel and a
stored table are two uses of one walk. `dx` in a visit is
`r_i - r_j` after the image, which is the negation of the vector the
table stores. EAM still asks for `Image::Half` because it wraps
coordinates into the cell first.

<!-- rgpot-doc: run -->
```python
import numpy as np
import rgpot

positions = np.array([[0.0, 0.0, 0.0], [1.2, 0.0, 0.0], [0.0, 1.3, 0.0]])
cell = np.eye(3) * 20.0
cutoff = 5.0
i, j, d = rgpot.neighbor_list(positions, cell, cutoff)
assert len(i) == 3
```

`positions` is `(natoms, 3)` and `cell` is `(3, 3)`, the ASE layout.
`full=True` asks for both directions. The binding is nanobind, in
`python/bind/bind_module.cpp`, and `rgpot.neighbor_list` re-exports it.

The names line up. C++ `Shape` / `Shell` is C `shape` / `shell` and
Cap'n Proto `format` (0 half, 1 full, 2 cluster). C++ `Cut` is C `cut`
and Cap'n Proto `cut`. `Code::Skin` is `RGPOT_NLIST_SKIN`. A batch is
`Systems` and `Batch` in C++, `rgpot_nlist_build_systems` in C, and
`NeighborRequest.systems` plus `NeighborList.system` on the wire.
`first` and `second` index the same positions in all three.

## A device kernel, checked on the host

No CUDA or HIP device is attached here. The launch that would run is
one atom per thread: thread `i` walks the 27 cells around atom `i`,
which is `visitDirected(cells, i, i + 1)`. `buildByAtom` does that
walk on the host and must return the same pairs as `build`. A device
port replaces the loop over atoms with a grid of the same body and
keeps `build` as the check. A cluster, or fewer than 512 atoms, stays
on the host `build`, which is also what a device
runtime would do when the grid degenerates.

## The Fortran gather uses the same list

`neighbor_table_t` used to ask vesin for a full list. It now calls
`rgpot_nlist_fortran_build`, which keeps an `rgpot_nlist_builder`.
The search radius is the cutoff plus the skin. While every atom stays
within half the skin the builder reuses that candidate list. Each call
still writes `r_j - r_i + S H` for the positions it was given and keeps
only pairs strictly inside the cutoff. `vec(3, npairs)` has the layout
of `Table::column_major_vectors()`: column `p` is the vector of pair `p`.
On 4000 FCC atoms and a cutoff of 5 with skin 0.5, a fresh search
takes 0.012 s and a reused call, which only refreshes the vectors,
takes 0.001 s. The Stillinger-Weber, EAM, water, Fe-He, Tersoff, EDIP
and Lenosky gather tests still match their finite-difference forces.
vesin remains in the neighbour benchmark and in the vendored tree for
that comparison.

## Calling the list from eOn

`docs/eon_nlist.md` is the short C and Cap'n Proto sketch. A builder
with a loose cut and a positive skin is the Verlet cache. The wire
request is `format = 1`, `cut = 1`.

## Re-run

`NeighbourListTest` passed, 570 assertions in 14 cases. The compiled
callers passed: Lennard-Jones, the three-body angle, the metatomic
label copy, the C caller, the batch of two systems, the C Verlet
builder, the Rust caller, and the two-process stream.
Morse, Stillinger-Weber, Tersoff, the EAM cell, Sutton-Chen, ZBL, the
LJ cluster, EMT, the half shell and the neighbour layer passed with
them.

`classical_scaling_bench`, one thread, median of seven warm calls.
One million atoms, streaming:

| potential | this run (s) | recorded (s) | isolated RSS (kB) | recorded RSS (kB) |
| --- | ---: | ---: | ---: | ---: |
| EAM | 5.163 | 4.920 | 139848 | 139796 |
| Lennard-Jones | 12.459 | 12.503 | 115108 | 115204 |

The resident set is the fresh-process figure from `classical_scaling_bench rss`.
The pair walk those potentials use is `rgpot::nlist::Neighbours`.
The public query uses that same walk.
