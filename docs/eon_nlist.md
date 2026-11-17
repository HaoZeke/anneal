# Calling the neighbour list from eOn

eOn potentials that today call `vesin_neighbors` can use the same request
through the C ABI. The Cap'n Proto schema carries the same request, but no
server in this repository answers it yet. Both speak the columns
`first`, `second`, `shift`, `vector`, `distance` (and `mask` for a
cluster). Indices are zero-based. The vector is `r_j - r_i + S H`, which
is the layout a Fortran `(3, npairs)` array already has.

## C ABI

A fresh list is two calls: one to learn the length, one to fill caller
buffers. A skin reuses an `rgpot_nlist_builder`. The cut has to be loose
when the skin is positive. A strict cut with a positive skin returns
`RGPOT_NLIST_SKIN`, because a strict cut stores no buffer.

<!-- rgpot-doc: run flags=-ICppCore link=rgpot -->
```c
#include "rgpot/nlist/rgpot_nlist.h"

#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>

int main(void) {
  double positions[] = {0.0, 0.0, 0.0, 1.2, 0.0, 0.0, 0.0, 1.3, 0.0};
  const size_t natoms = 3;
  const double box9[9] = {20, 0, 0, 0, 20, 0, 0, 0, 20};

  rgpot_nlist_request request;
  rgpot_nlist_request_init(&request);
  request.cutoff = 5.0;
  request.skin = 0.5;
  request.cut = 1;   /* loose: candidates out to cutoff + skin */
  request.shell = 1; /* full list, both directions */
  request.want_shifts = 1;
  request.want_vectors = 1;
  request.want_distances = 1;

  rgpot_nlist_builder *builder = NULL;
  if (rgpot_nlist_builder_open(&builder) != RGPOT_NLIST_OK) {
    return 1;
  }

  size_t needed = 0;
  int32_t rc = rgpot_nlist_builder_update(builder, positions, natoms, box9,
                                          &request, NULL, NULL, &needed, NULL);
  if (rc != RGPOT_NLIST_OK) {
    fprintf(stderr, "%s\n", rgpot_nlist_strerror(rc));
    return 1;
  }

  int32_t *first = malloc(needed * sizeof *first);
  int32_t *second = malloc(needed * sizeof *second);
  int32_t *shift = malloc(3 * needed * sizeof *shift);
  double *vector = malloc(3 * needed * sizeof *vector);
  double *distance = malloc(needed * sizeof *distance);

  rgpot_nlist_soa soa = {0};
  soa.bytes = (uint32_t)sizeof(soa);
  soa.first = first;
  soa.second = second;
  soa.shift = shift;
  soa.vector = vector;
  soa.distance = distance;
  soa.cap = needed;

  /* One atom moves by less than half the skin: the list is kept. */
  positions[3] += 0.1;
  size_t count = 0;
  int32_t reused = 0;
  rc = rgpot_nlist_builder_update(builder, positions, natoms, box9, &request,
                                  &soa, &count, &needed, &reused);
  rgpot_nlist_builder_close(builder);
  free(first);
  free(second);
  free(shift);
  free(vector);
  free(distance);
  if (rc != RGPOT_NLIST_OK || count != 6 || reused != 1) {
    fprintf(stderr, "rc %d count %zu reused %d\n", rc, count, reused);
    return 1;
  }
  return 0;
}
```

`reused == 1` means every atom was still within half the skin, so the
stored list was kept. The columns are then those of the last rebuild:
the pairs reach out to cutoff + skin, and `vector` and `distance` belong
to the positions of that rebuild, not of this call.
`rgpot_nlist_fortran_build` is the Fortran gather. Each
`rgpot_nlist_fortran_list` keeps its own builder. It rebuilds `r_j - r_i + S H` from the stored keys and the
positions of each call, and drops pairs that are not strictly inside the
cutoff.

## Cap'n Proto

The same request is one `NeighborRequest`. `format = 1` is a full list,
`cut = 1` is loose, and `skin` is the Verlet buffer. `mode = 1` asks for
one stored chunk. `mode = 2` is a stream that honours `cursor` and
`limit`: the client sends `cursor`, reads `next`, and repeats while
`more` is true. The reply is a `NeighborList` with the columns above.

potserv does not implement `Potential.neighbours`, and `calculate` does
not fill `PotentialResult.neighborList`. `CppCore/examples/nlist_stream.cc`
answers these messages between two processes over a socket, outside the
`Potential` interface.

<!-- rgpot-doc: skip Cap'n Proto text form of one NeighborRequest; nothing here parses it -->
```text
(cutoff = 5, skin = 0.5, format = 1, cut = 1,
 shifts = true, vectors = true, distances = true, mode = 1)
```
