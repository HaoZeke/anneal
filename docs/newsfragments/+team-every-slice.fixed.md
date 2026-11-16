A neighbour walk split across more slices than the OpenMP runtime started threads now walks every slice.
`nlist::Neighbours::forCellTeam`, the threaded `nlist` builds and `simd::fill_view` handed slice `t` only to thread `t`, so a build without OpenMP, a call inside another OpenMP region or a run under `OMP_THREAD_LIMIT` dropped the pairs of the missing threads: with `OMP_NUM_THREADS=3` and `OMP_THREAD_LIMIT=2`, the threaded Sutton-Chen energy came out wrong, and the macOS meson build, which has no OpenMP, failed `NeighbourLayerTest`.
`nlist::forEachSlice` runs the slices on the threads the runtime starts, so a sum taken in slice order is the same for any team.
A build inside another OpenMP region is no longer ruled out.
