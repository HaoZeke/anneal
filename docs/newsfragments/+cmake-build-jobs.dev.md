`potctl ci cmake-test` and `potctl ci bridge-client-build` give `cmake --build`
an explicit `--parallel` count: `CMAKE_BUILD_PARALLEL_LEVEL` when it is set,
the usable CPUs otherwise.
A bare `-j` let make start every ready compile at once, and with the whole
test registry to build, the macOS CMake legs (3 cores, 7 GB) spent 7 to 49
minutes compiling, where the meson legs of the same run built and tested in 3
to 6.
