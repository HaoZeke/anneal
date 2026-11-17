CMake's `nlist_rust` test passes the script the OpenMP libraries FindOpenMP
chose, and the runtime's directory as an rpath, so the shared library the
script links finds the OpenMP runtime with AppleClang, which has no
`-fopenmp`.
Called without them, as meson calls it, the script still tries `-fopenmp`.
