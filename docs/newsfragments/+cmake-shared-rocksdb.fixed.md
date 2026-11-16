A CMake build with the cache links the shared RocksDB library when there is one,
as meson does.
conda-forge's static RocksDB put jemalloc ahead of libc in every program that
links `librgpot`, and such a program crashed in `free` once it loaded `libmpi`,
as the MPI calculator group and collective RPC tests do.
