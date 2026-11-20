<!-- Copyright 2023--present rgpot developers -->

The `rgpot-pair` C kernels now build with GCC 4.8, the compiler in the
manylinux2014 aarch64 cross image. The build asks for `-std=gnu99` where the
compiler is not MSVC, and the Gupta buffers use `__thread` on GCC and Clang.
