The cargo-c install now places ``dlpack/dlpack.h`` (DLPack 1.3, the header
``dlpk`` 0.1.5 was translated from) beside ``anneal-core.h``. The C header
includes that file, so ``pkg-config --cflags anneal-core`` is enough to
compile a consumer. Meson and CMake install the same header.
