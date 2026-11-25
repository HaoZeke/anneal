The native library archives on the GitHub release carry
``include/dlpack/dlpack.h``. ``anneal-core.h`` includes ``<dlpack/dlpack.h>``,
and the archives held only the three anneal headers, so a C consumer of an
extracted archive failed to compile. The release job now compiles a consumer
against each archive before it uploads it.
