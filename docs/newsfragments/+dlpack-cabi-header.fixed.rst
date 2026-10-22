The installed C ABI ships the DLPack 1.3 header. ``anneal-core.h``
includes ``dlpack/dlpack.h``, and that header is the one dlpk 0.1.5
was translated from, so the FFI smoke compile can see
``DLManagedTensorVersioned``.
