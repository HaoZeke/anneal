Filtered ``cbindgen.toml`` exports to C ABI enums, structs, and ``anneal_*``
functions, combined Meson static-archive objects before localizing dependency
symbols, and enabled the ``capi`` feature in ``CMakeLists.txt`` so C and C++
consumers link ``anneal_box_minimize`` cleanly.
