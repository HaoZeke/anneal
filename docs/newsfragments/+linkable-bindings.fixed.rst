The generated C header keeps the enums, the structs and the four
``anneal_*`` functions. It no longer redefines same-named constants or
declares symbols the crate does not export. The Meson archive is one
object before ``eindir``, ``rgmin`` and ``rgsaddle`` symbols are made
local, so a later member can still resolve ``eindir_core_abi_stamp``.
CMake imports the crate with the ``capi`` feature.
