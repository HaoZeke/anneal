The C ABI stamp of `rgpot-core` moves to 1, 2 and 1: the Hessian-vector
entry points are additions, and `rgpot_potential_t` keeps its layout.
`scripts/abi_core_baseline.txt` lists the `rgpot_*` functions the
`rgpot-core` shared library exports.
The meson test `abi_core_symbols` fails when one of them is no longer
exported.
`scripts/abi_symbols.sh` takes the name pattern with `-p`, ahead of the
library and its baselines.
