<!-- Copyright 2023--present rgpot developers -->

The version-drift check fails when `rgpot-bundle/Cargo.lock` has lost the cog
markers around the `rgpot-core` version. The ABI stamp page records that a
whole-archive bundle exports 104 `rgpot_*` symbols, and `rgpot_so_exports` pins
that count next to the 51 of a normal link. The two added names are
`rgpot_builtin_muller_brown_new` and `rgpot_muller_brown_config_default`.
