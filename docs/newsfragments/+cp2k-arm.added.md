`CP2KPot` runs CP2K Quickstep through `libcp2kc_rgpot`, a shim over libcp2k
built with `-Dwith_cp2k=enabled`. `CP2KParams` (`PotentialConfig.cp2k`)
renders a default GPW deck or fills `@RGPOT_COORD@` and `@RGPOT_CELL@` in a
verbatim template. The shim keeps one force environment per orbital key, and an
unconverged SCF raises `rgpot::CP2KScfError` on every rank instead of aborting
the MPI world. potserv serves it as `CP2K`.
