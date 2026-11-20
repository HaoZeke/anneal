<!-- Copyright 2023--present rgpot developers -->

Every engine plugin built from this repository now exports
`rgpot_engine_abi_stamp`, which reports the engine ABI major, minor and layout
revision. The stamp is defined once, in `CppCore/rgpot/engine/EngineAbiStamp.cc`,
and the table-based libraries, the standalone Lennard-Jones, native-unit and
OpenMM plugins, and the test stub all add that file. A loader such as
`EnginePotential` in `rgpot-core` can now read the ABI minor and layout revision
of these plugins directly instead of assuming minor 0.
