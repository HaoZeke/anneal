`potentials-schema` is pinned at 7fa9046, and both vendored `Potentials.capnp` copies are that file with only the rgpot neighbour chunk applied.
`PotentialConfig` gains the upstream arms `xtb` @6, `tblite` @7, `uma` @8, `lj` @9, `morse` @10, `zbl` @11, `d3` @12, `d4` @13, `mopac` @14, `skala` @15 and `expr` @16, with their parameter structs, so rgpot numbers them as gpridge and every other consumer does.
`BuiltinParams` gains `config` @2, `PotSpec` gains `potential` @2 and `MetatomicParams` gains `lengthUnit` through `attachSystemExtras` (@13 to @18).
