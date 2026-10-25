# Water hexamer driver

`examples/water_hexamer.rs` searches (H2O)6. The full runs launch from the
v0.12.0 tag. A process with `PAPER_SMOKE=1` is a smoke and writes a file
whose name contains `SMOKE`. It is not a paper table.

## Conflicts with FIXES.md

IDEAS.md and OUTLINE-v2.md win where they differ.

- FIXES R6 logged 1200 PBE evaluations and 4000 GFN2 evaluations. The
  outline sets 4000 force evaluations per seed on every engine. This
  driver uses 4000.
- FIXES R6 named an RMSD gate of 0.1 angstrom in a proposed sentence.
  The outline sets the gate to half the minimum IRA RMSD between
  distinct references at that engine, computed after those references
  are relaxed and before any search seed. No references are deposited
  here, so the gate is unset and every seed is unassigned.
- The seed integers are 0-31, named by the outline. An earlier plan
  used 5100-5131 to stay in the high blocks of `paper/seeds.toml`.

## Search

One configuration, `Config::recommended_molecular`, with the molecular
groups kept. The outline's search row is that configuration. The
communicating, derived, and Good-Turing arms are the additional reruns
requested for the communicating stack. They do not replace the
molecular library with the atomic presets.

Groups are repacked from the isolated-water template in
`reference_coordinates`. That template is not an isomer reference.

A geometry is intact when every oxygen has exactly two hydrogens within
1.2 angstrom. A dissociated monomer is unassigned, not an isomer.
The label is species-aware IRA (`match_shapes_typed`, breadth 1.8),
the same function for a search geometry and a geometry under
`WATER_GOFEE_DIR`. The Kabsch helper is not the label. The gate is
half the minimum IRA RMSD between distinct deposited references,
computed once before any seed and before the engine is opened.
`WATER_ISOMER_DIR` holds `{prism,cage,book}.xyz` in O, H, H order,
relaxed at that engine. A build without the `ira` feature records
that reason and does not substitute the Kabsch helper. Primitive-ring
counts are not the isomer name.

With `ASE_PYTHON` set, the engine is the deposited calculator in
`examples/ase_objective.py`, not the rgpot library. GFN2-xTB is
tblite 0.7.0 `TBLite(method="GFN2-xTB")`. PBE is PySCF 2.14.0 RKS,
basis 6-31g*, xc PBE, grid level 3. PET-MAD is pet-mad 1.4.4,
checkpoint 1.0.2, CPU. ASE is 3.29.0. The cell is 40 angstrom with
periodic boundaries off, matching the relaxation that produced the
references. The script refuses any other package version. The helper keeps one ASE `Atoms` object and updates positions. tblite 0.7.0 aborts if a second `Atoms` is bound to the same calculator. Each
reference geometry is evaluated once and its energy is printed before
the seeds. Without `ASE_PYTHON` the driver still loads rgpot.

A non-finite opening quench is reported as `search_error` and does not drop the gate or the reference energies. Hit columns stay unscored on a smoke or a single arm. A success rate
waits on the full seed block from the v0.12.0 tag.

## Seeds and budget

`paper/seeds.toml` run `r6-water-hexamer`, seeds 0 through 31. Budget
4000 force evaluations. Smoke uses seed 0 and budget 12.

## Reproduction

Smoke, one engine and one arm:

```
ASE_PYTHON=/path/to/python PAPER_SMOKE=1 PAPER_OUT=water-hexamer-SMOKE.txt \
  WATER_ISOMER_DIR=/path/to/gfn2 \
  cargo run --offline --example water_hexamer --features ira,rgpot-ex,featomic,bank-rpc -- gfn2 recommended
```

Campaign shape, still not a tagged v0.12.0 result:

```
cargo run --offline --example water_hexamer --features rgpot-ex,featomic,bank-rpc -- gfn2 recommended
```

Replace `gfn2` with `pbe` or `petmad`, and `recommended` with
`communicating`, `derived`, or `good_turing`. `all all` is the default.
