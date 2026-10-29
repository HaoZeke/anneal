# Paper control drivers

Driver: `examples/paper_controls.rs`.

```text
PAPER_SMOKE=1 cargo run --example paper_controls -- <system> <arm>
```

Arms are `recommended`, `for_cluster`, `derived`, `soap-on`, and
`soap-off`. A smoke runs the first registered seed at 200 force
evaluations and writes a file whose name contains `SMOKE`. It is not a
paper table. The same command without `PAPER_SMOKE` is the full run and
launches from the v0.12.0 tag. A system whose paper budget is unset
refuses that full run.

Every log header carries the commit, CPU model, config digest
(`Config::resolved_sha256`), engine digest, seed, charged force count,
and wall time. The process also writes `RunManifest` JSON beside the
log. The manifest `build_identity` is the commit. The engine digest for
Lennard-Jones is `system_signature`'s `lennard-jones-reduced-v1` digest.
The Morse digest is the SHA-256 of `morse epsilon=1 r0=1 rho=<rho>`.

Success is an improvement whose energy is at most 1e-4 above a reference
that already appears in this repository. LJ31 and LJ76 have no such
reference, so those hits stay unscored. The search does not read the
reference.

## Conflicts with FIXES.md

IDEAS.md and OUTLINE-v2.md win.

- FIXES R1 asks for 200 seeds. The outline comparison table asks for 72
  LJ38 seeds and 48 LJ75 and LJ98 seeds. The driver uses the outline
  counts, taken from the start of the registered block 2000-2199. The
  rest of that block stays registered.
- FIXES R1 names Table 2 budgets, and the fix list also names 2.5e4
  Lennard-Jones 38 evaluations. The outline names 400000 for LJ38 and
  4000000 for LJ75 and LJ98. The driver uses the outline totals.
- FIXES R15 says to build GMIN in this VM. The outline puts GMIN and
  minima hopping on the optbench side. This binary does not build GMIN.
  `Config::for_cluster` is the in-tree screened basin-hopping control:
  `screen_steps` 25 and temperature 0.8, with no other field edited.
- FIXES R2 names held-out Morse and LJ landscapes and "the same budget
  rule as Table 2" without a number for LJ31, LJ76, LJ102, or Morse.
  Those paper budgets stay unset. References are copied only where
  `examples/cluster_bench.rs`, `examples/lj_ensemble_splice.rs`, or
  `examples/lattice_ensemble_ablation.rs` already records them.
- FIXES R10's second landscape is 12 LJ sites as 4 water-shaped groups.
  Those coordinates are not deposited in this binary. `lj12-groups`
  exits 2. `soap38` is the icosahedral 38-mer pair: seeds 5000-5063,
  budget 25000, the `ttt_lj38` icosahedral perturbation, soap on against
  soap off on `Config::recommended`.

## Systems

| system | n | seeds | paper budget | reference |
| --- | --- | --- | --- | --- |
| lj38 | 38 | 2000-2071 (72) | 400000 | -173.928427 |
| lj75 | 75 | 2000-2047 (48) | 4000000 | -397.492331 |
| lj98 | 98 | 2000-2047 (48) | 4000000 | -543.665361 |
| lj31 | 31 | 2200-2247 (48) | unset | unset |
| lj76 | 76 | 2200-2247 (48) | unset | unset |
| lj102 | 102 | 2200-2247 (48) | unset | -569.363652 |
| morse38-6, morse38-10, morse38-14 | 38 | 2200-2247 | unset | Doye-Wales bold entries in `cluster_bench` |
| morse55-6, morse55-10, morse55-14 | 55 | 2200-2247 | unset | same |
| morse80-6, morse80-10 | 80 | 2200-2247 | unset | unset |
| morse80-14 | 80 | 2200-2247 | unset | -340.811371 |
| soap38 | 38 | 5000-5063 (64) | 25000 | -173.928427 |

`recommended` and `derived` draw the preset sphere. `for_cluster` uses
that same start with the control configuration. `soap38` uses the
icosahedron. `derived` turns on `budget_window` and `bayes_screen` and
sets the threshold from `screen_steps` and `relax_steps`. `soap-off`
changes only `SoapProposalMode::Off` on `Config::recommended`.
