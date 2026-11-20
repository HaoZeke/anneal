# contact-cutoff-r1 gate

Verdict: PASS.

## Arms

DEVELOP is develop `61fa67cf`, tree `cff796c507a8fa1d0bfb28888c3570b61988055e`. CC is develop with `anneal-contact-cutoff-r1.mbox` applied by `git am -3` (one commit), tree `c5b961b7e51e83ae54d3f748f166ed74f0d5a432`. DEVELOP2 is a second, separate release build of a second checkout of develop, tree `cff796c507a8fa1d0bfb28888c3570b61988055e`, for the A/A band. The runner compared checkout and build trees against the expected values before the first run and after the last. Commit ids were not compared.

## Protocol

Each arm is a fresh release build (`build_arm.sh`, Rust 1.88.0) of `harness/`, which compiles the arm's own campaign drivers (`h2h_gmin_proof`, `h2h_sc`, `h2h_metal`, `h2h_morse`) from that arm's `examples/`. One wrapper is added around each driver's single `search(...)` call. It prints the best energy's bits and a SHA-256 of the outcome (`trace`). The runner checks that the driver's own row agrees with it on hit, charged count and the 8-decimal best. The harness lock is the arm's `Cargo.lock`, and every resolved package is pinned there. Sutton-Chen calls rgpot 3.7.0's `rgpot_sutton_chen_energy_forces`, the same library for all arms.

Runs are `run <system> <seed>` under `taskset -c 0`, one at a time, with every `*_THREADS` variable at 1 and no `H2H_*` or `ANNEAL_*` variable. The budget is the driver's own and was checked against the campaign `PAPER_BUDGETS` (`scripts/gmin_panel/run_panel.py`, `scripts/metal_h2h/run_sc.py`). There was no warm-up and no shortened run. Arm order reverses by seed: even seeds run DEVELOP, CC (part 1) or DEVELOP, CC, DEVELOP2 (part 2), and odd seeds run the reverse. A hit is a finite best at or below the reference plus 0.001. A miss is entered at the budget. Outside per call is `outside_s / charged` per row. Medians are Hyndman-Fan type 7.

Machine id (sha256) on every row: `3b11c160088e7af5c4f7ebd9da95e0bfe019a34698416701bb057c15fea10873`.
Harness blob (`harness/build.rs`) on every row: `6fa73f76c7c6760471932be4eae9aa56e176c096`.

## Rules, declared before the first run

Part 1 passes when best_bits, calls, charged and trace digest are identical for DEVELOP and CC on every cell and seed.

A part 2 cell fails if any of these holds:

- CC has fewer hits than DEVELOP. Any directional loss fails; exact Fisher one-sided p is reported.
- CC ranks worse on calls at one-sided Mann-Whitney p < 0.05. Misses are ranked worst.
- The CC/DEVELOP ratio of median outside per call is above 1.1 and above the A/A spread. The spread is the larger of DEVELOP2/DEVELOP and its reciprocal.

Overall PASS needs part 1 to pass and no part 2 cell to fail.

## Cells

Spec length in the code. `contact_cutoff(spec)` now returns
`CONTACT_CUTOFF * (spec.rcut_nn / SoapSpec::default().rcut_nn)`
(CC `src/soap.rs:86`). The search builds its SOAP spec with
`rcut_nn = SOAP_CUTOFF * cfg.length_scale`
(`src/methods/cluster_hopping/moves.rs:361`, `:1171`), so the spec
length of a cell is its `Config::length_scale`. Every campaign driver
builds its config with `Config::recommended(n)`:
`examples/h2h_gmin_proof.rs:68`, `examples/h2h_sc.rs:210`,
`examples/h2h_metal.rs:253` (au55 is `Kind::CopperGold { n_copper: 0 }`,
`:79-84`) and `examples/h2h_morse.rs:159`. `Config::recommended` passes
`LennardJonesPreset::REDUCED_SCALE`
(`src/methods/cluster_hopping/config.rs:1786-1791`), which is 1.0
(`src/methods/cluster_hopping/preset.rs:7`). The Morse driver uses its
pair length only when `H2H_PAIR_SCALE=1` (`examples/h2h_morse.rs:163-167`),
and this gate sets no `H2H_*` variable. So, as the code is written, every
campaign cell has spec length 1, including the Sutton-Chen cells.

The cells below are the panel cells whose geometry is not at the
Lennard-Jones length. These are the cells the gate was asked to test:

- Sutton-Chen in Cambridge units, nearest neighbour `a/sqrt(2)`
  (`scripts/metal_h2h/run_sc.py:71`): sc126-38, sc126-55, sc126-80,
  sc108-38, sc108-55, sc108-80, sc96-38, sc96-55, sc96-80
  (`scripts/metal_h2h/run_sc.py:38-46`, `examples/h2h_sc.rs:93-101`).
- Gupta gold, `r0 = 2.884` angstrom (`scripts/gmin_panel/run_panel.py:109`):
  au55 (`scripts/gmin_panel/run_panel.py:84`, `:102`).
- Morse, pair minimum at 1, `PairKind::length_scale` is
  `r_min / 2^(1/6)` (`src/potentials.rs:166-167`): m55r6, m55r14, m80r6,
  m80r14 (`scripts/gmin_panel/run_panel.py:85-88`, `:103-106`).

The remaining panel cells are Lennard-Jones (`scripts/gmin_panel/run_panel.py:76-83`).

Reach of the change. `contact_cutoff(spec)` is called from `target_421`,
`atom_w555`, `class_masses` and `ih_dominated` (`src/soap.rs:1306`,
`:1413`, `:1434`, `:1454`). In a search, these are reached through
`soap::step_away` (`src/methods/cluster_hopping/moves.rs:2638`) only when
`class` is set, and `class` is `cfg.soap_class_residual && packing`
(`moves.rs:1172`). Every preset sets `soap_class_residual` to false
(`config.rs:1835`, `:1943`, `:2228`, `:2428`, `:2548`).
`fivefold_openings` calls `ih_dominated` with `SoapSpec::default()`
(`src/soap.rs:2522`), which is unchanged.

## Part 1: bit identity at length 1

LJ38, LJ55 and LJ75 at budgets 400000, 1000000 and 4000000, seeds 2000-2002.

| Cell | Seed | best_bits DEVELOP | best_bits CC | calls | charged | trace | Identical |
| --- | --- | --- | --- | --- | --- | --- | --- |
| lj38 | 2000 | `c065bdb5abae6f95` | `c065bdb5abae6f95` | same (124155) | same (124155) | same | yes |
| lj38 | 2001 | `c065bdb5abae6fa2` | `c065bdb5abae6fa2` | same (111509) | same (111509) | same | yes |
| lj38 | 2002 | `c065bdb5abae6f97` | `c065bdb5abae6f97` | same (293837) | same (293837) | same | yes |
| lj55 | 2000 | `c07173f9bc2a0b2d` | `c07173f9bc2a0b2d` | same (5067) | same (5067) | same | yes |
| lj55 | 2001 | `c07173f9bc2a0b30` | `c07173f9bc2a0b30` | same (7542) | same (7570) | same | yes |
| lj55 | 2002 | `c07173f9bc2a0b31` | `c07173f9bc2a0b31` | same (2147) | same (2147) | same | yes |
| lj75 | 2000 | `c078c484175832b3` | `c078c484175832b3` | same (4000000) | same (4000000) | same | yes |
| lj75 | 2001 | `c078d7e09673e429` | `c078d7e09673e429` | same (3327845) | same (3327845) | same | yes |
| lj75 | 2002 | `c078c484175832b1` | `c078c484175832b1` | same (4000000) | same (4000000) | same | yes |

Part 1: PASS.

## Part 2: non-unit-length cells, n=12

| Cell | Budget | Hits D | Hits CC | Hits D2 | Fisher p (loss) | MW p calls (loss) | MW p calls (gain) | Median calls D | Median calls CC | Fail |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| sc126-38 | 400000 | 12/12 | 12/12 | 12/12 | 1.0000 | 0.5115 | 0.5115 | 113504.5 | 113504.5 | no |
| sc126-55 | 1000000 | 12/12 | 12/12 | 12/12 | 1.0000 | 0.5115 | 0.5115 | 18379.5 | 18379.5 | no |
| sc126-80 | 1000000 | 5/12 | 5/12 | 5/12 | 0.6599 | 0.5129 | 0.5129 | 1000000.0 | 1000000.0 | no |
| sc108-38 | 400000 | 7/12 | 7/12 | 7/12 | 0.6599 | 0.5120 | 0.5120 | 361031.5 | 361031.5 | no |
| sc108-55 | 1000000 | 5/12 | 5/12 | 5/12 | 0.6599 | 0.5129 | 0.5129 | 1000000.0 | 1000000.0 | no |
| sc108-80 | 1000000 | 3/12 | 3/12 | 3/12 | 0.6798 | 0.5152 | 0.5152 | 1000000.0 | 1000000.0 | no |
| sc96-38 | 400000 | 12/12 | 12/12 | 12/12 | 1.0000 | 0.5115 | 0.5115 | 50230.0 | 50230.0 | no |
| sc96-55 | 1000000 | 12/12 | 12/12 | 12/12 | 1.0000 | 0.5115 | 0.5115 | 23157.0 | 23157.0 | no |
| sc96-80 | 1000000 | 7/12 | 7/12 | 7/12 | 0.6599 | 0.5120 | 0.5120 | 989847.5 | 989847.5 | no |
| au55 | 1000000 | 4/12 | 4/12 | 4/12 | 0.6666 | 0.5137 | 0.5137 | 1000000.0 | 1000000.0 | no |
| m55r6 | 1000000 | 12/12 | 12/12 | 12/12 | 1.0000 | 0.5115 | 0.5115 | 22596.0 | 22596.0 | no |
| m55r14 | 1000000 | 9/12 | 9/12 | 9/12 | 0.6798 | 0.5116 | 0.5116 | 451561.5 | 451561.5 | no |
| m80r6 | 4000000 | 12/12 | 12/12 | 12/12 | 1.0000 | 0.5115 | 0.5115 | 62677.0 | 62677.0 | no |
| m80r14 | 4000000 | 6/12 | 6/12 | 6/12 | 0.6579 | 0.5123 | 0.5123 | 3901429.0 | 3901429.0 | no |

Outside time per charged call (s), median of the per-row values:

| Cell | D | CC | D2 | CC/D | D2/D (A/A) | A/A spread | Cost fail |
| --- | --- | --- | --- | --- | --- | --- | --- |
| sc126-38 | 2.150318e-06 | 2.156673e-06 | 2.197070e-06 | 1.0030 | 1.0217 | 1.0217 | no |
| sc126-55 | 4.314537e-06 | 4.348817e-06 | 4.428019e-06 | 1.0079 | 1.0263 | 1.0263 | no |
| sc126-80 | 1.132080e-05 | 1.108387e-05 | 1.090009e-05 | 0.9791 | 0.9628 | 1.0386 | no |
| sc108-38 | 2.584808e-06 | 2.582410e-06 | 2.574424e-06 | 0.9991 | 0.9960 | 1.0040 | no |
| sc108-55 | 5.085670e-06 | 5.137777e-06 | 5.150613e-06 | 1.0102 | 1.0128 | 1.0128 | no |
| sc108-80 | 1.004348e-05 | 1.039194e-05 | 1.033121e-05 | 1.0347 | 1.0286 | 1.0286 | no |
| sc96-38 | 2.290530e-06 | 2.278217e-06 | 2.235205e-06 | 0.9946 | 0.9758 | 1.0248 | no |
| sc96-55 | 4.116389e-06 | 4.072984e-06 | 4.146253e-06 | 0.9895 | 1.0073 | 1.0073 | no |
| sc96-80 | 9.796582e-06 | 1.005880e-05 | 9.995671e-06 | 1.0268 | 1.0203 | 1.0203 | no |
| au55 | 9.254160e-06 | 9.286048e-06 | 9.304109e-06 | 1.0034 | 1.0054 | 1.0054 | no |
| m55r6 | 5.024418e-06 | 4.956903e-06 | 4.996721e-06 | 0.9866 | 0.9945 | 1.0055 | no |
| m55r14 | 4.470638e-06 | 4.427448e-06 | 4.452637e-06 | 0.9903 | 0.9960 | 1.0040 | no |
| m80r6 | 1.167603e-05 | 1.177271e-05 | 1.162339e-05 | 1.0083 | 0.9955 | 1.0045 | no |
| m80r14 | 9.748361e-06 | 9.469508e-06 | 9.519216e-06 | 0.9714 | 0.9765 | 1.0241 | no |

Per seed, CC against DEVELOP (and DEVELOP2 against DEVELOP), counted over 12 seeds:

| Cell | best_bits same | charged same | trace same | A/A best_bits same | A/A charged same | Seeds that differ |
| --- | --- | --- | --- | --- | --- | --- |
| sc126-38 | 12/12 | 12/12 | 12/12 | 12/12 | 12/12 | none |
| sc126-55 | 12/12 | 12/12 | 12/12 | 12/12 | 12/12 | none |
| sc126-80 | 12/12 | 12/12 | 12/12 | 12/12 | 12/12 | none |
| sc108-38 | 12/12 | 12/12 | 12/12 | 12/12 | 12/12 | none |
| sc108-55 | 12/12 | 12/12 | 12/12 | 12/12 | 12/12 | none |
| sc108-80 | 12/12 | 12/12 | 12/12 | 12/12 | 12/12 | none |
| sc96-38 | 12/12 | 12/12 | 12/12 | 12/12 | 12/12 | none |
| sc96-55 | 12/12 | 12/12 | 12/12 | 12/12 | 12/12 | none |
| sc96-80 | 12/12 | 12/12 | 12/12 | 12/12 | 12/12 | none |
| au55 | 12/12 | 12/12 | 12/12 | 12/12 | 12/12 | none |
| m55r6 | 12/12 | 12/12 | 12/12 | 12/12 | 12/12 | none |
| m55r14 | 12/12 | 12/12 | 12/12 | 12/12 | 12/12 | none |
| m80r6 | 12/12 | 12/12 | 12/12 | 12/12 | 12/12 | none |
| m80r14 | 12/12 | 12/12 | 12/12 | 12/12 | 12/12 | none |

Gains (CC more hits, calls better at one-sided p < 0.05, or cost below 1/1.10 and outside the A/A spread): none.

Part 2: PASS.

## Special casing

None. The diff changes `src/soap.rs:86` (the return value of
`contact_cutoff`), rewrites `tests/contact_cutoff_length.rs`, and adds
`docs/newsfragments/+contact-cutoff.fixed.rst`. None of the added
lines names a system, a size, a cell, a potential or an element, and
the diff adds no branch.

## Result

PASS. No failing case. On part 2, CC equals DEVELOP in best_bits, charged and trace on all 168 cell-seeds, so the hit and calls columns are equal and the cost ratios are timing of identical work. The gate saw no change in behaviour on these cells, which is what the reach analysis under Cells predicts. It does not exercise the new cutoff.

## Files

- `rows.csv`: one row per run, in run order (`slot`).
- `provenance.csv`: binary hashes per arm, the rgpot library and the toolchain.
- `verdict.py`: writes this file from `rows.csv` and `provenance.csv`.

Binaries:

| Arm | File | sha256 |
| --- | --- | --- |
| DEVELOP | h2h_gmin_proof | `efc8f7473652af3b0fade7515789ae4d0c9379909aacfd8221a5d8e4c9b82911` |
| DEVELOP | h2h_sc | `837dfed9e18d5a503d932a437e7dce615db689b0a412b6d91422898e5fc402e6` |
| DEVELOP | h2h_metal | `d4c9d76e94dc4df0bc94bac57bbfb96f5f4aaf27591586a91f91723a194c3efa` |
| DEVELOP | h2h_morse | `837d93bdcc9bd7d0aa21bbe3eee15abd800f7fa319d54f3a4751af33b70be325` |
| CC | h2h_gmin_proof | `094c8d8ca37aea7eada199c0cde65890943cee4ae7ba394b73b0e99d6de9a8ae` |
| CC | h2h_sc | `d8497778cd2910fa95fac4034f54ebff8bcbb2924ce44586fbbc4f03364ff5a8` |
| CC | h2h_metal | `f0f10f1b8a68cbe9717cc3c05b0a1d59b3cbd0a2d46b5595e0eac4bb7204fc1c` |
| CC | h2h_morse | `e8e72627a55fa5051b21cd9ec5ccf7f5c6d9cfa585ee677a76d75a80908f97e5` |
| DEVELOP2 | h2h_gmin_proof | `a05910ed80eaff230386aa2e437a6e212e92f9dcc47effe6719ab29b18c51d99` |
| DEVELOP2 | h2h_sc | `07aad286b0ad9c6e127ab3bced70575fe7be894abc101497c7df164d4664caca` |
| DEVELOP2 | h2h_metal | `51ef741481dd1718dc99ac98782170963faf7816a28a7a454f12754a25b17286` |
| DEVELOP2 | h2h_morse | `bf1d6d33c4600d6b5b5563a358c87791c4fc36412f9694e270b759979f2c3623` |
| all | rgpot-3.7.0-cp312-abi3-manylinux_2_28_x86_64.whl | `ddc2280357786be3880f09cf40ba1f2069a8bff4bcdcb48babdaedd9c3095cfd` |
| all | librgpot.so.6 | `d80f3e4c6c845a6909d64aa55ca17bb93409cf29c19f18845b37a826c6c390d0` |
