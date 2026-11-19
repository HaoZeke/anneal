# Standalone minima hopping

Head-to-head of the anneal examples against the Goedecker-group minima
hopping package, distribution name `minimahopping`, tag 1.1.3, revision
`ff7e32bbd9a9b235f5f2a1dec11b311bf874300d`. This is the package described
by Gubler, Krummenacher, Huber, and Goedecker, SoftwareX 25, 101632
(2024). It is not `ase.optimize.minimahopping`.

The project name is not published on PyPI. The pin, the archive hash,
the sdist hash, and the wheel hash are in
`scripts/ase_panel/mhpkg_pin.json`. The install line is
`scripts/ase_panel/mhpkg-requirements.txt`.

numpy is pinned to 2.4.6. From numpy 2.5.0, `linalg.eig` returns a
complex eigenvector for the real symmetric inertia tensor, and the
package subtracts that vector into a real velocity array when it
removes torque. 2.4.6 is the last release whose result for that tensor
is real. The rest of the stack is the set that installed against that
numpy and ASE 3.29.0.

## What is compared

Both arms evaluate the same potential and count one call for each
energy-and-gradient pair.

| Cell | Potential | Binary | Budget | Cambridge minimum |
| --- | --- | --- | --- | --- |
| lj38, lj75, lj98, lj104 | `PairPotential::lennard_jones` | `h2h_gmin_proof` | paper list | Lennard-Jones table |
| m80r6, m80r14 | `PairPotential::morse` | `h2h_morse` | 4000000 | Morse table |
| au38 | `Gupta::copper_gold(0, 38)` | `h2h_metal` | 400000 | Gupta gold minimum |

The budget and the minimum are the values those examples already print
from `budget`. The harness refuses a cell when the binary and
`mhpkg_eval spec` disagree with that table. A hit is a quenched energy
at or below the minimum plus 0.001, the tolerance in
`scripts/target_tol.py`.

Seeds are 2000 through 2011. Both arms open the coordinate file written
by the example `coords` command for that seed, ten decimal places. The
order inside a seed pair is ABBA on the index `seed - 2000`: an even
index runs anneal in slot 0 and the package in slot 1, and an odd index
reverses the slots. A one-seed invocation keeps that index, so the
command stored on a row repeats the slot the campaign used.

A same-binary pair on lj38 runs the anneal example against itself under
the same ABBA schedule. Those rows are `aa0` and `aa1` in `aa_rows.csv`.
They are the noise band.

## Units

The documented package numbers are for a calculator in eV and Å. The
conversion to a native calculator is one rule:

- `s_E` is the number of native energy units in 1 eV.
- `s_L` is the number of native length units in 1 Å.
- `s_F = s_E / s_L`.
- `s_T = s_E`. The velocity draw uses `k_B` in eV/K and a mass of 1, and
  the integrator uses that same mass, so a temperature scales as an energy.
- `s_dt = 1 / sqrt(s_F)`, so a step of the Verlet update moves the same
  distance when forces and speeds are expressed in the native units.

Lennard-Jones identifies `epsilon` with 1 eV and `sigma` with 1 Å.
Morse identifies `epsilon` with 1 eV and `r_0` with 1 Å. Gupta gold is
already eV and Å. In all three cases every scale factor is 1, and the
numbers below are the values that are passed.

Rigid TIP4P is not run. Its energy unit in this repository is kJ/mol
and its length unit is Å, so `s_L = 1` and `s_E` is
`ase.units.eV / (ase.units.kJ / ase.units.mol)` with ASE 3.29.0, which
is 96.4853328824988. `s_F` and `s_T` equal that number and `s_dt` is
its inverse square root. The cell is still not run; the scale is
recorded only so the rule has a value for that unit system.

## Parameters

The cluster example at the pinned revision sets `T0=500`, `dt0=0.1`,
and `use_MPI=False`. Every other physics parameter below is the
constructor default in `minimahopping/mh/parameters.py` at that
revision. Where the narrative parameter page disagrees with the
constructor, the constructor is the one the example leaves in place.
`md_max_steps` is 10000 in the constructor.

| Name | Value | Scales as |
| --- | --- | --- |
| T0 | 500 | temperature |
| dt0 | 0.1 | time |
| Ediff0 | 0.1 | energy |
| beta_decrease | 1/1.05 | dimensionless |
| beta_increase | 1.05 | dimensionless |
| alpha_accept | 1/1.05 | dimensionless |
| alpha_reject | 1.05 | dimensionless |
| n_soft | 20 | count |
| soften_positions | 0.001 | length |
| soften_lattice | 0.001 | length |
| n_S_orbitals | 1 | count |
| n_P_orbitals | 0 | count |
| width_cutoff | 4 | length |
| dt_min | 0.0001 | time |
| mdmin | 2 | count |
| md_max_steps | 10000 | count |
| fmax | 0.005 | force |
| energy_threshold | 0.001 | energy |
| fingerprint_threshold | 0.05 | fingerprint distance |
| enhanced_feedback | false | flag |
| margin | 0.3 | dimensionless |
| opt_max_steps | 10000 | count |
| use_MPI | false | flag |
| fixed_cell_simulation | false | flag |

Three switches do not enter the acceptance or the integration. They
only open files, and they are off: `verbose_output` (the constructor
default; the cluster example turns it on to write trajectories),
`collect_md_data`, and `write_graph_output`. `new_start` is on so a
leftover restart file cannot resume a seed. The log level is warning.

The package sets every mass to 1 before the Maxwell-Boltzmann draw, and
the integrator divides by that same mass. The element does not change
the kinetic energy. It does change the covalent radius used for the
fragmentation check and the width of the overlap-matrix fingerprint.

Lennard-Jones and Morse have no chemical element. The symbol is the
entry in the package periodic table whose estimated bond
`2 * rcov * Bohr` is the smallest bond at least as long as the pair
minimum. That minimum is `2^(1/6)` for Lennard-Jones and 1 for Morse.
At this revision both resolve to Ne. Gold uses Au.

## How a run is made

Release binaries, `cargo +1.88.0 build --release --locked` of
`h2h_gmin_proof`, `h2h_morse`, `h2h_metal`, and `mhpkg_eval`. The
process is confined to cpu 0 with `taskset -c 0`. The thread variables
listed in the stored command are 1, and `LJOS_SEAT=benchbot`.

The anneal arm is one example process. `H2H_WALL_S` is 7200 and
`H2H_COORDS` is the shared start. Unset, those variables leave the
example on its existing path. A run that reaches the cap is a miss; the
row keeps the calls, the best energy, and the wall.

The package arm is one Python process with `use_MPI=False`, and the MPI
world size must be 1. The package still calls `MPI_Init` when it is
imported. A second process, `mhpkg_eval serve`, answers one coordinate
line with one `value_and_gradient`. Both processes inherit the cpu 0
affinity. There is no MPI launch and no thread pool. ASE 3.29.0 removed
the `communicator` keyword the release passes to
`MaxwellBoltzmannDistribution`. The harness rewrites `communicator='serial'`
to `comm=DummyMPI()`, which is the replacement ASE 3.28 applies for
that keyword.

A quenched point is an evaluation whose largest absolute force
component is at most `fmax`. The best energy is the lowest such point.
If the cap arrives before any quench finishes, the best energy is the
lowest energy evaluated. The startup quench is part of the search and
its calls are counted. There is no discarded warm-up hop. The first
seed of a session pays the one-time compilation of the package's
compiled kernels; the cache lives in the work directory and is not
shared with a smoke run.

Each row stores the package revision and one shell command that checks
out the harness revision, builds the examples, installs the pinned
requirements, and repeats that row. `verify` reruns one stored row and
checks the hit, the calls, the calls-to-hit, the stop, and the best
energy within `1e-8`. The wall is not required to match.

```text
python3 scripts/ase_panel/mhpkg.py run --cells lj38 --seeds 2000-2011 --panel h2h
python3 scripts/ase_panel/mhpkg.py run --cells lj38 --seeds 2000-2011 --panel aa
python3 scripts/ase_panel/mhpkg.py verify --cell lj38 --arm anneal --seed 2000
```

## Rigid TIP4P

`(H2O)n` for `n` from 8 to 21 is listed in `unsupported.csv` with the
Cambridge energy from `data/tip4p_ccd.csv` and the 2000000-call budget
used by `scripts/tip4p_h2h/run_h2h.py`. The same budget is the paper
budget recorded for the sizes that `run_campaign.py` and `run_radius.py`
name. The cell is not run.

The package moves Cartesian atoms. Its documented constraint example is
`FixAtoms`, and the example says that other ASE constraints are not all
tested. `biomode` is imported by the softening step and is never called,
and it does not integrate a rigid body. Rigid TIP4P here is a
six-coordinate angle-axis chart per molecule (`GminTip4p`). There is no
supported rigid-molecule interface, so those cells are not given a
Cartesian substitute.
