# CP2KPot

`CP2KPot` is the rgpot frontend for CP2K Quickstep. It renders a CP2K input
deck from `CP2KParams` and the first geometry, then loads `libcp2kc_rgpot`, a
thin shim over libcp2k, at runtime and calls the C ABI in `cp2kc_abi.h`.

| Carrier | Owner | Role |
| --- | --- | --- |
| `PotentialConfig.cp2k` | rgpot RPC clients | selects the CP2K backend arm |
| `CP2KParams` | rgpot clients, rendered by `CP2KDeck` | method and deck |
| `ForceInput` | every force call | coordinates, atomic numbers, box (Angstrom) |
| `cp2kc_session_calculate` | the shim | energy (Hartree) and forces (Hartree/Bohr) |

## Building the shim

The shim is the only target that links libcp2k and MPI. Build it against a
libcp2k that ships `libcp2k.pc`, for example the conda-forge `cp2k` package:

```bash
export PKG_CONFIG_PATH=/path/to/cp2k/prefix/lib/pkgconfig:$PKG_CONFIG_PATH
meson setup bbdir -Dwith_tests=true -Dwith_rpc=true -Dwith_mpi=enabled \
  -Dwith_cp2k=enabled
meson compile -C bbdir
meson test -C bbdir --suite cp2k
```

`bbdir/CppCore/rgpot/CP2KPot/libcp2kc_rgpot.so` carries an rpath to the
libcp2k library directory.

`CP2KPot`, the potserv `CP2K` backend and the shim build together, only with
`-Dwith_cp2k=enabled` (CMake `-DRGPOT_WITH_CP2K=ON`), which needs
`-Dwith_rpc=true` and libcp2k. Without it no CP2K code is compiled: potserv
exits with an error when asked for the `CP2K` backend, and every backend's
`configure()` refuses `PotentialConfig.cp2k` by name.

## Shim loading

| Source | Meaning |
| --- | --- |
| `CP2KParams.enginePath` | explicit path carried in the config message |
| `CP2KC_LIBRARY` | process-level shim path |
| `RGPOT_CP2K_ENGINE` | rgpot-specific shim path |
| `libcp2kc_rgpot.so`, `./libcp2kc_rgpot.so` | default probe names |

## The deck

With `inputTemplate` empty, `CP2KDeck` writes a GPW deck from the typed fields:
`&MGRID CUTOFF`/`REL_CUTOFF`, `CHARGE`, `MULTIPLICITY`, `UKS`, `&SCF` with
`EPS_SCF`, `MAX_SCF`, OT DIIS with `FULL_SINGLE_INVERSE` when `ot` is set, an
`&OUTER_SCF` when `outerMaxScf > 0`, the `&XC_FUNCTIONAL`, and one `&KIND` per
element with `basisSet` and `pseudopotential`. `extraDftSections` and
`extraForceEvalSections` are appended verbatim inside `&DFT` and
`&FORCE_EVAL`.

A non-empty `inputTemplate` is used as written, except that `@RGPOT_COORD@`
becomes one `Symbol x y z` row per atom and `@RGPOT_CELL@` becomes the `A`, `B`
and `C` rows, all in Angstrom. A template without both tokens is refused by
`setParams`. Every deck must set `IGNORE_CONVERGENCE_FAILURE` in `&SCF`;
otherwise CP2K calls `MPI_ABORT` on an unconverged SCF.

The deck is written on every rank to
`<workDir|$TMPDIR>/rgpot-cp2k-<pid>-g<group>-r<rank>-i<instance>/input.inp`,
and CP2K writes `cp2k.out` beside it. The directory is removed when the
session ends unless `keepWorkDir` is set. `dataDir` sets `CP2K_DATA_DIR`.

## Sessions and orbital keys

The session is created at the first force and recreated only when the
sequence of atomic numbers or the params change. Each step sets the positions,
sets the cell only when it changed, and evaluates energy and forces on the
same force environment, so CP2K's wavefunction extrapolation carries over.
`selectOrbitals(key)` gives each band image or bead its own force environment.

## Failures

An unconverged SCF returns code 2 from the shim on every rank of the
calculator, and `CP2KPot` throws `rgpot::CP2KScfError` without aborting. Other
failures follow the `CPMDPot` path: the message prints on every rank of the
calculator and `MPI_COMM_WORLD` aborts.
