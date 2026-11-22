<!-- Copyright 2023--present rgpot developers -->

# MDI compatibility frontend

The direct in-process API and the Cap'n Proto RPC server are rgpot's primary,
default paths. MDI is an optional compatibility layer for drivers that already
speak the MolSSI Driver Interface, including LAMMPS `fix mdi/qm`. It is disabled
unless the Cargo `mdi` feature or Meson `-Dwith_mdi=true` option is selected.

MDI is a frontend over `rgpot_potential_t`, not a potential backend. The
`rgpot-mdi` command constructs a potential through the same generic factory as
the other engine adapters, wraps it in `rgpot_potential_t`, and knows nothing
about LJ, EAM, CPMDC, or any other implementation.

## Build and run

The Meson option needs the Rust core:

<!-- rgpot-doc: skip requires the optional MDI runtime -->
```bash
set -euo pipefail
work=$(mktemp -d)
meson setup "$work/build-mdi" \
  -Dwith_rust_core=true \
  -Dwith_mdi=true
meson compile -C "$work/build-mdi" rgpot-mdi
```

CMake builds the same frontend with `-DRGPOT_WITH_MDI=ON`. It builds `rgpot-core`
with cargo and the Cargo feature `mdi`, then links `rgpot-mdi` and the
`rgpot-eval` helper the test uses. Neither tool is built on Windows or with
`-DRGPOT_RPC_CLIENT_ONLY=ON`, which is also true of Meson. The
`mdi_concrete_potentials` test is registered by both build systems when python
can import `mdi`.

Install the official MDI Library, either as a system library or from `pymdi`.
When it is not on the dynamic loader's search path, pass its shared object:

<!-- rgpot-doc: skip requires the optional MDI runtime -->
```bash
set -euo pipefail
work=$(mktemp -d)
meson setup "$work/build-mdi" \
  -Dwith_rust_core=true \
  -Dwith_mdi=true
meson compile -C "$work/build-mdi" rgpot-mdi
MDI_LIB=$(python -c \
  "import mdi, pathlib; print(pathlib.Path(mdi.__file__).with_name('libmdi.so'))")

"$work/build-mdi/engines/rgpot-mdi" \
  --pot "LJ u0 0.0104 psi 3.4 cutoff 8.5" \
  --mdi-library "$MDI_LIB" \
  -mdi "-role ENGINE -name RGPOT -method TCP -hostname 127.0.0.1 -port 8021"
```

The driver listens and the engine connects. A LAMMPS build with the MDI package
can use the same port through `fix mdi/qm`; use its `elements` keyword when the
potential needs atomic numbers. The MDI Library's TCP communicator is set to
`TCP_NODELAY` after connection. MDI Library 1.4.x is currently required for
TCP: MDI has no public socket-option API, so rgpot validates the 1.4
communicator ABI and stream socket before applying `TCP_NODELAY`.

TCP is refused on Windows. The MDI Library itself supports TCP there, but the
`TCP_NODELAY` step reads the 1.4 communicator and calls POSIX socket functions,
and rgpot has no validated Windows socket path. `rgpot_mdi_server_start` and
`serve_mdi` return an error that says so, before they load the library or
connect, when the options select `-method TCP` on a platform without POSIX
sockets. There is no fallback to a connection without `TCP_NODELAY`. Other MDI
methods are not affected by this check.

Rust embedders can enable `rgpot-core`'s `mdi` feature and call
`serve_mdi(potential, &MdiEngineConfig::default().with_options(...))`. C and
C++ embedders call `rgpot_mdi_server_start`, which every build exports: a
library compiled without the `mdi` feature returns `RGPOT_RPC_ERROR` and an
`rgpot_last_error()` message that names the feature. `RGPOT_MDI_LIBRARY`
supplies the shared-library path when neither API receives one explicitly.

## Commands and units

The engine registers these commands at `@DEFAULT`:

| Command | Wire value |
| --- | --- |
| `>NATOMS`, `<NATOMS` | one integer |
| `>ELEMENTS`, `>TYPES` | `N` integers |
| `>COORDS`, `<COORDS` | `3N` doubles, Bohr |
| `>CELL`, `<CELL` | nine row-major doubles, Bohr |
| `<ENERGY`, `<PE` | one double, Hartree |
| `<FORCES` | `3N` doubles, Hartree/Bohr |
| `EXIT` | terminate the engine |

`>ELEMENTS` and `>TYPES` both populate the generic potential's integer species
array; a driver should send the representation its potential expects. rgpot's
native Angstrom, eV, and eV/Angstrom conversion path is applied at the frontend
boundary. Coordinates, species, or cell changes invalidate the result. Energy,
potential energy, and forces from one unchanged geometry share one evaluation.

Malformed command ordering, non-finite geometry, invalid counts, potential
errors, and force tensors with the wrong shape or device terminate the
connection with a non-success status and an `rgpot_last_error()` message.
