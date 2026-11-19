# xTB backends: linked vs dlopen

rgpot exposes two GFN-xTB force loading strategies over the same single-point
semantics (Å / eV host units; Bohr / Hartree inside libxtb).

| Backend | Class | Build | Runtime |
|---------|-------|-------|---------|
| **Linked** | `rgpot::XTBPot` | `-Dwith_xtb=true` (pkg-config `xtb`) | NEEDED `libxtb` in `librgpot` / host (`xtbpot` is a static archive inside `librgpot`) |
| **dlopen** | `rgpot::XTBDlopen` | **always** (no libxtb at link time); engine needs `with_xtb` to *build* | `dlopen` `libxtb_engine.so`; set path below |

`GFNMethod` / `XTBConfig` / `XTBDlopenConfig` live in `XTBConfig.hpp` (no
`#include <xtb.h>`), so hosts can compile against the dlopen frontend without
pkg-config xtb. Linked `XTBPot.hpp` still includes libxtb and requires
`-Dwith_xtb=true`.

Default method: **GFN2-xTB** (`GFNMethod::GFN2xTB` / `RGPOT_XTB_METHOD_GFN2`).
The generic `libxtb_engine` reads the `BuiltinParams.config.xtb` arm when it
is present; method, accuracy, electronic temperature, iteration limit, charge
and UHF are passed to `XTBPot`. An absent arm keeps the defaults above.

libxtb does not expose a Hessian through its public C API. Consequently this
engine advertises `analytic_hvp = 0`; hosts that need exact curvature must use
an engine whose underlying implementation supplies it.

## ISO_C_BINDING C API (libxtb)

libxtb is implemented in Fortran but the public surface used here is the
**ISO_C_BINDING** C API (`xtb.h`): interoperable types, `bind(c)` procedures,
and opaque C pointers. Hosts (linked `XTBPot` and `libxtb_engine.so`) only call
that C contract.

Contract tests care about:

| Call pattern | C API |
|--------------|--------|
| Create | `xtb_newEnvironment` / `newCalculator` / `newResults` (+ `xtb_releaseOutput`) |
| First geometry | `xtb_newMolecule` + `xtb_loadGFN*xTB` |
| Updates | `xtb_updateMolecule` then `xtb_singlepoint`, reusing the results object inside `warm_move` |
| Destroy | `xtb_del*` in reverse order |
| Units | host Å/eV; API Bohr/Hartree (converted in pot) |

Catch tags: `[xtb][capi]` — multi-handle lifecycle, warm update, method switch,
linked↔dlopen parity through the same ISO C semantics.

## Engine selection (`XTBDlopen`)

A non-empty `XTBDlopenConfig::engine_path` is exclusive: that path is the only
candidate, and it must contain a directory separator (`/` or `\`). A bare
soname such as `libxtb_engine.so` is rejected. The loader would resolve it
through `LD_LIBRARY_PATH`, `DYLD_LIBRARY_PATH`, or `PATH`, and could open a
different `libxtb_engine`. If the path does not open, or the library does not
export the C ABI, construction throws `std::runtime_error`. The error names that path and the
loader diagnostic (`dlerror` on POSIX). `RGPOT_XTB_ENGINE`, the bare soname,
and the directory lists are not consulted.

When `engine_path` is empty, search order is:

1. env `RGPOT_XTB_ENGINE`
2. env `XTB_ENGINE`
3. bare soname on the loader path (`libxtb_engine.so` on Linux,
   `libxtb_engine.dylib` then `libxtb_engine.so` on Apple,
   `xtb_engine.dll` then `libxtb_engine.dll` on Windows)
4. `EON_POTENTIALS_PATH` / `RGPOT_ENGINE_PATH` directory lists

Construct throws `std::runtime_error` if no engine is found (same pattern
as `MetatomicDlopen`).

## eOn / thin host ship

eOn and other thin hosts build with `-Dwith_xtb=false` and still link
`XTBDlopen`. Provide `libxtb_engine.so` (built once in an xtb-enabled tree or
install) via `RGPOT_XTB_ENGINE` at runtime. No NEEDED `libxtb` on the host.
`XTBDlopen` is the static archive `xtb_dlopen`, which is not installed. A
full build links it into `librgpot`. A client-only build links it into
the Meson subproject dependency only; an installed client-only tree does
not contain it (`docs/eon_pkgconfig.md`).

eOn’s in-tree `client/potentials/XTBPot` with `-Dwith_xtb=true` remains the
**linked ship** comparison target (same xtb C API + GFN2).

## Tests

<!-- rgpot-doc: syntax needs the xtbbld pixi environment and a separate build tree -->
```bash
pixi run -e xtbbld meson setup bbdir-xtb -Dwith_xtb=true -Dwith_tests=true \
  -Dwith_rpc=false -Dwith_cache=false --buildtype=debug
pixi run -e xtbbld meson compile -C bbdir-xtb
pixi run -e xtbbld meson test -C bbdir-xtb --suite xtb --print-errorlogs
```

Catch tags: `[xtb]`, `[xtb][linked]`, `[xtb][dlopen]`. Dlopen tests set
`RGPOT_XTB_ENGINE` to the built engine full path. `XTBDlopenTest` also puts
that engine on the process loader path and requires a missing explicit
`engine_path` to throw anyway.

Thin-host compile check (no libxtb):

<!-- rgpot-doc: syntax configures a separate build tree -->
```bash
meson setup bbdir-thin -Dwith_xtb=false -Dwith_tests=true \
  -Dwith_rpc=false -Dwith_cache=false
meson compile -C bbdir-thin
# XTBDlopen is linked; constructing it without RGPOT_XTB_ENGINE throws.
```

## Timing compare

rgpot microbench (linked + dlopen, warm SCF state):

<!-- rgpot-doc: syntax needs an xtb-enabled build tree -->
```bash
export RGPOT_XTB_ENGINE=$PWD/bbdir-xtb/CppCore/rgpot/XTBPot/libxtb_engine.so
./bbdir-xtb/CppCore/xtb_backend_bench --warmup 5 --iters 50 \
  --json rgpot_xtb_internal.json
```

Full head-to-head vs eOn ship (subprocess JSON merge):

<!-- rgpot-doc: syntax builds rgpot and eOn and runs both benchmarks -->
```bash
bash scripts/run_xtb_backend_bench.sh --out-dir /path/to/scratch
```

See `scripts/compare_xtb_backends.py` for flags. Protocol: shared water GFN2
geometry, warmup force calls, then timed samples; report mean wall ms and
whether dlopen is as-fast-or-faster than eOn linked ship (5% band).

## Wavefunction reuse

`XTBPot` and `XTBDlopen` keep one calculator. The SCC wavefunction in the
results object is the next guess while no atom moves by more than
`XTBConfig::warm_move` angstroms. The default is 1. `reset()`, a larger
move, or a new composition drops that object, and the next single point
uses xtb's own guess. `reusedWavefunction()` reports which guess the last
successful call used. `setWarmMove` changes the guard on a live handle.
Zero and negative values disable reuse.

The guess is also dropped when the HOMO-LUMO gap is below
`XTBConfig::warm_min_gap` electronvolts. The default is 1. With
near-degenerate frontier orbitals the GFN SCC can have several
self-consistent solutions. On hot Si13 a carried-over guess settles about
5 eV below the solution that xtb reaches from its own guess. Below the
gap, then, the energy comes from xtb's guess and does not depend on the
call history. A previous wavefunction under the gap is not reused. A
reused one that converges under it is solved again from xtb's guess.
`setWarmMinGap` changes the guard on a live handle. Zero and negative
values reuse at any gap.

An SCC that does not converge never seeds another one. A reused guess
that fails is retried from xtb's guess. Each rung of the smeared-occupation
rescue also starts from xtb's guess. The last iterate of a near-zero-gap
SCC can carry several electrons per atom. Seeded from it, GFN2 settles on
a self-consistent state tens of eV below the physical one. `TBLitePot`
replaces its result object after a failed SCF for the same reason.

The C engine ABI (`CppCore/rgpot/XTBPot/xtb_c_abi.h`, each declared with
`RGPOT_XTB_API`) has the same controls:

<!-- rgpot-doc: fragment -->
```c
void rgpot_xtb_reset(RgpotXtbPot *pot);
void rgpot_xtb_set_warm_move(RgpotXtbPot *pot, double angstrom);
double rgpot_xtb_warm_move(const RgpotXtbPot *pot);
void rgpot_xtb_set_warm_min_gap(RgpotXtbPot *pot, double ev);
double rgpot_xtb_warm_min_gap(const RgpotXtbPot *pot);
int rgpot_xtb_reused_wavefunction(const RgpotXtbPot *pot);
```

`xtb_warm_bench` times that warm step against a cold guess on (H2O)16 and
(H2O)20. It is built next to `xtb_backend_bench`. The comparison is serial:

<!-- rgpot-doc: syntax needs root and an xtb-enabled build tree -->
```bash
sudo unshare --uts bash -c 'hostname bench && \
  OMP_NUM_THREADS=1 taskset -c 0 \
  ./bbdir-xtb/CppCore/xtb_warm_bench --repeats 6'
```
