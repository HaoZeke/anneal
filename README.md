

# rgpot

![img](https://raw.githubusercontent.com/OmniPotentRPC/rgpot/refs/heads/main/branding/logo/rgpot_logo.webp)

[![img](https://codecov.io/gh/OmniPotentRPC/rgpot/graph/badge.svg)](https://app.codecov.io/gh/OmniPotentRPC/rgpot)

`rgpot` is a potential-energy library and Cap'n Proto RPC server for atomistic
simulation codes.
Documentation: [rgpot.rgoswami.me](https://rgpot.rgoswami.me).
It gives clients one geometry carrier, one result carrier, and one extensible
backend-configuration carrier:

-   `ForceInput`: positions, atomic numbers, box, and unit strings
-   `PotentialResult`: energy, a flat force array, and optional outputs such as
    the Hessian or the stress
-   `PotentialConfig`: backend setup, with arms such as `none`, `nwchem`, and
    `cpmd`

XC response kernels (`rgpot::XcKernel`, meson `-Dwith_xckernel=true`) are
in-process only.
They are not a `Potential`, and there is no `PotentialConfig.xckernel` arm:
the operands are AO collocation arrays and named Libxc arrays, not `ForceInput`.
See `docs/xckernel.md`.
`pylibxc` is not on PyPI (`pylibxc2` there is an unrelated stub); the pixi
`xckernel` environment (`pixi install -e xckernel`, linux-64 only) takes
`libxc` and `pylibxc` from conda-forge.

`D3Pot` and `D4Pot` (meson `-Dwith_dftd3` / `-Dwith_dftd4`) are in-process
`Potential` summands: construct them and evaluate `ForceInput` to energy and
forces.
The schema has `PotentialConfig.d3` @12 and `PotentialConfig.d4` @13 arms.
Nothing applies them yet.
The `potserv` `D3` and `D4` backends run the default configuration.
`configure()` refuses both arms by name on every `potserv` backend.
A DFT host sums XcKernel + VV10 + D4 itself; dispersion is not folded into
the XC kernel.

`ExprPot` (meson `-Dwith_expr=true`) compiles a Lepton energy string over
named child `Potential` objects.
The parser is vendored OpenMM Lepton (`third_party/lepton`); there is no pixi
muparser feature.
The schema carries `PotentialConfig.expr` @16, but `potserv` does not apply it
and refuses it by name.

In-process `LJPot` vs pyeonclient `Matter` on the same LJ fixture is
`scripts/time_lj_rgpot_vs_pyeonclient.py` (on a host listed in
`RGPOT_REGEN_HOSTS`). On workstation-a
(2026-09-04, 1e6 calls) A (`LJPot::operator()`) was 31-33 ns/call and
B (in-process pyeonclient `Matter.forces`, not potserv) was 687 to 791
ns/call. `XcKernel` is not a `Potential` and is not part of that race.
See `docs/orgmode/howto/exprpot.org`.

From Python, `rgpot.ase.RgpotCalculator` is an ASE calculator over any bound
potential:
`atoms.calc = RgpotCalculator("lj", u0=0.0104, psi=3.4, cutoff=8.5)`.
See `docs/orgmode/howto/ase.org`.

The public wire schema lives in `CppCore/rgpot/rpc/Potentials.capnp` and is
shared by the C++ server, Python integration tests, and the Rust core crate.
That file is the pinned potentials-schema release byte for byte; the build
adds the rgpot-local fields listed in `CppCore/rgpot/rpc/local/series` and
installs the result.
Native rgpot units are eV and Angstrom.
RPC clients may request compatible units through `ForceInput.lengthUnit` and
`ForceInput.energyUnit`.

EAM, Lennard-Jones, Sutton-Chen, Stillinger-Weber, Tersoff, Morse and ZBL take
candidate pairs from one neighbour layer, `rgpot::nlist::Neighbours`.
It keeps a stored Verlet list or walks the cells without storing pairs.
With `OMP_NUM_THREADS` unset, empty or below 1, the layer runs on one thread in
the serial pair order.
A fixed thread count repeats bit for bit.
See `docs/neighbour_layer.md`.


## Python (PyPI)

`.github/workflows/wheels.yml` builds the wheels with cibuildwheel for Linux
x86<sub>64</sub> and aarch64 (manylinux) and for macOS 15.
CPython 3.12 and newer share one nanobind **abi3** wheel.
CPython 3.10 and 3.11 get version-specific wheels, and free-threaded CPython
3.14 gets its own wheel.
`requires-python` is `>=3.10`.

```bash
pip install rgpot
# Metatomic path (engine libs resolve from these packages at runtime):
pip install 'rgpot[metatomic]'
# or: pip install torch metatomic-torch metatensor-torch metatensor-core vesin
```

The wheel build turns on ExprPot (vendored Lepton), D3Pot, D4Pot, XcKernel and
the Fortran-backed potentials (`[tool.meson-python.args]` in
`pyproject.toml`).
`s-dftd3` / `dftd4` are built into the wheel; no conda install is required at
`pip install` time.
Lennard-Jones needs only `numpy`.
Metatomic loads its engine library with **dlopen**:

-   `rgpot.default_metatomic_engine_path()` looks in
    `rgpot/lib/torch-X.Y/` for the installed `torch` X.Y first
    (`librgpot_metatomic_engine.so`, then `libmetatomic_engine.so`), then in the
    other `torch-*` directories, then next to the package
-   the CI wheel build installs one engine, built against the torch that the
    isolated build installs (`torch>=2.7` in the build requirements)
-   `scripts/rgpot_pack_multi_abi_engines.sh` builds engines for torch 2.7
    through 2.13 (`RGPOT_TORCH_MAJORS` overrides the list) and adds them to a
    wheel under `rgpot/lib/torch-X.Y/`
-   torch 2.6 and older are not supported

```python
import numpy as np
import rgpot

print(rgpot.__version__)
print(rgpot.has_expr, rgpot.has_dftd3, rgpot.has_dftd4, rgpot.has_xckernel, rgpot.has_ipi)
print(rgpot.available_metatomic_engine_abis())

positions = np.array([[0.0, 0.0, 0.0], [1.1, 0.0, 0.0]])
atom_types = [0, 0]
box = np.diag([15.0, 20.0, 30.0])
energy, forces, variance = rgpot.LJPot()(positions, atom_types, box)
```

`ExprPot` takes a Lepton string and a dict that maps each name in it to a
child kind (`'lj'`, `'morse'`, `'d3'`, `'d4'`):

```python
import numpy as np
import rgpot

positions = np.array([[0.0, 0.0, 0.0], [1.1, 0.0, 0.0]])
box = np.diag([15.0, 20.0, 30.0])
pot = rgpot.ExprPot("0.5*lj + morse", {"lj": "lj", "morse": "morse"})
energy, forces, variance = pot(positions, [0, 0], box)
```

`rgpot.D3Pot()` needs `rgpot.has_dftd3`.
`rgpot.evaluate_metatomic(positions, atom_types, box, model_path="model.pt")`
needs a Metatomic engine.
See <https://pypi.org/project/rgpot/> .


## Build And Test

Meson is the primary build path.
Use the `rpctest` environment when running the RPC integration scripts because
they need pycapnp.

```bash
pixi shell -e rpctest
meson setup bbdir -Dwith_tests=true -Dwith_rpc=true --buildtype=debug
meson compile -C bbdir
meson test -C bbdir --print-errorlogs
python tests/rpc_integ.py --server-bin ./bbdir/CppCore/potserv
```

CMake is supported as well.
CMake's `RGPOT_WITH_CACHE` defaults to `ON` and needs RocksDB, unlike Meson's
`with_cache=false`, so turn it off unless RocksDB is installed (the pixi
`devbld` environment has it).
`docs/orgmode/reference/build_options.org` lists every option.

```bash
cmake -B build-cmake \
  -DRGPOT_BUILD_TESTS=ON \
  -DRGPOT_BUILD_EXAMPLES=ON \
  -DRGPOT_WITH_RPC=ON \
  -DRGPOT_WITH_CACHE=OFF
cmake --build build-cmake
ctest --test-dir build-cmake --output-on-failure
```

For client-only consumers that only need the Cap'n Proto schema and generated
RPC client types (`RGPOT_RPC_CLIENT_ONLY` turns the cache off):

```bash
cmake -B build_client \
  -DRGPOT_RPC_CLIENT_ONLY=ON \
  -DRGPOT_BUILD_TESTS=ON
cmake --build build_client
ctest --test-dir build_client --output-on-failure
```


## Backends

<table border="2" cellspacing="0" cellpadding="6" rules="groups" frame="hsides">


<colgroup>
<col  class="org-left" />

<col  class="org-left" />

<col  class="org-left" />
</colgroup>
<thead>
<tr>
<th scope="col" class="org-left">Backend</th>
<th scope="col" class="org-left">Server selector</th>
<th scope="col" class="org-left">Build / runtime notes</th>
</tr>
</thead>
<tbody>
<tr>
<td class="org-left"><code>LJPot</code></td>
<td class="org-left"><code>LJ</code></td>
<td class="org-left">Built-in 12-6 Lennard-Jones reference potential</td>
</tr>

<tr>
<td class="org-left"><code>LJClusterPot</code>, <code>MorsePot</code>, <code>ZBLPot</code></td>
<td class="org-left"><code>LJCluster</code>, <code>Morse</code>, <code>ZBL</code></td>
<td class="org-left">Built-in; <code>LJCluster</code> has free boundaries</td>
</tr>

<tr>
<td class="org-left"><code>CuH2Pot</code></td>
<td class="org-left"><code>CuH2</code></td>
<td class="org-left">Fortran-backed Cu-H EAM potential; needs <code>-Dwith_fortran_pots</code> (<code>auto</code>)</td>
</tr>

<tr>
<td class="org-left"><code>XTBPot</code></td>
<td class="org-left"><code>XTB</code>, <code>GFNFF</code>, <code>GFN0xTB</code>, <code>GFN1xTB</code></td>
<td class="org-left">Enable with <code>-Dwith_xtb=true</code>; use pixi env <code>xtbbld</code> or <code>tbbld</code>. <code>potserv</code> uses the linked <code>XTBPot</code>; <code>XTBDlopen</code> loads <code>libxtb_engine.so</code> (see <code>docs/xtb_backends.md</code>)</td>
</tr>

<tr>
<td class="org-left"><code>TBLitePot</code></td>
<td class="org-left"><code>TBLite</code> (or <code>TBLiteGFN2</code>), <code>TBLiteGFN1</code>, <code>TBLiteIPEA1</code></td>
<td class="org-left">Enable with <code>-Dwith_tblite=true</code>; use pixi env <code>tblitebld</code> or <code>tbbld</code></td>
</tr>

<tr>
<td class="org-left"><code>D3Pot</code></td>
<td class="org-left"><code>D3</code></td>
<td class="org-left">In-process Potential summand. Enable with <code>-Dwith_dftd3=true</code>; use pixi env <code>dftd3</code> or <code>dftd</code>. BJ/zero damping, functional key, explicit ATM flag. <code>PotentialConfig.d3</code> is not applied; every <code>potserv</code> backend refuses it. Goldens: <code>meson test --suite dftd</code></td>
</tr>

<tr>
<td class="org-left"><code>D4Pot</code></td>
<td class="org-left"><code>D4</code></td>
<td class="org-left">In-process Potential summand. Enable with <code>-Dwith_dftd4=true</code>; use pixi env <code>dftd4</code> or <code>dftd</code>. Functional key, charge, explicit ATM/many-body flag. <code>PotentialConfig.d4</code> is not applied; every <code>potserv</code> backend refuses it. Goldens: <code>meson test --suite dftd</code></td>
</tr>

<tr>
<td class="org-left"><code>ExprPot</code></td>
<td class="org-left">in-process</td>
<td class="org-left">Enable with <code>-Dwith_expr=true</code>. Vendored OpenMM Lepton (no pixi muparser). Named <code>Potential</code> children; construct-time fail-closed names. <code>potserv</code> refuses <code>PotentialConfig.expr</code>.</td>
</tr>

<tr>
<td class="org-left"><code>MetatomicPot</code></td>
<td class="org-left"><code>Metatomic:&lt;model_path&gt;[:&lt;device&gt;]</code></td>
<td class="org-left">Enable with <code>-Dwith_metatomic=true</code>; use pixi env <code>metatomicbld</code>. Pip engines: torch 2.7+</td>
</tr>

<tr>
<td class="org-left"><code>UmaPot</code></td>
<td class="org-left"><code>Uma:&lt;model.pt2&gt;[:&lt;task&gt;]</code></td>
<td class="org-left">Built with <code>-Dwith_metatomic=true</code>. UMA / OMol AOTInductor <code>.pt2</code>; <code>&lt;task&gt;</code> is one of <code>omol</code>, <code>omat</code>, <code>oc20</code>, <code>oc22</code>, <code>oc25</code>, <code>odac</code>, <code>omc</code>. Vesin neighbour lists, one package per composition, charge and spin. See <code>CppCore/rgpot/UmaPot/README.md</code>.</td>
</tr>

<tr>
<td class="org-left"><code>NWChemPot</code></td>
<td class="org-left"><code>NWChem[:basis[:theory[:xc]]]</code></td>
<td class="org-left">Frontend always builds; load <code>libnwchemc</code> from the split <code>nwchemc</code> project at runtime</td>
</tr>

<tr>
<td class="org-left"><code>CPMDPot</code></td>
<td class="org-left"><code>CPMD</code></td>
<td class="org-left">Frontend always builds; load <code>libcpmdc</code> from the split <code>cpmdc</code> project at runtime</td>
</tr>

<tr>
<td class="org-left"><code>CP2KPot</code></td>
<td class="org-left"><code>CP2K</code></td>
<td class="org-left">Enable with <code>-Dwith_cp2k=enabled</code> (needs <code>with_rpc</code> and pkg-config <code>libcp2k</code>)</td>
</tr>

<tr>
<td class="org-left"><code>SkalaPot</code></td>
<td class="org-left"><code>Skala</code>, <code>Skala:&lt;basis&gt;</code></td>
<td class="org-left">Skala XC on NWChem DFT via <code>libnwchemc</code>. Enable with <code>-Dwith_skala=true</code> (needs <code>with_rpc</code>)</td>
</tr>

<tr>
<td class="org-left"><code>MOPACPot</code></td>
<td class="org-left">none</td>
<td class="org-left">Not a <code>potserv</code> backend. Frontend always builds; load <code>libmopacc</code> from the split <a href="https://github.com/OmniPotentRPC/mopacc">mopacc</a> project at runtime (OpenMOPAC). Packed params. Default model AM1.</td>
</tr>

<tr>
<td class="org-left"><code>IPIPot</code></td>
<td class="org-left"><code>IPI:unix:&lt;path&gt;</code>, <code>IPI:tcp:&lt;host&gt;:&lt;port&gt;</code></td>
<td class="org-left">POSIX socket server for an i-PI force driver. <code>-Dwith_ipi=auto</code> (off on Windows). See <code>docs/orgmode/howto/ipi.org</code>.</td>
</tr>
</tbody>
</table>

Copyable construct-and-evaluate (meson `-Dwith_expr=true`).
`PotentialHandle::from_impl` / `rgpot_potential_new_eindir` wraps that
one `ExprPot` as one eindir objective for rgmin, rgsaddle, and anneal.

```cpp
#include "rgpot/ExprPot/ExprPot.hpp"
#include "rgpot/LennardJones/LJPot.hpp"
#include "rgpot/Morse/MorsePot.hpp"
#include "rgpot/types/AtomMatrix.hpp"

#include <array>
#include <memory>
#include <vector>

int main() {
  std::vector<rgpot::ExprPot::Term> terms;
  terms.emplace_back("lj", std::make_unique<rgpot::LJPot>());
  terms.emplace_back("morse", std::make_unique<rgpot::MorsePot>());
  rgpot::ExprPot pot("0.5*lj + morse", std::move(terms));

  rgpot::types::AtomMatrix positions{{1.0, 2.0, 3.0}, {1.5, 2.5, 3.5}};
  std::vector<int> atomTypes{0, 0};
  std::array<std::array<double, 3>, 3> box{{
      {15.0, 0.0, 0.0},
      {0.0, 20.0, 0.0},
      {0.0, 0.0, 30.0},
  }};
  auto [energy, forces, variance] = pot(positions, atomTypes, box);
  (void)forces;
  (void)variance;
  (void)energy;
}
```

When the build also has `-Dwith_dftd3=true`, swap the Morse child for
`D3Pot` and the string `"0.5*lj + d3"`.

Example server commands (`IPI:unix:rgpot` binds the socket `ipi_rgpot` in the
system temporary directory):

```bash
./bbdir/CppCore/potserv 12345 LJ
./bbdir/CppCore/potserv 12345 IPI:unix:rgpot
./bbdir/CppCore/potserv 12345 Metatomic:CppCore/tests/data/lj38/lennard-jones.pt
CPMDC_LIBRARY=/path/to/libcpmdc.so ./bbdir/CppCore/potserv 12345 CPMD
NWCHEMC_LIBRARY=/path/to/libnwchemc.so ./bbdir/CppCore/potserv 12345 NWChem
```


## CPMD And NWChem Configuration

CPMD and NWChem do not use ad hoc rgpot config files.
They use the `PotentialConfig` union over RPC:

-   `PotentialConfig.cpmd` carries `CPMDParams` for `CPMDPot`
-   `PotentialConfig.nwchem` carries `NWChemParams` for `NWChemPot`
-   geometry still arrives through `ForceInput` on every `calculate()` call

For CPMD, `tests/cpmd_params.py` contains pycapnp helpers for building
`CPMDParams` and `PotentialConfig.cpmd` messages.
The helper covers scalar fields, raw `inputBlocks`, and every structured
`CPMDInputSection` arm.
See `CppCore/rgpot/CPMDPot/README.md` for the engine lookup order and schema
field mapping.

The CPMD runtime path is:

1.  Build `libcpmdc.so` in the split `cpmdc` repository (or use the in-tree
    `libcpmdc_fake_engine.so` from a `-Dwith_tests=true` build for CI).
2.  Start `potserv <port> CPMD` with `CPMDC_LIBRARY`, `RGPOT_CPMDC_ENGINE`, or
    `RGPOT_CPMD_ENGINE` pointing at that shared library.
3.  Send `configure(PotentialConfig.cpmd)` once for method setup.
4.  Send `ForceInput` on each `calculate()` call for geometry and requested
    output units.

For NWChem, `tests/nwchem_params.py` builds `NWChemParams` /
`PotentialConfig.nwchem`.
The engine lookup order is `enginePath` on the params, then
`NWCHEMC_LIBRARY`, `RGPOT_NWCHEMC_ENGINE`, `RGPOT_NWCHEM_ENGINE`, then
`libnwchemc.so` on the loader path.
See `CppCore/rgpot/NWChemPot/README.md`.
The NWChem runtime path mirrors CPMD:

1.  Build `libnwchemc.so` in the split `nwchemc` repository (or
    `libnwchemc_fake_engine.so` from a tests build).
2.  Start `potserv <port> NWChem` with the library env vars above.
3.  `configure(PotentialConfig.nwchem)` once, then `calculate(ForceInput)`.

`potctl ci bridge-stress` starts `potserv` with a potential name, runs the
client-only `ctest` suite against it, then stops the server.


## Developer Tasks

Hooks use [prek](https://prek.j178.dev) through `prek.toml`.
`prek` is not a pixi dependency; install it first.
Common commands:

```sh
pixi r prek-install
pixi r prek
pixi r rust-test
pixi r -e rpctest python tests/test_cpmd_params.py
pixi r -e rpctest python tests/test_nwchem_params.py
pixi r -e rpctest python tests/test_rpc_integ_cpmd.py
pixi r -e rpctest python tests/test_rpc_integ_nwchem.py
# Full RPC + C ABI E2E (point env at built fake engines + potserv)
export RGPOT_POTSERV=/path/to/bbdir/CppCore/potserv
export NWCHEMC_LIBRARY=/path/to/bbdir/CppCore/libnwchemc_fake_engine.so
export RGPOT_CPMD_ENGINE=/path/to/bbdir/CppCore/libcpmdc_fake_engine.so
pixi r -e rpctest python tests/test_rpc_e2e_c_abi.py
pixi r -e rpctest python tests/rpc_integ.py \
  --server-bin ./bbdir/CppCore/potserv \
  --nwchem-smoke \
  --cpmd-smoke
```

The root README is generated from `readme_src.org`.
Project documentation sources live under `docs/orgmode/`.


## Contact

Ask questions, report bugs and suggest features on the [issue tracker](https://github.com/OmniPotentRPC/rgpot/issues).
Report security vulnerabilities privately, following [SECURITY.md](https://github.com/OmniPotentRPC/rgpot/blob/main/SECURITY.md).
Do not open a public issue for them.
The development setup is on the [contributing pages](https://rgpot.rgoswami.me/contributing/index.html).
Every contributor follows the [Code of Conduct](https://github.com/OmniPotentRPC/rgpot/blob/main/CODE_OF_CONDUCT.md).


# License

MIT, with backend-specific notes:
some potentials are adapted from eOn under BSD-3-Clause terms.
The unit expression parser in `CppCore/rgpot/units.cc` is derived from
metatomic-torch (BSD-3-Clause, metatensor developers).
The test tables `CppCore/tests/data/eam_table/Cu_u3.eam` and
`CuNi.eam.alloy` are copied unchanged from LAMMPS `potentials/` at
tag `stable_22Jul2025_update4` and are GPL-2.0 (LAMMPS).
Their copyright notice is in `CppCore/tests/data/eam_table/NOTICE`,
and the GPL-2.0 text is in `CppCore/tests/data/eam_table/COPYING`.
The reader sources stay MIT.
