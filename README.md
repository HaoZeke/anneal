

# rgpot

![img](https://raw.githubusercontent.com/OmniPotentRPC/rgpot/refs/heads/main/branding/logo/rgpot_logo.webp)

[![codecov](https://codecov.io/gh/OmniPotentRPC/rgpot/graph/badge.svg)](https://app.codecov.io/gh/OmniPotentRPC/rgpot)

`rgpot` is a potential-energy library and Cap'n Proto RPC server for atomistic
simulation codes.
Documentation: [rgpot.rgoswami.me](https://rgpot.rgoswami.me).
It gives clients one geometry carrier, one result carrier, and one extensible
backend-configuration carrier:

-   `ForceInput`: positions, atomic numbers, box, and unit strings
-   `PotentialResult`: energy plus flat force array
-   `PotentialConfig`: backend setup, with arms such as `none`, `nwchem`, and
    `cpmd`

XC response kernels (`rgpot::XcKernel`, meson `-Dwith_xckernel=true`) are
in-process only. They are not a `Potential`, and there is no
`PotentialConfig.xckernel` arm: operands are collocation plus named Libxc
arrays, not `ForceInput`. See `docs/xckernel.md`.
`pylibxc` is not on PyPI (`pylibxc2` is an unrelated stub); use the
conda-forge `libxc` package via `pixi install -e xckernel`.

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
named child `Potential` objects. The parser is vendored OpenMM Lepton;
there is no pixi muparser feature. The schema carries `PotentialConfig.expr`,
but `potserv` does not apply it and refuses it by name.

In-process `LJPot` vs pyeonclient `Matter` on the same LJ fixture is
`scripts/time_lj_rgpot_vs_pyeonclient.py` (on a host listed in
`RGPOT_REGEN_HOSTS`). On workstation-a
(2026-09-04, 1e6 calls) A (`LJPot::operator()`) was 31-33 ns/call and
B (in-process pyeonclient `Matter.forces`, not potserv) was 687 to 791
ns/call. `XcKernel` is not a `Potential` and is not part of that race.
See `docs/orgmode/howto/exprpot.org`.

The public wire schema lives in `CppCore/rgpot/rpc/Potentials.capnp` and is
shared by the C++ server, Python integration tests, and the Rust core crate.
Native rgpot units are eV and Angstrom.
EAM, Lennard-Jones, Sutton-Chen, Stillinger-Weber, Tersoff, Morse and
ZBL share one neighbour layer, with a stored Verlet list or a streaming
cell walk.
With `OMP_NUM_THREADS` unset, the energy and forces match the previous
one-thread result bit for bit. A set thread count stays within a relative
tolerance of 1e-9. See `docs/neighbour_layer.md`.
RPC clients may request compatible units through `ForceInput.lengthUnit` and
`ForceInput.energyUnit`.


## Python (PyPI)

Install the manylinux wheel (nanobind **abi3**, Python >= 3.12 for the stable
ABI extension; requires-python is >= 3.10 for pure metadata):

    pip install rgpot
    # Metatomic path (engine libs resolve from these packages at runtime):
    pip install 'rgpot[metatomic]'
    # or: pip install torch metatomic-torch metatensor-torch metatensor-core vesin

The PyPI wheel compiles ExprPot (vendored Lepton), D3Pot, D4Pot, and
XcKernel. `s-dftd3` / `dftd4` are linked and vendored; no conda install
is required at `pip install` time. Lennard-Jones needs only `numpy`.
Metatomic uses **dlopen** of a bundled
engine:

-   layout: `rgpot/lib/torch-X.Y/libmetatomic_engine.so`
-   picker: installed `torch` major (same multi-ABI idea as metatomic-torch)
-   **supported torch majors: 2.7 and newer** (wheels ship 2.7–2.13 engines)
-   torch 2.6 and older are not supported for Metatomic dlopen

```python
import numpy as np
import rgpot

print(rgpot.__version__)
print(rgpot.has_expr, rgpot.has_dftd3, rgpot.has_dftd4, rgpot.has_xckernel, rgpot.has_ipi)
print(rgpot.available_metatomic_engine_abis())  # e.g. 2.7 .. 2.13
# energy, forces, variance = rgpot.LJPot()(positions, atom_types, box)
# energy, forces, variance = rgpot.D3Pot()(positions, atom_types, box)
# energy, forces, variance = rgpot.ExprPot(
#     "0.5*lj + d3", {"lj": "lj", "d3": "d3"})(positions, atom_types, box)
# print(rgpot.XcKernel.catalog())
# energy, forces, variance = rgpot.evaluate_metatomic(
#     positions, atom_types, box, model_path="model.pt")
```

See <https://pypi.org/project/rgpot/>.


## Build And Test

Meson is the primary build path.
Use the `rpctest` environment when running the RPC integration scripts because
they need pycapnp.

    pixi shell -e rpctest
    meson setup bbdir -Dwith_tests=true -Dwith_rpc=true --buildtype=debug
    meson compile -C bbdir
    meson test -C bbdir --print-errorlogs
    python tests/rpc_integ.py --server-bin ./bbdir/CppCore/potserv

CMake is supported as well:

    cmake -B build \
      -DRGPOT_BUILD_TESTS=ON \
      -DRGPOT_BUILD_EXAMPLES=ON \
      -DRGPOT_WITH_RPC=ON
    cmake --build build
    ctest --test-dir build --output-on-failure

For client-only consumers that only need the Cap'n Proto schema and generated
RPC client types:

    cmake -B build_client \
      -DRGPOT_RPC_CLIENT_ONLY=ON \
      -DRGPOT_BUILD_TESTS=ON
    cmake --build build_client
    ctest --test-dir build_client --output-on-failure


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
<td class="org-left"><code>LJ</code></td>
<td class="org-left"><code>LJ</code></td>
<td class="org-left">Built-in 12-6 Lennard-Jones reference potential</td>
</tr>

<tr>
<td class="org-left"><code>CuH2Pot</code></td>
<td class="org-left"><code>CuH2</code></td>
<td class="org-left">Built-in Cu-H EAM potential</td>
</tr>

<tr>
<td class="org-left"><code>XTBPot</code></td>
<td class="org-left"><code>XTB</code>, <code>GFNFF</code>, <code>GFN0xTB</code>, <code>GFN1xTB</code></td>
<td class="org-left">Enable with <code>-Dwith_xtb=true</code>; use pixi env <code>xtbbld</code> or <code>tbbld</code>; linked <code>XTBPot</code> + dlopen <code>libxtb_engine.so</code> (see <code>docs/xtb_backends.md</code>)</td>
</tr>

<tr>
<td class="org-left"><code>TBLitePot</code></td>
<td class="org-left"><code>TBLite</code>, <code>TBLiteGFN1</code>, <code>TBLiteIPEA1</code></td>
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
<td class="org-left"><code>Metatomic:&lt;model_path&gt;</code></td>
<td class="org-left">Enable with <code>-Dwith_metatomic=true</code>; use pixi env <code>metatomicbld</code>. Pip engines: torch 2.7+</td>
</tr>

<tr>
<td class="org-left"><code>UmaPot</code></td>
<td class="org-left"><code>Uma:&lt;model_path&gt;</code>, <code>Uma:&lt;model_path&gt;:omol</code></td>
<td class="org-left">Enable with <code>-Dwith_metatomic=true</code>. AOTInductor <code>.pt2</code> plus vesin neighbor lists; charge/spin are per-call tensor inputs</td>
</tr>

<tr>
<td class="org-left"><code>NWChemPot</code></td>
<td class="org-left"><code>NWChem</code></td>
<td class="org-left">Frontend always builds; load <code>libnwchemc</code> from the split <code>nwchemc</code> project at runtime</td>
</tr>

<tr>
<td class="org-left"><code>CPMDPot</code></td>
<td class="org-left"><code>CPMD</code></td>
<td class="org-left">Frontend always builds; load <code>libcpmdc</code> from the split <code>cpmdc</code> project at runtime</td>
</tr>

<tr>
<td class="org-left"><code>UmaPot</code></td>
<td class="org-left"><code>Uma:&lt;model.pt2&gt;</code></td>
<td class="org-left">UMA / OMol AOTInductor <code>.pt2</code>. Vesin neighbors, one package per composition. See <code>CppCore/rgpot/UmaPot/README.md</code>.</td>
</tr>

<tr>
<td class="org-left"><code>SkalaPot</code></td>
<td class="org-left"><code>Skala</code></td>
<td class="org-left">Skala XC on NWChem DFT via <code>libnwchemc</code>.</td>
</tr>

<tr>
<td class="org-left"><code>MOPACPot</code></td>
<td class="org-left"><code>MOPAC</code></td>
<td class="org-left">Frontend always builds; load <code>libmopacc</code> from the split <a href="https://github.com/OmniPotentRPC/mopacc">mopacc</a> project at runtime (OpenMOPAC). Packed params. Default model AM1.</td>
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

When the build also has `-Dwith_dftd3=true`, swap the Morse child for
`D3Pot` and the string `"0.5*lj + d3"`.

Example server commands:

    ./bbdir/CppCore/potserv 12345 LJ
    ./bbdir/CppCore/potserv 12345 IPI:unix:/tmp/ipi_rgpot
    ./bbdir/CppCore/potserv 12345 Metatomic:CppCore/tests/data/lj38/lennard-jones.pt
    CPMDC_LIBRARY=/path/to/libcpmdc.so ./bbdir/CppCore/potserv 12345 CPMD
    NWCHEMC_LIBRARY=/path/to/libnwchemc.so ./bbdir/CppCore/potserv 12345 NWChem


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
`PotentialConfig.nwchem`. Engine lookup order is `NWCHEMC_LIBRARY`,
`RGPOT_NWCHEMC_ENGINE`, `RGPOT_NWCHEM_ENGINE`, then `enginePath` on params.
See `CppCore/rgpot/NWChemPot/README.md`. The NWChem runtime path mirrors CPMD:

1.  Build `libnwchemc.so` in the split `nwchemc` repository (or
    `libnwchemc_fake_engine.so` from a tests build).
2.  Start `potserv <port> NWChem` with the library env vars above.
3.  `configure(PotentialConfig.nwchem)` once, then `calculate(ForceInput)`.

`potctl` drives the same Potential Cap'n Proto RPC against `potserv` (bridge
stress and CI legs pass the potential name such as `NWChem` or `CPMD`).


## Developer Tasks

Hooks use [prek](https://prek.j178.dev) through `prek.toml`.
Common commands:

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

The root README is generated from `readme_src.org`.
Project documentation sources live under `docs/orgmode/`.


## Contact

Ask questions, report bugs and suggest features on the [issue tracker][issues].
Report security vulnerabilities privately, following [SECURITY.md][security].
Do not open a public issue for them.
The development setup is on the [contributing pages][contributing].
Every contributor follows the [Code of Conduct][conduct].

[issues]: https://github.com/OmniPotentRPC/rgpot/issues
[security]: https://github.com/OmniPotentRPC/rgpot/blob/main/SECURITY.md
[contributing]: https://rgpot.rgoswami.me/contributing/index.html
[conduct]: https://github.com/OmniPotentRPC/rgpot/blob/main/CODE_OF_CONDUCT.md


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
