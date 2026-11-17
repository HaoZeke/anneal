# XcKernel: in-process libxckernel contractions

`rgpot::XcKernel` provides contractions from
[libxckernel](https://github.com/susilehtola/libxckernel)
(arXiv:2608.26440, pin `d6a9d57ba3fe0f2763667ce168d0c0ef21cff4a4`).
It is **not** a geometry PES.

## What it is

- New C++ type under `CppCore/rgpot/XcKernel/`.
- Inputs: AO collocation (`chi`, `dchi`, optional `lapl_chi` / `hess_chi`),
  grid weights, named Libxc derivative arrays, and (for fxc) perturbed fields.
- Python `XcKernel.contract(chi, dchi=None, scal=None, ...)`: `dchi` may be
  omitted. `scal` is required; `None` raises `TypeError`. The stub default
  exists so a defaulted `dchi` is not followed by a required parameter.
  A nonzero return code from the C++ `contract` raises
  `RuntimeError("XcKernel.contract rc=N")`.
- Outputs: AO matrices accumulated `+=` (XC Fock at order 1, fxc contraction
  at order 2).
- Term ownership is XC-only. Coulomb, Hartree-Fock exact exchange, and
  range-separated exchange stay host-owned.

## What it is not

- Not a `Potential` subclass and not a `PotType`.
- No `PotentialConfig.xckernel` arm (rgpot-qf6b). The wire schema still
  carries `ForceInput` / `PotentialResult` / `PotentialConfig` for energy
  surfaces. Kernel operands are not those carriers, and no DFT host sends
  `chi` / `dchi` / weights over potserv in this slice.
- Grimme D3/D4 stay in-process `Potential` summands (`D3Pot` / `D4Pot`).
  Do not fold them into `contract` or `applyFxc`. The schema carries
  `PotentialConfig.d3` @12 / `.d4` @13, but nothing applies them: every potserv
  backend refuses both arms in `configure()`. A DFT host adds D4 as the Edisp
  term next to this kernel and VV10 (rgpot-8mse).
- The default wheel and default meson build do **not** ship the Python
  generator. `with_xckernel` defaults to false.

## Build

Meson option, same shape as `with_xtb`:

<!-- rgpot-doc: syntax configures a separate build tree -->
```bash
meson setup bbdir-xck -Dwith_xckernel=true -Dwith_tests=true
```

CI runs that configuration on Ubuntu (`potctl ci xckernel-test`, suite
`xckernel`, which includes `XcKernelGoldens`). The default meson matrix
leaves `with_xckernel` off, so this job is what keeps the BLAS pins from
regressing in silence.

Generate the C package **only on a host listed in `RGPOT_REGEN_HOSTS`**
(comma-separated). `XCKERNEL_ALLOW_REGEN=1` overrides the host check:

<!-- rgpot-doc: syntax regenerates the committed C package -->
```bash
pixi install -e xckernel
pixi run -e xckernel -- bash scripts/gen_libxckernel.sh
```

Families: `lda,gga,mgga_tau,mgga_lapl`. The generated package contains 36 C
kernels: four energy helpers, 12 Fock kernels, and 20 response kernels.
`XcKernel::catalog()` exposes the 32 Fock and response contractions through
derivative order two. Laplacian kernels require the AO Laplacians in
`lapl_chi`; missing Laplacians return error code 2 without changing the output.

## Stage B

`CppCore/rgpot/XcKernel/xckernel/evaluator.hpp` specializes `stage_b<double>`.
Each block of 256 grid points is one `dgemm`, or one `dsyr2k` when `U` and `V` are the same array, and the block results are added in long double.
`stage_a` still accumulates in long double.
The build links FlexiBLAS when pkg-config finds it, then pkg-config blas, then OpenBLAS (`openblas.pc` or `libopenblas`), then `libblas`.
On a release build linked to OpenBLAS 0.3.34 pthreads, with one thread bound to its own core, the random grid (nbf 200, 100000 points) median for `xck_gga_r_o1` was 859.8 ms against 35554.2 ms for the long-double loop.
Four threads on four distinct cores brought that median to 515.0 ms.
On the H2O grid, nbf is 7, and four threads are slower than one.
`xckernel_bench` prints the median, minimum, and maximum over its repetitions.

## Dependencies

Libxc implements the functional-derivative tower. Numerical evaluation needs
`pylibxc`. **`pylibxc` is not installable from PyPI**; the `pylibxc2` name
there is an unrelated empty stub. Use conda-forge `pylibxc` (libxc-feedstock
Python output) via the pixi feature `xckernel`. PySCF lives only in
`xckerneltest` for golden masters.

<!-- rgpot-doc: syntax installs pixi environments -->
```bash
pixi install -e xckernel        # libxc + sympy + numpy; `import pylibxc`
pixi install -e xckerneltest    # plus pyscf + scipy
# wrong: python -m pip install pylibxc
# wrong: python -m pip install pylibxc2
```

The compiled libxckernel runtime does not link pylibxc. The host passes
already-mixed derivative arrays.

## Golden masters

Fixtures live under `CppCore/tests/data/xckernel/`. Tests fail closed if a
named file is missing. Regenerator:

<!-- rgpot-doc: syntax rewrites the committed fixtures -->
```bash
pixi run -e xckerneltest -- python scripts/regen_xckernel_goldens.py
```

That script refuses to run unless `RGPOT_REGEN_HOSTS` lists the host or
`XCKERNEL_ALLOW_REGEN=1` is set. Each bound below is a maximum: a
relative error equal to it passes. Fock vs PySCF must not exceed
`1e-15`, fxc vs PySCF `1e-13`, TDA/RPA sigma vs PySCF `1e-17`.
The C-versus-NumPy and random-grid Fock pins are not a fixed `1e-16`.
Stage B contracts each grid block with `dsyr2k` when the two factors are
the same array, and with `dgemm` otherwise. A length-k dot product in
double, and the `dsyr2k` average of the two products, satisfies
`|fl(a^T b) - a^T b| <= k*eps*|a|^T|b|`, where `eps` is machine epsilon
(`2^{-52}`) and `|a|^T|b|` is the sum of absolute products. `k` is the
block length, `min(npts, 256)`, not the whole grid: block results are
added in long double and rounded once, so the double-precision sum does
not grow with `npts`. Each stage-B term adds its own bound. The
committed pin is a rounded double, so the comparison also allows half an
ulp of that pin. A fixed `1e-16` is below one ulp of these matrices and
rejects a correct OpenBLAS sum. Do not replace this bound with a looser
constant. `--pyscf` Fock compares
long-double stage A/B to live `nr_rks` and exits when `rel > 1e-15`.
`--tda-rpa` (also part of `--pyscf`) pins PySCF to one OpenMP thread,
replays committed MOs, reports live host-J without overwriting the
committed `tda_*_j.npy` / `rpa_*_j.npy` pins, writes `st_o2_p`
operands, and exits when live `gen_vind` / `gen_tdhf_operation`
drifted past `1e-17`. Multi-thread `gen_vind` jitters 1-2 ulp on this case, which is
already past that bar. A fresh RKS kernel on the same mol/xc is past
that bar (MO/energy noise ~1e-15); the `1e-17` gate is MO-replay
`gen_vind`, not a new SCF. Same-SCF extras and `MANIFEST.json` are
written before that exit. Do not invent looser values.

`scripts/compare_xckernel_tda_nwchemc.py` (same `RGPOT_REGEN_HOSTS` /
`XCKERNEL_ALLOW_REGEN=1` rule) calls `libnwchemc`
`nwchemc_energy` with `theory=tddft` / TDA on the same H2O/sto-3g
geometry and writes `tda_{lda,gga}_nwchemc_roots.npy`. That is the
second engine check of the pin operator (TDA.kernel on committed MOs).
The `1e-17` bar remains the C sigma contraction; the nwchemc
root residual is the measured engine value (`1e-6`).

TDA/RPA assembly is `XcKernel::tdaSigma` / `rpaSigma` over the singlet
`xck_*_st_o2_p` kernels plus host Coulomb. LDA wv is
`w * rho * (v2rho2_0 + v2rho2_1)` (one fused product, matching
`nr_rks_fxc_st`) and stage B tiles the grid at PySCF `BLKSIZE` (128).
GGA `st_o2_p` `applyFxc` uses the generated 7-term monomials through
the host long-double evaluator (`contract()`). Double tiled stage B
misses `1e-17` vs live `gen_vind` on this sto-3g pin.
Transition densities and ov projection follow the PySCF `lib.einsum`
contraction path (`qo,xov->vxq` then `vxq,pv->xpq` for the TDA DM;
`pv,xpq->vxq` then `vxq,qo->xov` for the ov block). Perturbed fields
follow PySCF `eval_rho` (`c0 = ao @ dm`, GGA `hermi=0` adds
`ao @ dm.T`). Coulomb `J` stays host-owned (pinned `tda_*_j.npy` /
`rpa_*_j.npy` from the `get_j` call inside `gen_vind` /
`gen_tdhf_operation`).
