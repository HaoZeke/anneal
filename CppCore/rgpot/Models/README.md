# Analytic model potentials

This directory holds built-in analytic model surfaces and their exact
Hessian-vector products.

`forceImpl` returns `-dV/dx`.
`hessianVectorImpl` returns `H v`, where `H` is the energy Hessian.

## Built-in model families

The shipped analytic families are:

- Eckart
- Muller-Brown
- Modified Muller-Brown
- LEPS (H + H2, H + Cl2, F + H2)
- N-dimensional quartic double well
- Coupled proton well

The non-LEPS models use eV for energy and angstrom for coordinates.
LEPS evaluates its published equations in hartree and bohr, then converts
energy, force, and Hessian-vector products to rgpot units.

Muller-Brown uses the first atom's x and y coordinates. Its z force and every
force on later atoms are zero. `MullerBrownConfig` exposes all six four-value
parameter arrays and defaults to the values of K. Müller and L. D. Brown,
*Theoret. Chim. Acta* **53**, 75-93 (1979),
doi:10.1007/BF00547608.

| preset | parameter reference |
|---|---|
| H + H2 | P. A. Whitlock, J. T. Muckerman and R. E. Roberts, *J. Chem. Phys.* **60**, 3658-3673 (1974), [doi:10.1063/1.1681586](https://doi.org/10.1063/1.1681586) |
| H + Cl2 | J. N. L. Connor, W. Jakubetz, J. Manz and J. C. Whitehead, *J. Chem. Phys.* **72**, 6209-6226 (1980), [doi:10.1063/1.439080](https://doi.org/10.1063/1.439080) |
| F + H2 | P. A. Whitlock and J. T. Muckerman, *J. Chem. Phys.* **61**, 4618-4629 (1974), [doi:10.1063/1.1681780](https://doi.org/10.1063/1.1681780) |

The H + Cl2 LEPS constants are:

| curve | De (hartree) | re (bohr) | exponent (bohr^-1) | Sato |
|---|---:|---:|---:|---:|
| HCl | 0.16963487116282344 | 2.4085504321417757 | 0.9886770766652772 | 0.067 |
| Cl2 | 0.09238779594959665 | 3.7565865631435686 | 1.0591804130726892 | -0.113 |

## C model API lifecycle

`ModelCapi.h` exposes opaque `rgpot_builtin_model*` handles.

Lifecycle:

1. Construct with one of:
   - `rgpot_builtin_model_new(kind)` for fixed built-ins
   - `rgpot_builtin_muller_brown_new(...)` for configurable Muller-Brown
   - `rgpot_builtin_double_well_new(...)` for a parameterized quartic well
   - `rgpot_builtin_coupled_proton_well_new(...)` for a parameterized coupled well
2. Read dimensions with `rgpot_builtin_model_dimensions(model)`.
3. Evaluate energies and forces with `rgpot_builtin_model_calculate(...)`.
4. Evaluate exact Hessian-vector products with `rgpot_builtin_model_hvp(...)`.
5. Free with `rgpot_builtin_model_free(model)`.

Every function reports failure through its status code and
`rgpot_builtin_model_last_error()`.

## External CBE frontend

H + CH4 is supplied by the external `cbe-pes` library through the optional
`CBEPot` linked frontend.

## CuH2 and embedded-atom models

The shipped `rgpot_cuh2.f90` is the eOn CuH2 potential and already reads its
dedicated `pot.par` chemistry. EAMTable and SuttonChen are metallic many-body
forms, so no parameter-set addition can represent that reactive Cu/H surface.
