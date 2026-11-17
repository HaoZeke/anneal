# Analytic model potentials

These models are regular rgpot built-ins. `forceImpl` returns `-dV/dx`, while
`hessianVectorImpl` returns the directional product of the energy Hessian,
`(d²V/dx²) v`.

Eckart, Müller–Brown, modified Müller–Brown, and the double-well families use
eV for energy and angstrom for coordinates. Their constructor parameters carry
the corresponding powers of those units. The quartic and coupled-well formulas
follow the analytic model definitions in spectral-instanton:

- `V = V0 ((x/x0)² - 1)²`
- `V = V0 ((x/x0)² - 1)² + ky (y - x²/x0)² / 2`
- bath coordinates add `omega² (qi - coupling q0)² / 2`

The LEPS implementations evaluate their published formulas in hartree and bohr,
then convert energy, forces, and Hessian-vector products to rgpot's eV and
angstrom convention. The F + H2 constants are the Whitlock–Muckerman Morse/Sato
parameters. The H + Cl2 curves use

| curve | De (hartree) | re (bohr) | exponent (bohr^-1) | Sato |
|---|---:|---:|---:|---:|
| HCl | 0.16963487116282344 | 2.4085504321417757 | 0.9886770766652772 | 0.067 |
| Cl2 | 0.09238779594959665 | 3.7565865631435686 | 1.0591804130726892 | -0.113 |

Those H + Cl2 parameters are from Connor, Mayne, and Whitehead,
J. Chem. Phys. 72, 6209 (1980).

H + CH4 is not duplicated here. CBE is supplied by the external `cbe-pes`
library and rgpot's optional `CBEPot` linked frontend.

`ModelCapi.h` constructs parameterized built-in handles for linked clients.
Its force call returns rgpot's force convention; its HVP call dispatches the
same `PotentialBase::hessianVectorImpl` hook used by the engine and RPC paths.

## CuH2 and embedded-atom models

The shipped `rgpot_cuh2.f90` is the eOn CuH2 potential and already reads its
dedicated `pot.par` chemistry. EAMTable and Sutton–Chen represent metallic
many-body parameterizations; neither can represent that reactive Cu/H surface
by adding a parameter set. No additional EAMTable or Sutton–Chen parameters are
therefore justified.
