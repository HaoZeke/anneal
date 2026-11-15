`rgpot.ase.RgpotCalculator` accepts any potential class the bindings expose,
by class name or kind string (`"morse"`, `"ljcluster"`, `"tersoffsic"`, ...;
`rgpot.ase.available_kinds()` lists them), and passes keyword arguments to
that class's constructor. It reports per-atom `energies` and `stresses` for
LJ, LJ-cluster and Morse, offers a finite-difference `stress` for kernels
without one (`numerical_stress=True`), rebuilds the potential on
`calc.set(...)`, and pickles, `copy`s and `deepcopy`s by rebuilding the
potential from its parameters. It matches ASE's `LennardJones`,
`MorsePotential` and an analytic `EAM` for EAM-Al to 1e-10 relative, and
runs under the ASE optimizers, cell filters, Velocity Verlet, NEB,
`Vibrations` and `SumCalculator`.
