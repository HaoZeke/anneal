`librgpot_gromacs` (`-Dwith_gromacs_dropin=enabled`, off by default) serves
any rgpot potential through the libcp2k C API of the GROMACS QM/MM interface.
A double-precision GROMACS 2024 to 2026 built with `-DGMX_CP2K=ON` against it
(`engines/gromacs/build_gromacs.sh`) evaluates the QM group in `gmx mdrun`
without PLUMED, with the energy in GROMACS' `Quantum En.` term. An `&RGPOT`
section of the `grompp -qmi` file chooses a `pot_spec` name and options or an
`rgpot_engine_*` library and its configuration; elements come from the
topology. Periodic (including triclinic) and isolated QM regions work; there
is no virial, so NPT is not supported, and MM charges do not act on the QM
region. Bad input stops `mdrun` with an `rgpot-gromacs:` message. New meson
tests `gromacs_dropin_api` and `gromacs_dropin_mdrun` compare with
`rgpot-eval` through real `mdrun` runs, and `engines/gromacs/bench_dropin.py`
measures the per-step cost.
