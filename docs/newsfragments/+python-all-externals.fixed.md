`from rgpot import *` now also imports `LammpsPot`, `GromacsPot`, `OpenMMPot`, `rgpot_lammps_potential`, `rgpot_gromacs_potential` and `rgpot_openmm_potential`.
They were exported but missing from `rgpot.__all__`.
