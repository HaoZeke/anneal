`MetatomicPot` now reports `ForceOut::variance` as a variance in squared units rather than an RMS: the square of the mean model `energy_uncertainty` in eV^2, or, when a multi-pass run has no model uncertainty, the mean-square force residual over orientations in (eV/Angstrom)^2.
The value is passed through unchanged by the C++, C and Python entry points.
Direct users who relied on the previously documented eV or eV/Angstrom must take the square root of `variance` to get the old units back.
