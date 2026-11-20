# Copyright (c) anneal developers
"""CODATA 2018 physical constants matching readcon-core and rgpot.

Derived from the SI 2019 exact definitions and CODATA 2018 SI values so
Python scripts and the paper audit share one CODATA edition with Rust
(readcon-core) and C++ (rgpot) without requiring SciPy at import time.
"""

from __future__ import annotations

# CODATA 2018 SI values (identical to readcon-core 0.14.8 src/units.rs).
HARTREE_JOULE = 4.359_744_722_207_1e-18
ELEMENTARY_CHARGE_C = 1.602_176_634e-19
BOHR_METER = 5.291_772_109_03e-11
ANGSTROM_METER = 1.0e-10
BOLTZMANN_JOULE_PER_K = 1.380_649e-23
AVOGADRO_MOL = 6.022_140_76e23
THERMOCHEMICAL_KCAL_J = 4184.0
KILOJOULE_J = 1000.0

HARTREE_EV = HARTREE_JOULE / ELEMENTARY_CHARGE_C
BOHR_ANGSTROM = BOHR_METER / ANGSTROM_METER
BOLTZMANN_EV_PER_K = BOLTZMANN_JOULE_PER_K / ELEMENTARY_CHARGE_C
KJ_PER_KCAL = THERMOCHEMICAL_KCAL_J / KILOJOULE_J
KCAL_PER_EV = ELEMENTARY_CHARGE_C * AVOGADRO_MOL / THERMOCHEMICAL_KCAL_J
