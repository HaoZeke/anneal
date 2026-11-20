#pragma once
// MIT License
// Copyright 2023--present rgpot developers
//
// Agreement bars for the dispersion references, from the unit tests of the
// reference libraries: s-dftd3 test/unit/test_dftd3.f90 and dftd4
// test/unit/test_dftd4.f90. Energies in Hartree, gradients in Hartree/Bohr.

#include <limits>

namespace rgpot::testtol {

inline constexpr double kEnergyThr = 100.0 * std::numeric_limits<double>::epsilon();
inline constexpr double kGradThr = 1.4901161193847656e-08; // sqrt(eps)

} // namespace rgpot::testtol
