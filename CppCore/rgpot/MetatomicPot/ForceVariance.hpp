// MIT License
// Copyright 2023--present rgpot developers

#pragma once

namespace rgpot {

// ForceOut::variance is a variance. Callers recover an RMS with one square
// root. Storing the RMS here makes that square root a fourth root: an SO(3)
// force scatter of 0.005 eV/A becomes 0.0707 eV/A.

// Mean-square force residual. `sum_of_squares` is the sum of squared
// component deviations and `n_components` counts those components.
inline double forceScatterVariance(double sum_of_squares, double n_components) {
  if (!(n_components > 0.0))
    return 0.0;
  return sum_of_squares / n_components;
}

// Model energy_uncertainty is an RMS-like energy scale (eV). The variance
// slot stores its square.
inline double modelUncertaintyVariance(double energy_uncertainty_mean) {
  return energy_uncertainty_mean * energy_uncertainty_mean;
}

} // namespace rgpot
