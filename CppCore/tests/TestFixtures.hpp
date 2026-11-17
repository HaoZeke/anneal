#pragma once
// MIT License
// Copyright 2023--present rgpot developers

/**
 * @brief Shared test geometries, cells, and thread-count helpers for
 * CppCore unit tests.
 */

#include "rgpot/types/AtomMatrix.hpp"

#include <algorithm>
#include <array>
#include <cmath>
#include <cstdlib>
#include <vector>

namespace rgpot::test_fixtures {

inline void setOmpThreads(const char *value) {
#if defined(_WIN32)
  _putenv_s("OMP_NUM_THREADS", value == nullptr ? "" : value);
#else
  if (value == nullptr || value[0] == '\0') {
    unsetenv("OMP_NUM_THREADS");
  } else {
    setenv("OMP_NUM_THREADS", value, 1);
  }
#endif
}

inline std::array<std::array<double, 3>, 3> cubicCell(double side) {
  return {{{side, 0.0, 0.0}, {0.0, side, 0.0}, {0.0, 0.0, side}}};
}

/// eOn's neb_morse reactant geometry (13 hydrogens, client/unit_tests/data/
/// systems/neb_morse/reactant.con), inlined because rgpot reads no .con.
inline rgpot::types::AtomMatrix nebMorseCluster() {
  return rgpot::types::AtomMatrix{
      {50.594227, 52.017165, 52.277482}, {51.578540, 52.642148, 51.195222},
      {51.243758, 52.919086, 52.218383}, {50.490127, 52.736378, 51.417663},
      {51.522643, 50.884535, 51.656125}, {50.927263, 51.743087, 51.272643},
      {51.240676, 50.960436, 50.573042}, {51.275727, 52.061877, 50.284160},
      {50.434110, 50.976423, 51.879062}, {51.695007, 51.919472, 52.038680},
      {51.363596, 52.199101, 53.064923}, {52.017483, 51.639074, 51.008389},
      {51.187843, 51.165217, 52.678225}};
}

/// The cell that ships with nebMorseCluster().
inline std::array<std::array<double, 3>, 3> nebMorseCell() {
  return {
      {{101.942400, 0.0, 0.0}, {0.0, 103.142600, 0.0}, {0.0, 0.0, 102.605500}}};
}

inline double maxForceNorm(const rgpot::types::AtomMatrix &forces) {
  double worst = 0.0;
  for (std::size_t i = 0; i < forces.rows(); ++i) {
    const double n2 = forces(i, 0) * forces(i, 0) +
                      forces(i, 1) * forces(i, 1) + forces(i, 2) * forces(i, 2);
    worst = std::max(worst, std::sqrt(n2));
  }
  return worst;
}

inline void appendDiamond(std::vector<double> &pos, int nrep, double a0) {
  const double basis[8][3] = {{0.0, 0.0, 0.0},    {0.5, 0.5, 0.0},
                              {0.5, 0.0, 0.5},    {0.0, 0.5, 0.5},
                              {0.25, 0.25, 0.25}, {0.75, 0.75, 0.25},
                              {0.75, 0.25, 0.75}, {0.25, 0.75, 0.75}};
  for (int ix = 0; ix < nrep; ++ix) {
    for (int iy = 0; iy < nrep; ++iy) {
      for (int iz = 0; iz < nrep; ++iz) {
        for (const auto &b : basis) {
          pos.push_back((ix + b[0]) * a0);
          pos.push_back((iy + b[1]) * a0);
          pos.push_back((iz + b[2]) * a0);
        }
      }
    }
  }
}

} // namespace rgpot::test_fixtures
