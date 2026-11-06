#pragma once
// MIT License
// Copyright 2023--present rgpot developers

// Row-major cell. Rows of H are the lattice vectors, matching ForceInput.

#include <cmath>
#include <stdexcept>

namespace rgpot::lr {

struct CellGeom {
  double H[3][3]{};
  double inv[3][3]{};
  double recip[3][3]{}; // b_i · a_j = delta_ij, no 2 pi. Rows are b_i.
  double volume = 0.0;

  // s = r * H^{-1}. k_phys = 2 pi * sum_i m_i recip[i].
  void fractional(const double r[3], double s[3]) const noexcept {
    for (int d = 0; d < 3; ++d) {
      s[d] = r[0] * inv[0][d] + r[1] * inv[1][d] + r[2] * inv[2][d];
    }
  }
};

inline CellGeom cell_from_box(const double *box) {
  if (box == nullptr) {
    throw std::invalid_argument("periodic electrostatics need a cell");
  }
  CellGeom g;
  for (int r = 0; r < 3; ++r) {
    for (int c = 0; c < 3; ++c) {
      g.H[r][c] = box[3 * r + c];
    }
  }
  const double *a0 = g.H[0];
  const double *a1 = g.H[1];
  const double *a2 = g.H[2];
  const double c0 = a1[1] * a2[2] - a1[2] * a2[1];
  const double c1 = a1[2] * a2[0] - a1[0] * a2[2];
  const double c2 = a1[0] * a2[1] - a1[1] * a2[0];
  const double det = a0[0] * c0 + a0[1] * c1 + a0[2] * c2;
  if (!(std::abs(det) > 0.0) || !std::isfinite(det)) {
    throw std::invalid_argument("cell is singular");
  }
  g.volume = std::abs(det);
  const double invdet = 1.0 / det;
  // Inverse of a row-vector matrix: columns of the cofactor over det.
  g.inv[0][0] = c0 * invdet;
  g.inv[1][0] = c1 * invdet;
  g.inv[2][0] = c2 * invdet;
  g.inv[0][1] = (a2[1] * a0[2] - a2[2] * a0[1]) * invdet;
  g.inv[1][1] = (a2[2] * a0[0] - a2[0] * a0[2]) * invdet;
  g.inv[2][1] = (a2[0] * a0[1] - a2[1] * a0[0]) * invdet;
  g.inv[0][2] = (a0[1] * a1[2] - a0[2] * a1[1]) * invdet;
  g.inv[1][2] = (a0[2] * a1[0] - a0[0] * a1[2]) * invdet;
  g.inv[2][2] = (a0[0] * a1[1] - a0[1] * a1[0]) * invdet;

  auto cross = [](const double *u, const double *v, double *o) {
    o[0] = u[1] * v[2] - u[2] * v[1];
    o[1] = u[2] * v[0] - u[0] * v[2];
    o[2] = u[0] * v[1] - u[1] * v[0];
  };
  cross(a1, a2, g.recip[0]);
  cross(a2, a0, g.recip[1]);
  cross(a0, a1, g.recip[2]);
  for (int i = 0; i < 3; ++i) {
    for (int c = 0; c < 3; ++c) {
      g.recip[i][c] *= invdet;
    }
  }
  return g;
}

} // namespace rgpot::lr
