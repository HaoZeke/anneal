#pragma once
// MIT License
// Copyright 2023--present rgpot developers
//
// A dense finite-difference Hessian, for checking analytic Hessian-vector
// products on systems of a few atoms. It costs 2 force calls per
// coordinate and step, which is why it stays in the tests.

#include <algorithm>
#include <array>
#include <cmath>
#include <cstddef>
#include <limits>
#include <vector>

#include "rgpot/ForceStructs.hpp"

/// @c product is @f$H_{h/2} v@f$ from central differences of the forces at
/// step @f$h/2@f$. @c bound is, per component, @f$|H_h v - H_{h/2} v|@f$,
/// three times the Richardson estimate of the @f$O(h^2)@f$ truncation left
/// in @c product, plus the rounding of the differenced forces.
struct DenseOracle {
  std::vector<double> product;
  std::vector<double> bound;
};

namespace dense_oracle_detail {

template <typename Pot>
std::vector<double> forces(const Pot &pot, const std::vector<double> &x,
                           const std::vector<int> &z, const std::array<double, 9> &box,
                           double &largest) {
  std::vector<double> f(x.size(), 0.0);
  rgpot::ForceInput in{.nAtoms = z.size(), .pos = x.data(), .atmnrs = z.data(),
                       .box = box.data()};
  rgpot::ForceOut out{.F = f.data(), .energy = 0.0, .variance = 0.0, .stress = {},
                      .has_stress = 0};
  pot.forceImpl(in, &out);
  for (double value : f) {
    largest = std::max(largest, std::abs(value));
  }
  return f;
}

template <typename Pot>
std::vector<double> product(const Pot &pot, const std::vector<double> &x,
                            const std::vector<int> &z, const std::array<double, 9> &box,
                            const std::vector<double> &v, double h, double &largest) {
  const std::size_t dim = x.size();
  std::vector<double> hv(dim, 0.0);
  for (std::size_t c = 0; c < dim; ++c) {
    std::vector<double> plus = x;
    std::vector<double> minus = x;
    plus[c] += h;
    minus[c] -= h;
    const auto fp = forces(pot, plus, z, box, largest);
    const auto fm = forces(pot, minus, z, box, largest);
    for (std::size_t r = 0; r < dim; ++r) {
      hv[r] += (fm[r] - fp[r]) / (2.0 * h) * v[c];
    }
  }
  return hv;
}

} // namespace dense_oracle_detail

/// The step is the cube root of machine epsilon times the shortest pair
/// distance, the balance of central-difference truncation and rounding on
/// the length scale the geometry sets.
template <typename Pot>
DenseOracle finiteDifferenceOracle(const Pot &pot, const std::vector<double> &x,
                                   const std::vector<int> &z,
                                   const std::array<double, 9> &box,
                                   const std::vector<double> &v) {
  double shortest = std::numeric_limits<double>::infinity();
  for (std::size_t i = 0; i < z.size(); ++i) {
    for (std::size_t j = i + 1; j < z.size(); ++j) {
      double r2 = 0.0;
      for (std::size_t k = 0; k < 3; ++k) {
        const double d = x[3 * i + k] - x[3 * j + k];
        r2 += d * d;
      }
      shortest = std::min(shortest, std::sqrt(r2));
    }
  }
  const double eps = std::numeric_limits<double>::epsilon();
  const double h = std::cbrt(eps) * shortest;
  double largest = 0.0;
  const auto coarse = dense_oracle_detail::product(pot, x, z, box, v, h, largest);
  const auto fine = dense_oracle_detail::product(pot, x, z, box, v, h / 2.0, largest);
  double vSum = 0.0;
  for (double value : v) {
    vSum += std::abs(value);
  }
  const double rounding = eps * largest / (h / 2.0) * vSum;
  DenseOracle out{fine, std::vector<double>(v.size())};
  for (std::size_t k = 0; k < v.size(); ++k) {
    out.bound[k] = std::abs(coarse[k] - fine[k]) + rounding;
  }
  return out;
}
