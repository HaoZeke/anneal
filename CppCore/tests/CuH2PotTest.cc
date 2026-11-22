// MIT License
// Copyright 2023--present rgpot developers
#include <algorithm>
#include <array>
#include <cmath>
#include <limits>
#include <vector>

#include <catch2/catch_all.hpp>

#include "DenseHessianOracle.hpp"
#include "rgpot/HessianVector.hpp"
#include "rgpot/fortran/FortranPots.hpp"
#include "rgpot/types/AtomMatrix.hpp"

TEST_CASE("CuH2Pot Energy and Forces", "[CuH2Pot]") {
  using rgpot::types::AtomMatrix;
  auto cuh2pot = rgpot::fortranpots::CuH2Pot();
  AtomMatrix positions{
      {0.63940268750835, 0.90484742551374, 6.97516498544584}, // Cu
      {3.19652040936288, 0.90417430354811, 6.97547796369474}, // Cu
      {8.98363230369760, 9.94703496017833, 7.83556854923689}, // H
      {7.64080177576300, 9.94703114803832, 7.83556986121272}  // H
  };
  std::vector<int> atmtypes{29, 29, 1, 1};
  std::array<std::array<double, 3>, 3> box{{
      {15.345599999999999, 0, 0}, //
      {0, 21.702000000000002, 0}, //
      {0, 0, 100.00000000000000}  //
  }};
  auto [energy, forces, variance] = cuh2pot(positions, atmtypes, box);
  (void)variance;

  double expected_energy = -2.7114096242662238;
  AtomMatrix expected_forces{
      {1.4919411183978113, -3.9273058476626193E-004, 1.8260603127768336E-004},
      {-1.4919411183978113, 3.9273058476626193E-004, -1.8260603127768336E-004},
      {-4.9118653085630006, -1.3944215503304855E-005, 4.7990036210569753E-006},
      {4.9118653085630006, 1.3944215503304855E-005, -4.7990036210569753E-006}};

  REQUIRE_THAT(energy, Catch::Matchers::WithinAbs(expected_energy, 1e-6));

  // For forces, you may need to loop over each element if there isn't a direct
  // Catch2 matcher for matrices.
  for (int i = 0; i < forces.rows(); ++i) {
    for (int j = 0; j < forces.cols(); ++j) {
      REQUIRE_THAT(forces(i, j),
                   Catch::Matchers::WithinAbs(expected_forces(i, j), 1e-6));
    }
  }
}

TEST_CASE("CuH2Pot hessianVectorImpl matches a dense finite-difference Hessian",
          "[CuH2Pot][HessianVector]") {
  // Four Cu and two H, every species pair at bonding distance. The 8 A
  // cell puts periodic images inside the 6.1 A cutoff; the 30 A cell
  // leaves an isolated cluster.
  const std::vector<double> x{10.0,  10.0,  10.0,  12.55, 10.08, 9.97,
                              11.31, 12.21, 10.04, 11.24, 10.71, 12.13,
                              11.05, 11.02, 8.38,  12.71, 11.43, 8.62};
  const std::vector<int> z{29, 29, 29, 29, 1, 1};
  std::vector<double> v(x.size());
  for (std::size_t k = 0; k < v.size(); ++k) {
    v[k] = std::cos(0.9 * static_cast<double>(k) + 0.2);
  }
  for (const double side : {30.0, 8.0}) {
    const std::array<double, 9> box{side, 0.0, 0.0, 0.0, side, 0.0, 0.0, 0.0, side};
    rgpot::fortranpots::CuH2Pot pot;
    rgpot::HessianVectorCache cache;
    std::vector<double> hv(v.size());
    cache.apply(pot,
                {.nAtoms = z.size(), .pos = x.data(), .atmnrs = z.data(),
                 .box = box.data()},
                v.data(), hv.data(), 0.0, rgpot::HvpScheme::Auto);
    CHECK(cache.counts().analytic == 1);
    CHECK(cache.counts().forceCalls == 0);
    const auto oracle = finiteDifferenceOracle(pot, x, z, box, v);
    for (std::size_t k = 0; k < v.size(); ++k) {
      CHECK(std::abs(hv[k] - oracle.product[k]) <= oracle.bound[k]);
    }
  }
}

namespace {

/// Pairs closer than `cutoff` through the nearest periodic image of an
/// orthorhombic cell whose nearest image is not the direct one, so only the
/// periodic path can see them.
std::size_t wrappedPairs(const std::vector<double> &x, const std::array<double, 3> &sides,
                         double cutoff) {
  std::size_t wrapped = 0;
  const std::size_t n = x.size() / 3;
  for (std::size_t i = 0; i < n; ++i) {
    for (std::size_t j = i + 1; j < n; ++j) {
      double r2 = 0.0;
      bool shifted = false;
      for (std::size_t k = 0; k < 3; ++k) {
        const double direct = x[3 * i + k] - x[3 * j + k];
        const double images = std::round(direct / sides[k]);
        shifted = shifted || images != 0.0;
        const double nearest = direct - images * sides[k];
        r2 += nearest * nearest;
      }
      if (shifted && std::sqrt(r2) < cutoff) {
        ++wrapped;
      }
    }
  }
  return wrapped;
}

} // namespace

TEST_CASE("CuH2Pot hessianVectorImpl follows pairs that wrap across a small cell",
          "[CuH2Pot][HessianVector]") {
  // The default kernel cutoff. Every side stays above it, so no atom meets
  // its own image and each wrapped pair is one that two distinct atoms share.
  constexpr double cutoff = 6.1;
  const std::array<double, 3> sides{6.3, 6.5, 6.7};
  const std::array<double, 9> box{sides[0], 0.0,      0.0,      0.0, sides[1],
                                  0.0,      0.0,      0.0,      sides[2]};
  const std::vector<double> cluster{10.0,  10.0,  10.0,  12.55, 10.08, 9.97,
                                    11.31, 12.21, 10.04, 11.24, 10.71, 12.13,
                                    11.05, 11.02, 8.38,  12.71, 11.43, 8.62};
  // The same atoms folded into the cell, which puts the cluster across the
  // boundary in every direction instead of beside it.
  std::vector<double> folded = cluster;
  for (std::size_t k = 0; k < folded.size(); ++k) {
    const double side = sides[k % 3];
    folded[k] -= side * std::floor(folded[k] / side);
  }
  const std::vector<int> z{29, 29, 29, 29, 1, 1};
  std::vector<double> v(cluster.size());
  for (std::size_t k = 0; k < v.size(); ++k) {
    v[k] = std::cos(0.9 * static_cast<double>(k) + 0.2);
  }
  const double eps = std::numeric_limits<double>::epsilon();
  double vSum = 0.0;
  for (const double value : v) {
    vSum += std::abs(value);
  }

  for (const auto &x : {cluster, folded}) {
    // The pairs this test exists for: nearest image differs from the direct
    // one and the pair still interacts.
    REQUIRE(wrappedPairs(x, sides, cutoff) > 0);

    rgpot::fortranpots::CuH2Pot pot;
    const rgpot::ForceInput in{.nAtoms = z.size(), .pos = x.data(), .atmnrs = z.data(),
                               .box = box.data()};
    std::vector<double> hv(v.size());
    pot.hessianVectorImpl(in, v.data(), hv.data());

    // Step from the length the energy and force scales set, h = cbrt(eps)
    // |E| / max|F|, the balance of the central difference's O(h^2)
    // truncation against the rounding of the forces.
    std::vector<double> f(x.size());
    rgpot::ForceOut out{.F = f.data(), .energy = 0.0, .variance = 0.0, .stress = {},
                        .has_stress = 0};
    pot.forceImpl(in, &out);
    double fMax = 0.0;
    for (const double value : f) {
      fMax = std::max(fMax, std::abs(value));
    }
    REQUIRE(fMax > 0.0);
    const double h = std::cbrt(eps) * std::abs(out.energy) / fMax;
    REQUIRE(h > 0.0);

    double largest = fMax;
    const auto coarse = dense_oracle_detail::product(pot, x, z, box, v, h, largest);
    const auto fine = dense_oracle_detail::product(pot, x, z, box, v, h / 2.0, largest);
    // |H_h v - H_{h/2} v| bounds the truncation left in H_{h/2} v; the
    // rounding is eps times the forces over the step, for every term of
    // the sum over directions.
    const double rounding = eps * largest / (h / 2.0) * vSum;
    double biggest = 0.0;
    double loosest = 0.0;
    for (std::size_t k = 0; k < v.size(); ++k) {
      const double bound = std::abs(coarse[k] - fine[k]) + rounding;
      CHECK(std::abs(hv[k] - fine[k]) <= bound);
      biggest = std::max(biggest, std::abs(hv[k]));
      loosest = std::max(loosest, bound);
    }
    // The check must resolve the product, not pass because it is loose.
    CHECK(biggest > loosest);

    // Without the wrap the cluster is isolated; the product must differ from
    // that by more than the bound, or the cell would not be exercised.
    const std::array<double, 9> open{300.0, 0.0, 0.0, 0.0, 300.0, 0.0, 0.0, 0.0, 300.0};
    const rgpot::ForceInput isolated{.nAtoms = z.size(), .pos = x.data(),
                                     .atmnrs = z.data(), .box = open.data()};
    std::vector<double> hvIsolated(v.size());
    pot.hessianVectorImpl(isolated, v.data(), hvIsolated.data());
    double shift = 0.0;
    for (std::size_t k = 0; k < v.size(); ++k) {
      shift = std::max(shift, std::abs(hv[k] - hvIsolated[k]));
    }
    CHECK(shift > loosest);
  }
}

TEST_CASE("CuH2Pot hessianVectorImpl reports the kernel's faults",
          "[CuH2Pot][HessianVector]") {
  const std::vector<double> x{10.0, 10.0, 10.0, 11.5, 10.0, 10.0};
  const std::vector<int> unknown{29, 26};
  const std::array<double, 9> box{30.0, 0.0, 0.0, 0.0, 30.0, 0.0, 0.0, 0.0, 30.0};
  const std::vector<double> v{0.1, 0.2, 0.3, -0.1, 0.0, 0.2};
  std::vector<double> hv(v.size());
  rgpot::fortranpots::CuH2Pot pot;
  CHECK_THROWS_WITH(pot.hessianVectorImpl({.nAtoms = unknown.size(),
                                           .pos = x.data(),
                                           .atmnrs = unknown.data(),
                                           .box = box.data()},
                                          v.data(), hv.data()),
                    Catch::Matchers::ContainsSubstring("atomic number 26"));
}
