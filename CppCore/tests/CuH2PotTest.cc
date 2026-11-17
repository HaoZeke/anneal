// MIT License
// Copyright 2023--present rgpot developers
#include <array>
#include <cmath>
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
