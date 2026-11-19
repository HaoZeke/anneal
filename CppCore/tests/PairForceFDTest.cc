// MIT License
// Copyright 2023--present rgpot developers
//
// Finite-difference forces for the potentials ManyBodyForceTest never
// reached: LJPot, LJClusterPot, MorsePot, ZBLPot and EMTPot. Each eOn/LAMMPS
// pin in their own test files checks one hard-coded force vector at one
// geometry; none of them checks that the analytic force is the negative
// gradient of the energy the same call returns, so a pair term and its
// derivative could drift apart (as the EAM pair force once did; see
// ManyBodyForceTest.cc) without failing anything. Each case here perturbs
// one Cartesian coordinate of one atom and compares the central-difference
// slope of the energy to the matching analytic force component.

#include <array>
#include <cmath>
#include <vector>

#include <catch2/catch_all.hpp>

#include "lsan_libomp.hpp"
#include "rgpot/EMT/EMTPot.hpp"
#include "rgpot/LennardJones/LJClusterPot.hpp"
#include "rgpot/LennardJones/LJPot.hpp"
#include "rgpot/Morse/MorsePot.hpp"
#include "rgpot/ZBL/ZBLPot.hpp"
#include "rgpot/types/AtomMatrix.hpp"

namespace {

using rgpot::types::AtomMatrix;

/// Central difference of the energy along one Cartesian coordinate of one
/// atom, checked against the analytic force component at that coordinate.
template <typename Pot>
void checkDerivative(Pot &pot, AtomMatrix xyz, const std::vector<int> &z,
                     const std::array<std::array<double, 3>, 3> &box,
                     std::size_t atom, int coord, double h = 1e-5) {
  const auto got = pot(xyz, z, box);
  const double analytic = -std::get<1>(got)(atom, coord);
  const double base = xyz(atom, coord);
  xyz(atom, coord) = base + h;
  const double ep = std::get<0>(pot(xyz, z, box));
  xyz(atom, coord) = base - h;
  const double em = std::get<0>(pot(xyz, z, box));
  const double numeric = (ep - em) / (2.0 * h);
  const double scale =
      std::max(1.0, std::max(std::abs(numeric), std::abs(analytic)));
  REQUIRE(std::abs(numeric - analytic) <= 2e-4 * scale);
}

std::array<std::array<double, 3>, 3> cubicCell(double side) {
  return {{{side, 0.0, 0.0}, {0.0, side, 0.0}, {0.0, 0.0, side}}};
}

} // namespace

TEST_CASE("LJPot force matches the energy derivative", "[lj][force]") {
  // A box well past twice the default 15 A cutoff, so the minimum image
  // convention keeps one pair per direction.
  rgpot::LJPot pot;
  AtomMatrix xyz(3, 3);
  xyz(0, 0) = 0.0;
  xyz(0, 1) = 0.0;
  xyz(0, 2) = 0.0;
  xyz(1, 0) = 1.3;
  xyz(1, 1) = 0.0;
  xyz(1, 2) = 0.0;
  xyz(2, 0) = 0.5;
  xyz(2, 1) = 1.1;
  xyz(2, 2) = 0.2;
  const std::vector<int> z{1, 1, 1};
  const auto box = cubicCell(40.0);
  checkDerivative(pot, xyz, z, box, 1, 0);
  checkDerivative(pot, xyz, z, box, 2, 1);
  checkDerivative(pot, xyz, z, box, 0, 2);
}

TEST_CASE("LJClusterPot force matches the energy derivative",
          "[ljcluster][force]") {
  rgpot::LJClusterPot pot;
  AtomMatrix xyz(3, 3);
  xyz(0, 0) = 0.0;
  xyz(0, 1) = 0.0;
  xyz(0, 2) = 0.0;
  xyz(1, 0) = 1.3;
  xyz(1, 1) = 0.0;
  xyz(1, 2) = 0.0;
  xyz(2, 0) = 0.5;
  xyz(2, 1) = 1.1;
  xyz(2, 2) = 0.2;
  const std::vector<int> z{1, 1, 1};
  const auto box = cubicCell(40.0);
  checkDerivative(pot, xyz, z, box, 1, 0);
  checkDerivative(pot, xyz, z, box, 0, 2);
}

TEST_CASE("MorsePot force matches the energy derivative", "[morse][force]") {
  rgpot::MorsePot pot;
  AtomMatrix xyz(2, 3);
  xyz(0, 0) = 0.0;
  xyz(0, 1) = 0.0;
  xyz(0, 2) = 0.0;
  xyz(1, 0) = 3.1;
  xyz(1, 1) = 0.2;
  xyz(1, 2) = -0.1;
  const std::vector<int> z{78, 78};
  const auto box = cubicCell(20.0);
  checkDerivative(pot, xyz, z, box, 1, 0);
  checkDerivative(pot, xyz, z, box, 1, 1);
}

TEST_CASE("ZBLPot force matches the energy derivative", "[zbl][force]") {
  // eOn's Si-Au dimer (ZBLPotTest.cc siAuDimer): the shared test geometry
  // keeps this case consistent with the dimer the LAMMPS pin checks.
  rgpot::ZBLPot pot;
  AtomMatrix xyz(2, 3);
  xyz(0, 0) = 9.40;
  xyz(0, 1) = 9.35;
  xyz(0, 2) = 9.30;
  xyz(1, 0) = 10.60;
  xyz(1, 1) = 10.65;
  xyz(1, 2) = 10.70;
  const std::vector<int> z{14, 79};
  const auto box = cubicCell(20.0);
  checkDerivative(pot, xyz, z, box, 1, 0);
  checkDerivative(pot, xyz, z, box, 0, 2);
}

TEST_CASE("EMTPot force matches the energy derivative", "[emt][force]") {
  // EMTPotTest.cc's copper cluster pin, reused so this case shares a known
  // geometry with the energy/force values already checked there.
  rgpot::EMTPot pot;
  AtomMatrix xyz(4, 3);
  const std::array<std::array<double, 3>, 4> geom = {{{8.1925, 8.1925, 8.1925},
                                                      {10.0, 10.0, 8.1925},
                                                      {10.0, 8.1925, 10.0},
                                                      {8.1925, 10.0, 10.0}}};
  for (int i = 0; i < 4; ++i) {
    for (int k = 0; k < 3; ++k) {
      xyz(static_cast<std::size_t>(i), k) =
          geom[static_cast<std::size_t>(i)][static_cast<std::size_t>(k)];
    }
  }
  const std::vector<int> z(4, 29);
  const auto box = cubicCell(20.0);
  checkDerivative(pot, xyz, z, box, 0, 0, 1e-4);
  checkDerivative(pot, xyz, z, box, 2, 1, 1e-4);
}
