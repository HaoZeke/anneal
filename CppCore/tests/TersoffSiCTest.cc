// MIT License
// Copyright 2023--present rgpot developers
//
// Tersoff Si-C, J. Tersoff, Phys. Rev. B 39, 5566 (1989),
// doi:10.1103/PhysRevB.39.5566. The erratum Phys. Rev. B 41, 3248 (1990),
// doi:10.1103/PhysRevB.41.3248.2, sets carbon B to 346.74 eV and d to
// 4.3484. Carbon beta is dimensionless and is not an erratum value.
// The diamond energies are the cohesive energies of that fit: 4.63 eV
// for silicon and 7.37 eV for carbon. The digits below are the closed
// form at a0 = 5.432 A (Si) and a0 = 3.5656 A (C), which are the lattice
// constants tabulated for this parameter set.

#include <algorithm>
#include <array>
#include <cmath>
#include <cstdint>
#include <cstring>
#include <limits>
#include <random>
#include <vector>

#include <catch2/catch_all.hpp>

#include "TestFixtures.hpp"
#include "rgpot/Tersoff/TersoffSiC.hpp"
#include "rgpot/nlist/HalfShell.hpp"
#include "rgpot/stress.hpp"
#include "rgpot/types/AtomMatrix.hpp"

using Catch::Matchers::WithinAbs;
using rgpot::TersoffSiC;
using rgpot::types::AtomMatrix;

namespace {

auto cubic(double a) { return rgpot::test_fixtures::cubicCell(a); }

std::vector<double> diamond(int nrep, double a0) {
  std::vector<double> pos;
  rgpot::test_fixtures::appendDiamond(pos, nrep, a0);
  return pos;
}

struct Homo {
  double big_a, big_b, lambda1, lambda2, beta, n, c, d, h;
};

constexpr Homo kSi = {1830.8,  471.18, 2.4799, 1.7322,  1.1e-6,
                      0.78734, 100390, 16.217, -0.59825};
constexpr Homo kC = {1393.6,  346.74, 3.4879, 2.2119,  1.5724e-7,
                     0.72751, 38049,  4.3484, -0.57058};

double gOf(const Homo &sp, double cth) {
  const double c2 = sp.c * sp.c;
  const double d2 = sp.d * sp.d;
  const double gap = sp.h - cth;
  const double den = d2 + gap * gap;
  return 1.0 + c2 / d2 - c2 / den;
}

/// Per-atom diamond energy. Four neighbours, fc = 1, zeta = 3 g(cos).
double diamondAtom(const Homo &sp, double a0) {
  const double r = a0 * std::sqrt(3.0) / 4.0;
  const double zeta = 3.0 * gOf(sp, -1.0 / 3.0);
  const double b = std::pow(
      1.0 + std::pow(sp.beta, sp.n) * std::pow(zeta, sp.n), -0.5 / sp.n);
  const double f_r = sp.big_a * std::exp(-sp.lambda1 * r);
  const double f_a = -sp.big_b * std::exp(-sp.lambda2 * r);
  return 2.0 * (f_r + b * f_a);
}

double energyOf(TersoffSiC &pot, const std::vector<double> &pos, int z,
                double box_a) {
  const std::size_t n = pos.size() / 3;
  AtomMatrix positions(n, 3);
  std::copy(pos.begin(), pos.end(), positions.data());
  const std::vector<int> types(n, z);
  auto [energy, forces, variance] = pot(positions, types, cubic(box_a));
  (void)variance;
  // The tabulated lattice constant is not the exact minimum, so individual
  // forces need not vanish. Pair forces still sum to zero.
  for (size_t col = 0; col < 3; ++col) {
    double sum = 0.0;
    for (size_t i = 0; i < forces.rows(); ++i) {
      sum += forces(i, col);
    }
    REQUIRE_THAT(sum, WithinAbs(0.0, 1e-8));
  }
  return energy;
}

} // namespace

TEST_CASE("Tersoff 1989 diamond cohesive energies", "[TersoffSiC]") {
  TersoffSiC pot;
  const double si = diamondAtom(kSi, 5.432);
  const double c = diamondAtom(kC, 3.5656);
  // Closed form at the tabulated lattice constants.
  REQUIRE_THAT(si, WithinAbs(-4.629595012655102, 1e-9));
  REQUIRE_THAT(c, WithinAbs(-7.370514205985597, 1e-9));

  // Eight-atom cells: the second neighbour stays outside the outer cutoff
  // after the minimum-image fold, so the crystal has four neighbours.
  const double e_si = energyOf(pot, diamond(1, 5.432), 14, 5.432);
  const double e_c = energyOf(pot, diamond(1, 3.5656), 6, 3.5656);
  REQUIRE_THAT(e_si / 8.0, WithinAbs(si, 1e-8));
  REQUIRE_THAT(e_c / 8.0, WithinAbs(c, 1e-8));

  const auto big = diamond(3, 5.432);
  double e_big = 0.0;
  std::vector<double> grad(big.size());
  std::vector<int> types(big.size() / 3, 14);
  const double box[9] = {3 * 5.432, 0, 0, 0, 3 * 5.432, 0, 0, 0, 3 * 5.432};
  pot.energyAndGradient(big.data(), types.data(), box, types.size(), &e_big,
                        grad.data());
  REQUIRE_THAT(e_big, WithinAbs(27.0 * e_si, 1e-6));
  REQUIRE(pot.getType() == rgpot::PotType::TersoffSi);
  REQUIRE(pot.paramsKey() == TersoffSiC{}.paramsKey());
}

TEST_CASE("Tersoff 1989 gradient matches a finite difference", "[TersoffSiC]") {
  const double xyz[4][3] = {{0.0, 0.0, 0.0},
                            {2.30, 0.15, 0.05},
                            {0.40, 2.25, 0.20},
                            {1.15, 0.90, 1.40}};
  const int types[4] = {14, 14, 14, 6};
  std::vector<double> pos(12);
  for (int i = 0; i < 4; ++i) {
    pos[3 * i] = xyz[i][0];
    pos[3 * i + 1] = xyz[i][1];
    pos[3 * i + 2] = xyz[i][2];
  }
  const double box[9] = {20, 0, 0, 0, 20, 0, 0, 0, 20};
  TersoffSiC pot;
  double energy = 0.0;
  std::vector<double> grad(12);
  pot.energyAndGradient(pos.data(), types, box, 4, &energy, grad.data());

  AtomMatrix positions(4, 3);
  std::copy(pos.begin(), pos.end(), positions.data());
  auto [e_call, forces, variance] =
      pot(positions, std::vector<int>{14, 14, 14, 6}, cubic(20.0));
  (void)variance;
  REQUIRE_THAT(e_call, WithinAbs(energy, 1e-12));
  for (int i = 0; i < 12; ++i) {
    REQUIRE_THAT(grad[static_cast<std::size_t>(i)],
                 WithinAbs(-forces.data()[i], 1e-12));
  }

  const double h = 1e-6;
  for (int k = 0; k < 12; ++k) {
    pos[static_cast<std::size_t>(k)] += h;
    double ep = 0.0;
    std::vector<double> scratch(12);
    pot.energyAndGradient(pos.data(), types, box, 4, &ep, scratch.data());
    pos[static_cast<std::size_t>(k)] -= 2.0 * h;
    double em = 0.0;
    pot.energyAndGradient(pos.data(), types, box, 4, &em, scratch.data());
    pos[static_cast<std::size_t>(k)] += h;
    REQUIRE_THAT(grad[static_cast<std::size_t>(k)],
                 WithinAbs((ep - em) / (2.0 * h), 2e-6));
  }
  double net[3] = {};
  for (int i = 0; i < 4; ++i) {
    net[0] += grad[3 * i];
    net[1] += grad[3 * i + 1];
    net[2] += grad[3 * i + 2];
  }
  for (double c : net) {
    REQUIRE_THAT(c, WithinAbs(0.0, 1e-8));
  }
}

TEST_CASE("Tersoff 1989 batch and periodic image", "[TersoffSiC]") {
  TersoffSiC pot;
  const int types[2] = {14, 6};
  const double box[9] = {30, 0, 0, 0, 30, 0, 0, 0, 30};
  const double pos[12] = {0, 0, 0, 1.9, 0, 0, 2, 2, 2, 2, 3.8, 2};
  double energies[2] = {};
  double grads[12];
  pot.energyAndGradientBatch(2, pos, types, box, 2, energies, grads);
  double e0 = 0.0;
  double e1 = 0.0;
  double g0[6];
  double g1[6];
  pot.energyAndGradient(pos, types, box, 2, &e0, g0);
  pot.energyAndGradient(pos + 6, types, box, 2, &e1, g1);
  REQUIRE_THAT(energies[0], WithinAbs(e0, 0.0));
  REQUIRE_THAT(energies[1], WithinAbs(e1, 0.0));

  const double L = 12.0;
  const double cell[9] = {L, 0, 0, 0, L, 0, 0, 0, L};
  const double centered[6] = {0, 0, 0, 2.2, 0, 0};
  const double wrapped[6] = {0.4, 0, 0, 0.4 + 2.2 - L, 0, 0};
  const int si[2] = {14, 14};
  pot.energyAndGradient(centered, si, cell, 2, &e0, g0);
  pot.energyAndGradient(wrapped, si, cell, 2, &e1, g1);
  REQUIRE_THAT(e1, WithinAbs(e0, 1e-10));

  const double skew[9] = {30, 0, 0, 3, 30, 0, 0, 1, 30};
  double e_skew = 0.0;
  pot.energyAndGradient(centered, si, skew, 2, &e_skew, g0);
  REQUIRE_THAT(e_skew, WithinAbs(e0, 1e-10));

  const int bad[2] = {14, 8};
  REQUIRE_THROWS_AS(pot.energyAndGradient(centered, bad, cell, 2, &e0, g0),
                    std::invalid_argument);
}

TEST_CASE("Tersoff 1989 held list matches a rebuild", "[TersoffSiC]") {
  const auto pos0 = diamond(3, 5.432);
  std::vector<double> pos = pos0;
  pos[0] += 0.07;
  pos[1] -= 0.04;
  pos[2] += 0.02;
  const std::size_t n = pos.size() / 3;
  std::vector<int> types(n, 14);
  const double box[9] = {3 * 5.432, 0, 0, 0, 3 * 5.432, 0, 0, 0, 3 * 5.432};
  TersoffSiC pot;
  double e0 = 0.0;
  std::vector<double> g0(pos.size());
  pot.energyAndGradient(pos.data(), types.data(), box, n, &e0, g0.data());
  rgpot::nlist::NeighborList held;
  held.skin = 0.4;
  double e1 = 0.0;
  std::vector<double> g1(pos.size());
  pot.energyAndGradient(pos.data(), types.data(), box, n, &e1, g1.data(),
                        &held);
  REQUIRE(std::memcmp(&e0, &e1, sizeof(double)) == 0);
  REQUIRE(std::memcmp(g0.data(), g1.data(), g0.size() * sizeof(double)) == 0);
  pot.energyAndGradient(pos.data(), types.data(), box, n, &e1, g1.data(),
                        &held);
  REQUIRE(std::memcmp(&e0, &e1, sizeof(double)) == 0);
  REQUIRE(std::memcmp(g0.data(), g1.data(), g0.size() * sizeof(double)) == 0);
}

namespace {

struct Eval {
  double energy = 0.0;
  std::vector<double> force;
};

Eval forceOf(const TersoffSiC &pot, const std::vector<double> &pos,
             const double *box) {
  const std::size_t n = pos.size() / 3;
  const std::vector<int> types(n, 14);
  Eval r;
  r.force.assign(pos.size(), 0.0);
  rgpot::ForceInput in{
      .nAtoms = n, .pos = pos.data(), .atmnrs = types.data(), .box = box};
  rgpot::ForceOut out{.F = r.force.data(),
                      .energy = 0.0,
                      .variance = 0.0,
                      .stress = {},
                      .has_stress = 0};
  pot.forceImpl(in, &out);
  r.energy = out.energy;
  return r;
}

bool sameBits(const Eval &a, const Eval &b) {
  return a.force.size() == b.force.size() &&
         std::memcmp(&a.energy, &b.energy, sizeof(double)) == 0 &&
         std::memcmp(a.force.data(), b.force.data(),
                     a.force.size() * sizeof(double)) == 0;
}

/// Uniform in [-amp, amp] per coordinate, from the raw mt19937 stream so
/// every standard library draws the same numbers.
std::vector<double> jitter(const std::vector<double> &pos, double amp,
                           std::uint32_t seed) {
  std::mt19937 gen(seed);
  std::vector<double> out = pos;
  for (double &x : out) {
    const double u = static_cast<double>(gen()) / 4294967295.0;
    x += amp * (2.0 * u - 1.0);
  }
  return out;
}

/// Cell of each atom on the cutoff-edged grid that orders the pairs. A
/// held list keeps the order of its build, so it repeats a rebuild bit for
/// bit while no atom changes cell.
std::vector<int> orderCells(const std::vector<double> &pos, double len) {
  const int nc = static_cast<int>(std::floor(len / 3.0));
  std::vector<int> cells(pos.size() / 3);
  for (std::size_t i = 0; i < cells.size(); ++i) {
    int c = 0;
    for (int k = 0; k < 3; ++k) {
      double x = std::fmod(pos[3 * i + static_cast<std::size_t>(k)], len);
      if (x < 0.0) {
        x += len;
      }
      c = c * nc + std::min(nc - 1, static_cast<int>(x / len * nc));
    }
    cells[i] = c;
  }
  return cells;
}

void heldSequence(int nrep) {
  constexpr double kA0 = 5.432;
  constexpr double kSkin = 0.4;
  const double len = nrep * kA0;
  double box[9] = {len, 0, 0, 0, len, 0, 0, 0, len};
  // Atoms sit on every a0/28 plane and the order grid has edge 4 a0 / 7,
  // so half a plane spacing keeps each atom 0.097 A from a cell face.
  std::vector<double> pos0 = diamond(nrep, kA0);
  for (double &x : pos0) {
    x += 0.5 * kA0 / 28.0;
  }
  const auto cells0 = orderCells(pos0, len);

  TersoffSiC fresh;
  TersoffSiC held;
  held.setHeldSkin(kSkin);
  REQUIRE(fresh.caps().reentrancy == rgpot::Reentrancy::PerInstance);
  REQUIRE(held.caps().reentrancy == rgpot::Reentrancy::PerInstance);
  REQUIRE_FALSE(held.heldList().built);

  // First call builds.
  REQUIRE(sameBits(forceOf(held, pos0, box), forceOf(fresh, pos0, box)));
  REQUIRE(held.heldList().built);
  REQUIRE(held.heldList().ref == pos0);

  // Same coordinates reuse the stored displacements.
  REQUIRE(sameBits(forceOf(held, pos0, box), forceOf(fresh, pos0, box)));
  REQUIRE(held.heldList().ref == pos0);

  // Inside half the skin and inside every cell: reuse, same bits.
  for (std::uint32_t seed : {11u, 12u, 13u}) {
    const auto pos = jitter(pos0, 0.05, seed);
    REQUIRE(orderCells(pos, len) == cells0);
    REQUIRE(sameBits(forceOf(held, pos, box), forceOf(fresh, pos, box)));
    REQUIRE(held.heldList().ref == pos0);
  }

  // Inside half the skin (|d| < 0.11 sqrt 3 < 0.2) but across cell faces:
  // reuse, and the bonds of a centre may be summed in another order.
  {
    const auto pos = jitter(pos0, 0.11, 21u);
    REQUIRE(orderCells(pos, len) != cells0);
    const Eval a = forceOf(held, pos, box);
    const Eval b = forceOf(fresh, pos, box);
    REQUIRE(held.heldList().ref == pos0);
    REQUIRE_THAT(a.energy, WithinAbs(b.energy, 1e-12 * std::fabs(b.energy)));
    for (std::size_t i = 0; i < a.force.size(); ++i) {
      REQUIRE_THAT(a.force[i], WithinAbs(b.force[i], 1e-12));
    }
  }

  // A random move past half the skin rebuilds at the new coordinates.
  {
    const auto pos = jitter(pos0, 0.3, 31u);
    REQUIRE(sameBits(forceOf(held, pos, box), forceOf(fresh, pos, box)));
    REQUIRE(held.heldList().ref == pos);
  }

  // One atom moved by 0.21 A, the rest still: rebuild.
  auto one = pos0;
  one[3 * 5] += 0.21;
  REQUIRE(sameBits(forceOf(held, one, box), forceOf(fresh, one, box)));
  REQUIRE(held.heldList().ref == one);
  // Back by 0.19 A from that build: reuse.
  auto back = one;
  back[3 * 5] -= 0.19;
  REQUIRE(orderCells(back, len) == orderCells(one, len));
  REQUIRE(sameBits(forceOf(held, back, box), forceOf(fresh, back, box)));
  REQUIRE(held.heldList().ref == one);

  // A new cell rebuilds even when no atom moved.
  box[0] *= 1.001;
  REQUIRE(sameBits(forceOf(held, back, box), forceOf(fresh, back, box)));
  REQUIRE(held.heldList().ref == back);
  REQUIRE(held.heldList().box[0] == box[0]);

  // A new atom count rebuilds.
  std::vector<double> fewer(back.begin(), back.end() - 3);
  REQUIRE(sameBits(forceOf(held, fewer, box), forceOf(fresh, fewer, box)));
  REQUIRE(held.heldList().n == fewer.size() / 3);
  box[0] = len;

  // Skin 0 goes back to a rebuild per call.
  held.setHeldSkin(0.0);
  REQUIRE(held.caps().reentrancy == rgpot::Reentrancy::PerInstance);
  REQUIRE(sameBits(forceOf(held, pos0, box), forceOf(fresh, pos0, box)));
  REQUIRE_FALSE(held.heldList().built);
}

} // namespace

TEST_CASE("Tersoff 1989 forceImpl keeps a skinned list on request",
          "[TersoffSiC]") {
  SECTION("512 atoms") { heldSequence(4); }
  SECTION("4096 atoms") { heldSequence(8); }
  TersoffSiC pot;
  REQUIRE_THROWS_AS(pot.setHeldSkin(-0.1), std::invalid_argument);
  REQUIRE_THROWS_AS(pot.setHeldSkin(std::numeric_limits<double>::quiet_NaN()),
                    std::invalid_argument);
  REQUIRE_THROWS_AS(pot.setHeldSkin(std::numeric_limits<double>::infinity()),
                    std::invalid_argument);
}

namespace {

// The worst error seen on these cells was 4.8e-11 eV/Angstrom^3.
constexpr double kStressTol = 5e-10; // eV/Angstrom^3

/// (1/V) dE/d(eps_row,col) by the fourth-order central difference. The
/// strain maps every position and every cell row r to r (I + eps).
double strainDifference(const std::vector<double> &pos,
                        const std::vector<int> &types,
                        const std::array<double, 9> &box, int row, int col) {
  constexpr double h = 2e-4;
  const std::size_t n = pos.size() / 3;
  const TersoffSiC pot;
  auto energyAt = [&](double eps) {
    std::vector<double> p = pos;
    std::array<double, 9> cell = box;
    for (std::size_t a = 0; a < n; ++a) {
      p[3 * a + col] += eps * pos[3 * a + row];
    }
    for (int k = 0; k < 3; ++k) {
      cell[3 * k + col] += eps * box[3 * k + row];
    }
    double e = 0.0;
    std::vector<double> g(p.size());
    pot.energyAndGradient(p.data(), types.data(), cell.data(), n, &e, g.data());
    return e;
  };
  return (8.0 * (energyAt(h) - energyAt(-h)) -
          (energyAt(2 * h) - energyAt(-2 * h))) /
         (12.0 * h * rgpot::cellVolume(box.data()));
}

/// Checks the forceImpl stress against strainDifference for all nine
/// entries and returns the differences, row-major. The forces must equal
/// the negated energyAndGradient gradient exactly, since only forceImpl
/// takes the strain path. A zero force is +0 on both paths, so the check
/// compares values, not sign bits.
std::array<double, 9> checkStressByStrain(const std::vector<double> &pos,
                                          const std::vector<int> &types,
                                          const std::array<double, 9> &box) {
  const std::size_t n = pos.size() / 3;
  const TersoffSiC pot;
  REQUIRE(pot.caps().stress);
  std::vector<double> forces(pos.size());
  rgpot::ForceOut out{forces.data(), 0.0, 0.0, {}, 0};
  pot.forceImpl({n, pos.data(), types.data(), box.data()}, &out);
  REQUIRE(out.has_stress == 1);
  double energy = 0.0;
  std::vector<double> grad(pos.size());
  pot.energyAndGradient(pos.data(), types.data(), box.data(), n, &energy,
                        grad.data());
  REQUIRE(std::memcmp(&energy, &out.energy, sizeof(double)) == 0);
  for (std::size_t i = 0; i < grad.size(); ++i) {
    REQUIRE(forces[i] == -grad[i]);
  }
  std::array<double, 9> fd{};
  for (int row = 0; row < 3; ++row) {
    for (int col = 0; col < 3; ++col) {
      fd[3 * row + col] = strainDifference(pos, types, box, row, col);
      CAPTURE(row, col, fd[3 * row + col]);
      REQUIRE_THAT(out.stress[3 * row + col],
                   WithinAbs(fd[3 * row + col], kStressTol));
    }
  }
  return fd;
}

/// Cubic positions of edge l carried to the same fractional sites of box.
std::vector<double> toCell(const std::vector<double> &cubicPos,
                           const std::array<double, 9> &box, double l) {
  std::vector<double> pos(cubicPos.size());
  for (std::size_t a = 0; a < pos.size() / 3; ++a) {
    for (int c = 0; c < 3; ++c) {
      pos[3 * a + c] =
          (cubicPos[3 * a] * box[c] + cubicPos[3 * a + 1] * box[3 + c] +
           cubicPos[3 * a + 2] * box[6 + c]) /
          l;
    }
  }
  return pos;
}

} // namespace

TEST_CASE("Tersoff 1989 stress matches a strain difference",
          "[TersoffSiC][stress]") {
  SECTION("compressed cubic silicon, 64 atoms") {
    const double a = 0.98 * 5.432;
    const std::vector<int> types(64, 14);
    const std::array<double, 9> box{2 * a, 0, 0, 0, 2 * a, 0, 0, 0, 2 * a};
    const auto fd = checkStressByStrain(diamond(2, a), types, box);
    for (int k : {0, 4, 8}) {
      REQUIRE(fd[k] < -1e-2);
    }
  }
  SECTION("sheared silicon, 64 atoms") {
    const double l = 2 * 5.432;
    const std::vector<int> types(64, 14);
    const std::array<double, 9> box{l, 0,        0,         0.10 * l, l,
                                    0, 0.05 * l, -0.07 * l, l};
    const auto fd =
        checkStressByStrain(toCell(diamond(2, 5.432), box, l), types, box);
    REQUIRE(std::abs(fd[1]) > 1e-3);
    REQUIRE(std::abs(fd[2]) > 1e-3);
    REQUIRE(std::abs(fd[5]) > 1e-3);
  }
  SECTION("sheared and rattled silicon carbide, 64 atoms") {
    // Zincblende SiC, carbon on the first four diamond sites. At a0 = 4.03
    // every Si-Si second neighbour is inside the 2.7 to 3.0 Angstrom taper
    // and at least 0.06 Angstrom from either end, where fc is smooth.
    const double l = 2 * 4.03;
    std::vector<int> types(64);
    for (std::size_t a = 0; a < types.size(); ++a) {
      types[a] = (a % 8 < 4) ? 6 : 14;
    }
    const std::array<double, 9> box{l, 0, 0, 0.03 * l, l, 0, 0, 0.02 * l, l};
    const auto pos = jitter(toCell(diamond(2, 4.03), box, l), 0.03, 41u);
    const auto fd = checkStressByStrain(pos, types, box);
    REQUIRE(std::abs(fd[1]) > 1e-4);
    REQUIRE(std::abs(fd[2]) > 1e-4);
    REQUIRE(std::abs(fd[5]) > 1e-4);
  }
}

TEST_CASE("Tersoff 1989 stress with a held list and without a pair",
          "[TersoffSiC][stress]") {
  const double a = 5.432;
  const auto pos = jitter(diamond(3, a), 0.05, 51u);
  const std::vector<int> types(pos.size() / 3, 14);
  const double box[9] = {3 * a, 0, 0, 0, 3 * a, 0, 0, 0, 3 * a};
  TersoffSiC fresh;
  TersoffSiC held;
  held.setHeldSkin(0.4);
  REQUIRE(held.caps().stress);
  std::vector<double> f0(pos.size());
  std::vector<double> f1(pos.size());
  rgpot::ForceOut o0{f0.data(), 0.0, 0.0, {}, 0};
  rgpot::ForceOut o1{f1.data(), 0.0, 0.0, {}, 0};
  fresh.forceImpl({types.size(), pos.data(), types.data(), box}, &o0);
  held.forceImpl({types.size(), pos.data(), types.data(), box}, &o1);
  held.forceImpl({types.size(), pos.data(), types.data(), box}, &o1);
  REQUIRE(o1.has_stress == 1);
  REQUIRE(std::memcmp(o0.stress, o1.stress, sizeof(o0.stress)) == 0);

  const double position[3] = {1.0, 2.0, 3.0};
  const int type = 14;
  const double cell[9] = {20, 0, 0, 0, 20, 0, 0, 0, 20};
  double force[3] = {};
  rgpot::ForceOut out{force, 1.0, 0.0, {1, 2, 3, 4, 5, 6, 7, 8, 9}, 0};
  fresh.forceImpl({1, position, &type, cell}, &out);
  REQUIRE(out.has_stress == 1);
  REQUIRE(out.energy == 0.0);
  for (double component : out.stress) {
    REQUIRE(component == 0.0);
  }
}

TEST_CASE("Tersoff 1989 names a missing species buffer",
          "[TersoffSiC][stress]") {
  const double position[3] = {1.0, 2.0, 3.0};
  const double cell[9] = {20, 0, 0, 0, 20, 0, 0, 0, 20};
  double force[3] = {};
  rgpot::ForceOut out{force, 0.0, 0.0, {}, 0};
  const TersoffSiC pot;
  REQUIRE_THROWS_WITH(pot.forceImpl({1, position, nullptr, cell}, &out),
                      "TersoffSiC: atomic numbers are required");
}
