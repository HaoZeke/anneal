// MIT License
// Copyright 2023--present rgpot developers
//
// Stillinger-Weber silicon, Stillinger and Weber, Phys. Rev. B 31, 5262
// (1985), doi:10.1103/PhysRevB.31.5262. Equation (2.8) gives the diamond
// energy at the pair minimum as -1.999993 epsilon per atom. The pair
// term of equation (2.7) is -1 at that minimum, so the crystal energy is
// -2 epsilon per atom (two bonds per atom) and the three-body term is
// identically zero at the tetrahedral angle.

#include <algorithm>
#include <array>
#include <cmath>
#include <cstdint>
#include <cstring>
#include <random>
#include <vector>

#include <catch2/catch_all.hpp>

#include "rgpot/StillingerWeber/SWSi.hpp"
#include "rgpot/nlist/HalfShell.hpp"
#include "rgpot/stress.hpp"
#include "rgpot/types/AtomMatrix.hpp"

using Catch::Matchers::WithinAbs;
using rgpot::StillingerWeberSi;
using rgpot::types::AtomMatrix;

namespace {

double df2du(double u) {
  const double s = 1.0 / (u - StillingerWeberSi::kCut);
  const double inv = 1.0 / u;
  const double um4 = inv * inv * inv * inv;
  const double g = StillingerWeberSi::kB * um4 - 1.0;
  const double es = std::exp(s);
  const double dgdu =
      -StillingerWeberSi::kP * StillingerWeberSi::kB * um4 * inv;
  return StillingerWeberSi::kA * es * (dgdu - g * s * s);
}

/// Reduced separation that minimises the two-body term.
double pairMinimumU() {
  double u = 1.12;
  for (int it = 0; it < 12; ++it) {
    const double h = 1e-7;
    const double d1 = df2du(u);
    const double d2 = (df2du(u + h) - df2du(u - h)) / (2.0 * h);
    u -= d1 / d2;
  }
  return u;
}

std::array<std::array<double, 3>, 3> cubic(double a) {
  return {{{a, 0.0, 0.0}, {0.0, a, 0.0}, {0.0, 0.0, a}}};
}

void appendDiamond(std::vector<double> &pos, int nrep, double a0) {
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

} // namespace

TEST_CASE("SW 1985 dimer at the pair minimum", "[SWSi]") {
  const double u = pairMinimumU();
  const double r = u * StillingerWeberSi::kSigma;
  const AtomMatrix positions{{0.0, 0.0, 0.0}, {r, 0.0, 0.0}};
  const std::vector<int> types{14, 14};
  const auto box = cubic(30.0);
  StillingerWeberSi pot;
  auto [energy, forces, variance] = pot(positions, types, box);
  (void)variance;
  // f2(u_min) = -1, so one pair is -epsilon. Eq. (2.7).
  REQUIRE_THAT(energy, WithinAbs(-StillingerWeberSi::kEpsilon, 1e-9));
  for (size_t i = 0; i < forces.rows(); ++i) {
    for (size_t j = 0; j < forces.cols(); ++j) {
      REQUIRE_THAT(forces(i, j), WithinAbs(0.0, 1e-8));
    }
  }
  REQUIRE(pot.getType() == rgpot::PotType::SWSi1985);
  REQUIRE(pot.paramsKey() == StillingerWeberSi{}.paramsKey());
}

TEST_CASE("SW 1985 diamond cohesive energy", "[SWSi]") {
  // Eq. (2.8): Phi/N = -1.999993 epsilon at the diamond minimum. The
  // tetrahedral angle makes the three-body term zero, and each atom has
  // two bonds, so the minimum is -2 epsilon per atom.
  const double u = pairMinimumU();
  const double a0 = u * StillingerWeberSi::kSigma * 4.0 / std::sqrt(3.0);
  std::vector<double> pos;
  appendDiamond(pos, 1, a0);
  const std::size_t n = pos.size() / 3;
  REQUIRE(n == 8);
  const auto box = cubic(a0);
  StillingerWeberSi pot;
  const std::vector<int> types(n, 14);
  AtomMatrix positions(n, 3);
  std::copy(pos.begin(), pos.end(), positions.data());
  auto [energy, forces, variance] = pot(positions, types, box);
  (void)variance;
  const double per_atom = energy / static_cast<double>(n);
  REQUIRE_THAT(per_atom, WithinAbs(-2.0 * StillingerWeberSi::kEpsilon, 1e-8));
  REQUIRE_THAT(per_atom / StillingerWeberSi::kEpsilon,
               WithinAbs(-1.999993, 1e-5));
  for (size_t i = 0; i < forces.rows(); ++i) {
    for (size_t j = 0; j < forces.cols(); ++j) {
      REQUIRE_THAT(forces(i, j), WithinAbs(0.0, 1e-7));
    }
  }

  // A 3x3x3 cell has 27 copies and a half-box larger than the cutoff, so
  // the minimum image adds no extra neighbours.
  std::vector<double> big;
  appendDiamond(big, 3, a0);
  const std::size_t nb = big.size() / 3;
  double energy_big = 0.0;
  std::vector<double> grad(big.size());
  const double cell[9] = {3 * a0, 0, 0, 0, 3 * a0, 0, 0, 0, 3 * a0};
  pot.energyAndGradient(big.data(), nullptr, cell, nb, &energy_big,
                        grad.data());
  REQUIRE_THAT(energy_big, WithinAbs(27.0 * energy, 1e-6));
}

TEST_CASE("SW 1985 trimer energies", "[SWSi]") {
  // Independent evaluation of equations (2.7) and (2.8) at r = u_min * sigma,
  // with the erg of equation (2.9). Equilateral: -3.756916215403288 eV.
  // Right angle, including the hypotenuse pair: -4.254103860175029 eV.
  const double r = pairMinimumU() * StillingerWeberSi::kSigma;
  const auto box = cubic(40.0);
  StillingerWeberSi pot;
  const std::vector<int> types{14, 14, 14};

  const AtomMatrix equil{
      {0.0, 0.0, 0.0}, {r, 0.0, 0.0}, {0.5 * r, 0.5 * r * std::sqrt(3.0), 0.0}};
  auto [e1, f1, v1] = pot(equil, types, box);
  (void)f1;
  (void)v1;
  REQUIRE_THAT(e1, WithinAbs(-3.756916215403288, 1e-8));

  const AtomMatrix right{{0.0, 0.0, 0.0}, {r, 0.0, 0.0}, {0.0, r, 0.0}};
  auto [e2, f2, v2] = pot(right, types, box);
  (void)f2;
  (void)v2;
  REQUIRE_THAT(e2, WithinAbs(-4.254103860175029, 1e-8));
}

TEST_CASE("SW 1985 gradient matches a finite difference", "[SWSi]") {
  const double xyz[5][3] = {{0.0, 0.0, 0.0},
                            {2.30, 0.10, 0.05},
                            {0.15, 2.45, 0.20},
                            {0.25, 0.10, 2.35},
                            {1.20, 1.10, 0.80}};
  std::vector<double> pos(15);
  for (int i = 0; i < 5; ++i) {
    pos[3 * i] = xyz[i][0];
    pos[3 * i + 1] = xyz[i][1];
    pos[3 * i + 2] = xyz[i][2];
  }
  const double box[9] = {20, 0, 0, 0, 20, 0, 0, 0, 20};
  StillingerWeberSi pot;
  double energy = 0.0;
  std::vector<double> grad(15);
  pot.energyAndGradient(pos.data(), nullptr, box, 5, &energy, grad.data());

  AtomMatrix positions(5, 3);
  std::copy(pos.begin(), pos.end(), positions.data());
  const std::vector<int> types(5, 14);
  const std::array<std::array<double, 3>, 3> cell = cubic(20.0);
  auto [e_call, forces, variance] = pot(positions, types, cell);
  (void)variance;
  REQUIRE_THAT(e_call, WithinAbs(energy, 1e-12));
  for (int i = 0; i < 15; ++i) {
    REQUIRE_THAT(grad[static_cast<std::size_t>(i)],
                 WithinAbs(-forces.data()[i], 1e-12));
  }

  const double h = 1e-6;
  for (int k = 0; k < 15; ++k) {
    pos[static_cast<std::size_t>(k)] += h;
    double ep = 0.0;
    std::vector<double> gp(15);
    pot.energyAndGradient(pos.data(), nullptr, box, 5, &ep, gp.data());
    pos[static_cast<std::size_t>(k)] -= 2.0 * h;
    double em = 0.0;
    pot.energyAndGradient(pos.data(), nullptr, box, 5, &em, gp.data());
    pos[static_cast<std::size_t>(k)] += h;
    const double fd = (ep - em) / (2.0 * h);
    REQUIRE_THAT(grad[static_cast<std::size_t>(k)], WithinAbs(fd, 2e-6));
  }

  double net[3] = {};
  for (int i = 0; i < 5; ++i) {
    net[0] += grad[3 * i];
    net[1] += grad[3 * i + 1];
    net[2] += grad[3 * i + 2];
  }
  for (double c : net) {
    REQUIRE_THAT(c, WithinAbs(0.0, 1e-8));
  }
}

TEST_CASE("SW 1985 batch matches single calls", "[SWSi]") {
  const double r = pairMinimumU() * StillingerWeberSi::kSigma;
  // Two dimers in one batch, different separations.
  std::vector<double> pos = {0, 0, 0, r, 0, 0, 1, 1, 1, 1 + 1.4 * r, 1, 1};
  const double box[9] = {40, 0, 0, 0, 40, 0, 0, 0, 40};
  double energies[2] = {};
  std::vector<double> grads(12);
  StillingerWeberSi pot;
  const int types[2] = {14, 14};
  pot.energyAndGradientBatch(2, pos.data(), types, box, 2, energies,
                             grads.data());

  double e0 = 0.0;
  double e1 = 0.0;
  std::vector<double> g0(6);
  std::vector<double> g1(6);
  pot.energyAndGradient(pos.data(), types, box, 2, &e0, g0.data());
  pot.energyAndGradient(pos.data() + 6, types, box, 2, &e1, g1.data());
  REQUIRE_THAT(energies[0], WithinAbs(e0, 0.0));
  REQUIRE_THAT(energies[1], WithinAbs(e1, 0.0));
  for (int i = 0; i < 6; ++i) {
    REQUIRE_THAT(grads[static_cast<std::size_t>(i)], WithinAbs(g0[i], 0.0));
    REQUIRE_THAT(grads[static_cast<std::size_t>(6 + i)], WithinAbs(g1[i], 0.0));
  }
}

TEST_CASE("SW 1985 periodic image of one pair", "[SWSi]") {
  const double r = pairMinimumU() * StillingerWeberSi::kSigma;
  const double L = 10.0;
  StillingerWeberSi pot;
  const int types[2] = {14, 14};
  const double box[9] = {L, 0, 0, 0, L, 0, 0, 0, L};
  const double centered[6] = {0.0, 0.0, 0.0, r, 0.0, 0.0};
  const double wrapped[6] = {0.2, 0.0, 0.0, 0.2 + r - L, 0.0, 0.0};
  double e0 = 0.0;
  double e1 = 0.0;
  double g0[6];
  double g1[6];
  pot.energyAndGradient(centered, types, box, 2, &e0, g0);
  pot.energyAndGradient(wrapped, types, box, 2, &e1, g1);
  REQUIRE_THAT(e1, WithinAbs(e0, 1e-10));
  REQUIRE_THAT(e0, WithinAbs(-StillingerWeberSi::kEpsilon, 1e-9));
}

TEST_CASE("SW 1985 held list matches a rebuild", "[SWSi]") {
  const double u = pairMinimumU();
  const double a0 = u * StillingerWeberSi::kSigma * 4.0 / std::sqrt(3.0);
  std::vector<double> pos;
  appendDiamond(pos, 3, a0);
  pos[0] += 0.07;
  pos[1] -= 0.04;
  pos[2] += 0.02;
  const std::size_t n = pos.size() / 3;
  const double box[9] = {3 * a0, 0, 0, 0, 3 * a0, 0, 0, 0, 3 * a0};
  StillingerWeberSi pot;
  double e0 = 0.0;
  std::vector<double> g0(pos.size());
  pot.energyAndGradient(pos.data(), nullptr, box, n, &e0, g0.data());
  rgpot::nlist::NeighborList held;
  held.skin = 0.4;
  double e1 = 0.0;
  std::vector<double> g1(pos.size());
  pot.energyAndGradient(pos.data(), nullptr, box, n, &e1, g1.data(), &held);
  REQUIRE(std::memcmp(&e0, &e1, sizeof(double)) == 0);
  REQUIRE(std::memcmp(g0.data(), g1.data(), g0.size() * sizeof(double)) == 0);
  pot.energyAndGradient(pos.data(), nullptr, box, n, &e1, g1.data(), &held);
  REQUIRE(std::memcmp(&e0, &e1, sizeof(double)) == 0);
  REQUIRE(std::memcmp(g0.data(), g1.data(), g0.size() * sizeof(double)) == 0);
}

TEST_CASE("SW 1985 skewed cell still sees one pair", "[SWSi]") {
  const double r = pairMinimumU() * StillingerWeberSi::kSigma;
  const double pos[6] = {5, 5, 5, 5 + r, 5, 5};
  const double box[9] = {30, 0, 0, 4, 30, 0, 0, 1, 30};
  double energy = 0.0;
  double grad[6];
  StillingerWeberSi{}.energyAndGradient(pos, nullptr, box, 2, &energy, grad);
  REQUIRE_THAT(energy, WithinAbs(-StillingerWeberSi::kEpsilon, 1e-9));
}

namespace {

// The worst error seen on these cells was 2.1e-11 eV/Angstrom^3.
constexpr double kStressTol = 5e-10; // eV/Angstrom^3

/// (1/V) dE/d(eps_row,col) by the fourth-order central difference. The
/// strain maps every position and every cell row r to r (I + eps).
double strainDifference(const std::vector<double> &pos,
                        const std::array<double, 9> &box, int row, int col) {
  constexpr double h = 2e-4;
  const std::size_t n = pos.size() / 3;
  const StillingerWeberSi pot;
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
    pot.energyAndGradient(p.data(), nullptr, cell.data(), n, &e, g.data());
    return e;
  };
  return (8.0 * (energyAt(h) - energyAt(-h)) -
          (energyAt(2 * h) - energyAt(-2 * h))) /
         (12.0 * h * rgpot::cellVolume(box.data()));
}

/// Checks forceImpl stress on (pos, box) against strainDifference on
/// (fdPos, fdBox) for all nine entries and returns the differences,
/// row-major. When the two cells differ, they are cells of one crystal.
std::array<double, 9> checkStressByStrain(const std::vector<double> &pos,
                                          const std::array<double, 9> &box,
                                          const std::vector<double> &fdPos,
                                          const std::array<double, 9> &fdBox) {
  const std::size_t n = pos.size() / 3;
  const StillingerWeberSi pot;
  REQUIRE(pot.caps().stress);
  std::vector<double> forces(pos.size());
  const std::vector<int> types(n, 14);
  rgpot::ForceOut out{forces.data(), 0.0, 0.0, {}, 0};
  pot.forceImpl({n, pos.data(), types.data(), box.data()}, &out);
  REQUIRE(out.has_stress == 1);
  std::array<double, 9> fd{};
  for (int row = 0; row < 3; ++row) {
    for (int col = 0; col < 3; ++col) {
      fd[3 * row + col] = strainDifference(fdPos, fdBox, row, col);
      CAPTURE(row, col, fd[3 * row + col]);
      REQUIRE_THAT(out.stress[3 * row + col],
                   WithinAbs(fd[3 * row + col], kStressTol));
    }
  }
  return fd;
}

std::array<double, 9> checkStressByStrain(const std::vector<double> &pos,
                                          const std::array<double, 9> &box) {
  return checkStressByStrain(pos, box, pos, box);
}

double diamondA0() {
  return pairMinimumU() * StillingerWeberSi::kSigma * 4.0 / std::sqrt(3.0);
}

} // namespace

TEST_CASE("SW 1985 stress matches a strain difference", "[SWSi][stress]") {
  SECTION("compressed cubic diamond, 216 atoms") {
    const double a = 0.98 * diamondA0();
    std::vector<double> pos;
    appendDiamond(pos, 3, a);
    const std::array<double, 9> box{3 * a, 0, 0, 0, 3 * a, 0, 0, 0, 3 * a};
    // Edges of three cutoffs put forceImpl on the half-shell list. A
    // strained 216-atom cell is not orthogonal and takes the all-pairs
    // fold, so the strain difference uses the 64-atom cell of the same
    // crystal. The stress is per volume and does not change.
    std::vector<double> fdPos;
    appendDiamond(fdPos, 2, a);
    const std::array<double, 9> fdBox{2 * a, 0, 0, 0, 2 * a, 0, 0, 0, 2 * a};
    const auto fd = checkStressByStrain(pos, box, fdPos, fdBox);
    // Compression lowers the energy on expansion, so sigma is negative.
    for (int k : {0, 4, 8}) {
      REQUIRE(fd[k] < -1e-2);
    }
  }
  SECTION("sheared diamond, 64 atoms") {
    const double l = 2 * diamondA0();
    std::vector<double> cubicPos;
    appendDiamond(cubicPos, 2, diamondA0());
    const std::array<double, 9> box{l, 0,        0,         0.10 * l, l,
                                    0, 0.05 * l, -0.07 * l, l};
    std::vector<double> pos(cubicPos.size());
    for (std::size_t a = 0; a < pos.size() / 3; ++a) {
      for (int c = 0; c < 3; ++c) {
        pos[3 * a + c] =
            (cubicPos[3 * a] * box[c] + cubicPos[3 * a + 1] * box[3 + c] +
             cubicPos[3 * a + 2] * box[6 + c]) /
            l;
      }
    }
    const auto fd = checkStressByStrain(pos, box);
    REQUIRE(std::abs(fd[1]) > 1e-3);
    REQUIRE(std::abs(fd[2]) > 1e-3);
    REQUIRE(std::abs(fd[5]) > 1e-3);
  }
  SECTION("rattled cubic diamond, 64 atoms") {
    const double a = diamondA0();
    std::vector<double> pos;
    appendDiamond(pos, 2, a);
    std::mt19937_64 gen(20261030);
    for (double &x : pos) {
      x += 0.4 * (static_cast<double>(gen() >> 11) * 0x1.0p-53 - 0.5);
    }
    const std::array<double, 9> box{2 * a, 0, 0, 0, 2 * a, 0, 0, 0, 2 * a};
    const auto fd = checkStressByStrain(pos, box);
    REQUIRE(std::abs(fd[1]) > 1e-4);
    REQUIRE(std::abs(fd[2]) > 1e-4);
    REQUIRE(std::abs(fd[5]) > 1e-4);
  }
}

TEST_CASE("SW 1985 stress is zero without a pair", "[SWSi][stress]") {
  const double position[3] = {1.0, 2.0, 3.0};
  const int type = 14;
  const double box[9] = {20, 0, 0, 0, 20, 0, 0, 0, 20};
  double forces[3] = {};
  rgpot::ForceOut out{forces, 1.0, 0.0, {1, 2, 3, 4, 5, 6, 7, 8, 9}, 0};
  StillingerWeberSi{}.forceImpl({1, position, &type, box}, &out);
  REQUIRE(out.has_stress == 1);
  REQUIRE(out.energy == 0.0);
  for (double component : out.stress) {
    REQUIRE(component == 0.0);
  }
}
