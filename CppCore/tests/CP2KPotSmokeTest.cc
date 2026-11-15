// MIT License
// Copyright 2023--present rgpot developers

// CP2KPot against the real libcp2k through libcp2kc_rgpot. Registered only
// with -Dwith_cp2k; every rank of an MPI launch runs the same sequence.

#include <catch2/catch_test_macros.hpp>

#include <capnp/message.h>

#include <array>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <string>
#include <vector>

#include "rgpot/CP2KPot/CP2KPot.hpp"
#include "rgpot/CalculatorGroup.hpp"
#include "rgpot/rpc/Potentials.capnp.h"
#include "rgpot/types/AtomMatrix.hpp"

namespace {

using Box = std::array<std::array<double, 3>, 3>;
constexpr double kA = 5.431; // Angstrom, diamond silicon

rgpot::types::AtomMatrix diamond_si8() {
  constexpr double frac[8][3] = {{0.00, 0.00, 0.00}, {0.00, 0.50, 0.50},
                                 {0.50, 0.00, 0.50}, {0.50, 0.50, 0.00},
                                 {0.25, 0.25, 0.25}, {0.25, 0.75, 0.75},
                                 {0.75, 0.25, 0.75}, {0.75, 0.75, 0.25}};
  rgpot::types::AtomMatrix pos(8, 3);
  for (int i = 0; i < 8; ++i)
    for (int k = 0; k < 3; ++k)
      pos(i, k) = frac[i][k] * kA;
  // Break the symmetry so atom 0 carries a force worth checking.
  pos(0, 0) += 0.05;
  pos(0, 1) += 0.03;
  pos(0, 2) -= 0.02;
  return pos;
}

Box cubic() { return {{{kA, 0.0, 0.0}, {0.0, kA, 0.0}, {0.0, 0.0, kA}}}; }

void reduced_cutoff(::CP2KParams::Builder p) {
  // RGPOT_CP2K_SMOKE_KEEP keeps input.inp and cp2k.out for inspection.
  p.setKeepWorkDir(std::getenv("RGPOT_CP2K_SMOKE_KEEP") != nullptr);
  p.setCutoffRy(300.0);
  p.setRelCutoffRy(40.0);
  // OT stalls near 1e-7 on this cell without restarts; an outer loop of
  // 30-step inner cycles converges it. EPS_SCF 1e-9 (the Leiden setting)
  // keeps the SCF error below the 1e-6 eV rotation check.
  p.setEpsScf(1.0e-9);
  p.setMaxScf(30);
  p.setOuterMaxScf(10);
  // Tighter Gaussian screening than CP2K's default 1e-10, so a rigid
  // rotation of the cell changes the energy by less than 1e-6 eV.
  p.initExtraDftSections(1).set(
      0, "    &QS\n      EPS_DEFAULT 1.0E-14\n    &END QS");
}

const std::vector<int> kZ(8, 14);

} // namespace

TEST_CASE("CP2KPot gives consistent energies and forces for diamond Si8",
          "[cp2k][smoke]") {
  ::capnp::MallocMessageBuilder msg;
  auto p = msg.initRoot<::CP2KParams>();
  reduced_cutoff(p);
  rgpot::CP2KPot pot(p.asReader());
  REQUIRE(pot.available());

  auto pos = diamond_si8();
  const auto [energy, forces, variance] = pot(pos, kZ, cubic());
  std::printf("Si8 E = %.10f eV, F0 = (%.6f, %.6f, %.6f) eV/A\n", energy,
              forces(0, 0), forces(0, 1), forces(0, 2));
  REQUIRE(std::isfinite(energy));
  for (int k = 0; k < 3; ++k) {
    double total = 0.0;
    for (int i = 0; i < 8; ++i)
      total += forces(i, k);
    REQUIRE(std::abs(total) < 1e-4);
  }
  REQUIRE(std::abs(forces(0, 0)) > 1e-2);

  // Central difference of atom 0, h = 1e-3 Angstrom.
  constexpr double h = 1.0e-3;
  for (int k = 0; k < 3; ++k) {
    auto plus = pos;
    auto minus = pos;
    plus(0, k) += h;
    minus(0, k) -= h;
    const double ep = std::get<0>(pot(plus, kZ, cubic()));
    const double em = std::get<0>(pot(minus, kZ, cubic()));
    const double fd = -(ep - em) / (2.0 * h);
    std::printf("atom 0 axis %d: analytic %.6f, central difference %.6f\n", k,
                forces(0, k), fd);
    REQUIRE(std::abs(fd - forces(0, k)) < 1e-2);
  }

  // The same crystal rotated rigidly: every cell row becomes a general
  // vector, which checks that the shim hands libcp2k the rows as cell
  // vectors.
  const double c = std::cos(0.3), s = std::sin(0.3);
  const double cb = std::cos(0.2), sb = std::sin(0.2);
  const double rz[3][3] = {{c, -s, 0}, {s, c, 0}, {0, 0, 1}};
  const double rx[3][3] = {{1, 0, 0}, {0, cb, -sb}, {0, sb, cb}};
  double r[3][3] = {};
  for (int i = 0; i < 3; ++i)
    for (int j = 0; j < 3; ++j)
      for (int k = 0; k < 3; ++k)
        r[i][j] += rx[i][k] * rz[k][j];
  auto rotate = [&](const double v[3], double out[3]) {
    for (int i = 0; i < 3; ++i)
      out[i] = r[i][0] * v[0] + r[i][1] * v[1] + r[i][2] * v[2];
  };
  Box box = cubic();
  Box rbox{};
  for (int row = 0; row < 3; ++row)
    rotate(box[row].data(), rbox[row].data());
  rgpot::types::AtomMatrix rpos(8, 3);
  for (int i = 0; i < 8; ++i) {
    const double v[3] = {pos(i, 0), pos(i, 1), pos(i, 2)};
    double w[3];
    rotate(v, w);
    for (int k = 0; k < 3; ++k)
      rpos(i, k) = w[k];
  }
  const double erot = std::get<0>(pot(rpos, kZ, rbox));
  std::printf("rotated E = %.10f eV (difference %.3e)\n", erot, erot - energy);
  REQUIRE(std::abs(erot - energy) < 1e-6);
}

TEST_CASE("CP2KPot reports an unconverged SCF without aborting",
          "[cp2k][smoke]") {
  ::capnp::MallocMessageBuilder msg;
  auto p = msg.initRoot<::CP2KParams>();
  reduced_cutoff(p);
  p.setMaxScf(1);
  p.setOuterMaxScf(0);
  rgpot::CP2KPot pot(p.asReader());
  REQUIRE(pot.available());
  REQUIRE_THROWS_AS(pot(diamond_si8(), kZ, cubic()), rgpot::CP2KScfError);
  REQUIRE_FALSE(rgpot::mpiAbortRequested());
}
