// MIT License
// Copyright 2023--present rgpot developers
//
// dftd4 ISO C API smoke (dftd4.h). Golden pins live on rgpot-0r9w.
// This suite only checks that D4Pot evaluates the same water fixture as
// D3Pot, that the explicit ATM (mdb) flag changes the energy, and that
// charge is forwarded to the C API.

#include <catch2/catch_test_macros.hpp>
#include <catch2/matchers/catch_matchers_floating_point.hpp>

#include <array>
#include <cmath>
#include <fstream>
#include <iterator>
#include <regex>
#include <stdexcept>
#include <string>
#include <type_traits>
#include <vector>

#include "rgpot/D4Pot/D4Pot.hpp"
#include "rgpot/types/AtomMatrix.hpp"

using rgpot::types::AtomMatrix;
using Catch::Matchers::WithinAbs;

// Water (A): O + 2H -- same fixture as D3PotTest
static const double kWaterPos[] = {
    0.00000000, 0.00000000,  0.11779000,  // O
    0.00000000, 0.75545000,  -0.47116000, // H
    0.00000000, -0.75545000, -0.47116000  // H
};
static const int kWaterZ[] = {8, 1, 1};

static void fill_water(AtomMatrix &positions, std::vector<int> &atmtypes,
                       std::array<std::array<double, 3>, 3> &box) {
  for (int i = 0; i < 3; ++i)
    for (int j = 0; j < 3; ++j)
      positions(static_cast<size_t>(i), static_cast<size_t>(j)) =
          kWaterPos[i * 3 + j];
  atmtypes.assign(kWaterZ, kWaterZ + 3);
  box = {{{100.0, 0.0, 0.0}, {0.0, 100.0, 0.0}, {0.0, 0.0, 100.0}}};
}

TEST_CASE("D4Pot is a Potential and PotentialConfig.d4 keeps its defaults",
          "[d4][schema]") {
  STATIC_REQUIRE(
      std::is_base_of_v<rgpot::Potential<rgpot::D4Pot>, rgpot::D4Pot>);
  std::ifstream in("CppCore/rgpot/rpc/Potentials.capnp");
  REQUIRE(in);
  const std::string schema((std::istreambuf_iterator<char>(in)),
                           std::istreambuf_iterator<char>());
  // Upstream aligns the arm columns, so match any run of blanks.
  REQUIRE(std::regex_search(schema, std::regex(R"(\bd4\s+@13\s*:D4Params;)")));
  const auto begin = schema.find("struct D4Params {");
  REQUIRE(begin != std::string::npos);
  const std::string params =
      schema.substr(begin, schema.find('}', begin) - begin);
  const rgpot::D4Config defaults;
  REQUIRE(defaults.functional == "pbe");
  REQUIRE(std::regex_search(
      params, std::regex(R"(functional\s+@0\s*:Text\s*=\s*"pbe";)")));
  REQUIRE(defaults.charge == 0.0);
  REQUIRE(std::regex_search(
      params, std::regex(R"(charge\s+@1\s*:Float64\s*=\s*0\.0;)")));
  REQUIRE(defaults.atm);
  REQUIRE(
      std::regex_search(params, std::regex(R"(atm\s+@2\s*:Bool\s*=\s*true;)")));
}

TEST_CASE("D4Pot default is PBE charge 0 with ATM on", "[d4][config]") {
  rgpot::D4Pot pot;
  REQUIRE(pot.getType() == rgpot::PotType::D4);
  REQUIRE(pot.config().functional == "pbe");
  REQUIRE(pot.config().charge == 0.0);
  REQUIRE(pot.config().atm);
}

TEST_CASE("D4Pot rejects an empty functional key", "[d4][config]") {
  rgpot::D4Config cfg;
  cfg.functional.clear();
  REQUIRE_THROWS_AS(rgpot::D4Pot(cfg), std::invalid_argument);
}

TEST_CASE("D4Pot water energy is finite and ATM flips it", "[d4][atm]") {
  AtomMatrix positions(3, 3);
  std::vector<int> atmtypes;
  std::array<std::array<double, 3>, 3> box{};
  fill_water(positions, atmtypes, box);

  rgpot::D4Config on;
  on.functional = "pbe";
  on.charge = 0.0;
  on.atm = true;
  rgpot::D4Config off = on;
  off.atm = false;

  rgpot::D4Pot pot_on(on);
  rgpot::D4Pot pot_off(off);
  auto [e_on, f_on, v_on] = pot_on(positions, atmtypes, box);
  auto [e_off, f_off, v_off] = pot_off(positions, atmtypes, box);
  (void)v_on;
  (void)v_off;

  REQUIRE(std::isfinite(e_on));
  REQUIRE(std::isfinite(e_off));
  REQUIRE(e_on != e_off);
  REQUIRE(std::abs(e_on - e_off) > 1e-12);

  REQUIRE(f_on.rows() == 3);
  REQUIRE(f_off.rows() == 3);
  for (size_t i = 0; i < 3; ++i)
    for (size_t j = 0; j < 3; ++j) {
      REQUIRE(std::isfinite(f_on(i, j)));
      REQUIRE(std::isfinite(f_off(i, j)));
    }

  double fx = 0, fy = 0, fz = 0;
  for (size_t i = 0; i < 3; ++i) {
    fx += f_on(i, 0);
    fy += f_on(i, 1);
    fz += f_on(i, 2);
  }
  REQUIRE_THAT(fx, WithinAbs(0.0, 1e-8));
  REQUIRE_THAT(fy, WithinAbs(0.0, 1e-8));
  REQUIRE_THAT(fz, WithinAbs(0.0, 1e-8));
}

TEST_CASE("D4Pot charge is forwarded and flips water energy", "[d4][charge]") {
  AtomMatrix positions(3, 3);
  std::vector<int> atmtypes;
  std::array<std::array<double, 3>, 3> box{};
  fill_water(positions, atmtypes, box);

  rgpot::D4Config zero;
  zero.functional = "pbe";
  zero.charge = 0.0;
  zero.atm = true;
  rgpot::D4Config cation = zero;
  cation.charge = 1.0;

  rgpot::D4Pot pot0(zero);
  rgpot::D4Pot pot1(cation);
  auto [e0, f0, v0] = pot0(positions, atmtypes, box);
  auto [e1, f1, v1] = pot1(positions, atmtypes, box);
  (void)f0;
  (void)f1;
  (void)v0;
  (void)v1;

  REQUIRE(std::isfinite(e0));
  REQUIRE(std::isfinite(e1));
  REQUIRE(e0 != e1);
  REQUIRE(std::abs(e0 - e1) > 1e-12);
}

TEST_CASE("D4Pot warm update_structure path is stable", "[d4][capi]") {
  rgpot::D4Pot pot;
  AtomMatrix positions(3, 3);
  std::vector<int> atmtypes;
  std::array<std::array<double, 3>, 3> box{};
  fill_water(positions, atmtypes, box);

  auto [e1, f1, v1] = pot(positions, atmtypes, box);
  positions(1, 1) += 0.01;
  auto [e2, f2, v2] = pot(positions, atmtypes, box);
  (void)f1;
  (void)f2;
  (void)v1;
  (void)v2;
  REQUIRE(std::isfinite(e1));
  REQUIRE(std::isfinite(e2));
  REQUIRE(e1 != e2);
}

TEST_CASE("D4Pot reads the species again when only the species change",
          "[d4]") {
  AtomMatrix positions(3, 3);
  std::vector<int> water;
  std::array<std::array<double, 3>, 3> box{};
  fill_water(positions, water, box);
  const std::vector<int> swapped = {16, 1, 1};

  rgpot::D4Pot reused;
  (void)reused(positions, water, box);
  const auto [e_reused, f_reused, v_reused] = reused(positions, swapped, box);

  rgpot::D4Pot fresh;
  const auto [e_fresh, f_fresh, v_fresh] = fresh(positions, swapped, box);

  REQUIRE_THAT(e_reused, WithinAbs(e_fresh, 1e-12));
  for (size_t i = 0; i < 3; ++i) {
    for (size_t k = 0; k < 3; ++k) {
      REQUIRE_THAT(f_reused(i, k), WithinAbs(f_fresh(i, k), 1e-12));
    }
  }
}

TEST_CASE("D4Pot of a molecule reads no box", "[d4][box]") {
  const std::vector<double> pos(kWaterPos, kWaterPos + 9);
  auto run = [&](rgpot::D4Pot &pot, const double *box) {
    std::vector<double> forces(9, 0.0);
    const rgpot::ForceInput in{3, pos.data(), kWaterZ, box};
    rgpot::ForceOut out{};
    out.F = forces.data();
    pot.forceBatch(rgpot::ForceBatch{.nSystems = 1, .in = &in, .out = &out});
    return std::make_pair(out.energy, forces);
  };
  const double wide[9] = {100.0, 0.0, 0.0, 0.0, 100.0, 0.0, 0.0, 0.0, 100.0};

  rgpot::D4Pot boxed;
  const auto with_box = run(boxed, wide);

  // The first call builds the structure, the second updates it; neither
  // may read a box that was not passed.
  rgpot::D4Pot open;
  const auto first = run(open, nullptr);
  const auto warm = run(open, nullptr);
  REQUIRE(first.first == with_box.first);
  REQUIRE(first.second == with_box.second);
  REQUIRE(warm.first == with_box.first);
  REQUIRE(warm.second == with_box.second);

  rgpot::D4Config lattice_config;
  lattice_config.periodic = true;
  rgpot::D4Pot lattice(lattice_config);
  REQUIRE_THROWS_AS(run(lattice, nullptr), std::invalid_argument);
}
