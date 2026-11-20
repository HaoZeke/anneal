// MIT License
// Copyright 2023--present rgpot developers
//
// Golden masters for D3 BJ ATM on/off and D4 vs s-dftd3 / dftd4 library
// refs (rgpot-0r9w). Pins live under CppCore/tests/data/dftd/. Fail closed
// if a named file is missing. Tolerances are the library unit-test bars:
// energy 100*eps, gradient sqrt(eps) (Hartree, Hartree/Bohr).

#include <catch2/catch_test_macros.hpp>
#include <catch2/matchers/catch_matchers_floating_point.hpp>

#include <array>
#include <cmath>
#include <filesystem>
#include <fstream>
#include <limits>
#include <sstream>
#include <string>
#include <vector>

#include "dftd_tolerances.hpp"
#include "npy_io.hpp"
#include "rgpot/units.hpp"

#ifdef RGPOT_HAS_DFTD3
#include "rgpot/D3Pot/D3Pot.hpp"
#endif
#ifdef RGPOT_HAS_DFTD4
#include "rgpot/D4Pot/D4Pot.hpp"
#endif

using rgpot::types::AtomMatrix;
using rgpot::testio::NpyArray;
using rgpot::testio::load_npy;
using rgpot::testio::load_npz;
using Catch::Matchers::WithinAbs;

namespace {

constexpr const char *kData = "CppCore/tests/data/dftd";

using rgpot::testtol::kEnergyThr;
using rgpot::testtol::kGradThr;

const char *kRequired[] = {
    "CppCore/tests/data/dftd/MANIFEST.json",
    "CppCore/tests/data/dftd/geometry.npz",
    "CppCore/tests/data/dftd/water_octamer.xyz",
    "CppCore/tests/data/dftd/d3_bj_pbe_atm_off_energy.npy",
    "CppCore/tests/data/dftd/d3_bj_pbe_atm_off_grad.npy",
    "CppCore/tests/data/dftd/d3_bj_pbe_atm_on_energy.npy",
    "CppCore/tests/data/dftd/d3_bj_pbe_atm_on_grad.npy",
    "CppCore/tests/data/dftd/d4_pbe_energy.npy",
    "CppCore/tests/data/dftd/d4_pbe_grad.npy",
};

void require_file(const std::string &path) {
  REQUIRE(std::filesystem::exists(path));
  REQUIRE(std::filesystem::file_size(path) > 0);
}

std::string slurp(const std::string &path) {
  require_file(path);
  std::ifstream in(path);
  std::ostringstream ss;
  ss << in.rdbuf();
  return ss.str();
}

struct Geom {
  std::size_t nat = 0;
  AtomMatrix positions;
  std::vector<int> numbers;
  std::array<std::array<double, 3>, 3> box{};
};

Geom load_geom() {
  const std::string path = std::string(kData) + "/geometry.npz";
  require_file(path);
  auto arrays = load_npz(path);
  const auto &pos = arrays.at("positions");
  const auto &num = arrays.at("numbers");
  REQUIRE(pos.shape.size() == 2);
  REQUIRE(pos.shape[1] == 3);
  REQUIRE(num.shape.size() == 1);
  REQUIRE(num.shape[0] == pos.shape[0]);
  Geom g;
  g.nat = pos.shape[0];
  g.positions = AtomMatrix(g.nat, 3);
  g.numbers.resize(g.nat);
  for (std::size_t i = 0; i < g.nat; ++i) {
    g.numbers[i] = static_cast<int>(std::lround(num.data[i]));
    for (std::size_t j = 0; j < 3; ++j) {
      g.positions(i, j) = pos.data[3 * i + j];
    }
  }
  g.box = {{{100.0, 0.0, 0.0}, {0.0, 100.0, 0.0}, {0.0, 0.0, 100.0}}};
  return g;
}

struct Pin {
  double energy_ha = 0.0;
  std::vector<double> grad; // Hartree/Bohr, row-major n*3
};

Pin load_pin(const std::string &stem, std::size_t nat) {
  const std::string epath = std::string(kData) + "/" + stem + "_energy.npy";
  const std::string gpath = std::string(kData) + "/" + stem + "_grad.npy";
  require_file(epath);
  require_file(gpath);
  auto e = load_npy(epath);
  auto g = load_npy(gpath);
  REQUIRE(e.data.size() == 1);
  REQUIRE(g.shape.size() == 2);
  REQUIRE(g.shape[0] == nat);
  REQUIRE(g.shape[1] == 3);
  REQUIRE(g.data.size() == 3 * nat);
  Pin p;
  p.energy_ha = e.data[0];
  p.grad = g.data;
  return p;
}

void check_vs_pin(double energy_ev, const AtomMatrix &forces, const Pin &pin) {
  // Convert the library pin once with the same factors D3Pot/D4Pot apply.
  const double e_ev = pin.energy_ha * rgpot::units::HARTREE_TO_EV;
  REQUIRE_THAT(energy_ev, WithinAbs(e_ev, kEnergyThr * rgpot::units::HARTREE_TO_EV));
  REQUIRE(forces.rows() == pin.grad.size() / 3);
  REQUIRE(forces.cols() == 3);
  const double gtol = kGradThr * std::abs(rgpot::units::NEG_GRAD_TO_FORCE);
  for (std::size_t i = 0; i < forces.rows(); ++i) {
    for (std::size_t j = 0; j < 3; ++j) {
      const double f_ref = pin.grad[3 * i + j] * rgpot::units::NEG_GRAD_TO_FORCE;
      REQUIRE_THAT(forces(i, j), WithinAbs(f_ref, gtol));
    }
  }
}

} // namespace

TEST_CASE("dftd golden fixtures exist (fail closed)", "[dftd][golden]") {
  for (const char *p : kRequired) {
    require_file(p);
  }
  const std::string man = slurp("CppCore/tests/data/dftd/MANIFEST.json");
  REQUIRE(man.find("sha256") != std::string::npos);
  REQUIRE(man.find("d3_bj_pbe_atm_off_energy.npy") != std::string::npos);
  REQUIRE(man.find("d3_bj_pbe_atm_on_energy.npy") != std::string::npos);
  REQUIRE(man.find("d4_pbe_energy.npy") != std::string::npos);
  REQUIRE(man.find("100*epsilon") != std::string::npos);
}

TEST_CASE("dftd ATM on pin differs from ATM off pin", "[dftd][golden][atm]") {
  auto geom = load_geom();
  auto off = load_pin("d3_bj_pbe_atm_off", geom.nat);
  auto on = load_pin("d3_bj_pbe_atm_on", geom.nat);
  REQUIRE(std::isfinite(off.energy_ha));
  REQUIRE(std::isfinite(on.energy_ha));
  REQUIRE(on.energy_ha != off.energy_ha);
  REQUIRE(std::abs(on.energy_ha - off.energy_ha) > kEnergyThr);
  double g2 = 0.0, dg2 = 0.0;
  REQUIRE(off.grad.size() == on.grad.size());
  for (std::size_t i = 0; i < off.grad.size(); ++i) {
    g2 += off.grad[i] * off.grad[i];
    const double d = on.grad[i] - off.grad[i];
    dg2 += d * d;
  }
  REQUIRE(g2 > 0.0);
  REQUIRE(dg2 > 0.0);
}

#ifdef RGPOT_HAS_DFTD3
TEST_CASE("D3Pot BJ PBE ATM off matches s-dftd3 pin", "[dftd][golden][d3]") {
  auto geom = load_geom();
  auto pin = load_pin("d3_bj_pbe_atm_off", geom.nat);
  rgpot::D3Config cfg;
  cfg.damping = rgpot::D3Damping::BJ;
  cfg.functional = "pbe";
  cfg.atm = false;
  rgpot::D3Pot pot(cfg);
  auto [energy, forces, var] = pot(geom.positions, geom.numbers, geom.box);
  (void)var;
  check_vs_pin(energy, forces, pin);
}

TEST_CASE("D3Pot BJ PBE ATM on matches s-dftd3 pin", "[dftd][golden][d3]") {
  auto geom = load_geom();
  auto pin = load_pin("d3_bj_pbe_atm_on", geom.nat);
  rgpot::D3Config cfg;
  cfg.damping = rgpot::D3Damping::BJ;
  cfg.functional = "pbe";
  cfg.atm = true;
  rgpot::D3Pot pot(cfg);
  auto [energy, forces, var] = pot(geom.positions, geom.numbers, geom.box);
  (void)var;
  check_vs_pin(energy, forces, pin);
}
#endif

#ifdef RGPOT_HAS_DFTD4
TEST_CASE("D4Pot PBE default matches dftd4 pin", "[dftd][golden][d4]") {
  auto geom = load_geom();
  auto pin = load_pin("d4_pbe", geom.nat);
  rgpot::D4Config cfg;
  cfg.functional = "pbe";
  cfg.charge = 0.0;
  cfg.atm = true;
  rgpot::D4Pot pot(cfg);
  auto [energy, forces, var] = pot(geom.positions, geom.numbers, geom.box);
  (void)var;
  check_vs_pin(energy, forces, pin);
}
#endif
