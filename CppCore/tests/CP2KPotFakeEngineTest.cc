// MIT License
// Copyright 2023--present rgpot developers

// CP2KPot against cp2kc_fake_engine (RGPOT_CP2K_ENGINE), whose surface is
// E = 1/2 k sum |r_i - r0_i|^2 in Hartree and Bohr.

#include <catch2/catch_test_macros.hpp>

#include <capnp/message.h>

#include <array>
#include <cmath>
#include <cstdlib>
#include <filesystem>
#include <string>
#include <vector>

#include "rgpot/CP2KPot/CP2KPot.hpp"
#include "rgpot/CalculatorGroup.hpp"
#include "rgpot/NWChemPot/DynLib.hpp"
#include "rgpot/rpc/Potentials.capnp.h"
#include "rgpot/types/AtomMatrix.hpp"
#include "rgpot/units.hpp"

namespace {

using Box = std::array<std::array<double, 3>, 3>;
constexpr double kSpring = 0.5;

// One dlopen of the engine path shares the globals CP2KPot's handle sees.
struct Fake {
  rgpot::DynLib lib;
  int (*session_creates)() = nullptr;
  int (*env_creates)() = nullptr;
  int (*set_cell_calls)() = nullptr;
  int (*live_sessions)() = nullptr;
  long long (*last_key)() = nullptr;
  void (*last_cell)(double *) = nullptr;
  const char *(*last_deck)() = nullptr;
  const char *(*last_dir)() = nullptr;

  Fake() {
    const char *path = std::getenv("RGPOT_CP2K_ENGINE");
    REQUIRE(path != nullptr);
    lib.open(path);
    session_creates = lib.sym<int (*)()>("cp2kc_fake_session_creates");
    env_creates = lib.sym<int (*)()>("cp2kc_fake_env_creates");
    set_cell_calls = lib.sym<int (*)()>("cp2kc_fake_set_cell_calls");
    live_sessions = lib.sym<int (*)()>("cp2kc_fake_live_sessions");
    last_key = lib.sym<long long (*)()>("cp2kc_fake_last_key");
    last_cell = lib.sym<void (*)(double *)>("cp2kc_fake_last_cell");
    last_deck = lib.sym<const char *(*)()>("cp2kc_fake_last_deck");
    last_dir = lib.sym<const char *(*)()>("cp2kc_fake_last_dir");
  }
};

bool close_rel(double got, double want, double rel = 1e-12) {
  return std::abs(got - want) <= rel * std::max(1.0, std::abs(want));
}

Box cubic(double a) { return {{{a, 0.0, 0.0}, {0.0, a, 0.0}, {0.0, 0.0, a}}}; }

rgpot::types::AtomMatrix water_positions() {
  constexpr double xyz[3][3] = {
      {0.0, 0.0, 0.1173}, {0.0, 0.7572, -0.4692}, {0.0, -0.7572, -0.4692}};
  rgpot::types::AtomMatrix p(3, 3);
  for (int i = 0; i < 3; ++i)
    for (int k = 0; k < 3; ++k)
      p(i, k) = xyz[i][k];
  return p;
}

} // namespace

TEST_CASE("CP2KPot converts the shim's atomic units to eV and eV/Angstrom",
          "[cp2k][abi]") {
  rgpot::CP2KPot pot;
  REQUIRE(pot.available());
  const auto pos = water_positions();
  const std::vector<int> z{8, 1, 1};
  const auto [energy, forces, variance] = pot(pos, z, cubic(10.0));

  double e_ha = 0.0;
  for (int i = 0; i < 3; ++i) {
    const double r0[3] = {0.1 * i, -0.2 * i, 0.3 * i};
    for (int k = 0; k < 3; ++k) {
      const double d = pos(i, k) * rgpot::units::ANGSTROM_TO_BOHR - r0[k];
      e_ha += 0.5 * kSpring * d * d;
      const double f_ev_ang =
          -kSpring * d * rgpot::units::HARTREE_BOHR_TO_EV_ANGSTROM;
      REQUIRE(close_rel(forces(i, k), f_ev_ang));
    }
  }
  REQUIRE(close_rel(energy, e_ha * rgpot::units::HARTREE_TO_EV));
  REQUIRE(variance == 0.0);
}

TEST_CASE("CP2KPot hands the shim the cell rows in Bohr", "[cp2k][abi]") {
  Fake fake;
  rgpot::CP2KPot pot;
  const Box box = {{{10.0, 0.0, 0.0}, {1.5, 11.0, 0.0}, {0.5, -0.25, 12.0}}};
  (void)pot(water_positions(), {8, 1, 1}, box);
  double cell[9];
  fake.last_cell(cell);
  for (int r = 0; r < 3; ++r)
    for (int c = 0; c < 3; ++c)
      REQUIRE(close_rel(cell[3 * r + c],
                        box[r][c] * rgpot::units::ANGSTROM_TO_BOHR));
}

TEST_CASE("CP2KPot keeps the force environment across positions and cells",
          "[cp2k][abi]") {
  Fake fake;
  rgpot::CP2KPot pot;
  const std::vector<int> z{8, 1, 1};
  auto pos = water_positions();
  const int sessions0 = fake.session_creates();
  const int envs0 = fake.env_creates();
  const int cells0 = fake.set_cell_calls();

  (void)pot(pos, z, cubic(10.0));
  REQUIRE(fake.session_creates() == sessions0 + 1);
  REQUIRE(fake.env_creates() == envs0 + 1);
  REQUIRE(fake.set_cell_calls() == cells0 + 1);

  // A new position reuses the environment and leaves the cell alone.
  pos(1, 0) += 0.05;
  (void)pot(pos, z, cubic(10.0));
  REQUIRE(fake.session_creates() == sessions0 + 1);
  REQUIRE(fake.env_creates() == envs0 + 1);
  REQUIRE(fake.set_cell_calls() == cells0 + 1);

  // A new box sets the cell on the same environment.
  (void)pot(pos, z, cubic(10.5));
  REQUIRE(fake.session_creates() == sessions0 + 1);
  REQUIRE(fake.env_creates() == envs0 + 1);
  REQUIRE(fake.set_cell_calls() == cells0 + 2);
}

TEST_CASE("CP2KPot rebuilds the deck when the elements change", "[cp2k][abi]") {
  Fake fake;
  rgpot::CP2KPot pot;
  const auto pos = water_positions();
  const int sessions0 = fake.session_creates();
  const int envs0 = fake.env_creates();

  (void)pot(pos, {8, 1, 1}, cubic(10.0));
  const std::string first_dir = pot.runDirectory();
  REQUIRE(std::filesystem::exists(first_dir + "/input.inp"));
  REQUIRE(std::string(fake.last_deck()).find("&KIND O") != std::string::npos);

  (void)pot(pos, {16, 1, 1}, cubic(10.0));
  REQUIRE(fake.session_creates() == sessions0 + 2);
  REQUIRE(fake.env_creates() == envs0 + 2);
  const std::string deck = fake.last_deck();
  REQUIRE(deck.find("&KIND S\n") != std::string::npos);
  REQUIRE(deck.find("&KIND O\n") == std::string::npos);
}

TEST_CASE("CP2KPot keeps one environment per orbital key", "[cp2k][abi]") {
  Fake fake;
  rgpot::CP2KPot pot;
  REQUIRE(pot.keepsOrbitalsPerKey());
  const std::vector<int> z{8, 1, 1};
  auto pos = water_positions();
  const int envs0 = fake.env_creates();

  (void)pot(pos, z, cubic(10.0));
  REQUIRE(fake.last_key() == -1000000);
  REQUIRE(fake.env_creates() == envs0 + 1);

  for (long long key : {3LL, 7LL, 3LL, 7LL}) {
    pot.selectOrbitals(key);
    pos(0, 2) += 0.01;
    (void)pot(pos, z, cubic(10.0));
    REQUIRE(fake.last_key() == key);
  }
  // Keys 3 and 7 each got one environment; key 0 kept the first.
  REQUIRE(fake.env_creates() == envs0 + 3);
}

TEST_CASE("CP2KPot reports an unconverged SCF as CP2KScfError", "[cp2k][abi]") {
  rgpot::CP2KPot pot;
  const std::vector<int> z{8, 1, 1};
  const auto pos = water_positions();
#if !defined(_WIN32)
  setenv("RGPOT_CP2K_FAKE_SCF_FAIL", "1", 1);
#endif
  REQUIRE_THROWS_AS(pot(pos, z, cubic(10.0)), rgpot::CP2KScfError);
#if !defined(_WIN32)
  unsetenv("RGPOT_CP2K_FAKE_SCF_FAIL");
#endif
  // The point was rejected, not the potential: the next call succeeds.
  const auto [energy, forces, variance] = pot(pos, z, cubic(10.0));
  REQUIRE(std::isfinite(energy));
  REQUIRE_FALSE(rgpot::mpiAbortRequested());
}

TEST_CASE("CP2KPot removes its run directory unless asked to keep it",
          "[cp2k][abi]") {
  Fake fake;
  const auto parent =
      std::filesystem::temp_directory_path() / "rgpot-cp2k-fake-workdir";
  std::filesystem::remove_all(parent);
  std::filesystem::create_directories(parent);
  std::string removed;
  std::string kept;
  for (const bool keep : {false, true}) {
    ::capnp::MallocMessageBuilder msg;
    auto p = msg.initRoot<::CP2KParams>();
    p.setWorkDir(parent.string());
    p.setKeepWorkDir(keep);
    {
      rgpot::CP2KPot pot(p.asReader());
      (void)pot(water_positions(), {8, 1, 1}, cubic(10.0));
      const std::string dir = pot.runDirectory();
      REQUIRE(dir.rfind(parent.string(), 0) == 0);
      REQUIRE(std::filesystem::exists(dir + "/input.inp"));
      (keep ? kept : removed) = dir;
    }
  }
  REQUIRE_FALSE(std::filesystem::exists(removed));
  REQUIRE(std::filesystem::exists(kept + "/input.inp"));
  REQUIRE(fake.live_sessions() == 0);
  std::filesystem::remove_all(parent);
}

TEST_CASE("CP2KPot refuses a template without geometry tokens", "[cp2k][abi]") {
  ::capnp::MallocMessageBuilder msg;
  auto p = msg.initRoot<::CP2KParams>();
  p.setInputTemplate("&FORCE_EVAL\n&SUBSYS\n&COORD\n@RGPOT_COORD@\n&END COORD"
                     "\n&END SUBSYS\n&END FORCE_EVAL\n");
  rgpot::CP2KPot pot;
  REQUIRE_FALSE(pot.setParams(p.asReader()));
  REQUIRE_FALSE(pot.available());
  REQUIRE(pot.loadError().find("@RGPOT_CELL@") != std::string::npos);

  ::capnp::MallocMessageBuilder cfg_msg;
  auto cfg = cfg_msg.initRoot<::PotentialConfig>();
  cfg.initCp2k().setCutoffRy(400.0);
  std::string message;
  REQUIRE(pot.setPotentialConfig(cfg.asReader(), &message));
  REQUIRE(pot.available());
  REQUIRE(message.find("cp2k arm") != std::string::npos);

  auto other = cfg_msg.initRoot<::PotentialConfig>();
  other.initCpmd();
  REQUIRE_FALSE(pot.setPotentialConfig(other.asReader(), &message));
}

TEST_CASE("CP2KPot paramsKey and the session follow the params",
          "[cp2k][abi]") {
  Fake fake;
  ::capnp::MallocMessageBuilder msg;
  auto p = msg.initRoot<::CP2KParams>();
  rgpot::CP2KPot a(p.asReader());
  rgpot::CP2KPot b(p.asReader());
  REQUIRE(a.paramsKey() == b.paramsKey());

  const std::vector<int> z{8, 1, 1};
  (void)a(water_positions(), z, cubic(10.0));
  const int sessions = fake.session_creates();
  // The same params keep the live session.
  REQUIRE(a.setParams(p.asReader()));
  (void)a(water_positions(), z, cubic(10.0));
  REQUIRE(fake.session_creates() == sessions);

  p.setCharge(1);
  REQUIRE(a.setParams(p.asReader()));
  REQUIRE(a.paramsKey() != b.paramsKey());
  (void)a(water_positions(), z, cubic(10.0));
  REQUIRE(fake.session_creates() == sessions + 1);
  REQUIRE(std::string(fake.last_deck()).find("CHARGE 1\n") !=
          std::string::npos);

  ::capnp::MallocMessageBuilder out_msg;
  auto out = out_msg.initRoot<::CP2KParams>();
  a.getParams(out);
  REQUIRE(out.getCharge() == 1);
  REQUIRE(std::string(out.getBasisSet().cStr()) == "DZVP-MOLOPT-SR-GTH");
}
