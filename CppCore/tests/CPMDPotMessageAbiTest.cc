// MIT License
// Copyright 2023--present rgpot developers

#include <catch2/catch_test_macros.hpp>
#include <catch2/matchers/catch_matchers_floating_point.hpp>

#include <capnp/message.h>

#include <array>
#include <cstdlib>
#include <string>
#include <vector>

#include "cpmd_stress_oracle.hpp"
#include "rgpot/CPMDPot/CPMDPot.hpp"
#include "rgpot/NWChemPot/DynLib.hpp"
#include "rgpot/rpc/Potentials.capnp.h"
#include "rgpot/types/AtomMatrix.hpp"
#include "rgpot/units.hpp"

using Catch::Matchers::WithinAbs;

namespace {

// The fake engine is one shared object in this process, so a second
// dlopen of the same path shares its globals with the one CPMDPot holds.
struct FakeEngineCounter {
  using CountFn = int (*)(void);
  rgpot::DynLib lib;
  CountFn count = nullptr;

  FakeEngineCounter() {
    const char *path = std::getenv("RGPOT_CPMD_ENGINE");
    REQUIRE(path != nullptr);
    lib.open(path);
    count = lib.sym<CountFn>("cpmdc_fake_session_create_count");
  }

  int sessions() const { return count(); }
};

// Puts the process environment back so later cases still find the engine.
class RestoreEnv {
public:
  explicit RestoreEnv(const char *name) : name_(name) {
    if (const char *value = std::getenv(name)) {
      had_ = true;
      value_ = value;
    }
  }

  void clear() const { unsetenv(name_); }

  ~RestoreEnv() {
    if (had_)
      setenv(name_, value_.c_str(), 1);
    else
      unsetenv(name_);
  }

private:
  const char *name_;
  bool had_ = false;
  std::string value_;
};

} // namespace

TEST_CASE("CPMDPot passes serialized CPMDParams to cpmdc engine",
          "[cpmd][abi]") {
  ::capnp::MallocMessageBuilder msg;
  auto p = msg.initRoot<::CPMDParams>();
  p.setFunctional("BLYP");
  p.setCutOffRy(70.0);
  p.setCharge(0);
  p.setMultiplicity(1);

  rgpot::CPMDPot pot(p.asReader());
  REQUIRE(pot.available());

  rgpot::types::AtomMatrix positions(1, 3);
  positions(0, 0) = 0.0;
  positions(0, 1) = 0.0;
  positions(0, 2) = 0.0;
  std::vector<int> atmtypes{8};
  std::array<std::array<double, 3>, 3> box = {
      {{20.0, 0.0, 0.0}, {0.0, 21.0, 0.0}, {0.0, 0.0, 23.0}}};

  auto [energy, forces, variance] = pot(positions, atmtypes, box);
  (void)variance;

  const double cell_zz = 23.0;
  const double hartree = cpmd_stress_oracle::sessionHartree(cell_zz);
  REQUIRE_THAT(energy, WithinAbs(0.773, 1e-12));
  REQUIRE_THAT(energy, WithinAbs(hartree * rgpot::units::HARTREE_TO_EV, 1e-12));
  REQUIRE_THAT(forces(0, 0), WithinAbs(0.011, 1e-12));
  REQUIRE_THAT(forces(0, 1), WithinAbs(0.012, 1e-12));
  REQUIRE_THAT(forces(0, 2), WithinAbs(0.013, 1e-12));

  const double pos[3] = {0.0, 0.0, 0.0};
  const int atm = 8;
  const double flat_box[9] = {20.0, 0.0, 0.0, 0.0,    21.0,
                              0.0,  0.0, 0.0, cell_zz};
  std::array<double, 3> raw_forces{};
  rgpot::ForceOut raw{};
  raw.F = raw_forces.data();
  const rgpot::ForceInput in{
      .nAtoms = 1, .pos = pos, .atmnrs = &atm, .box = flat_box};
  pot.forceImpl(in, &raw);
  REQUIRE_THAT(raw.energy,
               WithinAbs(hartree * rgpot::units::HARTREE_TO_EV, 1e-12));
  REQUIRE(raw.has_stress == 1);
  for (int i = 0; i < 9; ++i) {
    REQUIRE_THAT(raw.stress[i],
                 WithinAbs(cpmd_stress_oracle::stressEvPerAngstrom3(i), 1e-9));
  }
}

// A session carries the engine's converged wavefunction. Identical
// params must keep the live session; changed params must replace it.
TEST_CASE("CPMDPot keeps its engine session across identical setParams",
          "[cpmd][abi]") {
  FakeEngineCounter counter;

  ::capnp::MallocMessageBuilder msg;
  auto p = msg.initRoot<::CPMDParams>();
  p.setFunctional("BLYP");
  p.setCutOffRy(70.0);
  p.setCharge(0);
  p.setMultiplicity(1);

  const int before = counter.sessions();
  rgpot::CPMDPot pot(p.asReader());
  REQUIRE(pot.available());
  REQUIRE(counter.sessions() == before + 1);

  REQUIRE(pot.setParams(p.asReader()));
  REQUIRE(pot.setParams(p.asReader()));
  REQUIRE(pot.available());
  REQUIRE(counter.sessions() == before + 1);

  ::capnp::MallocMessageBuilder cfg_msg;
  auto cfg = cfg_msg.initRoot<::PotentialConfig>();
  auto cp = cfg.initCpmd();
  cp.setFunctional("BLYP");
  cp.setCutOffRy(70.0);
  cp.setCharge(0);
  cp.setMultiplicity(1);
  REQUIRE(pot.setPotentialConfig(cfg.asReader()));
  REQUIRE(counter.sessions() == before + 1);

  p.setCutOffRy(80.0);
  REQUIRE(pot.setParams(p.asReader()));
  REQUIRE(pot.available());
  REQUIRE(counter.sessions() == before + 2);
}

// paramsKey feeds the result-cache key: equal params hash equal, any
// field change hashes different, and the salt keeps it off the
// parameter-free default of 0.
TEST_CASE("CPMDPot paramsKey follows the serialized params", "[cpmd][abi]") {
  ::capnp::MallocMessageBuilder msg;
  auto p = msg.initRoot<::CPMDParams>();
  p.setFunctional("BLYP");
  p.setCutOffRy(70.0);
  p.setCharge(0);
  p.setMultiplicity(1);

  rgpot::CPMDPot a(p.asReader());
  rgpot::CPMDPot b(p.asReader());
  REQUIRE(a.paramsKey() != 0);
  REQUIRE(a.paramsKey() == b.paramsKey());

  const auto key_before = a.paramsKey();
  p.setCutOffRy(80.0);
  REQUIRE(a.setParams(p.asReader()));
  REQUIRE(a.paramsKey() != key_before);
  REQUIRE(a.paramsKey() != b.paramsKey());

  p.setCutOffRy(70.0);
  REQUIRE(a.setParams(p.asReader()));
  REQUIRE(a.paramsKey() == key_before);

  rgpot::CPMDPot defaults;
  REQUIRE(defaults.paramsKey() != 0);
  REQUIRE(defaults.paramsKey() != b.paramsKey());
}

// selectOrbitals applies immediately to a live session, so callers can
// intentionally select several keys before the next force.
TEST_CASE("CPMDPot applies orbital selection eagerly", "[cpmd][abi]") {
  rgpot::DynLib lib;
  const char *path = std::getenv("RGPOT_CPMD_ENGINE");
  REQUIRE(path != nullptr);
  lib.open(path);
  using KeyFn = long long (*)(void);
  using OrbitalFn = double (*)(void);
  auto last_key = lib.sym<KeyFn>("cpmdc_fake_last_orbital_key");
  auto last_initial_orbital =
      lib.sym<OrbitalFn>("cpmdc_fake_last_initial_orbital_state");
  auto last_final_orbital =
      lib.sym<OrbitalFn>("cpmdc_fake_last_final_orbital_state");
  constexpr long long kNone = -1000000;

  ::capnp::MallocMessageBuilder msg;
  auto p = msg.initRoot<::CPMDParams>();
  p.setFunctional("BLYP");
  p.setCutOffRy(70.0);
  rgpot::CPMDPot pot(p.asReader());
  REQUIRE(pot.available());
  REQUIRE(pot.keepsOrbitalsPerKey());

  rgpot::types::AtomMatrix positions = rgpot::types::AtomMatrix::Zero(1, 3);
  std::vector<int> atmtypes{8};
  std::array<std::array<double, 3>, 3> box = {
      {{20.0, 0.0, 0.0}, {0.0, 21.0, 0.0}, {0.0, 0.0, 23.0}}};

  (void)pot(positions, atmtypes, box);
  REQUIRE(last_key() == kNone);

  for (long long key : {3LL, 0LL, 3LL, 41LL}) {
    pot.selectOrbitals(key);
    positions(0, 0) += 0.01;
    (void)pot(positions, atmtypes, box);
    REQUIRE(last_key() == key);
  }

  // A rejected key does not replace the key re-applied to a new session.
  REQUIRE_THROWS_WITH(pot.selectOrbitals(-99),
                      "CPMD engine refused the orbital key");
  p.setCutOffRy(80.0);
  REQUIRE(pot.setParams(p.asReader()));
  positions(0, 0) += 0.01;
  (void)pot(positions, atmtypes, box);
  REQUIRE(last_key() == 41);

  // select(seed), select(slot), then force(slot) starts from seed state.
  constexpr long long seed = 7;
  constexpr long long slot = 9;
  pot.selectOrbitals(seed);
  positions(0, 0) += 0.01;
  (void)pot(positions, atmtypes, box);
  const double seed_state = last_final_orbital();
  pot.selectOrbitals(seed);
  pot.selectOrbitals(slot);
  positions(0, 0) += 0.01;
  (void)pot(positions, atmtypes, box);
  REQUIRE(last_key() == slot);
  REQUIRE_THAT(last_initial_orbital(), WithinAbs(seed_state, 1e-12));
}

TEST_CASE("CPMDPot copies orbitals between keys", "[cpmd][abi]") {
  rgpot::DynLib lib;
  const char *path = std::getenv("RGPOT_CPMD_ENGINE");
  REQUIRE(path != nullptr);
  lib.open(path);
  using OrbitalFn = double (*)(void);
  auto last_initial_orbital =
      lib.sym<OrbitalFn>("cpmdc_fake_last_initial_orbital_state");
  auto last_final_orbital =
      lib.sym<OrbitalFn>("cpmdc_fake_last_final_orbital_state");

  ::capnp::MallocMessageBuilder msg;
  auto p = msg.initRoot<::CPMDParams>();
  p.setFunctional("BLYP");
  p.setCutOffRy(70.0);
  rgpot::CPMDPot pot(p.asReader());
  REQUIRE(pot.available());
  REQUIRE(pot.keepsOrbitalsPerKey());

  rgpot::types::AtomMatrix positions = rgpot::types::AtomMatrix::Zero(1, 3);
  std::vector<int> atmtypes{8};
  std::array<std::array<double, 3>, 3> box = {
      {{20.0, 0.0, 0.0}, {0.0, 21.0, 0.0}, {0.0, 0.0, 23.0}}};

  constexpr long long seed = 101;
  constexpr long long slot = 202;
  pot.selectOrbitals(seed);
  positions(0, 0) += 0.01;
  (void)pot(positions, atmtypes, box);
  const double seed_state = last_final_orbital();

  pot.selectOrbitals(slot);
  box[2][2] = 24.0;
  positions(0, 0) += 0.01;
  (void)pot(positions, atmtypes, box);
  REQUIRE(last_final_orbital() != seed_state);

  // Copying into the active slot refreshes the state used by the next force.
  REQUIRE(pot.copyOrbitals(seed, slot));
  box[2][2] = 23.0;
  positions(0, 0) += 0.01;
  (void)pot(positions, atmtypes, box);
  REQUIRE_THAT(last_initial_orbital(), WithinAbs(seed_state, 1e-12));
}

// A refused key remembered before any session exists has to fail configure
// without leaving that session up. The same params must try again.
TEST_CASE("CPMDPot retries a refused orbital key on the same params",
          "[cpmd][abi]") {
  const char *raw = std::getenv("RGPOT_CPMD_ENGINE");
  REQUIRE(raw != nullptr);
  const std::string engine_path(raw);

  RestoreEnv engine_env("RGPOT_CPMD_ENGINE");
  RestoreEnv library_env("CPMDC_LIBRARY");
  RestoreEnv alt_env("RGPOT_CPMDC_ENGINE");

  rgpot::DynLib lib;
  lib.open(engine_path);
  using CountFn = int (*)(void);
  using KeyFn = long long (*)(void);
  auto creates = lib.sym<CountFn>("cpmdc_fake_session_create_count");
  auto last_key = lib.sym<KeyFn>("cpmdc_fake_last_orbital_key");

  engine_env.clear();
  library_env.clear();
  alt_env.clear();

  ::capnp::MallocMessageBuilder msg;
  auto p = msg.initRoot<::CPMDParams>();
  p.setFunctional("BLYP");
  p.setCutOffRy(70.0);
  rgpot::CPMDPot pot(p.asReader());
  REQUIRE_FALSE(pot.available());

  // No live session, so the refusal is remembered for configure.
  pot.selectOrbitals(-99);

  p.setEnginePath(engine_path);
  const int before = creates();
  REQUIRE_FALSE(pot.setParams(p.asReader()));
  REQUIRE(creates() == before + 1);
  REQUIRE_FALSE(pot.available());

  // Identical params must not take the live-session early return.
  REQUIRE_FALSE(pot.setParams(p.asReader()));
  REQUIRE(creates() == before + 2);
  REQUIRE_FALSE(pot.available());

  pot.selectOrbitals(41);
  REQUIRE(pot.setParams(p.asReader()));
  REQUIRE(pot.available());
  REQUIRE(creates() == before + 3);
  REQUIRE(pot.setParams(p.asReader()));
  REQUIRE(creates() == before + 3);

  rgpot::types::AtomMatrix positions = rgpot::types::AtomMatrix::Zero(1, 3);
  std::vector<int> atmtypes{8};
  std::array<std::array<double, 3>, 3> box = {
      {{20.0, 0.0, 0.0}, {0.0, 21.0, 0.0}, {0.0, 0.0, 23.0}}};
  (void)pot(positions, atmtypes, box);
  REQUIRE(last_key() == 41);
}

TEST_CASE("CPMDPot rejects a copy onto a refused orbital key", "[cpmd][abi]") {
  rgpot::DynLib lib;
  const char *path = std::getenv("RGPOT_CPMD_ENGINE");
  REQUIRE(path != nullptr);
  lib.open(path);
  using OrbitalFn = double (*)(void);
  auto last_initial_orbital =
      lib.sym<OrbitalFn>("cpmdc_fake_last_initial_orbital_state");
  auto last_final_orbital =
      lib.sym<OrbitalFn>("cpmdc_fake_last_final_orbital_state");

  ::capnp::MallocMessageBuilder msg;
  auto p = msg.initRoot<::CPMDParams>();
  p.setFunctional("BLYP");
  p.setCutOffRy(70.0);
  rgpot::CPMDPot pot(p.asReader());
  REQUIRE(pot.available());
  REQUIRE(pot.keepsOrbitalsPerKey());

  rgpot::types::AtomMatrix positions = rgpot::types::AtomMatrix::Zero(1, 3);
  std::vector<int> atmtypes{8};
  std::array<std::array<double, 3>, 3> box = {
      {{20.0, 0.0, 0.0}, {0.0, 21.0, 0.0}, {0.0, 0.0, 23.0}}};

  constexpr long long seed = 11;
  pot.selectOrbitals(seed);
  (void)pot(positions, atmtypes, box);
  const double seed_state = last_final_orbital();

  REQUIRE_FALSE(pot.copyOrbitals(seed, -99));
  REQUIRE_FALSE(pot.copyOrbitals(-99, 12));

  box[2][2] = 24.0;
  (void)pot(positions, atmtypes, box);
  REQUIRE_THAT(last_initial_orbital(), WithinAbs(seed_state, 1e-12));
}

// Naming a key does not store orbitals. Only a converged evaluation does,
// so a later visit still starts from the orbitals already stored.
TEST_CASE("CPMDPot does not store an orbital key before it converges",
          "[cpmd][abi]") {
  rgpot::DynLib lib;
  const char *path = std::getenv("RGPOT_CPMD_ENGINE");
  REQUIRE(path != nullptr);
  lib.open(path);
  using OrbitalFn = double (*)(void);
  auto last_initial_orbital =
      lib.sym<OrbitalFn>("cpmdc_fake_last_initial_orbital_state");
  auto last_final_orbital =
      lib.sym<OrbitalFn>("cpmdc_fake_last_final_orbital_state");

  ::capnp::MallocMessageBuilder msg;
  auto p = msg.initRoot<::CPMDParams>();
  p.setFunctional("BLYP");
  p.setCutOffRy(70.0);
  rgpot::CPMDPot pot(p.asReader());
  REQUIRE(pot.available());

  rgpot::types::AtomMatrix positions = rgpot::types::AtomMatrix::Zero(1, 3);
  std::vector<int> atmtypes{8};
  std::array<std::array<double, 3>, 3> box = {
      {{20.0, 0.0, 0.0}, {0.0, 21.0, 0.0}, {0.0, 0.0, 23.0}}};

  constexpr long long held = 1;
  constexpr long long fresh = 2;
  pot.selectOrbitals(held);
  (void)pot(positions, atmtypes, box);
  const double first = last_final_orbital();

  pot.selectOrbitals(held);
  pot.selectOrbitals(fresh);
  pot.selectOrbitals(held);
  box[2][2] = 24.0;
  (void)pot(positions, atmtypes, box);
  const double second = last_final_orbital();
  REQUIRE(second != first);

  pot.selectOrbitals(fresh);
  box[2][2] = 23.0;
  (void)pot(positions, atmtypes, box);
  REQUIRE_THAT(last_initial_orbital(), WithinAbs(second, 1e-12));
}
