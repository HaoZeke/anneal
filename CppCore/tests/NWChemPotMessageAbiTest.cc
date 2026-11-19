// MIT License
// Copyright 2023--present rgpot developers

#include <catch2/catch_test_macros.hpp>
#include <catch2/matchers/catch_matchers_floating_point.hpp>
#include <catch2/matchers/catch_matchers_string.hpp>

#include <capnp/message.h>

#include <array>
#include <cmath>
#include <cstdlib>
#include <filesystem>
#include <string>
#include <vector>

#include "RestoreEnv.hpp"
#include "rgpot/NWChemPot/DynLib.hpp"
#include "rgpot/NWChemPot/NWChemPot.hpp"
#include "rgpot/rpc/Potentials.capnp.h"
#include "rgpot/types/AtomMatrix.hpp"
#include "rgpot/units.hpp"

using Catch::Matchers::ContainsSubstring;
using Catch::Matchers::WithinAbs;

namespace {

using rgpot::test_support::RestoreEnv;

class ScopedTempCurrentDir {
public:
  ScopedTempCurrentDir()
      : original_(std::filesystem::current_path()),
        temp_(std::filesystem::temp_directory_path() /
              ("rgpot-nwchem-missing-" + std::to_string(std::rand()))) {
    std::filesystem::create_directories(temp_);
    std::filesystem::current_path(temp_);
  }

  const std::filesystem::path &path() const { return temp_; }

  ~ScopedTempCurrentDir() {
    std::error_code ec;
    std::filesystem::current_path(original_, ec);
    std::filesystem::remove_all(temp_, ec);
  }

  ScopedTempCurrentDir(const ScopedTempCurrentDir &) = delete;
  ScopedTempCurrentDir &operator=(const ScopedTempCurrentDir &) = delete;

private:
  std::filesystem::path original_;
  std::filesystem::path temp_;
};

struct RejectParamsGuard {
  using RejectParamsFn = void (*)(int);
  RejectParamsFn fn = nullptr;

  explicit RejectParamsGuard(RejectParamsFn reject_fn) : fn(reject_fn) {
    fn(1);
  }

  ~RejectParamsGuard() {
    if (fn)
      fn(0);
  }

  RejectParamsGuard(const RejectParamsGuard &) = delete;
  RejectParamsGuard &operator=(const RejectParamsGuard &) = delete;
};

std::string configured_engine_path() {
  const char *names[] = {"NWCHEMC_LIBRARY", "RGPOT_NWCHEMC_ENGINE",
                         "RGPOT_NWCHEM_ENGINE"};
  for (const char *name : names) {
    if (const char *value = std::getenv(name); value && *value) {
      return value;
    }
  }
  return {};
}

} // namespace

TEST_CASE("NWChemPot passes serialized NWChemParams to nwchemc engine",
          "[nwchem][abi]") {
  if (!rgpot::NWChemPot::probeAvailable()) {
    SKIP("libnwchemc not available");
  }

  ::capnp::MallocMessageBuilder msg;
  auto p = msg.initRoot<::NWChemParams>();
  p.setBasis("6-31g");
  p.setTheory("dft");
  p.setScfType("blyp");
  p.setCharge(-1);
  p.setMultiplicity(2);

  rgpot::NWChemPot pot(p.asReader());
  REQUIRE(pot.available());

  rgpot::types::AtomMatrix positions(1, 3);
  positions(0, 0) = 0.0;
  positions(0, 1) = 0.0;
  positions(0, 2) = 0.0;
  std::vector<int> atmtypes{8};
  std::array<std::array<double, 3>, 3> box = {
      {{20.0, 0.0, 0.0}, {0.0, 20.0, 0.0}, {0.0, 0.0, 20.0}}};

  auto [energy, forces, variance] = pot(positions, atmtypes, box);
  (void)variance;

  REQUIRE(std::isfinite(energy));
  REQUIRE(energy != 0.0);
  for (int j = 0; j < 3; ++j)
    REQUIRE(std::isfinite(forces(0, j)));

  const double fake_e = 0.25 * rgpot::units::HARTREE_TO_EV;
  // In-tree fake returns fixed Ha/grad; real NWChem does not.
  if (std::abs(energy - fake_e) < 1e-9) {
    REQUIRE_THAT(energy, WithinAbs(fake_e, 1e-12));
    REQUIRE_THAT(forces(0, 0),
                 WithinAbs(0.001 * rgpot::units::NEG_GRAD_TO_FORCE, 1e-12));
    REQUIRE_THAT(forces(0, 1),
                 WithinAbs(0.002 * rgpot::units::NEG_GRAD_TO_FORCE, 1e-12));
    REQUIRE_THAT(forces(0, 2),
                 WithinAbs(0.003 * rgpot::units::NEG_GRAD_TO_FORCE, 1e-12));
  } else {
    // Real embed: O- / BLYP / 6-31G is bound (negative eV scale).
    REQUIRE(energy < 0.0);
  }
}

TEST_CASE("NWChemPot preserves typed DFT convergence parameters",
          "[nwchem][abi]") {
  if (!rgpot::NWChemPot::probeAvailable()) {
    SKIP("libnwchemc not available");
  }

  ::capnp::MallocMessageBuilder input_message;
  auto input = input_message.initRoot<::NWChemParams>();
  input.setTheory("dft");
  auto stanzas = input.initInputStanzas(1);
  stanzas[0].setKind(::NWChemInputStanza::Kind::DFT);
  auto dft = stanzas[0].initDft();
  dft.setXc("hyb_gga_xc_wb97x_v");
  dft.setEnergyConv(1.0e-6);

  rgpot::NWChemPot pot(input.asReader());
  REQUIRE(pot.available());

  ::capnp::MallocMessageBuilder output_message;
  auto output = output_message.initRoot<::NWChemParams>();
  pot.getParams(output);
  const auto restored_stanzas = output.asReader().getInputStanzas();

  REQUIRE(restored_stanzas.size() == 1);
  REQUIRE(restored_stanzas[0].getKind() ==
          ::NWChemInputStanza::Kind::DFT);
  REQUIRE(restored_stanzas[0].getDft().getXc() ==
          "hyb_gga_xc_wb97x_v");
  REQUIRE_THAT(restored_stanzas[0].getDft().getEnergyConv(),
               WithinAbs(1.0e-6, 1.0e-15));
}

TEST_CASE("NWChemPot reports a loaded engine configuration refusal",
          "[nwchem][abi]") {
  const char *path = std::getenv("RGPOT_NWCHEM_ENGINE");
  REQUIRE(path != nullptr);

  rgpot::NWChemPot pot;
  REQUIRE(pot.available());

  rgpot::DynLib engine;
  engine.open(path);
  auto reject_params = engine.sym<RejectParamsGuard::RejectParamsFn>(
      "nwchemc_fake_reject_params");

  ::capnp::MallocMessageBuilder message;
  auto config = message.initRoot<::PotentialConfig>();
  auto params = config.initNwchem();
  params.setBasis("6-31g");
  params.setTheory("dft");
  params.setScfType("rhf");
  params.setCharge(0);
  params.setMultiplicity(1);

  std::string result;
  {
    RejectParamsGuard reject_guard(reject_params);
    REQUIRE_FALSE(pot.setPotentialConfig(config.asReader(), &result));
  }

  REQUIRE(pot.available());
  REQUIRE(result == "rgpot params nwchem arm rejected by embed: "
                    "basis=6-31g theory=dft scfType=rhf charge=0 mult=1");
}

TEST_CASE("NWChemPot reports a missing engine in setPotentialConfig",
          "[nwchem][abi]") {
  const char *raw_live = std::getenv("RGPOT_NWCHEM_ENGINE");
  REQUIRE(raw_live != nullptr);
  const std::filesystem::path live_engine = std::filesystem::absolute(raw_live);

  RestoreEnv nwchem_library("NWCHEMC_LIBRARY");
  RestoreEnv nwchemc_engine("RGPOT_NWCHEMC_ENGINE");
  RestoreEnv nwchem_engine("RGPOT_NWCHEM_ENGINE");
  nwchem_library.clear();
  nwchemc_engine.clear();
  nwchem_engine.set("/nonexistent/env-libnwchemc.so");

  ScopedTempCurrentDir workdir;
#ifndef _WIN32
  std::filesystem::create_symlink(live_engine,
                                  workdir.path() / "libnwchemc.so");
  std::filesystem::create_symlink(live_engine,
                                  workdir.path() / "libnwchemc.dylib");
#endif

  rgpot::NWChemPot pot;
  rgpot::types::AtomMatrix positions = rgpot::types::AtomMatrix::Zero(1, 3);
  std::vector<int> atmtypes{8};
  std::array<std::array<double, 3>, 3> box = {
      {{20.0, 0.0, 0.0}, {0.0, 20.0, 0.0}, {0.0, 0.0, 20.0}}};
  REQUIRE_THROWS_WITH(
      pot(positions, atmtypes, box),
      ContainsSubstring("NWChem engine (libnwchemc) not loaded") &&
          ContainsSubstring("/nonexistent/env-libnwchemc.so"));

  nwchem_engine.set(live_engine.string().c_str());
  REQUIRE(rgpot::NWChemPot().available());

  ::capnp::MallocMessageBuilder message;
  auto config = message.initRoot<::PotentialConfig>();
  auto params = config.initNwchem();
  params.setBasis("6-31g");
  params.setTheory("dft");
  params.setScfType("rhf");
  params.setCharge(0);
  params.setMultiplicity(1);
  params.setEnginePath("/nonexistent/libnwchemc.so");

  std::string result;
  REQUIRE_FALSE(pot.setPotentialConfig(config.asReader(), &result));
  REQUIRE_FALSE(pot.available());
  REQUIRE(result == "rgpot params nwchem arm failed (engine not loaded): "
                    "basis=6-31g theory=dft scfType=rhf charge=0 mult=1 "
                    "enginePath=/nonexistent/libnwchemc.so");

  ::capnp::MallocMessageBuilder stored_message;
  auto stored = stored_message.initRoot<::NWChemParams>();
  pot.getParams(stored);
  REQUIRE(stored.asReader().getBasis() == "6-31g");
  REQUIRE(stored.asReader().getTheory() == "dft");
  REQUIRE(stored.asReader().getEnginePath() == "/nonexistent/libnwchemc.so");
}

TEST_CASE("NWChemPot rejects a nonexistent explicit engine path after a load",
          "[nwchem][abi]") {
  const char *raw_live = std::getenv("RGPOT_NWCHEM_ENGINE");
  REQUIRE(raw_live != nullptr);
  const std::string live_engine = std::filesystem::absolute(raw_live).string();
  const std::string missing_engine = "/nonexistent/other-libnwchemc.so";

  RestoreEnv nwchem_library("NWCHEMC_LIBRARY");
  RestoreEnv nwchemc_engine("RGPOT_NWCHEMC_ENGINE");
  RestoreEnv nwchem_engine("RGPOT_NWCHEM_ENGINE");
  nwchem_library.clear();
  nwchemc_engine.clear();
  nwchem_engine.set(live_engine.c_str());

  rgpot::NWChemPot pot;
  REQUIRE(pot.available());

  ::capnp::MallocMessageBuilder valid_message;
  auto valid_config = valid_message.initRoot<::PotentialConfig>();
  auto valid = valid_config.initNwchem();
  valid.setBasis("6-31g");
  valid.setTheory("dft");
  valid.setScfType("rhf");
  valid.setCharge(0);
  valid.setMultiplicity(1);
  valid.setEnginePath(live_engine.c_str());

  std::string result;
  REQUIRE(pot.setPotentialConfig(valid_config.asReader(), &result));
  REQUIRE(pot.available());

  ::capnp::MallocMessageBuilder missing_message;
  auto missing_config = missing_message.initRoot<::PotentialConfig>();
  auto missing = missing_config.initNwchem();
  missing.setBasis("6-31g");
  missing.setTheory("dft");
  missing.setScfType("rhf");
  missing.setCharge(0);
  missing.setMultiplicity(1);
  missing.setEnginePath(missing_engine.c_str());

  REQUIRE_FALSE(pot.setPotentialConfig(missing_config.asReader(), &result));
  REQUIRE_FALSE(pot.available());
  REQUIRE_THAT(result, ContainsSubstring("engine not loaded") &&
                           ContainsSubstring(missing_engine));

  rgpot::types::AtomMatrix positions = rgpot::types::AtomMatrix::Zero(1, 3);
  std::vector<int> atmtypes{8};
  std::array<std::array<double, 3>, 3> box = {
      {{20.0, 0.0, 0.0}, {0.0, 20.0, 0.0}, {0.0, 0.0, 20.0}}};
  REQUIRE_THROWS_WITH(pot(positions, atmtypes, box),
                      ContainsSubstring("NWChem engine (libnwchemc) not "
                                        "loaded") &&
                          ContainsSubstring(missing_engine));

  REQUIRE(pot.setPotentialConfig(valid_config.asReader(), &result));
  REQUIRE(pot.available());
}

// NWChemPot has no session handle: available() checks only that libnwchemc
// loaded with nwchemc_energy_gradient and nwchemc_set_params, and setParams
// keeps both when nwchemc_set_params returns nonzero.
TEST_CASE("NWChemPot setPotentialConfig distinguishes missing engine from "
          "refused params",
          "[nwchem][abi]") {
  const std::string engine_path = configured_engine_path();
  if (engine_path.empty()) {
    SKIP("explicit nwchemc engine path not set in environment");
  }

  rgpot::DynLib lib(engine_path);
  using VersionFn = const char *(*)(void);
  using CountFn = int (*)(void);
  auto version = lib.sym_optional<VersionFn>("nwchemc_version");
  auto reject_params = lib.sym_optional<RejectParamsGuard::RejectParamsFn>(
      "nwchemc_fake_reject_params");
  auto set_params_count =
      lib.sym_optional<CountFn>("nwchemc_fake_set_params_count");
  const bool is_fake =
      version && std::string(version()).rfind("nwchemc-fake/", 0) == 0;
  if (is_fake) {
    REQUIRE(reject_params != nullptr);
    REQUIRE(set_params_count != nullptr);
  } else if (!reject_params || !set_params_count) {
    SKIP("real libnwchemc does not expose fake refusal controls");
  }
  reject_params(0);

  RestoreEnv nwchem_library("NWCHEMC_LIBRARY");
  RestoreEnv nwchemc_engine("RGPOT_NWCHEMC_ENGINE");
  RestoreEnv nwchem_engine("RGPOT_NWCHEM_ENGINE");
  nwchem_library.clear();
  nwchemc_engine.clear();
  nwchem_engine.clear();

  rgpot::NWChemPot pot;
  if (pot.available()) {
    SKIP("default libnwchemc on loader search path masks missing engine path");
  }

  ::capnp::MallocMessageBuilder cfg_msg;
  auto cfg = cfg_msg.initRoot<::PotentialConfig>();
  auto nw = cfg.initNwchem();
  nw.setBasis("6-31g");
  nw.setTheory("dft");
  nw.setScfType("blyp");
  nw.setCharge(-1);
  nw.setMultiplicity(2);
  nw.setEnginePath("/nonexistent/libnwchemc.so");

  const int before = set_params_count();
  std::string message;
  REQUIRE_FALSE(pot.setPotentialConfig(cfg.asReader(), &message));
  REQUIRE_FALSE(pot.available());
  REQUIRE(set_params_count() == before);
  REQUIRE(message == "rgpot params nwchem arm failed (engine not loaded): "
                     "basis=6-31g theory=dft scfType=blyp charge=-1 mult=2 "
                     "enginePath=/nonexistent/libnwchemc.so");

  const std::string summary =
      "basis=6-31g theory=dft scfType=blyp charge=-1 mult=2 enginePath=" +
      engine_path;
  nw.setEnginePath(engine_path);
  {
    RejectParamsGuard reject_guard(reject_params);
    REQUIRE_FALSE(pot.setPotentialConfig(cfg.asReader(), &message));
    REQUIRE(pot.available());
    REQUIRE(set_params_count() == before + 1);
    REQUIRE(message == "rgpot params nwchem arm rejected by embed: " + summary);

    REQUIRE_FALSE(pot.setPotentialConfig(cfg.asReader(), &message));
    REQUIRE(pot.available());
    REQUIRE(set_params_count() == before + 2);
    REQUIRE(message == "rgpot params nwchem arm rejected by embed: " + summary);
  }

  REQUIRE(pot.setPotentialConfig(cfg.asReader(), &message));
  REQUIRE(pot.available());
  REQUIRE(set_params_count() == before + 3);
  REQUIRE(message == "rgpot params applied (nwchem arm): " + summary);

  ::capnp::MallocMessageBuilder none_msg;
  auto none_cfg = none_msg.initRoot<::PotentialConfig>();
  none_cfg.setNone();
  REQUIRE(pot.setPotentialConfig(none_cfg.asReader(), &message));
  REQUIRE(set_params_count() == before + 3);
  REQUIRE(message == "no-op (rgpot params: none)");

  ::capnp::MallocMessageBuilder cpmd_msg;
  auto cpmd_cfg = cpmd_msg.initRoot<::PotentialConfig>();
  cpmd_cfg.initCpmd();
  REQUIRE_FALSE(pot.setPotentialConfig(cpmd_cfg.asReader(), &message));
  REQUIRE(set_params_count() == before + 3);
  REQUIRE(message == "rgpot params arm cpmd not handled by NWChemPot "
                     "(use matching backend pot)");
}
