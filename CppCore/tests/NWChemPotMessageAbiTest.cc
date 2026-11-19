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
