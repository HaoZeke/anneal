// MIT License
// Copyright 2023--present rgpot developers

#include <catch2/catch_test_macros.hpp>
#include <catch2/matchers/catch_matchers_floating_point.hpp>

#include <capnp/message.h>

#include <array>
#include <cmath>
#include <cstdlib>
#include <string>
#include <vector>

#include "rgpot/NWChemPot/DynLib.hpp"
#include "rgpot/NWChemPot/NWChemPot.hpp"
#include "rgpot/rpc/Potentials.capnp.h"
#include "rgpot/types/AtomMatrix.hpp"
#include "rgpot/units.hpp"

using Catch::Matchers::WithinAbs;

namespace {

class RestoreEnv {
public:
  explicit RestoreEnv(const char *name) : name_(name) {
    if (const char *value = std::getenv(name)) {
      had_ = true;
      value_ = value;
    }
  }

  void clear() const {
#ifdef _WIN32
    _putenv_s(name_, "");
#else
    unsetenv(name_);
#endif
  }

  ~RestoreEnv() {
#ifdef _WIN32
    _putenv_s(name_, had_ ? value_.c_str() : "");
#else
    if (had_)
      setenv(name_, value_.c_str(), 1);
    else
      unsetenv(name_);
#endif
  }

  RestoreEnv(const RestoreEnv &) = delete;
  RestoreEnv &operator=(const RestoreEnv &) = delete;

private:
  const char *name_;
  bool had_ = false;
  std::string value_;
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
  RestoreEnv nwchem_library("NWCHEMC_LIBRARY");
  RestoreEnv nwchemc_engine("RGPOT_NWCHEMC_ENGINE");
  RestoreEnv nwchem_engine("RGPOT_NWCHEM_ENGINE");
  nwchem_library.clear();
  nwchemc_engine.clear();
  nwchem_engine.clear();

  rgpot::NWChemPot pot;
  REQUIRE_FALSE(pot.available());

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
}
