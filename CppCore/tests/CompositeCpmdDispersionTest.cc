// MIT License
// Copyright 2023--present rgpot developers
//
// A DFT engine plus a dispersion term, built from PotentialConfig.composite
// as a host would send it: a cpmd term (the gradient test engine stands in
// for cpmdc) beside D4 BLYP when dftd4 is built, and beside D2 otherwise,
// with CPMD's own D2 subtracted. The sum, its stress and its per-term record
// must match the terms evaluated alone.

#include <array>
#include <cmath>
#include <memory>
#include <string>
#include <vector>

#include <capnp/message.h>
#include <catch2/catch_all.hpp>

#include "cpmd_stress_oracle.hpp"
#include "dftd_tolerances.hpp"
#include "rgpot/CPMDPot/CPMDPot.hpp"
#include "rgpot/D2Pot/D2CpmdPot.hpp"
#include "rgpot/D2Pot/D2Pot.hpp"
#include "rgpot/rpc/CompositeConfigure.hpp"
#ifdef RGPOT_HAS_DFTD4
#include "rgpot/D4Pot/D4Pot.hpp"
#endif

namespace {

const std::vector<double> kPos{0.0, 0.0, 0.1178, 0.0, 0.75545, -0.47116, 0.0, -0.75545, -0.47116,
                               3.0, 0.2, 0.1};
const std::vector<int> kZ{8, 1, 1, 8};
const std::array<double, 9> kBox{12.0, 0.0, 0.0, 0.0, 12.5, 0.0, 0.0, 0.0, 13.0};

struct Result {
  double energy = 0.0;
  std::vector<double> forces;
  std::array<double, 9> stress{};
  bool has_stress = false;
};

Result run(rgpot::PotentialBase &pot) {
  Result r;
  r.forces.assign(kPos.size(), 0.0);
  const rgpot::ForceInput in{kZ.size(), kPos.data(), kZ.data(), kBox.data()};
  rgpot::ForceOut out{};
  out.F = r.forces.data();
  pot.forceBatch(rgpot::ForceBatch{.nSystems = 1, .in = &in, .out = &out});
  r.energy = out.energy;
  r.has_stress = out.has_stress != 0;
  for (int k = 0; k < 9; ++k) {
    r.stress[k] = out.stress[k];
  }
  return r;
}

void setCpmd(::CPMDParams::Builder p) {
  p.setFunctional("BLYP");
  p.setCutOffRy(70.0);
}

} // namespace

TEST_CASE("a cpmd term and a dispersion term sum through the config path",
          "[composite][cpmd][dispersion]") {
  capnp::MallocMessageBuilder builder;
  auto cfg = builder.initRoot<::PotentialConfig>();
  auto terms = cfg.initComposite().initTerms(3);
  terms[0].setName("cpmd");
  setCpmd(terms[0].initPot().initCpmd());
  terms[1].setName("undo cpmd d2");
  terms[1].setWeight(-1.0);
  terms[1].initPot().initBuiltin().setName("d2cpmd");
  terms[2].setName("dispersion");
  auto disp = terms[2].initPot().initBuiltin();
#ifdef RGPOT_HAS_DFTD4
  disp.setName("d4");
  auto d4 = disp.initConfig().initD4();
  d4.setFunctional("blyp");
  d4.setPeriodic(true);
  rgpot::D4Pot alone(rgpot::D4Config{.functional = "blyp", .periodic = true});
#else
  disp.setName("d2");
  rgpot::D2Pot alone;
#endif
  auto applied = rgpot::rpc::configureComposite(cfg.asReader());
  INFO(applied.message);
  REQUIRE(applied.ok);
  rgpot::SumPot &sum = *applied.replacement;

  ::capnp::MallocMessageBuilder cpmd_msg;
  auto cpmd_params = cpmd_msg.initRoot<::CPMDParams>();
  setCpmd(cpmd_params);
  rgpot::CPMDPot cpmd(cpmd_params.asReader());
  REQUIRE(cpmd.available());
  rgpot::D2CpmdPot cpmd_d2;
  const Result engine = run(cpmd);
  const Result undo = run(cpmd_d2);
  const Result dispersion = run(alone);
  REQUIRE(engine.energy == cpmd_stress_oracle::kGradientHartree * rgpot::units::HARTREE_TO_EV);
  // CPMDPot writes the stress cpmdc reports without declaring it in its
  // caps, so the sum carries a stress because every term wrote one.
  REQUIRE(engine.has_stress);
  REQUIRE(undo.has_stress);
  REQUIRE(dispersion.has_stress);

  const Result total = run(sum);
  using rgpot::testtol::kEnergyThr;
  using rgpot::testtol::kGradThr;
  using rgpot::units::BOHR_TO_ANGSTROM;
  using rgpot::units::HARTREE_PER_BOHR3_TO_EV_PER_ANGSTROM3;
  using rgpot::units::HARTREE_TO_EV;
  const double ethr = kEnergyThr * HARTREE_TO_EV;
  REQUIRE(std::abs(total.energy - (engine.energy - undo.energy + dispersion.energy)) <= ethr);
  for (std::size_t k = 0; k < kPos.size(); ++k) {
    const double expected = engine.forces[k] - undo.forces[k] + dispersion.forces[k];
    REQUIRE(std::abs(total.forces[k] - expected) <= kGradThr * HARTREE_TO_EV / BOHR_TO_ANGSTROM);
  }
  REQUIRE(total.has_stress);
  for (int k = 0; k < 9; ++k) {
    const double expected = engine.stress[k] - undo.stress[k] + dispersion.stress[k];
    REQUIRE(std::abs(total.stress[k] - expected) <=
            kGradThr * HARTREE_PER_BOHR3_TO_EV_PER_ANGSTROM3);
  }

  const auto &rec = sum.lastComponents();
  REQUIRE(rec.size() == 3);
  REQUIRE(rec[0].name == "cpmd");
  REQUIRE(rec[0].energy == engine.energy);
  REQUIRE(rec[1].weight == -1.0);
  REQUIRE(std::abs(rec[1].energy - undo.energy) <= ethr);
  REQUIRE(std::abs(rec[2].energy - dispersion.energy) <= ethr);
  REQUIRE(rec[2].forces.size() == kPos.size());
}
