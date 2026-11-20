#pragma once
// MIT License
// Copyright 2023--present rgpot developers

/**
 * @brief PotentialConfig.d4 for the D4 backend and composite terms.
 *
 * D4Pot loads its damping parameters at construction, so applying the arm
 * builds a replacement pot. Include only when RGPOT_HAS_DFTD4 is defined.
 */

#include <memory>
#include <sstream>
#include <stdexcept>
#include <string>

#include "rgpot/D4Pot/D4Pot.hpp"
#include "rgpot/rpc/PotentialConfigArm.hpp"
#include "rgpot/rpc/Potentials.capnp.h"

namespace rgpot::rpc {

/// The D4Config @p params names. The functional is a dftd4 method key, whose
/// rational damping parameters dftd4 loads from its own table.
[[nodiscard]] inline D4Config d4ConfigFrom(const ::D4Params::Reader &params) {
  D4Config c;
  c.functional = params.getFunctional().cStr();
  c.charge = params.getCharge();
  c.atm = params.getAtm();
  c.periodic = params.getPeriodic();
  return c;
}

struct D4ConfigureResult {
  bool ok = false;
  /// Set when the arm was accepted; the caller swaps it in.
  std::unique_ptr<D4Pot> replacement;
  std::string message;
};

/// none is a no-op, d4 rebuilds the pot, every other arm is refused. A
/// functional dftd4 has no parameters for is refused with dftd4's message.
[[nodiscard]] inline D4ConfigureResult configureD4(const ::PotentialConfig::Reader &cfg) {
  D4ConfigureResult result;
  switch (cfg.which()) {
  case ::PotentialConfig::NONE:
    result.ok = true;
    result.message = "no-op (rgpot params: none)";
    return result;
  case ::PotentialConfig::D4: {
    try {
      const D4Config next = d4ConfigFrom(cfg.getD4());
      result.replacement = std::make_unique<D4Pot>(next);
      std::ostringstream summary;
      summary << "functional=" << next.functional << " charge=" << next.charge
              << " atm=" << (next.atm ? "true" : "false")
              << " periodic=" << (next.periodic ? "true" : "false");
      result.ok = true;
      result.message = "rgpot params applied (d4 arm): " + summary.str();
    } catch (const std::exception &error) {
      result.message = std::string("rgpot params d4 arm rejected: ") + error.what();
    }
    return result;
  }
  default:
    result.message = "rgpot params arm " + potentialConfigArm(cfg) +
                     " not handled by D4Pot (use matching backend pot)";
    return result;
  }
}

} // namespace rgpot::rpc
