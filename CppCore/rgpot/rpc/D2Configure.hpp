#pragma once
// MIT License
// Copyright 2023--present rgpot developers

/**
 * @brief PotentialConfig.d2 for a D2Pot backend and a composite term.
 *
 * D2Pot fixes its parameters at construction, so applying the d2 arm builds
 * a replacement pot.
 */

#include <memory>
#include <sstream>
#include <stdexcept>
#include <string>

#include "rgpot/D2Pot/D2Pot.hpp"
#include "rgpot/rpc/PotentialConfigArm.hpp"
#include "rgpot/rpc/Potentials.capnp.h"

namespace rgpot::rpc {

/// The D2Config @p params names: s6, d and sr from the fields, or from
/// dftd3's table when functional is set. Throws std::invalid_argument for an
/// unknown functional.
[[nodiscard]] inline D2Config d2ConfigFrom(const ::D2Params::Reader &params) {
  D2Config c;
  if (params.hasFunctional() && params.getFunctional().size() > 0) {
    c = d2_config_for(params.getFunctional().cStr());
  } else {
    c.s6 = params.getS6();
    c.d = params.getD();
  }
  c.sr = params.getSr();
  c.cutoff = params.getCutoff();
  return c;
}

struct D2ConfigureResult {
  bool ok = false;
  /// Set when the d2 arm was accepted; the caller swaps it in.
  std::unique_ptr<D2Pot> replacement;
  std::string message;
};

/// none is a no-op, d2 rebuilds the pot, every other arm is refused.
[[nodiscard]] inline D2ConfigureResult configureD2(const ::PotentialConfig::Reader &cfg) {
  D2ConfigureResult result;
  switch (cfg.which()) {
  case ::PotentialConfig::NONE:
    result.ok = true;
    result.message = "no-op (rgpot params: none)";
    return result;
  case ::PotentialConfig::D2: {
    try {
      const D2Config next = d2ConfigFrom(cfg.getD2());
      result.replacement = std::make_unique<D2Pot>(next);
      std::ostringstream summary;
      summary << "s6=" << next.s6 << " d=" << next.d << " sr=" << next.sr
              << " cutoff=" << next.cutoff;
      result.ok = true;
      result.message = "rgpot params applied (d2 arm): " + summary.str();
    } catch (const std::invalid_argument &error) {
      result.message = std::string("rgpot params d2 arm rejected: ") + error.what();
    }
    return result;
  }
  default:
    result.message = "rgpot params arm " + potentialConfigArm(cfg) +
                     " not handled by D2Pot (use matching backend pot)";
    return result;
  }
}

} // namespace rgpot::rpc
