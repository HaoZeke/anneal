#pragma once
// MIT License
// Copyright 2023--present rgpot developers

#include <memory>
#include <stdexcept>
#include <string>

#include "rgpot/Models/ModelPots.hpp"
#include "rgpot/rpc/PotentialConfigArm.hpp"
#include "rgpot/rpc/Potentials.capnp.h"

namespace rgpot::rpc {

/// Every field is on the wire, so the result never depends on the pot being
/// replaced; the rule for empty lists is in mullerBrownConfigFromLists.
[[nodiscard]] inline MullerBrownConfig
mullerBrownConfigFrom(const ::MullerBrownParams::Reader &params) {
  return mullerBrownConfigFromLists(params.getAmplitude(), params.getA(),
                                    params.getB(), params.getC(),
                                    params.getX0(), params.getY0());
}

struct MullerBrownConfigureResult {
  bool ok = false;
  std::unique_ptr<MullerBrownPot> replacement;
  std::string message;
};

[[nodiscard]] inline MullerBrownConfigureResult
configureMullerBrown(const ::PotentialConfig::Reader &cfg) {
  MullerBrownConfigureResult result;
  if (cfg.which() == ::PotentialConfig::NONE) {
    result.ok = true;
    result.message = "no-op (rgpot params: none)";
    return result;
  }
  if (cfg.which() != ::PotentialConfig::MULLER_BROWN) {
    result.message =
        "rgpot params arm " + potentialConfigArm(cfg) +
        " not handled by MullerBrownPot (use matching backend pot)";
    return result;
  }
  try {
    result.replacement = std::make_unique<MullerBrownPot>(
        mullerBrownConfigFrom(cfg.getMullerBrown()));
    result.ok = true;
    result.message = "rgpot params applied (mullerBrown arm)";
  } catch (const std::invalid_argument &error) {
    result.message =
        std::string("rgpot params mullerBrown arm rejected: ") + error.what();
  }
  return result;
}

} // namespace rgpot::rpc
