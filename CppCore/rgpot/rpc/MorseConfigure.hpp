#pragma once
// MIT License
// Copyright 2023--present rgpot developers

/**
 * @brief PotentialConfig.morse for a MorsePot backend.
 *
 * MorsePot fixes its parameters at construction, so applying the morse arm
 * builds a replacement pot. switch_width is not on the wire and carries
 * over from the pot being replaced.
 */

#include <cmath>
#include <memory>
#include <sstream>
#include <stdexcept>
#include <string>

#include "rgpot/Morse/MorsePot.hpp"
#include "rgpot/rpc/PotentialConfigArm.hpp"
#include "rgpot/rpc/Potentials.capnp.h"

namespace rgpot::rpc {

[[nodiscard]] inline MorseConfig morseConfigFrom(const ::MorseParams::Reader &params,
                                                 MorseConfig base = {}) {
  base.De = params.getDe();
  base.a = params.getA();
  base.re = params.getRe();
  base.cutoff = params.getCutoff();
  return base;
}

struct MorseConfigureResult {
  bool ok = false;
  /// Set when the morse arm was accepted; the caller swaps it in.
  std::unique_ptr<MorsePot> replacement;
  std::string message;
};

/// none is a no-op, morse rebuilds the pot, every other arm is refused.
[[nodiscard]] inline MorseConfigureResult
configureMorse(const ::PotentialConfig::Reader &cfg, const MorsePot &current) {
  MorseConfigureResult result;
  switch (cfg.which()) {
  case ::PotentialConfig::NONE:
    result.ok = true;
    result.message = "no-op (rgpot params: none)";
    return result;
  case ::PotentialConfig::MORSE: {
    const MorseConfig next = morseConfigFrom(cfg.getMorse(), current.config());
    if (!std::isfinite(next.De) || !std::isfinite(next.a) || !std::isfinite(next.re) ||
        !std::isfinite(next.cutoff) || !(next.cutoff > 0.0)) {
      result.message = "rgpot params morse arm rejected: de, a, re and cutoff must be "
                       "finite and cutoff positive";
      return result;
    }
    std::ostringstream summary;
    summary << "de=" << next.De << " a=" << next.a << " re=" << next.re
            << " cutoff=" << next.cutoff;
    try {
      result.replacement = std::make_unique<MorsePot>(next);
    } catch (const std::invalid_argument &error) {
      result.message = "rgpot params morse arm rejected (" + summary.str() + "): " + error.what();
      return result;
    }
    result.ok = true;
    result.message = "rgpot params applied (morse arm): " + summary.str();
    return result;
  }
  default:
    result.message = "rgpot params arm " + potentialConfigArm(cfg) +
                     " not handled by MorsePot (use matching backend pot)";
    return result;
  }
}

} // namespace rgpot::rpc
