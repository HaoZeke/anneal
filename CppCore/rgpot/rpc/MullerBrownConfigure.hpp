#pragma once
// MIT License
// Copyright 2023--present rgpot developers

#include <algorithm>
#include <memory>
#include <stdexcept>
#include <string>

#include "rgpot/Models/ModelPots.hpp"
#include "rgpot/rpc/PotentialConfigArm.hpp"
#include "rgpot/rpc/Potentials.capnp.h"

namespace rgpot::rpc {

inline void readMullerBrownParameter(const capnp::List<double>::Reader &source,
                                     std::array<double, 4> &target,
                                     const char *name) {
  if (source.size() == 0)
    return;
  if (source.size() != target.size())
    throw std::invalid_argument(std::string("Muller-Brown ") + name +
                                " must contain four values");
  std::copy(source.begin(), source.end(), target.begin());
}

/// Every field is on the wire, so the result never depends on the pot being
/// replaced: an empty list is the published default, four values override it.
[[nodiscard]] inline MullerBrownConfig
mullerBrownConfigFrom(const ::MullerBrownParams::Reader &params) {
  MullerBrownConfig config;
  readMullerBrownParameter(params.getAmplitude(), config.amplitude,
                           "amplitude");
  readMullerBrownParameter(params.getA(), config.a, "a");
  readMullerBrownParameter(params.getB(), config.b, "b");
  readMullerBrownParameter(params.getC(), config.c, "c");
  readMullerBrownParameter(params.getX0(), config.x0, "x0");
  readMullerBrownParameter(params.getY0(), config.y0, "y0");
  return config;
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
    result.message = "rgpot params arm " + potentialConfigArm(cfg) +
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
