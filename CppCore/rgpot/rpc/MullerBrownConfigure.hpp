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

[[nodiscard]] inline MullerBrownConfig
mullerBrownConfigFrom(const ::MullerBrownParams::Reader &params,
                      MullerBrownConfig base = {}) {
  readMullerBrownParameter(params.getAmplitude(), base.amplitude, "amplitude");
  readMullerBrownParameter(params.getA(), base.a, "a");
  readMullerBrownParameter(params.getB(), base.b, "b");
  readMullerBrownParameter(params.getC(), base.c, "c");
  readMullerBrownParameter(params.getX0(), base.x0, "x0");
  readMullerBrownParameter(params.getY0(), base.y0, "y0");
  return base;
}

struct MullerBrownConfigureResult {
  bool ok = false;
  std::unique_ptr<MullerBrownPot> replacement;
  std::string message;
};

[[nodiscard]] inline MullerBrownConfigureResult configureMullerBrown(
    const ::PotentialConfig::Reader &cfg, const MullerBrownPot &current) {
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
        mullerBrownConfigFrom(cfg.getMullerBrown(), current.config()));
    result.ok = true;
    result.message = "rgpot params applied (mullerBrown arm)";
  } catch (const std::invalid_argument &error) {
    result.message =
        std::string("rgpot params mullerBrown arm rejected: ") + error.what();
  }
  return result;
}

} // namespace rgpot::rpc
