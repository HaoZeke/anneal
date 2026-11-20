#pragma once
// MIT License
// Copyright 2023--present rgpot developers

/**
 * @brief PotentialConfig.d2 and d2cpmd for the D2 backends and composite
 * terms.
 *
 * D2Pot and D2CpmdPot fix their parameters at construction, so applying an
 * arm builds a replacement pot.
 */

#include <memory>
#include <sstream>
#include <stdexcept>
#include <string>

#include "rgpot/D2Pot/D2CpmdPot.hpp"
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

/// The D2CpmdConfig @p params names, s6 from CPMD's set_s6 when functional
/// is set. Throws std::invalid_argument for an unknown functional.
[[nodiscard]] inline D2CpmdConfig d2CpmdConfigFrom(const ::D2CpmdParams::Reader &params) {
  D2CpmdConfig c;
  if (params.hasFunctional() && params.getFunctional().size() > 0) {
    c = d2_cpmd_config_for(params.getFunctional().cStr());
  } else {
    c.s6 = params.getS6();
  }
  c.alpha = params.getAlpha();
  c.vdweps = params.getVdweps();
  c.symmetry = params.getSymmetry();
  return c;
}

template <typename Pot> struct D2ConfigureResult {
  bool ok = false;
  /// Set when the arm was accepted; the caller swaps it in.
  std::unique_ptr<Pot> replacement;
  std::string message;
};

/// none is a no-op, d2 rebuilds the pot, every other arm is refused.
[[nodiscard]] inline D2ConfigureResult<D2Pot> configureD2(const ::PotentialConfig::Reader &cfg) {
  D2ConfigureResult<D2Pot> result;
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

/// none is a no-op, d2cpmd rebuilds the pot, every other arm is refused.
[[nodiscard]] inline D2ConfigureResult<D2CpmdPot>
configureD2Cpmd(const ::PotentialConfig::Reader &cfg) {
  D2ConfigureResult<D2CpmdPot> result;
  switch (cfg.which()) {
  case ::PotentialConfig::NONE:
    result.ok = true;
    result.message = "no-op (rgpot params: none)";
    return result;
  case ::PotentialConfig::D2CPMD: {
    try {
      const D2CpmdConfig next = d2CpmdConfigFrom(cfg.getD2cpmd());
      result.replacement = std::make_unique<D2CpmdPot>(next);
      std::ostringstream summary;
      summary << "s6=" << next.s6 << " alpha=" << next.alpha << " vdweps=" << next.vdweps
              << " symmetry=" << next.symmetry;
      result.ok = true;
      result.message = "rgpot params applied (d2cpmd arm): " + summary.str();
    } catch (const std::invalid_argument &error) {
      result.message = std::string("rgpot params d2cpmd arm rejected: ") + error.what();
    }
    return result;
  }
  default:
    result.message = "rgpot params arm " + potentialConfigArm(cfg) +
                     " not handled by D2CpmdPot (use matching backend pot)";
    return result;
  }
}

} // namespace rgpot::rpc
