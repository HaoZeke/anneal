#pragma once
// MIT License
// Copyright 2023--present rgpot developers

/**
 * @brief Potentials built from PotentialSpec, and PotentialConfig.composite.
 *
 * A composite term names its child with a PotentialSpec: the builtin arm
 * for the in-process potentials (its config carries their parameters) and
 * the engine arms (cpmd) for the DFT frontends. A disabled term is recorded
 * in the sum and never constructed, so switching off an engine term does
 * not load the engine.
 */

#include <cctype>
#include <cmath>
#include <memory>
#include <sstream>
#include <stdexcept>
#include <string>

#include "rgpot/CPMDPot/CPMDPot.hpp"
#include "rgpot/LennardJones/LJPot.hpp"
#include "rgpot/Morse/MorsePot.hpp"
#include "rgpot/SumPot/SumPot.hpp"
#include "rgpot/ZBL/ZBLPot.hpp"
#include "rgpot/rpc/D2Configure.hpp"
#ifdef RGPOT_HAS_DFTD4
#include "rgpot/rpc/D4Configure.hpp"
#endif
#include "rgpot/rpc/MorseConfigure.hpp"
#include "rgpot/rpc/PotentialConfigArm.hpp"
#include "rgpot/rpc/Potentials.capnp.h"

namespace rgpot::rpc {

[[nodiscard]] inline std::unique_ptr<SumPot>
compositeFromParams(const ::CompositeParams::Reader &params);

namespace detail {

inline std::string lowered(std::string s) {
  for (char &c : s) {
    c = static_cast<char>(std::tolower(static_cast<unsigned char>(c)));
  }
  return s;
}

/// Refuses a builtin config arm that does not belong to the named backend.
inline void expectArm(const std::string &name, const ::PotentialConfig::Reader &cfg,
                      ::PotentialConfig::Which arm) {
  if (cfg.which() != ::PotentialConfig::NONE && cfg.which() != arm) {
    throw std::invalid_argument("builtin '" + name + "' cannot take the " +
                                potentialConfigArm(cfg) + " config arm");
  }
}

inline std::unique_ptr<PotentialBase> builtinPotential(const ::BuiltinParams::Reader &b) {
  const std::string name = lowered(b.getName().cStr());
  const auto cfg = b.getConfig();
  if (name == "lj") {
    expectArm(name, cfg, ::PotentialConfig::LJ);
    LJConfig c;
    if (cfg.isLj()) {
      const auto p = cfg.getLj();
      c.u0 = p.getU0();
      c.psi = p.getPsi();
      c.cutoff = p.getCutoff();
    }
    return std::make_unique<LJPot>(c);
  }
  if (name == "morse") {
    expectArm(name, cfg, ::PotentialConfig::MORSE);
    return std::make_unique<MorsePot>(cfg.isMorse() ? morseConfigFrom(cfg.getMorse())
                                                    : MorseConfig{});
  }
  if (name == "zbl") {
    expectArm(name, cfg, ::PotentialConfig::ZBL);
    ZBLConfig c;
    if (cfg.isZbl()) {
      c.cut_inner = cfg.getZbl().getCutInner();
      c.cut_global = cfg.getZbl().getCutGlobal();
    }
    return std::make_unique<ZBLPot>(c);
  }
  if (name == "d2") {
    expectArm(name, cfg, ::PotentialConfig::D2);
    return std::make_unique<D2Pot>(cfg.isD2() ? d2ConfigFrom(cfg.getD2()) : D2Config{});
  }
  if (name == "d2cpmd") {
    expectArm(name, cfg, ::PotentialConfig::D2CPMD);
    return std::make_unique<D2CpmdPot>(cfg.isD2cpmd() ? d2CpmdConfigFrom(cfg.getD2cpmd())
                                                      : D2CpmdConfig{});
  }
#ifdef RGPOT_HAS_DFTD4
  if (name == "d4") {
    expectArm(name, cfg, ::PotentialConfig::D4);
    return std::make_unique<D4Pot>(cfg.isD4() ? d4ConfigFrom(cfg.getD4()) : D4Config{});
  }
#endif
  if (name == "composite" || name == "sum") {
    expectArm(name, cfg, ::PotentialConfig::COMPOSITE);
    if (!cfg.isComposite()) {
      return std::make_unique<SumPot>();
    }
    return compositeFromParams(cfg.getComposite());
  }
  throw std::invalid_argument("builtin '" + name +
                              "' is not a potential this build can place in a composite");
}

} // namespace detail

/// Builds the potential @p spec names. Throws std::invalid_argument for a
/// backend this build cannot construct in process.
[[nodiscard]] inline std::unique_ptr<PotentialBase>
potentialFromSpec(const ::PotentialSpec::Reader &spec) {
  switch (spec.which()) {
  case ::PotentialSpec::BUILTIN:
    return detail::builtinPotential(spec.getBuiltin());
  case ::PotentialSpec::CPMD:
    return std::make_unique<CPMDPot>(spec.getCpmd());
  default:
    throw std::invalid_argument("PotentialSpec arm " +
                                std::to_string(static_cast<unsigned>(spec.which())) +
                                " cannot be placed in a composite by this build");
  }
}

/// The SumPot @p params describe, in term order.
[[nodiscard]] inline std::unique_ptr<SumPot>
compositeFromParams(const ::CompositeParams::Reader &params) {
  auto sum = std::make_unique<SumPot>();
  for (const auto term : params.getTerms()) {
    SumTerm spec{.name = term.getName().cStr(),
                 .weight = term.getWeight(),
                 .enabled = term.getEnabled()};
    if (!term.hasPot() && spec.enabled) {
      throw std::invalid_argument("composite term '" + spec.name + "' has no pot");
    }
    std::unique_ptr<PotentialBase> pot;
    if (spec.enabled) {
      try {
        pot = potentialFromSpec(term.getPot());
      } catch (const std::invalid_argument &e) {
        throw std::invalid_argument("composite term '" + spec.name + "': " + e.what());
      }
    }
    sum->add(std::move(spec), std::move(pot));
  }
  return sum;
}

struct CompositeConfigureResult {
  bool ok = false;
  /// Set when the composite arm was accepted; the caller swaps it in.
  std::unique_ptr<SumPot> replacement;
  std::string message;
};

/// none is a no-op, composite rebuilds the sum, every other arm is refused.
[[nodiscard]] inline CompositeConfigureResult
configureComposite(const ::PotentialConfig::Reader &cfg) {
  CompositeConfigureResult result;
  switch (cfg.which()) {
  case ::PotentialConfig::NONE:
    result.ok = true;
    result.message = "no-op (rgpot params: none)";
    return result;
  case ::PotentialConfig::COMPOSITE: {
    try {
      result.replacement = compositeFromParams(cfg.getComposite());
    } catch (const std::exception &e) {
      result.message = std::string("rgpot params composite arm rejected: ") + e.what();
      return result;
    }
    std::ostringstream summary;
    const auto &sum = *result.replacement;
    for (std::size_t i = 0; i < sum.size(); ++i) {
      const SumTerm &t = sum.term(i);
      summary << (i == 0 ? "" : " ") << t.name << "(weight=" << t.weight
              << (t.enabled ? "" : ", disabled") << ")";
    }
    result.ok = true;
    result.message = "rgpot params applied (composite arm): " + summary.str();
    return result;
  }
  default:
    result.message = "rgpot params arm " + potentialConfigArm(cfg) +
                     " not handled by the composite backend (use matching backend pot)";
    return result;
  }
}

} // namespace rgpot::rpc
