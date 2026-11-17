#pragma once
// MIT License
// Copyright 2023--present rgpot developers

/**
 * @brief NWChem potential under rgpot `PotentialConfig` user parameters.
 *
 * User path: Cap'n Proto `PotentialConfig` / `NWChemParams` only. The frontend
 * serializes that message and passes it through C ABI symbols resolved by
 * dlopen.
 */

#include "rgpot/Potential.hpp"
#include "rgpot/rpc/Potentials.capnp.h"

#include <string>

namespace rgpot {

class NWChemPot : public Potential<NWChemPot> {
public:
  NWChemPot();
  /** Apply schema defaults, then optional Cap'n Proto NWChemParams. */
  explicit NWChemPot(const ::NWChemParams::Reader &params);
  ~NWChemPot() override;

  NWChemPot(const NWChemPot &) = delete;
  NWChemPot &operator=(const NWChemPot &) = delete;

  void forceImpl(const ForceInput &in, ForceOut *out) const override;

  /// The dlopen'd engine keeps global session state: serialize
  /// process-wide.
  [[nodiscard]] PotCaps caps() const noexcept override {
    return {.reentrancy = Reentrancy::ProcessSerial, .periodic = false};
  }


  /** Apply NWChem arm payload as the next Cap'n Proto message for the engine. */
  bool setParams(const ::NWChemParams::Reader &params);

  /** Write last applied NWChem params back to Cap'n Proto. */
  void getParams(::NWChemParams::Builder out) const;

  /**
   * Apply full rgpot PotentialConfig; only succeeds for none or nwchem arms.
   */
  bool setPotentialConfig(const ::PotentialConfig::Reader &cfg,
                          std::string *message_out = nullptr);

  bool available() const;
  static bool probeAvailable();
  static bool abiAvailable();

private:
  struct Impl;
  Impl *impl_;
};

} // namespace rgpot
