#pragma once
// MIT License
// Copyright 2023--present rgpot developers

#include "rgpot/Potential.hpp"

namespace rgpot {

/// Linked frontend for the external cbe-pes H + CH4 library.
class CBEPot final : public Potential<CBEPot> {
public:
  CBEPot();
  void forceImpl(const ForceInput &in, ForceOut *out) const override;
  void hessianVectorImpl(const ForceInput &in, const double *direction,
                         double *product) const override;
  [[nodiscard]] PotCaps caps() const noexcept override {
    return {.reentrancy = Reentrancy::ProcessSerial,
            .periodic = false,
            .analyticHvp = true};
  }
};

} // namespace rgpot
