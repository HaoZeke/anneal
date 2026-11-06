#pragma once
// MIT License
// Copyright 2023--present rgpot developers

// Optional short-range repulsion from SO3LR NLHRepulsionSparse.
// phi(r) = sum_k a_k exp(-b_k r), screened Coulomb, switched off by 1.5 A.

#include <memory>

#include "rgpot/LongRange/Pairs.hpp"
#include "rgpot/Potential.hpp"

namespace rgpot::lr {

struct NlhConfig {
  double cutoff = 2.0;
  double skin = 0.5;
  double switch_off = 1.5;
};

class NlhPot final : public Potential<NlhPot> {
public:
  explicit NlhPot(NlhConfig cfg = {});

  void forceImpl(const ForceInput &in, ForceOut *out) const override;

  [[nodiscard]] PotCaps caps() const noexcept override {
    return {.reentrancy = Reentrancy::PerInstance, .stress = true};
  }

  [[nodiscard]] uint64_t paramsKey() const noexcept override { return params_; }

  void set_neighbours(std::shared_ptr<DualList> lists) { lists_ = std::move(lists); }

private:
  NlhConfig cfg_;
  std::shared_ptr<DualList> lists_;
  uint64_t params_ = 0;
};

} // namespace rgpot::lr
