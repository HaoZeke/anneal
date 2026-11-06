#pragma once
// MIT License
// Copyright 2023--present rgpot developers

// Additive composition. Each child is a Potential evaluated through
// ForceInput; energies, forces, and Cauchy stress are summed. A CPMD,
// metatomic, or other backend is one child. Long-range summands are the
// others. Children that own a DualList can share one list.

#include <memory>
#include <vector>

#include "rgpot/LongRange/Pairs.hpp"
#include "rgpot/Potential.hpp"

namespace rgpot::lr {

class SumPot final : public Potential<SumPot> {
public:
  SumPot();

  void add(std::unique_ptr<PotentialBase> term);
  void share_neighbours(std::shared_ptr<DualList> lists);

  void forceImpl(const ForceInput &in, ForceOut *out) const override;

  [[nodiscard]] PotCaps caps() const noexcept override {
    return {.reentrancy = Reentrancy::PerInstance, .stress = true};
  }

  [[nodiscard]] std::size_t size() const noexcept { return terms_.size(); }

private:
  mutable std::vector<std::unique_ptr<PotentialBase>> terms_;
};

} // namespace rgpot::lr
