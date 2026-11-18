#pragma once
// MIT License
// Copyright 2023--present rgpot developers

// Additive composition for the long-range summands. rgpot::SumPot does the
// weighted sum; this class adds one pair list shared by the children that
// own a DualList.

#include <memory>

#include "rgpot/LongRange/Pairs.hpp"
#include "rgpot/SumPot/SumPot.hpp"

namespace rgpot::lr {

class SumPot final : public rgpot::SumPot {
public:
  SumPot() = default;

  void shareNeighbours(std::shared_ptr<DualList> lists);
};

} // namespace rgpot::lr
