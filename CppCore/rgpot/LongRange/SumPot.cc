// MIT License
// Copyright 2023--present rgpot developers

#include "rgpot/LongRange/SumPot.hpp"

#include "rgpot/LongRange/Dispersion.hpp"
#include "rgpot/LongRange/Electro.hpp"
#include "rgpot/LongRange/Nlh.hpp"

namespace rgpot::lr {

void SumPot::shareNeighbours(std::shared_ptr<DualList> lists) {
  for (std::size_t i = 0; i < size(); ++i) {
    PotentialBase *term = child(i);
    if (auto *c = dynamic_cast<CoulombPot *>(term)) {
      c->setNeighbours(lists);
    } else if (auto *d = dynamic_cast<DispersionPot *>(term)) {
      d->setNeighbours(lists);
    } else if (auto *n = dynamic_cast<NlhPot *>(term)) {
      n->setNeighbours(lists);
    }
  }
}

} // namespace rgpot::lr
