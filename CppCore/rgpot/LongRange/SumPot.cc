// MIT License
// Copyright 2023--present rgpot developers

#include "rgpot/LongRange/SumPot.hpp"

#include "rgpot/LongRange/Dispersion.hpp"
#include "rgpot/LongRange/Electro.hpp"
#include "rgpot/LongRange/Nlh.hpp"
#include "rgpot/types/AtomMatrix.hpp"

#include <algorithm>
#include <stdexcept>

namespace rgpot::lr {

SumPot::SumPot() : Potential(PotType::LrSum) {}

void SumPot::add(std::unique_ptr<PotentialBase> term) {
  if (!term) {
    throw std::invalid_argument("SumPot cannot take a null term");
  }
  terms_.push_back(std::move(term));
}

void SumPot::share_neighbours(std::shared_ptr<DualList> lists) {
  for (auto &term : terms_) {
    if (auto *c = dynamic_cast<CoulombPot *>(term.get())) {
      c->set_neighbours(lists);
    } else if (auto *d = dynamic_cast<DispersionPot *>(term.get())) {
      d->set_neighbours(lists);
    } else if (auto *n = dynamic_cast<NlhPot *>(term.get())) {
      n->set_neighbours(lists);
    }
  }
}

void SumPot::forceImpl(const ForceInput &in, ForceOut *out) const {
  std::fill(out->F, out->F + 3 * in.nAtoms, 0.0);
  out->energy = 0.0;
  out->variance = 0.0;
  out->has_stress = 0;
  std::fill(out->stress, out->stress + 9, 0.0);
  for (const auto &term : terms_) {
    ForceOut part{};
    std::vector<double> f(3 * in.nAtoms, 0.0);
    part.F = f.data();
    term->forceBatch(ForceBatch{.nSystems = 1, .in = &in, .out = &part});
    out->energy += part.energy;
    out->variance += part.variance;
    for (std::size_t i = 0; i < f.size(); ++i) {
      out->F[i] += f[i];
    }
    if (part.has_stress) {
      for (int k = 0; k < 9; ++k) {
        out->stress[k] += part.stress[k];
      }
      out->has_stress = 1;
    }
  }
}

} // namespace rgpot::lr
