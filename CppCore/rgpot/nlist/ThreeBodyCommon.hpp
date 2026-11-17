#pragma once
// MIT License
// Copyright 2023--present rgpot developers

/**
 * @brief Shared neighbour-update, bond-collection, and slice-reduction
 * helpers for three-body potentials (Stillinger-Weber and Tersoff).
 */

#include "rgpot/nlist/HalfShell.hpp"
#include "rgpot/nlist/NeighbourList.hpp"

#include <algorithm>
#include <cstddef>
#include <cstdint>
#include <cstring>
#include <vector>

namespace rgpot::nlist {

/// Update @p cells for a three-body cutoff and synchronize @p cache when it
/// carries a positive skin and the cell list rebuilt.
inline void updateThreeBodyCells(Neighbours &cells, const double *positions,
                                 std::size_t n_atoms, const double *box,
                                 double cutoff, Mode mode,
                                 NeighborList *cache) {
  const double skin = (cache != nullptr && cache->skin > 0.0)
                          ? cache->skin
                          : (mode == Mode::Stream ? 0.0 : cutoff / 5.0);
  const auto before = cells.builds();
  updateCells(cells, positions, n_atoms, box, cutoff, skin, Image::Round, mode,
              {true, true, true});
  if (cache != nullptr && cache->skin > 0.0 &&
      (cells.builds() != before || !cache->built)) {
    cache->ref.assign(positions, positions + 3 * n_atoms);
    std::memcpy(cache->box, box, 9 * sizeof(double));
    cache->n = n_atoms;
    cache->cutoff = cutoff;
    cache->skin_used = cache->skin;
    cache->built = true;
  }
}

/// Collect and sort by partner index all neighbour bonds of atom @p i.
template <typename Bond, typename AcceptFn>
inline void collectAtomBonds(const Neighbours &cells, int i,
                             std::vector<Bond> &nb, AcceptFn &&accept) {
  nb.clear();
  if (cells.streaming()) {
    cells.forAtom(static_cast<std::int32_t>(i), accept);
  } else {
    visitDirected(cells, i, static_cast<std::int64_t>(i) + 1, accept);
  }
  std::sort(nb.begin(), nb.end(),
            [](const Bond &a, const Bond &b) { return a.j < b.j; });
}

/// Clamp cosine of two bond vectors to [-1, 1].
inline double clampCosine(double c) noexcept {
  if (c > 1.0) {
    return 1.0;
  }
  if (c < -1.0) {
    return -1.0;
  }
  return c;
}

} // namespace rgpot::nlist
