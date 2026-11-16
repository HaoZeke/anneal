#pragma once
// MIT License
// Copyright 2023--present rgpot developers

#include "rgpot/nlist/NeighbourList.hpp"
#include "rgpot/simd/pods.hpp"

#include <algorithm>
#include <array>
#include <cstddef>
#include <cstdint>
#include <vector>

namespace rgpot::simd {

/// Cutoff, skin and periodicity for one half-list build.
struct PairSpec {
  double cutoff = 0;
  double skin = 1.0;
  std::array<bool, 3> periodic{{true, true, true}};
};

/// Indices from one neighbour-layer evaluation. The view points into this
/// object. image is the general-cell fold when the box is not orthorhombic.
struct CachedPairs {
  std::vector<std::int32_t> rows;
  std::vector<std::int32_t> nbr;
  nlist::MinimumImage image{};
  PairView view{};
};

namespace detail_fill {

inline void append_half(std::vector<std::int32_t> &ij, std::int32_t i,
                        std::int32_t j, double r2, double cutoff2) {
  if (i < j && r2 <= cutoff2 && r2 > 0.0) {
    ij.push_back(i);
    ij.push_back(j);
  }
}

} // namespace detail_fill

/// Build a half-list for the dispatched pair kernels. Candidates come
/// from the shared neighbour layer. `reuse` keeps the Verlet list across
/// calls. Several threads walk disjoint atoms and the rows are sorted, so
/// the list does not depend on which thread found a pair. fold is the
/// orthorhombic round, with a zero inverse on an axis the options leave
/// unfolded.
inline void fill_view(CachedPairs &out, const double *R, std::size_t n,
                      const double *box, const PairSpec &opt,
                      nlist::Mode mode = nlist::Mode::Auto,
                      nlist::Neighbours *reuse = nullptr) {
  nlist::Neighbours local;
  nlist::Neighbours &found = reuse != nullptr ? *reuse : local;
  nlist::updateCells(found, R, n, box, opt.cutoff, opt.skin, nlist::Image::Round, mode,
                     opt.periodic);
  // Keep candidates out to the skin. The force kernel still drops pairs
  // past opt.cutoff, using the positions of the call that evaluates them.
  // A reused list therefore still has a pair that was outside the cutoff
  // and has since moved inside it.
  const double search = opt.cutoff + opt.skin;
  const double cutoff2 = search * search;
  const int nt = n > 1 ? nlist::threadsRequested() : 1;
  std::vector<std::int32_t> ij;
  if (nt <= 1) {
    nlist::visitDirected(found, 0, static_cast<std::int64_t>(n),
                      [&](std::int32_t i, std::int32_t j, double, double, double,
                          double r2) {
                        detail_fill::append_half(ij, i, j, r2, cutoff2);
                      });
  } else {
    std::vector<std::vector<std::int32_t>> part(static_cast<std::size_t>(nt));
    nlist::forEachSlice(nt, [&](int t) {
      const auto range = nlist::partition(static_cast<std::int64_t>(n), t, nt);
      auto &mine = part[static_cast<std::size_t>(t)];
      nlist::visitDirected(found, range.begin, range.end,
                        [&](std::int32_t i, std::int32_t j, double, double,
                            double, double r2) {
                          detail_fill::append_half(mine, i, j, r2, cutoff2);
                        });
    });
    std::size_t total = 0;
    for (const auto &mine : part) {
      total += mine.size();
    }
    ij.reserve(total);
    for (const auto &mine : part) {
      ij.insert(ij.end(), mine.begin(), mine.end());
    }
  }
  out.rows.assign(n + 1, 0);
  for (std::size_t p = 0; p < ij.size(); p += 2) {
    out.rows[static_cast<std::size_t>(ij[p]) + 1] += 1;
  }
  for (std::size_t i = 0; i < n; ++i) {
    out.rows[i + 1] += out.rows[i];
  }
  out.nbr.assign(ij.size() / 2, 0);
  std::vector<std::int32_t> cursor(out.rows.begin(), out.rows.end() - 1);
  for (std::size_t p = 0; p < ij.size(); p += 2) {
    const auto i = static_cast<std::size_t>(ij[p]);
    out.nbr[static_cast<std::size_t>(cursor[i]++)] = ij[p + 1];
  }
  // Each row is sorted so a rebuild and a threaded walk sum in one order.
  for (std::size_t i = 0; i < n; ++i) {
    std::sort(out.nbr.begin() + out.rows[i], out.nbr.begin() + out.rows[i + 1]);
  }
  PairView v{};
  v.R = R;
  v.rows = out.rows.data();
  v.nbr = out.nbr.data();
  v.n = n;
  v.cutoff2 = opt.cutoff * opt.cutoff;
  const bool ortho = box[1] == 0.0 && box[2] == 0.0 && box[3] == 0.0 &&
                     box[5] == 0.0 && box[6] == 0.0 && box[7] == 0.0;
  if (ortho) {
    v.fold = 1;
    v.w[0] = box[0];
    v.w[1] = box[4];
    v.w[2] = box[8];
    for (int k = 0; k < 3; ++k) {
      const double w = v.w[k];
      v.inv[k] = (opt.periodic[static_cast<std::size_t>(k)] && w != 0.0)
                     ? 1.0 / w
                     : 0.0;
    }
  } else {
    out.image = nlist::MinimumImage(box, opt.periodic);
    v.fold = 3;
    v.general = &out.image;
  }
  out.view = v;
}

} // namespace rgpot::simd
