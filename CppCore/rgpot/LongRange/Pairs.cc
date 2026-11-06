// MIT License
// Copyright 2023--present rgpot developers

#include "rgpot/LongRange/Pairs.hpp"

#include <algorithm>
#include <cmath>
#include <stdexcept>

#include <linkcell.hpp>

namespace rgpot::lr {

struct DualList::Skin {
  linkcell::SkinList list;
  explicit Skin(const linkcell::Cell &cell, double cutoff, double skin)
      : list(cell, cutoff, skin, true) {}
};

namespace {

linkcell::Cell cell_of(const double *box9) {
  return linkcell::Cell::from_vectors({box9[0], box9[1], box9[2]},
                                      {box9[3], box9[4], box9[5]},
                                      {box9[6], box9[7], box9[8]});
}

void vacuum_box(const double *pos, std::size_t n, double reach, double *box9) {
  double lo[3] = {pos[0], pos[1], pos[2]};
  double hi[3] = {lo[0], lo[1], lo[2]};
  for (std::size_t i = 1; i < n; ++i) {
    for (int c = 0; c < 3; ++c) {
      lo[c] = std::min(lo[c], pos[3 * i + c]);
      hi[c] = std::max(hi[c], pos[3 * i + c]);
    }
  }
  // L > 2 * extent and L/2 > reach, so the minimum image of every pair
  // inside the cutoff is the Cartesian pair.
  std::fill(box9, box9 + 9, 0.0);
  for (int c = 0; c < 3; ++c) {
    const double extent = hi[c] - lo[c];
    const double L = 2.0 * std::max(extent, reach) + 1.0;
    box9[4 * c] = L;
  }
}

} // namespace

DualList::DualList(double short_cut, double long_cut, double skin)
    : short_cut_(short_cut), long_cut_(long_cut), skin_(skin) {
  if (!(short_cut > 0.0) || !(long_cut > 0.0) || !(skin >= 0.0)) {
    throw std::invalid_argument("cutoffs must be positive and skin non-negative");
  }
  if (short_cut > long_cut) {
    throw std::invalid_argument("short cutoff exceeds the long cutoff");
  }
}

DualList::~DualList() = default;

void DualList::fill(Skin &skin, double cutoff, const double *pos, std::size_t n,
                    const double *box9, std::vector<Pair> *out, bool *rebuilt) {
  const auto cell = cell_of(box9);
  const auto &rows = skin.list.update(cell, pos, n, true);
  *rebuilt = skin.list.rebuilt();
  const auto &vec = skin.list.vectors();
  out->clear();
  out->reserve(rows.size());
  for (std::size_t t = 0; t < rows.size(); ++t) {
    const double x = vec[t][0];
    const double y = vec[t][1];
    const double z = vec[t][2];
    const double r2 = x * x + y * y + z * z;
    if (!(r2 > 0.0) || r2 >= cutoff * cutoff) {
      continue;
    }
    Pair p;
    p.i = rows[t].i;
    p.j = rows[t].j;
    p.x = x;
    p.y = y;
    p.z = z;
    p.r = std::sqrt(r2);
    out->push_back(p);
  }
}

void DualList::update(const double *pos, std::size_t n, const double *box, bool periodic) {
  short_.clear();
  long_.clear();
  short_rebuilt_ = false;
  long_rebuilt_ = false;
  if (n < 2) {
    return;
  }
  if (pos == nullptr) {
    throw std::invalid_argument("positions are null");
  }
  const double *use = box;
  if (!periodic || box == nullptr) {
    double need[9];
    vacuum_box(pos, n, long_cut_ + skin_, need);
    if (!have_vacuum_) {
      std::copy(need, need + 9, vacuum_);
      have_vacuum_ = true;
    } else {
      for (int c = 0; c < 3; ++c) {
        vacuum_[4 * c] = std::max(vacuum_[4 * c], need[4 * c]);
      }
    }
    use = vacuum_;
  }
  const auto cell = cell_of(use);
  if (!short_skin_) {
    short_skin_ = std::make_unique<Skin>(cell, short_cut_, skin_);
    long_skin_ = std::make_unique<Skin>(cell, long_cut_, skin_);
  }
  fill(*short_skin_, short_cut_, pos, n, use, &short_, &short_rebuilt_);
  fill(*long_skin_, long_cut_, pos, n, use, &long_, &long_rebuilt_);
}

} // namespace rgpot::lr
