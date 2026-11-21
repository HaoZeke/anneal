// MIT License
// Copyright 2023--present rgpot developers

#include "rgpot/LongRange/Pairs.hpp"

#include "rgpot/nlist/NeighbourList.hpp"

#include <algorithm>
#include <cmath>
#include <stdexcept>
#include <vector>

namespace rgpot::lr {

namespace {

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

struct DualList::Skin {
  nlist::Builder builder;
};

DualList::DualList(double short_cut, double long_cut, double skin)
    : short_cut_(short_cut), long_cut_(long_cut), skin_(skin) {
  if (!(short_cut > 0.0) || !(long_cut > 0.0) || !(skin >= 0.0)) {
    throw std::invalid_argument(
        "cutoffs must be positive and skin non-negative");
  }
  if (short_cut > long_cut) {
    throw std::invalid_argument("short cutoff exceeds the long cutoff");
  }
}

DualList::~DualList() = default;

const std::vector<Pair> &DualList::short_pairs() const noexcept {
  return short_;
}

const std::vector<Pair> &DualList::long_pairs() const noexcept { return long_; }

void DualList::fill(Skin &skin, double cutoff, const double *pos, std::size_t n,
                    const double *box9, std::vector<Pair> *out, bool *rebuilt) {
  nlist::Query query;
  query.cutoff = cutoff;
  query.skin = skin_;
  query.shape = nlist::Pairs{nlist::Shell::Half};
  query.cut = nlist::Cut::Loose;
  query.fields.shifts = true;
  query.fields.vectors = false;
  query.fields.distances = false;
  skin.builder.update(pos, n, box9, query);
  *rebuilt = !skin.builder.reused();

  const nlist::View candidates = skin.builder.table().view();
  const double cut2 = cutoff * cutoff;
  out->clear();
  out->reserve(candidates.first.size());
  for (std::size_t p = 0; p < candidates.first.size(); ++p) {
    const int i = candidates.first[p];
    const int j = candidates.second[p];
    const double x =
        pos[3 * j] - pos[3 * i] +
        static_cast<double>(candidates.shift[3 * p]) * box9[0] +
        static_cast<double>(candidates.shift[3 * p + 1]) * box9[3] +
        static_cast<double>(candidates.shift[3 * p + 2]) * box9[6];
    const double y =
        pos[3 * j + 1] - pos[3 * i + 1] +
        static_cast<double>(candidates.shift[3 * p]) * box9[1] +
        static_cast<double>(candidates.shift[3 * p + 1]) * box9[4] +
        static_cast<double>(candidates.shift[3 * p + 2]) * box9[7];
    const double z =
        pos[3 * j + 2] - pos[3 * i + 2] +
        static_cast<double>(candidates.shift[3 * p]) * box9[2] +
        static_cast<double>(candidates.shift[3 * p + 1]) * box9[5] +
        static_cast<double>(candidates.shift[3 * p + 2]) * box9[8];
    const double r2 = x * x + y * y + z * z;
    if (r2 > 0.0 && r2 < cut2) {
      out->push_back(Pair{i, j, x, y, z, std::sqrt(r2)});
    }
  }
}

void DualList::update(const double *pos, std::size_t n, const double *box,
                      bool periodic) {
  short_rebuilt_ = false;
  long_rebuilt_ = false;
  if (n == 0) {
    short_.clear();
    long_.clear();
    return;
  }
  if (n == 1 && (!periodic || box == nullptr)) {
    short_.clear();
    long_.clear();
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
  if (!short_skin_) {
    short_skin_ = std::make_unique<Skin>();
    long_skin_ = std::make_unique<Skin>();
  }
  fill(*short_skin_, short_cut_, pos, n, use, &short_, &short_rebuilt_);
  fill(*long_skin_, long_cut_, pos, n, use, &long_, &long_rebuilt_);
}

} // namespace rgpot::lr
