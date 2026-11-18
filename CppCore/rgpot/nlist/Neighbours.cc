// MIT License
// Copyright 2023--present rgpot developers

#include "rgpot/nlist/Neighbours.hpp"

#include <algorithm>
#include <cmath>
#include <cstdlib>
#include <cstring>
#include <string>

#if defined(_WIN32)
#ifndef WIN32_LEAN_AND_MEAN
#define WIN32_LEAN_AND_MEAN
#endif
#include <windows.h>
#elif defined(__APPLE__)
#include <mach/mach.h>
#else
#include <fstream>
#endif

namespace rgpot::nlist {
namespace {

int axisCount(double length, double edge) {
  if (!(length > 0.0) || !(edge > 0.0)) {
    return 1;
  }
  auto n = static_cast<long>(std::floor(length / edge));
  if (n < 1) {
    n = 1;
  }
  while (n > 1 && length / static_cast<double>(n) < edge) {
    --n;
  }
  return static_cast<int>(n);
}

} // namespace

std::uint64_t availableBytes() {
#if defined(_WIN32)
  MEMORYSTATUSEX status{};
  status.dwLength = sizeof(status);
  if (!GlobalMemoryStatusEx(&status)) {
    return 0;
  }
  return static_cast<std::uint64_t>(status.ullAvailPhys);
#elif defined(__APPLE__)
  vm_statistics64_data_t vm{};
  mach_msg_type_number_t count = HOST_VM_INFO64_COUNT;
  if (host_statistics64(mach_host_self(), HOST_VM_INFO64,
                        reinterpret_cast<host_info64_t>(&vm),
                        &count) != KERN_SUCCESS) {
    return 0;
  }
  const auto page = static_cast<std::uint64_t>(vm_page_size);
  return (static_cast<std::uint64_t>(vm.free_count) +
          static_cast<std::uint64_t>(vm.inactive_count)) *
         page;
#else
  std::ifstream in("/proc/meminfo");
  std::string key;
  std::uint64_t kb = 0;
  std::string unit;
  while (in >> key >> kb >> unit) {
    if (key == "MemAvailable:") {
      return kb * 1024ULL;
    }
  }
  return 0;
#endif
}

bool Neighbours::reusable(const double *xyz, std::size_t n, const double *box9,
                          const Spec &spec, bool &wrapped) const {
  wrapped = false;
  if (!built_ || n != n_) {
    return false;
  }
  SpecKey key;
  key.cutoff = spec.cutoff;
  key.skin = spec.skin;
  key.image = spec.image;
  key.mode = spec.mode;
  key.periodic = spec.periodic;
  if (!(key == key_) || std::memcmp(box_, box9, sizeof(box_)) != 0) {
    return false;
  }
  if (allPairs()) {
    return true;
  }
  // A streaming walk trusts the bins. Atoms that have moved may sit in
  // another cell, so the bins are reused only for an identical geometry.
  // A stored list keeps its candidates out to the skin.
  if (streaming_) {
    return std::memcmp(ref_.data(), xyz, n * 3 * sizeof(double)) == 0;
  }
  const double limit2 = 0.25 * spec.skin * spec.skin;
  for (std::size_t a = 0; a < n; ++a) {
    double dx = xyz[3 * a] - ref_[3 * a];
    double dy = xyz[3 * a + 1] - ref_[3 * a + 1];
    double dz = xyz[3 * a + 2] - ref_[3 * a + 2];
    if (dx * dx + dy * dy + dz * dz <= limit2) {
      continue;
    }
    fold(dx, dy, dz);
    if (!(dx * dx + dy * dy + dz * dz <= limit2)) {
      return false;
    }
    wrapped = true;
  }
  return true;
}

void Neighbours::buildCells(const double *xyz, std::size_t n) {
  const double edge = search_ > 0.0 ? search_ : 1.0;
  // A grid that does not fit is one cell. An open axis can instead be
  // binned on the atom span, which is what a cluster in a huge box needs.
  // A periodic axis keeps the cell length. One cell remains when the
  // tight grid does not fit either. A grid that fits is unchanged, so
  // subtracting a zero origin leaves those bins as they were.
  double origin[3] = {0.0, 0.0, 0.0};
  double span[3] = {len_[0], len_[1], len_[2]};
  auto set_axes = [&]() {
    nx_ = axisCount(span[0], edge);
    ny_ = axisCount(span[1], edge);
    nz_ = axisCount(span[2], edge);
  };
  auto bytes_of = [&]() {
    return static_cast<double>(nx_) * static_cast<double>(ny_) *
           static_cast<double>(nz_) * static_cast<double>(sizeof(std::int32_t));
  };
  set_axes();
  const auto avail = availableBytes();
  if (avail > 0 && bytes_of() > static_cast<double>(avail)) {
    bool any_open = false;
    for (int k = 0; k < 3; ++k) {
      if (periodic_[k] || n == 0) {
        continue;
      }
      any_open = true;
      double lo = xyz[static_cast<std::size_t>(k)];
      double hi = lo;
      for (std::size_t a = 1; a < n; ++a) {
        const double v = xyz[3 * a + static_cast<std::size_t>(k)];
        if (v < lo) {
          lo = v;
        }
        if (v > hi) {
          hi = v;
        }
      }
      const double extent = hi - lo;
      span[k] = extent > edge ? extent : edge;
      origin[k] = lo;
    }
    if (any_open) {
      set_axes();
    }
    if (bytes_of() > static_cast<double>(avail)) {
      nx_ = ny_ = nz_ = 1;
    }
  }
  const auto ncells = static_cast<std::size_t>(nx_) *
                      static_cast<std::size_t>(ny_) *
                      static_cast<std::size_t>(nz_);
  cell_start_.assign(ncells + 1, 0);
  cell_of_.resize(n);
  auto bin = [](double x, double length, int naxis) {
    if (!(length > 0.0) || naxis <= 1) {
      return 0;
    }
    double t = x / length * static_cast<double>(naxis);
    if (!(t >= 0.0)) {
      t = 0.0;
    }
    auto i = static_cast<int>(t);
    if (i >= naxis) {
      i = naxis - 1;
    }
    if (i < 0) {
      i = 0;
    }
    return i;
  };
  for (std::size_t a = 0; a < n; ++a) {
    const int ix = bin(xyz[3 * a] - origin[0], span[0], nx_);
    const int iy = bin(xyz[3 * a + 1] - origin[1], span[1], ny_);
    const int iz = bin(xyz[3 * a + 2] - origin[2], span[2], nz_);
    const int id = (ix * ny_ + iy) * nz_ + iz;
    cell_of_[a] = id;
    ++cell_start_[static_cast<std::size_t>(id) + 1];
  }
  for (std::size_t c = 0; c < ncells; ++c) {
    cell_start_[c + 1] += cell_start_[c];
  }
  cell_atoms_.assign(n, 0);
  std::vector<std::int32_t> fill(cell_start_.begin(), cell_start_.end() - 1);
  for (std::size_t a = 0; a < n; ++a) {
    const auto c = static_cast<std::size_t>(cell_of_[a]);
    cell_atoms_[static_cast<std::size_t>(fill[c]++)] =
        static_cast<std::int32_t>(a);
  }
}

std::uint64_t Neighbours::candidateBound() const {
  // Integer adds over the stencil. A dilute system above kTeamMinAtoms
  // has almost none, and a region would cost more than the count, so
  // this stays on the calling thread. The gather below is the build.
  std::uint64_t bound = 0;
  const auto n = n_;
  for (std::size_t a = 0; a < n; ++a) {
    const int cell = cell_of_[a];
    int iz = cell % nz_;
    int iy = (cell / nz_) % ny_;
    int ix = cell / (nz_ * ny_);
    int seen[27];
    int nseen = 0;
    std::uint64_t here = 0;
    for (int ox = -1; ox <= 1; ++ox) {
      for (int oy = -1; oy <= 1; ++oy) {
        for (int oz = -1; oz <= 1; ++oz) {
          const int id = cellId(ix + ox, iy + oy, iz + oz);
          if (id < 0) {
            continue;
          }
          bool dup = false;
          for (int s = 0; s < nseen; ++s) {
            if (seen[s] == id) {
              dup = true;
              break;
            }
          }
          if (dup) {
            continue;
          }
          seen[nseen++] = id;
          const auto c = static_cast<std::size_t>(id);
          here +=
              static_cast<std::uint64_t>(cell_start_[c + 1] - cell_start_[c]);
        }
      }
    }
    if (here > 0) {
      bound += here - 1;
    }
  }
  return bound;
}

void Neighbours::buildStored(const double *xyz, std::size_t n,
                             std::uint64_t bound) {
  (void)xyz;
  const double search2 = search_ * search_;
  // Partners of one atom, unsorted. The caller sorts and appends them.
  auto gather = [&](std::size_t a, std::vector<std::int32_t> &out) {
    const int cell = cell_of_[a];
    int iz = cell % nz_;
    int iy = (cell / nz_) % ny_;
    int ix = cell / (nz_ * ny_);
    int seen[27];
    int nseen = 0;
    for (int ox = -1; ox <= 1; ++ox) {
      for (int oy = -1; oy <= 1; ++oy) {
        for (int oz = -1; oz <= 1; ++oz) {
          const int id = cellId(ix + ox, iy + oy, iz + oz);
          if (id < 0) {
            continue;
          }
          bool dup = false;
          for (int s = 0; s < nseen; ++s) {
            if (seen[s] == id) {
              dup = true;
              break;
            }
          }
          if (dup) {
            continue;
          }
          seen[nseen++] = id;
          const auto c = static_cast<std::size_t>(id);
          for (std::int32_t q = cell_start_[c]; q < cell_start_[c + 1]; ++q) {
            const auto j = cell_atoms_[static_cast<std::size_t>(q)];
            if (j == static_cast<std::int32_t>(a)) {
              continue;
            }
            double dx = 0;
            double dy = 0;
            double dz = 0;
            if (!foldPairWithin(static_cast<std::int32_t>(a), j, search2, dx,
                                dy, dz)) {
              continue;
            }
            const double r2 = dx * dx + dy * dy + dz * dz;
            if (r2 <= search2) {
              out.push_back(j);
            }
          }
        }
      }
    }
  };
  // A region costs more than a short gather. The pair kernels use 4096
  // before they thread a half list. A dilute system can still report
  // about one stencil candidate per atom, which is the same cheap pass,
  // so the gather also stays serial unless the stencil holds more
  // candidates than atoms.
  constexpr std::uint64_t kMinParallelCandidates = 4096;
  const bool worth = bound >= kMinParallelCandidates && bound > n;
  const int nt = worth ? teamSize(n) : 1;
  if (nt <= 1) {
    off_.assign(n + 1, 0);
    nbr_.clear();
    std::vector<std::int32_t> row;
    for (std::size_t a = 0; a < n; ++a) {
      off_[a] = static_cast<std::int32_t>(nbr_.size());
      row.clear();
      gather(a, row);
      std::sort(row.begin(), row.end());
      nbr_.insert(nbr_.end(), row.begin(), row.end());
    }
    off_[n] = static_cast<std::int32_t>(nbr_.size());
    return;
  }
  // One buffer per slice, reserved once, written only by the slice that
  // owns it. Joining in slice order is the atom order of one thread.
  struct Part {
    std::vector<std::int32_t> nbr;
    std::vector<std::int32_t> count;
  };
  std::vector<Part> parts(static_cast<std::size_t>(nt));
  forEachSlice(nt, [&](int t) {
    const auto range = partition(static_cast<std::int64_t>(n), t, nt);
    auto &part = parts[static_cast<std::size_t>(t)];
    const auto len = static_cast<std::size_t>(range.end - range.begin);
    part.count.assign(len, 0);
    part.nbr.reserve(len * 64);
    std::vector<std::int32_t> row;
    row.reserve(64);
    std::size_t k = 0;
    for (std::int64_t i = range.begin; i < range.end; ++i, ++k) {
      row.clear();
      gather(static_cast<std::size_t>(i), row);
      std::sort(row.begin(), row.end());
      part.count[k] = static_cast<std::int32_t>(row.size());
      part.nbr.insert(part.nbr.end(), row.begin(), row.end());
    }
  });
  std::size_t total = 0;
  for (const Part &part : parts) {
    total += part.nbr.size();
  }
  nbr_.clear();
  nbr_.reserve(total);
  off_.assign(n + 1, 0);
  std::size_t cursor = 0;
  std::size_t atom = 0;
  for (const Part &part : parts) {
    nbr_.insert(nbr_.end(), part.nbr.begin(), part.nbr.end());
    for (const std::int32_t c : part.count) {
      off_[atom] = static_cast<std::int32_t>(cursor);
      cursor += static_cast<std::size_t>(c);
      ++atom;
    }
  }
  off_[n] = static_cast<std::int32_t>(nbr_.size());
}

void Neighbours::update(const double *xyz, std::size_t n, const double *box9,
                        const Spec &spec) {
  xyz_ = xyz;
  bool wrapped = false;
  if (reusable(xyz, n, box9, spec, wrapped)) {
    reused_ = true;
    // The tile walk takes the image of a pair from the bins of its atoms,
    // and a wrapped atom left its bin on the far side of the cell.
    if (wrapped) {
      buildCells(xyz, n);
    }
    return;
  }
  reused_ = false;
  n_ = n;
  std::memcpy(box_, box9, sizeof(box_));
  len_[0] = box9[0];
  len_[1] = box9[4];
  len_[2] = box9[8];
  for (int k = 0; k < 3; ++k) {
    periodic_[k] = spec.periodic[static_cast<std::size_t>(k)] && len_[k] > 0.0;
    inv_[k] = periodic_[k] ? 1.0 / len_[k] : 0.0;
  }
  const bool sheared = box9[1] != 0.0 || box9[2] != 0.0 || box9[3] != 0.0 ||
                       box9[5] != 0.0 || box9[6] != 0.0 || box9[7] != 0.0;
  image_ = spec.image;
  use_general_ = image_ == Image::General || sheared;
  if (use_general_) {
    general_ = MinimumImage(box9, spec.periodic);
  }
  search_ = spec.cutoff + spec.skin;
  if (!(search_ > 0.0)) {
    search_ = 0.0;
  }
  key_.cutoff = spec.cutoff;
  key_.skin = spec.skin;
  key_.image = spec.image;
  key_.mode = spec.mode;
  key_.periodic = spec.periodic;
  off_.clear();
  nbr_.clear();
  streaming_ = true;
  if (n < 2 || !(search_ > 0.0)) {
    streaming_ = false;
    mode_ = Mode::Stored;
    off_.assign(n + 1, 0);
    nbr_.clear();
    built_ = true;
    ref_.assign(xyz, xyz + 3 * n);
    return;
  }
  // Every pair is a candidate, so the walks need neither bins nor a list,
  // and the coordinates never invalidate the update. allPairs() reads
  // this state back.
  if (std::isinf(search_)) {
    mode_ = Mode::Stream;
    built_ = true;
    return;
  }
  buildCells(xyz, n);
  // A stream never stores the pairs, so the candidate count is unused.
  // Counting it walks every stencil the search is about to walk.
  if (spec.mode == Mode::Stream) {
    streaming_ = true;
    mode_ = Mode::Stream;
    built_ = true;
    ref_.assign(xyz, xyz + 3 * n);
    return;
  }
  const std::uint64_t bound = candidateBound();
  const auto avail = availableBytes();
  const auto index_bytes = static_cast<std::uint64_t>(sizeof(std::int32_t));
  // Zero means the probe failed, not that the machine has no free memory.
  // An explicit stored list, and Auto, still keep the index in that case.
  const bool bounded = bound <= static_cast<std::uint64_t>(0x7fffffff);
  const bool over_budget = avail > 0 && bound > avail / index_bytes;
  const bool fits = bounded && !over_budget;
  // The stream keeps the bins. The list adds one index per candidate.
  // Auto stores only when that index fits. A short cutoff that does not
  // fit the 27-slot stencil keeps the smaller mask in the same cache.
  const bool want_store =
      spec.mode == Mode::Stored ? fits : spec.mode == Mode::Auto && fits;
  if (want_store) {
    buildStored(xyz, n, bound);
    streaming_ = false;
    mode_ = Mode::Stored;
    ++builds_;
  } else {
    streaming_ = true;
    mode_ = Mode::Stream;
  }
  built_ = true;
  ref_.assign(xyz, xyz + 3 * n);
}

} // namespace rgpot::nlist
