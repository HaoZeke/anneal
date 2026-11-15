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

#if defined(_OPENMP)
#include <omp.h>
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

int threadsRequested() {
#if defined(_OPENMP)
  const char *env = std::getenv("OMP_NUM_THREADS");
  if (env == nullptr || env[0] == '\0') {
    return 1;
  }
  char *end = nullptr;
  const long parsed = std::strtol(env, &end, 10);
  if (end == env || parsed < 1) {
    return 1;
  }
  if (parsed > 0x7fffffffL) {
    return 0x7fffffff;
  }
  return static_cast<int>(parsed);
#else
  return 1;
#endif
}

Range partition(std::int64_t n, int thread, int nthreads) {
  if (nthreads < 1) {
    nthreads = 1;
  }
  if (thread < 0 || thread >= nthreads || n <= 0) {
    return {};
  }
  const auto nt = static_cast<std::int64_t>(nthreads);
  const auto t = static_cast<std::int64_t>(thread);
  Range out;
  out.begin = (n * t) / nt;
  out.end = (n * (t + 1)) / nt;
  return out;
}

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
                          const Spec &spec) const {
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
  // A streaming walk trusts the bins. Atoms that have moved may sit in
  // another cell, so the bins are reused only for an identical geometry.
  // A stored list keeps its candidates out to the skin.
  if (streaming_) {
    return std::memcmp(ref_.data(), xyz, n * 3 * sizeof(double)) == 0;
  }
  return withinSkin(ref_.data(), xyz, n, spec.skin);
}

void Neighbours::buildCells(const double *xyz, std::size_t n) {
  const double edge = search_ > 0.0 ? search_ : 1.0;
  nx_ = axisCount(len_[0], edge);
  ny_ = axisCount(len_[1], edge);
  nz_ = axisCount(len_[2], edge);
  const double bytes =
      static_cast<double>(nx_) * static_cast<double>(ny_) *
      static_cast<double>(nz_) * static_cast<double>(sizeof(std::int32_t));
  const auto avail = availableBytes();
  if (avail > 0 && bytes > static_cast<double>(avail)) {
    nx_ = ny_ = nz_ = 1;
  }
  const auto ncells = static_cast<std::size_t>(nx_) * static_cast<std::size_t>(ny_) *
                      static_cast<std::size_t>(nz_);
  cell_start_.assign(ncells + 1, 0);
  cell_of_.resize(n);
  for (std::size_t a = 0; a < n; ++a) {
    int ix = 0;
    int iy = 0;
    int iz = 0;
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
    ix = bin(xyz[3 * a], len_[0], nx_);
    iy = bin(xyz[3 * a + 1], len_[1], ny_);
    iz = bin(xyz[3 * a + 2], len_[2], nz_);
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
    cell_atoms_[static_cast<std::size_t>(fill[c]++)] = static_cast<std::int32_t>(a);
  }
}

std::uint64_t Neighbours::candidateBound() const {
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
          here += static_cast<std::uint64_t>(cell_start_[c + 1] - cell_start_[c]);
        }
      }
    }
    if (here > 0) {
      bound += here - 1;
    }
  }
  return bound;
}


void Neighbours::buildStored(const double *xyz, std::size_t n) {
  (void)xyz;
  off_.assign(n + 1, 0);
  nbr_.clear();
  std::vector<std::int32_t> gather;
  const double search2 = search_ * search_;
  for (std::size_t a = 0; a < n; ++a) {
    off_[a] = static_cast<std::int32_t>(nbr_.size());
    const int cell = cell_of_[a];
    int iz = cell % nz_;
    int iy = (cell / nz_) % ny_;
    int ix = cell / (nz_ * ny_);
    int seen[27];
    int nseen = 0;
    gather.clear();
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
              gather.push_back(j);
            }
          }
        }
      }
    }
    std::sort(gather.begin(), gather.end());
    nbr_.insert(nbr_.end(), gather.begin(), gather.end());
  }
  off_[n] = static_cast<std::int32_t>(nbr_.size());
}

void Neighbours::update(const double *xyz, std::size_t n, const double *box9,
                        const Spec &spec) {
  xyz_ = xyz;
  if (reusable(xyz, n, box9, spec)) {
    reused_ = true;
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
  const bool want_store = spec.mode == Mode::Stored ? fits
                          : spec.mode == Mode::Auto && fits;
  if (want_store) {
    buildStored(xyz, n);
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
