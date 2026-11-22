// MIT License
// Copyright 2023--present rgpot developers

#include "rgpot/nlist/Neighbours.hpp"

#include <algorithm>
#include <cmath>
#include <cstdlib>
#include <cstring>
#include <limits>
#include <stdexcept>
#include <string>
#include <utility>

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

#if defined(__GNUC__) || defined(__clang__)
__attribute__((visibility("hidden")))
#endif
void invertCell(const double *cell, double inv[9], double dual_norm[3]);

namespace {

int axisCount(double length, double edge) {
  if (!(length > 0.0) || !(edge > 0.0) || !(length > edge)) {
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

bool orthorhombicCell(const double box[9]) {
  return box[1] == 0.0 && box[2] == 0.0 && box[3] == 0.0 && box[5] == 0.0 &&
         box[6] == 0.0 && box[7] == 0.0;
}

struct FractionalPlan {
  int axes[3]{1, 1, 1};
  double origin[3]{0.0, 0.0, 0.0};
  double span[3]{1.0, 1.0, 1.0};
  bool prefix_ready = false;
  bool cache_previous = false;
};

struct FractionalWorkspace {
  std::vector<double> coordinate;
  std::vector<std::int32_t> cell;
  std::vector<std::int32_t> start;
  double box[9]{};
  double inverse[9]{};
  double dual_norm[3]{};
  bool inverse_valid = false;
  double cached_box[9]{};
  bool cached_periodic[3]{};
  double cached_edge = 0.0;
  int cached_axes[3]{1, 1, 1};
  bool cells_valid = false;
};

FractionalWorkspace &fractionalWorkspace() {
  // update() and its cell build are one synchronous operation. A workspace
  // per calling thread is therefore reusable without changing Neighbours'
  // public object layout or sharing mutable data between update threads.
  thread_local FractionalWorkspace workspace;
  return workspace;
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

void prepareFractionalCells(const double *xyz, std::size_t n,
                            const double *box9, const bool periodic[3],
                            double edge, FractionalPlan &plan) {
  auto &workspace = fractionalWorkspace();
  if (workspace.cells_valid && workspace.cached_edge == edge &&
      workspace.coordinate.size() == 3 * n &&
      std::memcmp(workspace.cached_box, box9, sizeof(workspace.cached_box)) ==
          0 &&
      std::memcmp(workspace.cached_periodic, periodic,
                  sizeof(workspace.cached_periodic)) == 0 &&
      std::memcmp(workspace.coordinate.data(), xyz, 3 * n * sizeof(double)) ==
          0) {
    plan.axes[0] = workspace.cached_axes[0];
    plan.axes[1] = workspace.cached_axes[1];
    plan.axes[2] = workspace.cached_axes[2];
    plan.prefix_ready = true;
    plan.cache_previous = true;
    return;
  }
  workspace.cells_valid = false;
  double inv[9]{};
  double dual_norm[3]{};
  if (workspace.inverse_valid &&
      std::memcmp(workspace.box, box9, sizeof(workspace.box)) == 0) {
    std::memcpy(inv, workspace.inverse, sizeof(inv));
    std::memcpy(dual_norm, workspace.dual_norm, sizeof(dual_norm));
  } else {
    invertCell(box9, inv, dual_norm);
    std::memcpy(workspace.box, box9, sizeof(workspace.box));
    std::memcpy(workspace.inverse, inv, sizeof(workspace.inverse));
    std::memcpy(workspace.dual_norm, dual_norm, sizeof(workspace.dual_norm));
    workspace.inverse_valid = true;
  }
  auto &fractional = workspace.coordinate;

  double max_err[3] = {0.0, 0.0, 0.0};
  double lo[3] = {0.0, 0.0, 0.0};
  double hi[3] = {0.0, 0.0, 0.0};
  const double ref[3] = {
      n > 0 ? xyz[0] : 0.0,
      n > 0 ? xyz[1] : 0.0,
      n > 0 ? xyz[2] : 0.0,
  };
  constexpr double eps = std::numeric_limits<double>::epsilon();
  constexpr double kFracEdgeSlack = 1.0 + 16.0 * eps;
  const bool lower_triangular = inv[1] == 0.0 && inv[2] == 0.0 && inv[5] == 0.0;
  auto project = [&](double px, double py, double pz, double s[3]) {
    if (lower_triangular) {
      s[0] = (px * inv[0] + py * inv[3]) + pz * inv[6];
      s[1] = py * inv[4] + pz * inv[7];
      s[2] = pz * inv[8];
      return;
    }
    s[0] = (px * inv[0] + py * inv[3]) + pz * inv[6];
    s[1] = (px * inv[1] + py * inv[4]) + pz * inv[7];
    s[2] = (px * inv[2] + py * inv[5]) + pz * inv[8];
  };

  if (periodic[0] && periodic[1] && periodic[2]) {
    const double coordinate_bound[3] = {
        2.0 * (std::abs(box9[0]) + std::abs(box9[3]) + std::abs(box9[6])),
        2.0 * (std::abs(box9[1]) + std::abs(box9[4]) + std::abs(box9[7])),
        2.0 * (std::abs(box9[2]) + std::abs(box9[5]) + std::abs(box9[8])),
    };
    for (int k = 0; k < 3; ++k) {
      const double error = 2.0 * eps *
                           (coordinate_bound[0] * std::abs(inv[k]) +
                            coordinate_bound[1] * std::abs(inv[3 + k]) +
                            coordinate_bound[2] * std::abs(inv[6 + k]));
      const double frac_edge =
          (edge * dual_norm[k]) * kFracEdgeSlack + 2.0 * error + 2.0 * eps;
      plan.axes[k] = axisCount(1.0, frac_edge);
    }
    auto grid_bytes = [&]() {
      return static_cast<double>(plan.axes[0]) *
             static_cast<double>(plan.axes[1]) *
             static_cast<double>(plan.axes[2]) *
             static_cast<double>(sizeof(std::int32_t));
    };
    const auto avail = availableBytes();
    if (avail > 0 && grid_bytes() > static_cast<double>(avail)) {
      plan.axes[0] = plan.axes[1] = plan.axes[2] = 1;
    }
    auto prepare_grid = [&]() {
      const auto ncells = static_cast<std::size_t>(plan.axes[0]) *
                          static_cast<std::size_t>(plan.axes[1]) *
                          static_cast<std::size_t>(plan.axes[2]);
      workspace.cell.resize(n);
      workspace.start.assign(ncells + 1, 0);
    };
    prepare_grid();
    bool within_bound = true;
    for (std::size_t a = 0; a < n; ++a) {
      const double px = xyz[3 * a] - ref[0];
      const double py = xyz[3 * a + 1] - ref[1];
      const double pz = xyz[3 * a + 2] - ref[2];
      double s[3];
      project(px, py, pz, s);
      if (std::abs(px) > coordinate_bound[0] ||
          std::abs(py) > coordinate_bound[1] ||
          std::abs(pz) > coordinate_bound[2]) {
        within_bound = false;
      }
      int axis[3]{};
      for (int k = 0; k < 3; ++k) {
        if (!std::isfinite(s[k])) {
          throw std::invalid_argument("fractional coordinate must be finite");
        }
        const double wrapped = s[k] - std::floor(s[k]);
        axis[k] = std::clamp(
            static_cast<int>(wrapped * static_cast<double>(plan.axes[k])), 0,
            plan.axes[k] - 1);
      }
      const int id =
          (axis[0] * plan.axes[1] + axis[1]) * plan.axes[2] + axis[2];
      workspace.cell[a] = id;
      ++workspace.start[static_cast<std::size_t>(id) + 1];
    }
    if (!within_bound) {
      const int provisional[3] = {
          plan.axes[0],
          plan.axes[1],
          plan.axes[2],
      };
      double max_delta[3] = {0.0, 0.0, 0.0};
      for (std::size_t a = 0; a < n; ++a) {
        max_delta[0] = std::max(max_delta[0], std::abs(xyz[3 * a] - ref[0]));
        max_delta[1] =
            std::max(max_delta[1], std::abs(xyz[3 * a + 1] - ref[1]));
        max_delta[2] =
            std::max(max_delta[2], std::abs(xyz[3 * a + 2] - ref[2]));
      }
      for (int k = 0; k < 3; ++k) {
        const double error = 2.0 * eps *
                             (max_delta[0] * std::abs(inv[k]) +
                              max_delta[1] * std::abs(inv[3 + k]) +
                              max_delta[2] * std::abs(inv[6 + k]));
        const double frac_edge =
            (edge * dual_norm[k]) * kFracEdgeSlack + 2.0 * error + 2.0 * eps;
        plan.axes[k] = axisCount(1.0, frac_edge);
      }
      if (avail > 0 && grid_bytes() > static_cast<double>(avail)) {
        plan.axes[0] = plan.axes[1] = plan.axes[2] = 1;
      }
      const bool changed = plan.axes[0] != provisional[0] ||
                           plan.axes[1] != provisional[1] ||
                           plan.axes[2] != provisional[2];
      if (!changed) {
        return;
      }
      prepare_grid();
      for (std::size_t a = 0; a < n; ++a) {
        const double px = xyz[3 * a] - ref[0];
        const double py = xyz[3 * a + 1] - ref[1];
        const double pz = xyz[3 * a + 2] - ref[2];
        double s[3];
        project(px, py, pz, s);
        int axis[3]{};
        for (int k = 0; k < 3; ++k) {
          const double wrapped = s[k] - std::floor(s[k]);
          axis[k] = std::clamp(
              static_cast<int>(wrapped * static_cast<double>(plan.axes[k])), 0,
              plan.axes[k] - 1);
        }
        const int id =
            (axis[0] * plan.axes[1] + axis[1]) * plan.axes[2] + axis[2];
        workspace.cell[a] = id;
        ++workspace.start[static_cast<std::size_t>(id) + 1];
      }
    }
    return;
  }

  fractional.resize(3 * n);
  double max_delta[3] = {0.0, 0.0, 0.0};
  double max_fraction[3] = {0.0, 0.0, 0.0};
  bool finite = true;

  for (std::size_t a = 0; a < n; ++a) {
    const double px = xyz[3 * a] - ref[0];
    const double py = xyz[3 * a + 1] - ref[1];
    const double pz = xyz[3 * a + 2] - ref[2];
    double s[3];
    project(px, py, pz, s);
    max_delta[0] = std::max(max_delta[0], std::abs(px));
    max_delta[1] = std::max(max_delta[1], std::abs(py));
    max_delta[2] = std::max(max_delta[2], std::abs(pz));
    for (int k = 0; k < 3; ++k) {
      fractional[3 * a + static_cast<std::size_t>(k)] = s[k];
      finite = finite && std::isfinite(s[k]);
      max_fraction[k] = std::max(max_fraction[k], std::abs(s[k]));
      if (a == 0 || s[k] < lo[k]) {
        lo[k] = s[k];
      }
      if (a == 0 || s[k] > hi[k]) {
        hi[k] = s[k];
      }
    }
  }
  if (!finite) {
    throw std::invalid_argument("fractional coordinate must be finite");
  }

  const bool needs_wrapping = (periodic[0] && max_fraction[0] > 2.0) ||
                              (periodic[1] && max_fraction[1] > 2.0) ||
                              (periodic[2] && max_fraction[2] > 2.0);
  if (needs_wrapping) {
    lo[0] = lo[1] = lo[2] = 0.0;
    hi[0] = hi[1] = hi[2] = 0.0;
    for (std::size_t a = 0; a < n; ++a) {
      double px = xyz[3 * a] - ref[0];
      double py = xyz[3 * a + 1] - ref[1];
      double pz = xyz[3 * a + 2] - ref[2];
      double s[3] = {
          fractional[3 * a],
          fractional[3 * a + 1],
          fractional[3 * a + 2],
      };
      double err[3] = {
          2.0 * eps *
              (std::abs(px * inv[0]) + std::abs(py * inv[3]) +
               std::abs(pz * inv[6])),
          2.0 * eps *
              (std::abs(px * inv[1]) + std::abs(py * inv[4]) +
               std::abs(pz * inv[7])),
          2.0 * eps *
              (std::abs(px * inv[2]) + std::abs(py * inv[5]) +
               std::abs(pz * inv[8])),
      };
      for (int pass = 0; pass < 3; ++pass) {
        double m[3] = {0.0, 0.0, 0.0};
        bool wrap = false;
        for (int k = 0; k < 3; ++k) {
          if (periodic[k] && std::abs(s[k]) > 2.0) {
            m[k] = std::round(s[k]);
            wrap = wrap || m[k] != 0.0;
          }
        }
        if (!wrap) {
          break;
        }
        const double before_abs[3] = {
            std::abs(px * inv[0]) + std::abs(py * inv[3]) +
                std::abs(pz * inv[6]),
            std::abs(px * inv[1]) + std::abs(py * inv[4]) +
                std::abs(pz * inv[7]),
            std::abs(px * inv[2]) + std::abs(py * inv[5]) +
                std::abs(pz * inv[8]),
        };
        const double mx = m[0] * box9[0] + m[1] * box9[3] + m[2] * box9[6];
        const double my = m[0] * box9[1] + m[1] * box9[4] + m[2] * box9[7];
        const double mz = m[0] * box9[2] + m[1] * box9[5] + m[2] * box9[8];
        if (!std::isfinite(mx) || !std::isfinite(my) || !std::isfinite(mz)) {
          throw std::invalid_argument("fractional coordinate must be finite");
        }
        const double image_xyz[3] = {
            std::abs(m[0] * box9[0]) + std::abs(m[1] * box9[3]) +
                std::abs(m[2] * box9[6]),
            std::abs(m[0] * box9[1]) + std::abs(m[1] * box9[4]) +
                std::abs(m[2] * box9[7]),
            std::abs(m[0] * box9[2]) + std::abs(m[1] * box9[5]) +
                std::abs(m[2] * box9[8]),
        };
        px -= mx;
        py -= my;
        pz -= mz;
        project(px, py, pz, s);
        for (int k = 0; k < 3; ++k) {
          const double after_abs = std::abs(px * inv[k]) +
                                   std::abs(py * inv[3 + k]) +
                                   std::abs(pz * inv[6 + k]);
          const double image_abs = image_xyz[0] * std::abs(inv[k]) +
                                   image_xyz[1] * std::abs(inv[3 + k]) +
                                   image_xyz[2] * std::abs(inv[6 + k]);
          if (!std::isfinite(s[k]) || !std::isfinite(image_abs) ||
              !std::isfinite(after_abs)) {
            throw std::invalid_argument("fractional coordinate must be finite");
          }
          err[k] += 2.0 * eps * (before_abs[k] + image_abs + after_abs);
        }
      }
      for (int k = 0; k < 3; ++k) {
        fractional[3 * a + static_cast<std::size_t>(k)] = s[k];
        max_err[k] = std::max(max_err[k], err[k]);
        if (!periodic[k]) {
          if (a == 0 || s[k] < lo[k]) {
            lo[k] = s[k];
          }
          if (a == 0 || s[k] > hi[k]) {
            hi[k] = s[k];
          }
        }
      }
    }
  }
  for (int k = 0; k < 3; ++k) {
    const double unwrapped_error =
        2.0 * eps *
        (max_delta[0] * std::abs(inv[k]) + max_delta[1] * std::abs(inv[3 + k]) +
         max_delta[2] * std::abs(inv[6 + k]));
    max_err[k] = std::max(max_err[k], unwrapped_error);
  }

  for (int k = 0; k < 3; ++k) {
    const double base_edge =
        (edge * dual_norm[k]) * kFracEdgeSlack + 2.0 * max_err[k] + 2.0 * eps;
    if (periodic[k]) {
      plan.axes[k] = axisCount(1.0, base_edge);
      continue;
    }
    if (n > 0) {
      const double extent = hi[k] - lo[k];
      if (!std::isfinite(extent)) {
        throw std::invalid_argument("fractional coordinate must be finite");
      }
      const double frac_edge = base_edge + 2.0 * eps * extent;
      plan.origin[k] = lo[k];
      plan.span[k] = extent > frac_edge ? extent : frac_edge;
      plan.axes[k] = axisCount(plan.span[k], frac_edge);
    }
  }
  const double bytes = static_cast<double>(plan.axes[0]) *
                       static_cast<double>(plan.axes[1]) *
                       static_cast<double>(plan.axes[2]) *
                       static_cast<double>(sizeof(std::int32_t));
  const auto avail = availableBytes();
  if (avail > 0 && bytes > static_cast<double>(avail)) {
    plan.axes[0] = plan.axes[1] = plan.axes[2] = 1;
  }
  const auto ncells = static_cast<std::size_t>(plan.axes[0]) *
                      static_cast<std::size_t>(plan.axes[1]) *
                      static_cast<std::size_t>(plan.axes[2]);
  workspace.cell.resize(n);
  workspace.start.assign(ncells + 1, 0);
  auto bin = [](double x, double length, int naxis) {
    if (!(length > 0.0) || naxis <= 1) {
      return 0;
    }
    double t = x / length * static_cast<double>(naxis);
    if (!(t >= 0.0)) {
      t = 0.0;
    }
    return std::clamp(static_cast<int>(t), 0, naxis - 1);
  };
  for (std::size_t a = 0; a < n; ++a) {
    int axis[3]{};
    for (int k = 0; k < 3; ++k) {
      const double s = fractional[3 * a + static_cast<std::size_t>(k)];
      if (periodic[k]) {
        const double wrapped = s - std::floor(s);
        axis[k] = std::clamp(
            static_cast<int>(wrapped * static_cast<double>(plan.axes[k])), 0,
            plan.axes[k] - 1);
      } else {
        axis[k] = bin(s - plan.origin[k], plan.span[k], plan.axes[k]);
      }
    }
    const int id = (axis[0] * plan.axes[1] + axis[1]) * plan.axes[2] + axis[2];
    workspace.cell[a] = id;
    ++workspace.start[static_cast<std::size_t>(id) + 1];
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
  const bool sheared = !orthorhombicCell(box9);
  if (!sheared) {
    // Keep the established orthorhombic path in place: it neither constructs
    // a transactional neighbour object nor touches fractional workspace.
    xyz_ = xyz;
    bool wrapped = false;
    if (reusable(xyz, n, box9, spec, wrapped)) {
      reused_ = true;
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
      periodic_[k] =
          spec.periodic[static_cast<std::size_t>(k)] && len_[k] > 0.0;
      inv_[k] = periodic_[k] ? 1.0 / len_[k] : 0.0;
    }
    image_ = spec.image;
    use_general_ = image_ == Image::General;
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
    if (std::isinf(search_)) {
      mode_ = Mode::Stream;
      built_ = true;
      return;
    }
    buildCells(xyz, n);
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
    const bool bounded = bound <= static_cast<std::uint64_t>(0x7fffffff);
    const bool over_budget = avail > 0 && bound > avail / index_bytes;
    const bool fits = bounded && !over_budget;
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
    return;
  }

  auto build_fractional_cells = [&](const FractionalPlan &plan) {
    auto &workspace = fractionalWorkspace();
    const int previous_axes[3] = {nx_, ny_, nz_};
    nx_ = plan.axes[0];
    ny_ = plan.axes[1];
    nz_ = plan.axes[2];
    const auto ncells = static_cast<std::size_t>(nx_) *
                        static_cast<std::size_t>(ny_) *
                        static_cast<std::size_t>(nz_);
    cell_of_.swap(workspace.cell);
    cell_start_.swap(workspace.start);
    if (!plan.prefix_ready) {
      for (std::size_t c = 0; c < ncells; ++c) {
        cell_start_[c + 1] += cell_start_[c];
      }
    }
    cell_atoms_.assign(n, 0);
    std::vector<std::int32_t> fill(cell_start_.begin(), cell_start_.end() - 1);
    for (std::size_t a = 0; a < n; ++a) {
      const auto c = static_cast<std::size_t>(cell_of_[a]);
      cell_atoms_[static_cast<std::size_t>(fill[c]++)] =
          static_cast<std::int32_t>(a);
    }
    if (plan.cache_previous && ref_.size() == 3 * n) {
      workspace.coordinate.assign(ref_.begin(), ref_.end());
      std::memcpy(workspace.cached_box, box_, sizeof(workspace.cached_box));
      std::memcpy(workspace.cached_periodic, periodic_,
                  sizeof(workspace.cached_periodic));
      workspace.cached_edge = search_;
      workspace.cached_axes[0] = previous_axes[0];
      workspace.cached_axes[1] = previous_axes[1];
      workspace.cached_axes[2] = previous_axes[2];
      workspace.cells_valid = true;
    } else {
      workspace.cells_valid = false;
    }
  };

  bool wrapped = false;
  if (reusable(xyz, n, box9, spec, wrapped)) {
    if (wrapped) {
      FractionalPlan plan;
      prepareFractionalCells(xyz, n, box_, periodic_, search_, plan);
      build_fractional_cells(plan);
    }
    xyz_ = xyz;
    reused_ = true;
    return;
  }

  bool next_periodic[3]{};
  for (int k = 0; k < 3; ++k) {
    next_periodic[k] =
        spec.periodic[static_cast<std::size_t>(k)] && box9[4 * k] > 0.0;
  }
  const bool same_general = built_ && use_general_ &&
                            key_.periodic == spec.periodic &&
                            std::memcmp(box_, box9, sizeof(box_)) == 0;
  MinimumImage next_general =
      same_general ? general_ : MinimumImage(box9, spec.periodic);
  double next_search = spec.cutoff + spec.skin;
  if (!(next_search > 0.0)) {
    next_search = 0.0;
  }
  const bool fractional_path =
      next_periodic[0] || next_periodic[1] || next_periodic[2];
  FractionalPlan plan;
  plan.cache_previous = same_general;
  if (fractional_path && n >= 2 && next_search > 0.0 &&
      !std::isinf(next_search)) {
    prepareFractionalCells(xyz, n, box9, next_periodic, next_search, plan);
  }

  xyz_ = xyz;
  reused_ = false;
  n_ = n;
  std::memcpy(box_, box9, sizeof(box_));
  len_[0] = box9[0];
  len_[1] = box9[4];
  len_[2] = box9[8];
  for (int k = 0; k < 3; ++k) {
    periodic_[k] = next_periodic[k];
    inv_[k] = periodic_[k] ? 1.0 / len_[k] : 0.0;
  }
  image_ = spec.image;
  use_general_ = true;
  general_ = std::move(next_general);
  search_ = next_search;
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
    built_ = true;
    ref_.assign(xyz, xyz + 3 * n);
    return;
  }
  if (std::isinf(search_)) {
    mode_ = Mode::Stream;
    built_ = true;
    return;
  }
  if (fractional_path) {
    build_fractional_cells(plan);
  } else {
    buildCells(xyz, n);
  }
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
  const bool bounded = bound <= static_cast<std::uint64_t>(0x7fffffff);
  const bool over_budget = avail > 0 && bound > avail / index_bytes;
  const bool fits = bounded && !over_budget;
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
