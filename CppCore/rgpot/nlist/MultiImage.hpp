#pragma once
// MIT License
// Copyright 2023--present rgpot developers

/**
 * @brief Every periodic image inside a cutoff.
 *
 * A stored pair list keeps one image (the minimum image) of each partner.
 * That spans the full cutoff sphere only while 2 * cutoff is below every
 * periodic face spacing. Past that, a pair can have multiple periodic images
 * inside the cutoff, and an atom can see translations of itself.
 *
 * This header walks those images directly. The walk is a plain O(N^2) pair
 * scan and, for each pair, a cube of lattice shifts O((2e + 1)^3), where e is
 * the per-axis image extent. For short-face slabs this is intentionally slow;
 * a cell-list acceleration for this path is left out of scope. The product of
 * the per-axis shift counts is bounded, so three thin faces are refused. A
 * fractional separation that is not finite, or whose nearest integer plus the
 * extent does not fit in an `int` index, is refused the same way.
 */

#include <array>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <limits>
#include <stdexcept>
#include <string>

namespace rgpot::nlist {

/// Perpendicular spacing of the face opposite lattice axis `axis`.
inline double faceSpacing(const double *box, int axis) noexcept {
  const double *u = box + 3 * ((axis + 1) % 3);
  const double *v = box + 3 * ((axis + 2) % 3);
  const double cx = u[1] * v[2] - u[2] * v[1];
  const double cy = u[2] * v[0] - u[0] * v[2];
  const double cz = u[0] * v[1] - u[1] * v[0];
  const double area = std::sqrt(cx * cx + cy * cy + cz * cz);
  const double vol = std::abs(box[0] * (box[4] * box[8] - box[5] * box[7]) -
                              box[1] * (box[3] * box[8] - box[5] * box[6]) +
                              box[2] * (box[3] * box[7] - box[4] * box[6]));
  if (!(area > 0.0) || !(vol > 0.0) || !std::isfinite(area) ||
      !std::isfinite(vol)) {
    return 0.0;
  }
  return vol / area;
}

/// True when one minimum image is not the whole shell on a periodic side.
inline bool
cutoffExceedsHalfFace(const double *box, double cutoff,
                      const std::array<bool, 3> &periodic) noexcept {
  for (int axis = 0; axis < 3; ++axis) {
    if (!periodic[static_cast<std::size_t>(axis)]) {
      continue;
    }
    const double spacing = faceSpacing(box, axis);
    if (!(2.0 * cutoff < spacing)) {
      return true;
    }
  }
  return false;
}

namespace detail {

inline bool invertColumns(const double m[9], double inv[9]) noexcept {
  const double det = m[0] * (m[4] * m[8] - m[5] * m[7]) -
                     m[3] * (m[1] * m[8] - m[2] * m[7]) +
                     m[6] * (m[1] * m[5] - m[2] * m[4]);
  if (!(std::abs(det) > 0.0) || !std::isfinite(det)) {
    return false;
  }
  const double id = 1.0 / det;
  inv[0] = (m[4] * m[8] - m[5] * m[7]) * id;
  inv[1] = (m[2] * m[7] - m[1] * m[8]) * id;
  inv[2] = (m[1] * m[5] - m[2] * m[4]) * id;
  inv[3] = (m[5] * m[6] - m[3] * m[8]) * id;
  inv[4] = (m[0] * m[8] - m[2] * m[6]) * id;
  inv[5] = (m[2] * m[3] - m[0] * m[5]) * id;
  inv[6] = (m[3] * m[7] - m[4] * m[6]) * id;
  inv[7] = (m[1] * m[6] - m[0] * m[7]) * id;
  inv[8] = (m[0] * m[4] - m[1] * m[3]) * id;
  return true;
}

inline bool positiveHalfSpace(long long h, long long k, long long l) noexcept {
  return h > 0 || (h == 0 && (k > 0 || (k == 0 && l > 0)));
}

constexpr int kMaxImageExtent = 1024;

/// Lattice shifts one pair may visit: the product over the three axes of
/// `2 * extent + 1`. One thin face may still reach the per-axis cap when the
/// other faces are wide. Three thin faces are refused here. 64 cubed.
constexpr std::int64_t kMaxImageCount = 64 * 64 * 64;

/// Image loops index with `int`. A centre is usable only when `centre ±
/// extent` stays inside these limits.
constexpr long long kMinImageIndex = std::numeric_limits<int>::min();
constexpr long long kMaxImageIndex = std::numeric_limits<int>::max();

/// `std::llround` is specified only inside the range of `long long`.
constexpr double kMinRoundableShift =
    static_cast<double>(std::numeric_limits<long long>::min());
constexpr double kMaxRoundableShift =
    static_cast<double>(std::numeric_limits<long long>::max());

/// Nearest lattice index of a fractional separation. Throws the same
/// `std::invalid_argument` as a bad cell when the value is not finite, when
/// `std::llround` cannot represent it, or when the image window around the
/// rounded centre does not fit in `int`.
inline int imageCenter(double fractional, int extent) {
  if (!std::isfinite(fractional)) {
    throw std::invalid_argument("EAM table: non-finite fractional separation");
  }
  if (!(fractional > kMinRoundableShift) ||
      !(fractional < kMaxRoundableShift)) {
    throw std::invalid_argument(
        "EAM table: fractional separation exceeds lattice index range");
  }
  const long long shift = std::llround(fractional);
  const long long span = static_cast<long long>(extent);
  if (shift < kMinImageIndex + span || shift > kMaxImageIndex - span) {
    throw std::invalid_argument(
        "EAM table: fractional separation exceeds lattice index range");
  }
  return static_cast<int>(shift);
}

inline int extentForAxis(double cutoff, double spacing, int axis) {
  if (!(spacing > 0.0) || !std::isfinite(spacing)) {
    throw std::invalid_argument(
        "EAM table: non-positive periodic face spacing");
  }
  const double shells = std::ceil(cutoff / spacing);
  if (!(shells >= 0.0) || !std::isfinite(shells)) {
    throw std::invalid_argument("EAM table: invalid image-extent shell count");
  }
  if (shells > static_cast<double>(kMaxImageExtent)) {
    throw std::invalid_argument("EAM table: image extent on axis " +
                                std::to_string(axis) + " exceeds cap " +
                                std::to_string(kMaxImageExtent) +
                                " (cutoff=" + std::to_string(cutoff) +
                                ", spacing=" + std::to_string(spacing) + ")");
  }
  // An image inside the cutoff has |h - f| < cutoff / spacing along the
  // axis, where f is the fractional separation and h the lattice shift,
  // because the face normal component of a vector is at most its length.
  // The walk is centred on lround(f), so |h - lround(f)| is below
  // cutoff / spacing + 1/2, which is at most ceil(cutoff / spacing).
  return static_cast<int>(shells);
}

} // namespace detail

/// Unordered pairs `(i, j)` with `j >= i`, one call per image inside the
/// cutoff. `fn` is `fn(i, j, dx, dy, dz, r2)`.
/// Throws `std::invalid_argument` for a singular cell, an image extent or
/// image count past the cap, a non-finite fractional separation, or a
/// separation whose nearest integer plus the extent does not fit in `int`.
template <typename Fn>
void visitHalfImages(const double *pos, std::size_t n, const double *box,
                     const std::array<bool, 3> &periodic, double cutoff,
                     Fn &&fn) {
  if (n < 1 || !(cutoff > 0.0) || !std::isfinite(cutoff)) {
    return;
  }
  // Columns of `columns` are the lattice vectors, so r = columns * fractional.
  const double columns[9] = {box[0], box[1], box[2], box[3], box[4],
                             box[5], box[6], box[7], box[8]};
  double inverse[9];
  if (!detail::invertColumns(columns, inverse)) {
    throw std::invalid_argument("EAM table: periodic cell is singular");
  }
  int extent[3];
  for (int axis = 0; axis < 3; ++axis) {
    if (!periodic[static_cast<std::size_t>(axis)]) {
      extent[axis] = 0;
      continue;
    }
    extent[axis] = detail::extentForAxis(cutoff, faceSpacing(box, axis), axis);
  }
  std::int64_t image_count = 1;
  for (int axis = 0; axis < 3; ++axis) {
    const std::int64_t span = 2 * static_cast<std::int64_t>(extent[axis]) + 1;
    if (span < 1 || image_count > detail::kMaxImageCount / span) {
      throw std::invalid_argument("EAM table: image count exceeds cap " +
                                  std::to_string(detail::kMaxImageCount) +
                                  " (extents " + std::to_string(extent[0]) +
                                  " " + std::to_string(extent[1]) + " " +
                                  std::to_string(extent[2]) + ")");
    }
    image_count *= span;
  }
  const double cut2 = cutoff * cutoff;
  for (std::size_t i = 0; i < n; ++i) {
    const double xi = pos[3 * i];
    const double yi = pos[3 * i + 1];
    const double zi = pos[3 * i + 2];
    for (std::size_t j = i; j < n; ++j) {
      const double dx0 = xi - pos[3 * j];
      const double dy0 = yi - pos[3 * j + 1];
      const double dz0 = zi - pos[3 * j + 2];
      const double f0 = inverse[0] * dx0 + inverse[3] * dy0 + inverse[6] * dz0;
      const double f1 = inverse[1] * dx0 + inverse[4] * dy0 + inverse[7] * dz0;
      const double f2 = inverse[2] * dx0 + inverse[5] * dy0 + inverse[8] * dz0;
      const int c0 = detail::imageCenter(f0, extent[0]);
      const int c1 = detail::imageCenter(f1, extent[1]);
      const int c2 = detail::imageCenter(f2, extent[2]);
      // The sums fit in int. The counters are wide so the increment after
      // the maximum int, which ends a window on that limit, is defined.
      const int h_lo = c0 - extent[0];
      const int h_hi = c0 + extent[0];
      const int k_lo = c1 - extent[1];
      const int k_hi = c1 + extent[1];
      const int l_lo = c2 - extent[2];
      const int l_hi = c2 + extent[2];
      for (long long h = h_lo; h <= h_hi; ++h) {
        for (long long k = k_lo; k <= k_hi; ++k) {
          for (long long l = l_lo; l <= l_hi; ++l) {
            if (!periodic[0] && h != 0) {
              continue;
            }
            if (!periodic[1] && k != 0) {
              continue;
            }
            if (!periodic[2] && l != 0) {
              continue;
            }
            if (i == j) {
              if (h == 0 && k == 0 && l == 0) {
                continue;
              }
              if (!detail::positiveHalfSpace(h, k, l)) {
                continue;
              }
            }
            const double sx = h * box[0] + k * box[3] + l * box[6];
            const double sy = h * box[1] + k * box[4] + l * box[7];
            const double sz = h * box[2] + k * box[5] + l * box[8];
            const double dx = dx0 - sx;
            const double dy = dy0 - sy;
            const double dz = dz0 - sz;
            const double r2 = dx * dx + dy * dy + dz * dz;
            if (!(r2 > 0.0) || !(r2 < cut2)) {
              continue;
            }
            fn(static_cast<std::int32_t>(i), static_cast<std::int32_t>(j), dx,
               dy, dz, r2);
          }
        }
      }
    }
  }
}

} // namespace rgpot::nlist
