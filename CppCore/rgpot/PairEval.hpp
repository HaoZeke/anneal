#pragma once
// MIT License
// Copyright 2023--present rgpot developers

/**
 * @brief Shared free-box fallback for free-boundary potentials.
 */

namespace rgpot {

/// Return @p box when it is non-null with positive diagonal entries, else a
/// 1e6 Angstrom cube for free-boundary cell walks.
inline const double *freeBoxOrCell(const double *box) noexcept {
  static constexpr double free_box[9] = {1.0e6, 0.0, 0.0, 0.0, 1.0e6,
                                         0.0,   0.0, 0.0, 1.0e6};
  const bool box_ok =
      box != nullptr && box[0] > 0.0 && box[4] > 0.0 && box[8] > 0.0;
  return box_ok ? box : free_box;
}

} // namespace rgpot
