#pragma once
// MIT License
// Copyright 2023--present rgpot developers

#include <array>
#include <cstdint>

namespace rgpot::nlist {

/// Nearest periodic image for row-major lattice vectors: d - n H.
///
/// The working type is double. Each product-sum is rgpot::simd::mul_add:
/// a hardware fused multiply-add when the processor has one, and the
/// exactly rounded TwoProduct/TwoSum emulation otherwise. This translation
/// unit is compiled with contraction off, so the folded bits do not depend
/// on the instruction set. A cell whose infinity-norm
/// condition estimate would spend more than half the double significand is
/// Minkowski-reduced (Nguyen and Stehlé, ACM Trans. Algorithms 5, 46
/// (2009), doi:10.1145/1597036.1597050) and the image is taken in that
/// basis. A reduced basis that is still too ill-conditioned searches the
/// 27 images of coefficients in {-1, 0, 1}. Singular or non-finite cells
/// throw. An index that does not fit in an int64 throws.
///
/// The rounded image, d less the nearest lattice point in fractional
/// coordinates, is returned without a search when a bound from its
/// fractional coordinates shows that no other image is as short. The
/// search would return the same bits.
class MinimumImage {
public:
  MinimumImage() = default;
  MinimumImage(const double *cell, const std::array<bool, 3> &periodic);
  void fold(double &dx, double &dy, double &dz) const;
  /// fold for a caller that keeps only images of squared length at most
  /// `within2`. False means every image is longer, and the displacement is
  /// left unspecified. True leaves the bits of fold, which may still be
  /// longer. A neighbour search uses this to skip far pairs without a
  /// search over images.
  [[nodiscard]] bool foldWithin(double &dx, double &dy, double &dz,
                                double within2) const;
  /// Folds on the calling thread, since it started, that searched over
  /// images. A fold the bounds settle does not count. Tests read it to pin
  /// the paths that need no search.
  [[nodiscard]] static std::uint64_t searches() noexcept;

private:
  std::array<double, 9> cell_{};
  std::array<double, 9> inverse_{};
  std::array<double, 3> dual_norm_{};
  std::array<bool, 3> periodic_{};
  bool narrow_{false};
};

} // namespace rgpot::nlist
