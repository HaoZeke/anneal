// MIT License
// Copyright 2023--present rgpot developers

#include <algorithm>
#include <array>
#include <cmath>
#include <cstdint>
#include <cstring>
#include <limits>
#include <random>
#include <stdexcept>
#include <vector>

#include <catch2/catch_all.hpp>

#include "rgpot/nlist/MinimumImage.hpp"

using Catch::Matchers::WithinAbs;

namespace {

/// The previous long-double search, kept as the agreement oracle.
class LongDoubleMinimumImage {
public:
  LongDoubleMinimumImage(const double *cell,
                         const std::array<bool, 3> &periodic)
      : periodic_(periodic) {
    for (int k = 0; k < 9; ++k) {
      if (!std::isfinite(cell[k]))
        throw std::invalid_argument("periodic cell must be finite");
      cell_[k] = cell[k];
    }
    const auto &a = cell_;
    const long double det = a[0] * (a[4] * a[8] - a[5] * a[7]) -
                            a[1] * (a[3] * a[8] - a[5] * a[6]) +
                            a[2] * (a[3] * a[7] - a[4] * a[6]);
    if (!std::isfinite(det) || det == 0.0L)
      throw std::invalid_argument("periodic cell must be nonsingular");
    inverse_ = {
        (a[4] * a[8] - a[5] * a[7]) / det, (a[2] * a[7] - a[1] * a[8]) / det,
        (a[1] * a[5] - a[2] * a[4]) / det, (a[5] * a[6] - a[3] * a[8]) / det,
        (a[0] * a[8] - a[2] * a[6]) / det, (a[2] * a[3] - a[0] * a[5]) / det,
        (a[3] * a[7] - a[4] * a[6]) / det, (a[1] * a[6] - a[0] * a[7]) / det,
        (a[0] * a[4] - a[1] * a[3]) / det};
    long double cell_norm = 0.0L, inverse_norm = 0.0L;
    for (int row = 0; row < 3; ++row) {
      long double cell_sum = 0.0L, inverse_sum = 0.0L;
      for (int col = 0; col < 3; ++col) {
        cell_sum += std::abs(cell_[3 * row + col]);
        inverse_sum += std::abs(inverse_[3 * row + col]);
      }
      cell_norm = std::max(cell_norm, cell_sum);
      inverse_norm = std::max(inverse_norm, inverse_sum);
    }
    const long double eps = std::numeric_limits<long double>::epsilon();
    if (!(cell_norm * inverse_norm * eps <= std::sqrt(eps)))
      throw std::invalid_argument(
          "periodic cell is too ill-conditioned for image bounds");
    for (int k = 0; k < 3; ++k) {
      dual_norm_[k] =
          std::hypot(inverse_[k], inverse_[3 + k], inverse_[6 + k]);
      if (!std::isfinite(dual_norm_[k]))
        throw std::overflow_error("periodic cell inverse is not finite");
    }
  }

  void fold(double &dx, double &dy, double &dz) const {
    if (!std::isfinite(dx) || !std::isfinite(dy) || !std::isfinite(dz))
      throw std::invalid_argument("pair displacement must be finite");
    std::array<long double, 3> reduced{dx, dy, dz};
    std::array<long double, 3> initial{};
    for (int k = 0; k < 3; ++k) {
      if (periodic_[k])
        initial[k] = std::round(reduced[0] * inverse_[k] +
                                reduced[1] * inverse_[3 + k] +
                                reduced[2] * inverse_[6 + k]);
      checkedIndex(initial[k]);
    }
    for (int c = 0; c < 3; ++c)
      for (int k = 0; k < 3; ++k)
        reduced[c] -= initial[k] * cell_[3 * k + c];
    auto best = reduced;
    long double best2 =
        best[0] * best[0] + best[1] * best[1] + best[2] * best[2];
    const long double radius = std::sqrt(best2);
    std::array<std::int64_t, 3> low{}, high{};
    std::uint64_t count = 1;
    for (int k = 0; k < 3; ++k) {
      if (periodic_[k]) {
        const long double f = reduced[0] * inverse_[k] +
                              reduced[1] * inverse_[3 + k] +
                              reduced[2] * inverse_[6 + k];
        const long double extent = radius * dual_norm_[k];
        low[k] = checkedIndex(std::floor(f - extent) - 1.0L);
        high[k] = checkedIndex(std::ceil(f + extent) + 1.0L);
      }
      const auto width = static_cast<std::uint64_t>(high[k]) -
                         static_cast<std::uint64_t>(low[k]) + 1;
      if (width == 0 || count > static_cast<std::uint64_t>(
                                    std::numeric_limits<std::int64_t>::max()) /
                                    width)
        throw std::overflow_error("periodic image search count is too large");
      count *= width;
    }
    for (auto i = low[0]; i <= high[0]; ++i)
      for (auto j = low[1]; j <= high[1]; ++j)
        for (auto k = low[2]; k <= high[2]; ++k) {
          std::array<long double, 3> candidate{};
          for (int c = 0; c < 3; ++c)
            candidate[c] = reduced[c] - i * cell_[c] - j * cell_[3 + c] -
                           k * cell_[6 + c];
          const long double r2 = candidate[0] * candidate[0] +
                                 candidate[1] * candidate[1] +
                                 candidate[2] * candidate[2];
          if (r2 < best2) {
            best = candidate;
            best2 = r2;
          }
        }
    dx = static_cast<double>(best[0]);
    dy = static_cast<double>(best[1]);
    dz = static_cast<double>(best[2]);
  }

private:
  static std::int64_t checkedIndex(long double value) {
    const long double limit = std::ldexp(1.0L, 63);
    if (!std::isfinite(value) || value <= -limit + 1.0L ||
        value >= limit - 1.0L)
      throw std::overflow_error("periodic image index is out of range");
    return static_cast<std::int64_t>(value);
  }
  std::array<long double, 9> cell_{};
  std::array<long double, 9> inverse_{};
  std::array<long double, 3> dual_norm_{};
  std::array<bool, 3> periodic_{};
};

void expectSameImage(const LongDoubleMinimumImage &old_image,
                     const rgpot::nlist::MinimumImage &image, double x,
                     double y, double z, double tol) {
  double ox = x, oy = y, oz = z;
  double nx = x, ny = y, nz = z;
  old_image.fold(ox, oy, oz);
  image.fold(nx, ny, nz);
  CAPTURE(x, y, z, ox, oy, oz, nx, ny, nz);
  const double ro2 = ox * ox + oy * oy + oz * oz;
  const double rn2 = nx * nx + ny * ny + nz * nz;
  const double ro = std::sqrt(ro2);
  const double rn = std::sqrt(rn2);
  const double comp = std::max(
      {std::abs(nx - ox), std::abs(ny - oy), std::abs(nz - oz)});
  if (comp <= tol) {
    // Same lattice image. Its length is the minimum-image contract.
    REQUIRE(rn <= ro + tol);
    return;
  }
  // A second lattice image of the same length sits on the boundary of
  // the Wigner-Seitz cell. Either vector is a minimum image. Accept it
  // only when the lengths match to the ordinary contract, so a longer
  // image cannot hide inside the looser tolerance used for far inputs.
  REQUIRE_THAT(rn2, WithinAbs(ro2, 1e-9 * std::max(1.0, ro2)));
}

} // namespace

TEST_CASE("Double minimum image agrees with the long-double search",
          "[PairList][triclinic]") {
  // 1e-9 is the contract stated on MinimumImage: ordinary cells, skewed
  // cells, and near-ill-conditioned cells that still pass the guard.
  constexpr double kTol = 1e-9;
  const std::array<std::array<double, 9>, 3> cells{{
      {10.0, 0.0, 0.0, 0.0, 10.0, 0.0, 0.0, 0.0, 10.0},
      {4.0, 0.2, 0.1, 0.3, 5.0, -0.2, 0.1, -0.3, 6.0},
      {4.0, 0.0, 0.0, 15.6, 0.5, 0.0, 0.2, 0.3, 5.0},
  }};
  std::mt19937_64 rng(7);
  std::uniform_real_distribution<double> local(-12.0, 12.0);
  std::uniform_real_distribution<double> far(-1.0e8, 1.0e8);
  for (const auto &cell : cells) {
    for (unsigned mask = 0; mask < 8; ++mask) {
      std::array<bool, 3> periodic{};
      for (int k = 0; k < 3; ++k)
        periodic[k] = (mask & (1u << k)) != 0;
      CAPTURE(cell, mask);
      const LongDoubleMinimumImage old_image(cell.data(), periodic);
      const rgpot::nlist::MinimumImage image(cell.data(), periodic);
      for (int n = 0; n < 12; ++n)
        expectSameImage(old_image, image, local(rng), local(rng), local(rng),
                        kTol);
      if (mask == 7) {
        // Far inputs are near 1e8. The double fold tracks the long-double
        // image to a few dozen units in the last place of that input.
        for (int n = 0; n < 8; ++n) {
          const double x = far(rng);
          const double y = far(rng);
          const double z = far(rng);
          const double mag = std::max({std::abs(x), std::abs(y), std::abs(z), 1.0});
          const double far_tol = std::max(
              kTol, 64.0 * std::numeric_limits<double>::epsilon() * mag);
          expectSameImage(old_image, image, x, y, z, far_tol);
        }
      }
    }
    const std::array<bool, 3> periodic{true, true, true};
    const LongDoubleMinimumImage old_image(cell.data(), periodic);
    const rgpot::nlist::MinimumImage image(cell.data(), periodic);
    for (int ix = -2; ix <= 2; ++ix)
      for (int iy = -2; iy <= 2; ++iy)
        for (int iz = -2; iz <= 2; ++iz)
          expectSameImage(old_image, image, ix * 0.5 * cell[0],
                          iy * 0.5 * cell[4], iz * 0.5 * cell[8], kTol);
  }

  // One shear stays on the double path (condition about 5e7). Where
  // long double is wider than double, a second shear sits just inside
  // that guard (condition about 1e9) and takes the double-double search.
  // Where long double is double, the guard is the double guard, so the
  // second shear sits just inside it (condition about 6.7e7).
  // Displacements stay near the short lattice vector so the image box
  // stays small.
  const double near_guard =
      std::numeric_limits<long double>::digits >
              std::numeric_limits<double>::digits
          ? 2.0e-9
          : 3.0e-8;
  const std::array<std::array<double, 9>, 2> shears{{
      {1.0, 1.0, 0.0, 0.0, 4.0e-8, 0.0, 0.0, 0.0, 1.0},
      {1.0, 1.0, 0.0, 0.0, near_guard, 0.0, 0.0, 0.0, 1.0},
  }};
  std::uniform_real_distribution<double> tiny(-1.0e-8, 1.0e-8);
  const std::array<bool, 3> periodic{true, true, true};
  for (const auto &cell : shears) {
    CAPTURE(cell);
    const LongDoubleMinimumImage old_image(cell.data(), periodic);
    const rgpot::nlist::MinimumImage image(cell.data(), periodic);
    for (int n = 0; n < 16; ++n)
      expectSameImage(old_image, image, tiny(rng), tiny(rng), tiny(rng), kTol);
  }
  const double past_guard[9] = {1.0, 1.0, 0.0, 0.0, 5.0e-10, 0.0, 0.0, 0.0,
                                1.0};
  // Non-singular. The released fold Minkowski-reduces a cell this thin
  // and still returns an image, instead of rejecting the search.
  rgpot::nlist::MinimumImage thin(past_guard, periodic);
  double tx = 1.0e-12;
  double ty = 1.0e-12;
  double tz = 1.0e-12;
  thin.fold(tx, ty, tz);
  REQUIRE(std::isfinite(tx));
  REQUIRE(std::isfinite(ty));
  REQUIRE(std::isfinite(tz));
}

TEST_CASE("foldWithin drops only displacements with no image inside the radius",
          "[PairList][triclinic]") {
  // The first cell is the kernel cell of the ASE morse-bulk check.
  const std::array<std::array<double, 9>, 4> cells{{
      {21.6, 0.0, 0.0, 1.728, 21.6, 0.0, 0.864, -1.296, 21.6},
      {4.0, 0.2, 0.1, 0.3, 5.0, -0.2, 0.1, -0.3, 6.0},
      {6.0, 0.0, 0.0, 2.7, 5.4, 0.0, -1.9, 2.2, 5.1},
      {10.0, 0.0, 0.0, 0.0, 10.0, 0.0, 0.0, 0.0, 10.0},
  }};
  std::mt19937_64 rng(5);
  const auto unit = [&rng] {
    return static_cast<double>(rng() >> 11) * 0x1p-53;
  };
  std::uint64_t dropped = 0;
  std::uint64_t kept = 0;
  for (const auto &cell : cells) {
    for (unsigned mask = 1; mask < 8; ++mask) {
      std::array<bool, 3> periodic{};
      for (int k = 0; k < 3; ++k)
        periodic[k] = (mask & (1u << k)) != 0;
      const LongDoubleMinimumImage oracle(cell.data(), periodic);
      const rgpot::nlist::MinimumImage image(cell.data(), periodic);
      for (int t = 0; t < 400; ++t) {
        const double reach = t % 8 == 0 ? 300.0 : 30.0;
        const double d[3] = {reach * (2.0 * unit() - 1.0),
                             reach * (2.0 * unit() - 1.0),
                             reach * (2.0 * unit() - 1.0)};
        const double radius = 0.5 + 12.0 * unit();
        double g[3] = {d[0], d[1], d[2]};
        image.fold(g[0], g[1], g[2]);
        double h[3] = {d[0], d[1], d[2]};
        const bool near = image.foldWithin(h[0], h[1], h[2], radius * radius);
        CAPTURE(cell, mask, d[0], d[1], d[2], radius);
        if (near) {
          ++kept;
          REQUIRE(std::memcmp(g, h, sizeof(g)) == 0);
          continue;
        }
        ++dropped;
        double o[3] = {d[0], d[1], d[2]};
        oracle.fold(o[0], o[1], o[2]);
        REQUIRE(o[0] * o[0] + o[1] * o[1] + o[2] * o[2] > radius * radius);
        REQUIRE(g[0] * g[0] + g[1] * g[1] + g[2] * g[2] > radius * radius);
      }
    }
  }
  REQUIRE(dropped > 0);
  REQUIRE(kept > 0);
}

TEST_CASE("Benchmark minimum image fold", "[.][benchmark]") {
  const double triclinic[9] = {4.0, 0.2, 0.1, 0.3, 5.0, -0.2, 0.1, -0.3, 6.0};
  const std::array<bool, 3> periodic{true, true, true};
  const LongDoubleMinimumImage old_image(triclinic, periodic);
  const rgpot::nlist::MinimumImage image(triclinic, periodic);
  std::mt19937_64 rng(1);
  std::uniform_real_distribution<double> u(-20.0, 20.0);
  constexpr int kN = 20000;
  std::vector<double> disp(3 * kN);
  for (double &v : disp)
    v = u(rng);
  BENCHMARK("long-double fold, 20000 displacements") {
    double sink = 0.0;
    for (int i = 0; i < kN; ++i) {
      double dx = disp[3 * i], dy = disp[3 * i + 1], dz = disp[3 * i + 2];
      old_image.fold(dx, dy, dz);
      sink += dx + dy + dz;
    }
    return sink;
  };
  BENCHMARK("double fold, 20000 displacements") {
    double sink = 0.0;
    for (int i = 0; i < kN; ++i) {
      double dx = disp[3 * i], dy = disp[3 * i + 1], dz = disp[3 * i + 2];
      image.fold(dx, dy, dz);
      sink += dx + dy + dz;
    }
    return sink;
  };

}
