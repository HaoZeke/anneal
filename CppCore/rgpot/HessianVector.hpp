#pragma once
// MIT License
// Copyright 2023--present rgpot developers

/**
 * @file HessianVector.hpp
 * @brief Hessian-vector products of a C++ potential with a cached centre.
 *
 * The product is @f$H v@f$, with @f$H@f$ the Hessian of the energy with
 * respect to the @f$3N@f$ positions and @f$v@f$ laid out like them. This
 * is the C++ form of @c rgpot_potential_hvp, for potentials used without
 * the Rust core (the Python module among them): the same schemes, default
 * step, centre cache and counters.
 *
 * A @c HessianVectorCache keeps the forces of the last centre it
 * evaluated, keyed on the bits of the positions, atomic numbers and cell.
 * @c k forward products at one centre cost @c k+1 force calls instead of
 * @c 2k. A potential with a member
 * @code
 * void hessianVectorImpl(const ForceInput &in, const double *v,
 *                        double *hv) const;
 * @endcode
 * is used directly by @c HvpScheme::Auto and makes no force call.
 *
 * | Scheme   | Force calls per product       | Error        |
 * |----------|-------------------------------|--------------|
 * | Forward  | 1, plus 1 for each new centre | O(h)         |
 * | Central  | 2                             | O(h^2)       |
 * | Analytic | 0                             | the kernel's |
 *
 * @c h is the displacement along the unit direction @f$v/|v|@f$; the
 * difference is scaled back by @f$|v|@f$, so the product scales with
 * @f$|v|@f$; the forward scheme is not linear in @f$v@f$, because its
 * @f$O(h)@f$ term keeps its sign when @f$v@f$ does. The forward error is @f$(h/2) T(u,u) |v|@f$ with @f$T@f$ the
 * third derivative; removing it takes the second evaluation that the
 * central scheme makes.
 *
 * @ingroup rgpot_cpp
 */

#include <algorithm>
#include <array>
#include <atomic>
#include <bit>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <memory>
#include <mutex>
#include <optional>
#include <stdexcept>
#include <string>
#include <utility>
#include <vector>

#include "rgpot/ForceStructs.hpp"

namespace rgpot {

/// How a product is computed. The values match @c rgpot_hvp_scheme_t.
enum class HvpScheme : std::int32_t {
  Auto = 0,     //!< Analytic when the potential has it, otherwise Forward.
  Forward = 1,  //!< One displaced force call against the cached centre.
  Central = 2,  //!< Two displaced force calls, one on each side.
  Analytic = 3, //!< The potential's hessianVectorImpl. Refused without it.
};

/// Displacement length used when a caller passes a step of 0.
inline constexpr double kHvpDefaultStep = 1.0e-3;

/// Work done by a cache. The fields match @c rgpot_hvp_counts_t.
struct HvpCounts {
  std::uint64_t products{0};     //!< Products answered.
  std::uint64_t forceCalls{0};   //!< Force evaluations made.
  std::uint64_t centreReused{0}; //!< Requests served from the centre.
  std::uint64_t analytic{0};     //!< Products from hessianVectorImpl.
};

/// Energy, variance and forces (@c 3*nAtoms, row-major) at a centre.
struct HvpCentre {
  double energy{0.0};
  double variance{0.0};
  std::vector<double> forces;
};

/// A kernel with the force entry point every rgpot potential has.
template <typename Pot>
concept ForceKernel = requires(Pot &pot, const ForceInput &in, ForceOut *out) {
  pot.forceImpl(in, out);
};

/// A kernel that also knows its Hessian-vector product.
template <typename Pot>
concept AnalyticHessianVector =
    requires(const Pot &pot, const ForceInput &in, const double *v, double *hv) {
      pot.hessianVectorImpl(in, v, hv);
    };

/**
 * @class HessianVectorCache
 * @brief Centre forces and counters for Hessian-vector products.
 *
 * Safe to share between threads; the lock covers the cached centre, not
 * the force calls. The potential passed to each call must be the one the
 * cache was filled with, or @c invalidate() must be called in between.
 */
class HessianVectorCache {
public:
  HessianVectorCache() = default;
  HessianVectorCache(const HessianVectorCache &) = delete;
  HessianVectorCache &operator=(const HessianVectorCache &) = delete;

  /**
   * @brief Energy and forces at @a in.
   *
   * Served from the cached centre when its positions, species and cell
   * carry the same bits, otherwise one force call that becomes the centre.
   * A search that needs the centre forces asks here, so the products that
   * follow reuse them.
   */
  template <ForceKernel Pot>
  HvpCentre centre(Pot &pot, const ForceInput &in) {
    const Geometry g = readGeometry(in);
    return *centreOf(pot, in, g);
  }

  /**
   * @brief Write @f$H v@f$ to @a hv.
   *
   * @a v and @a hv hold @c 3*in.nAtoms doubles and must not overlap.
   * @a step 0 selects @c kHvpDefaultStep. A zero direction gives a zero
   * product without a call.
   *
   * @throws std::invalid_argument for a malformed request.
   * @throws std::domain_error when the kernel returns a non-finite value.
   */
  template <ForceKernel Pot>
  void apply(Pot &pot, const ForceInput &in, const double *v, double *hv,
             double step = 0.0, HvpScheme scheme = HvpScheme::Auto) {
    const Geometry g = readGeometry(in);
    const std::size_t dim = g.positions.size();
    if (dim != 0 && (v == nullptr || hv == nullptr)) {
      throw std::invalid_argument("hessianVector: direction or output is null");
    }
    if (dim != 0) {
      const auto vb = reinterpret_cast<std::uintptr_t>(v);
      const auto hb = reinterpret_cast<std::uintptr_t>(hv);
      const std::uintptr_t bytes = dim * sizeof(double);
      if (vb < hb + bytes && hb < vb + bytes) {
        throw std::invalid_argument(
            "hessianVector: direction and output overlap");
      }
    }
    for (std::size_t i = 0; i < dim; ++i) {
      if (!std::isfinite(v[i])) {
        throw std::invalid_argument("direction: non-finite value at element " +
                                    std::to_string(i));
      }
    }
    const double h = step == 0.0 ? kHvpDefaultStep : step;
    if (!(std::isfinite(h) && h > 0.0)) {
      throw std::invalid_argument("step must be positive and finite, got " +
                                  std::to_string(step));
    }
    if (scheme == HvpScheme::Auto) {
      scheme = AnalyticHessianVector<Pot> ? HvpScheme::Analytic
                                          : HvpScheme::Forward;
    }
    if (scheme != HvpScheme::Forward && scheme != HvpScheme::Central &&
        scheme != HvpScheme::Analytic) {
      throw std::invalid_argument("unknown scheme " +
                                  std::to_string(static_cast<int>(scheme)));
    }
    if (scheme == HvpScheme::Analytic && !AnalyticHessianVector<Pot>) {
      throw std::invalid_argument(
          "HvpScheme::Analytic: the potential has no hessianVectorImpl");
    }
    const double length = norm(v, dim);
    if (length == 0.0) {
      std::fill_n(hv, dim, 0.0);
      products_.fetch_add(1, std::memory_order_relaxed);
      return;
    }
    auto displaced = [&](double sign) {
      std::vector<double> x(g.positions);
      for (std::size_t i = 0; i < dim; ++i) {
        x[i] += sign * h * (v[i] / length);
      }
      return x;
    };
    if (scheme == HvpScheme::Analytic) {
      if constexpr (AnalyticHessianVector<Pot>) {
        std::as_const(pot).hessianVectorImpl(in, v, hv);
        requireFinite(hv, dim, "hessianVectorImpl");
        analytic_.fetch_add(1, std::memory_order_relaxed);
      }
    } else if (scheme == HvpScheme::Central) {
      const HvpCentre plus = forcesAt(pot, in, displaced(1.0));
      const HvpCentre minus = forcesAt(pot, in, displaced(-1.0));
      const double scale = length / (2.0 * h);
      for (std::size_t i = 0; i < dim; ++i) {
        hv[i] = (minus.forces[i] - plus.forces[i]) * scale;
      }
    } else {
      const auto c = centreOf(pot, in, g);
      const HvpCentre plus = forcesAt(pot, in, displaced(1.0));
      const double scale = length / h;
      for (std::size_t i = 0; i < dim; ++i) {
        hv[i] = (c->forces[i] - plus.forces[i]) * scale;
      }
    }
    products_.fetch_add(1, std::memory_order_relaxed);
  }

  /// Drop the cached centre. Call it when the potential changes while the
  /// geometry does not.
  void invalidate() {
    std::lock_guard<std::mutex> lock(mu_);
    centre_.reset();
  }

  [[nodiscard]] HvpCounts counts() const noexcept {
    return {products_.load(std::memory_order_relaxed),
            forceCalls_.load(std::memory_order_relaxed),
            centreReused_.load(std::memory_order_relaxed),
            analytic_.load(std::memory_order_relaxed)};
  }

  /// Zero the counters. The cached centre stays.
  void resetCounts() noexcept {
    products_.store(0, std::memory_order_relaxed);
    forceCalls_.store(0, std::memory_order_relaxed);
    centreReused_.store(0, std::memory_order_relaxed);
    analytic_.store(0, std::memory_order_relaxed);
  }

private:
  /// @c std::bit_cast of every coordinate and cell entry: float equality
  /// would serve the forces of +0.0 at -0.0.
  struct Key {
    std::vector<std::uint64_t> positions;
    std::optional<std::vector<int>> atomicNumbers;
    std::optional<std::array<std::uint64_t, 9>> box;
    bool operator==(const Key &) const = default;
  };

  struct Geometry {
    std::vector<double> positions;
    Key key;
  };

  struct Cached {
    Key key;
    std::shared_ptr<const HvpCentre> centre;
  };

  static Geometry readGeometry(const ForceInput &in) {
    const std::size_t dim = 3 * in.nAtoms;
    if (dim != 0 && in.pos == nullptr) {
      throw std::invalid_argument("positions: data pointer is null");
    }
    Geometry g;
    g.positions.assign(in.pos, in.pos + dim);
    g.key.positions.reserve(dim);
    for (std::size_t i = 0; i < dim; ++i) {
      if (!std::isfinite(in.pos[i])) {
        throw std::invalid_argument("positions: non-finite value at element " +
                                    std::to_string(i));
      }
      g.key.positions.push_back(std::bit_cast<std::uint64_t>(in.pos[i]));
    }
    if (in.atmnrs != nullptr) {
      g.key.atomicNumbers.emplace(in.atmnrs, in.atmnrs + in.nAtoms);
    }
    if (in.box != nullptr) {
      std::array<std::uint64_t, 9> bits{};
      for (std::size_t i = 0; i < 9; ++i) {
        bits[i] = std::bit_cast<std::uint64_t>(in.box[i]);
      }
      g.key.box = bits;
    }
    return g;
  }

  /// Euclidean norm, scaled so that large or tiny entries neither
  /// overflow nor underflow the sum of squares.
  static double norm(const double *v, std::size_t n) {
    double scale = 0.0;
    for (std::size_t i = 0; i < n; ++i) {
      scale = std::max(scale, std::abs(v[i]));
    }
    if (scale == 0.0) {
      return 0.0;
    }
    double sum = 0.0;
    for (std::size_t i = 0; i < n; ++i) {
      const double x = v[i] / scale;
      sum += x * x;
    }
    return scale * std::sqrt(sum);
  }

  static void requireFinite(const double *x, std::size_t n, const char *what) {
    for (std::size_t i = 0; i < n; ++i) {
      if (!std::isfinite(x[i])) {
        throw std::domain_error(std::string(what) +
                                ": non-finite value at element " +
                                std::to_string(i));
      }
    }
  }

  template <ForceKernel Pot>
  std::shared_ptr<const HvpCentre> centreOf(Pot &pot, const ForceInput &in,
                                            const Geometry &g) {
    {
      std::lock_guard<std::mutex> lock(mu_);
      if (centre_ && centre_->key == g.key) {
        centreReused_.fetch_add(1, std::memory_order_relaxed);
        return centre_->centre;
      }
    }
    auto c = std::make_shared<const HvpCentre>(forcesAt(pot, in, g.positions));
    std::lock_guard<std::mutex> lock(mu_);
    centre_ = Cached{g.key, c};
    return c;
  }

  /// Forces at @a positions, with the species and cell of @a in.
  template <ForceKernel Pot>
  HvpCentre forcesAt(Pot &pot, const ForceInput &in,
                     const std::vector<double> &positions) {
    HvpCentre r;
    r.forces.assign(positions.size(), 0.0);
    ForceInput moved{.nAtoms = in.nAtoms,
                     .pos = positions.data(),
                     .atmnrs = in.atmnrs,
                     .box = in.box};
    ForceOut out{.F = r.forces.data(),
                 .energy = 0.0,
                 .variance = 0.0,
                 .stress = {},
                 .has_stress = 0};
    forceCalls_.fetch_add(1, std::memory_order_relaxed);
    pot.forceImpl(moved, &out);
    if (!std::isfinite(out.energy)) {
      throw std::domain_error("forceImpl: energy is not finite");
    }
    requireFinite(r.forces.data(), r.forces.size(), "forceImpl forces");
    r.energy = out.energy;
    r.variance = out.variance;
    return r;
  }

  mutable std::mutex mu_;
  std::optional<Cached> centre_;
  std::atomic<std::uint64_t> products_{0};
  std::atomic<std::uint64_t> forceCalls_{0};
  std::atomic<std::uint64_t> centreReused_{0};
  std::atomic<std::uint64_t> analytic_{0};
};

} // namespace rgpot
