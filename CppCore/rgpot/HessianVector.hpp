#pragma once
// MIT License
// Copyright 2023--present rgpot developers

/**
 * @file HessianVector.hpp
 * @brief Hessian-vector products and dense Hessians of a C++ potential,
 * with a cached centre.
 *
 * The product is @f$H v@f$, with @f$H@f$ the Hessian of the energy with
 * respect to the @f$3N@f$ positions and @f$v@f$ laid out like them. This
 * is the C++ form of @c rgpot_potential_hvp and @c rgpot_potential_hessian,
 * for potentials used without the Rust core (the Python module among
 * them): the same schemes, default step, centre cache and counters.
 *
 * A @c HessianVectorCache keeps the forces of the last centre it
 * evaluated in a @c FusedCache: keyed on the bits of the positions, the
 * atomic numbers, the cell and the @c fusedSettings() of the potential, and
 * holding the outcome of the force call, which is a failure as often as a
 * result. A failed centre is not evaluated again at the same bits; a changed
 * setting of the potential, or @c invalidate(), is the way to retry it.
 * @c k forward products at one centre cost @c k+1 force calls instead of
 * @c 2k. A potential that sets @c caps().analyticHvp and provides
 * @code
 * void hessianVectorImpl(const ForceInput &in, const double *v,
 *                        double *hv) const;
 * @endcode
 * is used directly by @c HvpScheme::Auto and makes no force call.
 *
 * | Scheme   | Force calls per product       | Error          |
 * |----------|-------------------------------|----------------|
 * | Forward  | 1, plus 1 for each new centre | O(h)           |
 * | Central  | 2                             | O(h^2)         |
 * | Analytic | 0                             | the kernel's   |
 * | Ad       | 0                             | rounding only  |
 *
 * @c h is the displacement along the unit direction @f$v/|v|@f$; the
 * difference is scaled back by @f$|v|@f$, so the product scales with
 * @f$|v|@f$; the forward scheme is not linear in @f$v@f$, because its
 * @f$O(h)@f$ term keeps its sign when @f$v@f$ does. The forward error is
 * @f$(h/2) T(u,u) |v|@f$ with @f$T@f$ the third derivative; removing it takes
 * the second evaluation that the central scheme makes.
 *
 * @c HvpScheme::Auto falls back to @c Forward when the potential has no
 * analytic product; @c HvpScheme::Exact never differences and refuses
 * instead. Either way @c apply() returns the scheme that ran and
 * @c derivativeCounts() counts it, so a fallback is always visible.
 * A dense Hessian (@c hessian()) takes an explicit scheme: @c Analytic,
 * @c Ad, @c Exact (analytic, else dual numbers, else refused) or
 * @c Central, which differences the forces and runs only when asked for.
 * @c derivativeCaps() reports which of these a potential has.
 *
 * @ingroup rgpot_cpp
 */

#include <algorithm>
#include <array>
#include <atomic>
#include <bit>
#include <cmath>
#include <concepts>
#include <cstddef>
#include <cstdint>
#include <initializer_list>
#include <limits>
#include <memory>
#include <mutex>
#include <optional>
#include <stdexcept>
#include <string>
#include <utility>
#include <vector>

#include "rgpot/ForceStructs.hpp"
#include "rgpot/FusedCache.hpp"
#include "rgpot/PotHelpers.hpp"
#include "rgpot/FusedSettings.hpp"

namespace rgpot {

/// How a product is computed. The values match @c rgpot_hvp_scheme_t.
enum class HvpScheme : std::int32_t {
  Auto = 0,     //!< Analytic when the potential has it, otherwise Forward.
  Forward = 1,  //!< One displaced force call against the cached centre.
  Central = 2,  //!< Two displaced force calls, one on each side.
  Analytic = 3, //!< The potential's hessianVectorImpl. Refused without it.
  Ad = 4,       //!< The potential's hessianVectorAdImpl. Refused without it.
  Exact = 5,    //!< Analytic, else Ad. Refused when it has neither.
};

/// How a dense Hessian is computed. The values match
/// @c rgpot_hessian_scheme_t and share the numbering of @c HvpScheme; there
/// is no automatic choice, so differencing runs only when asked for.
enum class HessianScheme : std::int32_t {
  Central = 2,  //!< Central differences of the forces, symmetrised.
  Analytic = 3, //!< The potential's hessianImpl. Refused without it.
  Ad = 4,       //!< The potential's hessianAdImpl. Refused without it.
  Exact = 5,    //!< Analytic, else Ad. Refused when it has neither.
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

/// Products and Hessians by the scheme that computed them. The fields match
/// @c rgpot_derivative_counts_t. A product of a zero direction counts under
/// the scheme its request resolved to, so @c hvpAnalytic can exceed
/// @c HvpCounts::analytic, which counts calls of hessianVectorImpl.
struct DerivativeCounts {
  std::uint64_t hvpForward{0};        //!< Products by forward differences.
  std::uint64_t hvpCentral{0};        //!< Products by central differences.
  std::uint64_t hvpAnalytic{0};       //!< Products by hessianVectorImpl.
  std::uint64_t hvpAd{0};             //!< Products from hessianVectorAdImpl.
  std::uint64_t hessianCentral{0};    //!< Hessians by central differences.
  std::uint64_t hessianAnalytic{0};   //!< Hessians from hessianImpl.
  std::uint64_t hessianAd{0};         //!< Hessians from hessianAdImpl.
  std::uint64_t hessianForceCalls{0}; //!< Force calls of central Hessians.
};

/// Which ways a potential has to a product and to a dense Hessian, with the
/// force-call cost of the differenced ones. The fields match
/// @c rgpot_derivative_caps_t. Analytic and dual-number ways make no force
/// call.
struct DerivativeCaps {
  bool hvpAnalytic{false};     //!< caps().analyticHvp.
  bool hvpAd{false};           //!< caps().adHvp.
  bool hvpFd{true};            //!< Forward and central differences.
  bool hessianAnalytic{false}; //!< caps().analyticHessian.
  bool hessianAd{false};       //!< caps().adHessian.
  bool hessianFd{true};        //!< Central differences.
  /// Force calls of a forward product at a cached centre; a new centre
  /// adds one.
  std::uint32_t hvpForwardForceCalls{1};
  std::uint32_t hvpCentralForceCalls{2}; //!< Of a central product.
  /// Of a central Hessian, per atom: two per coordinate.
  std::uint32_t hessianCentralForceCallsPerAtom{6};
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

/// A kernel that reports an analytic Hessian-vector capability bit.
template <typename Pot>
concept AnalyticHessianVector =
    requires(const Pot &pot) {
      { pot.caps().analyticHvp } -> std::convertible_to<bool>;
    };

/// A kernel that implements the analytic Hessian-vector product hook.
template <typename Pot>
concept AnalyticHessianVectorKernel =
    requires(const Pot &pot, const ForceInput &in, const double *v,
             double *hv) { pot.hessianVectorImpl(in, v, hv); };

/// A kernel that reports a dual-number Hessian-vector capability bit.
template <typename Pot>
concept AdHessianVector = requires(const Pot &pot) {
  { pot.caps().adHvp } -> std::convertible_to<bool>;
};

/// A kernel that implements the dual-number Hessian-vector product hook.
template <typename Pot>
concept AdHessianVectorKernel =
    requires(const Pot &pot, const ForceInput &in, const double *v,
             double *hv) { pot.hessianVectorAdImpl(in, v, hv); };

/// A kernel that reports an analytic dense-Hessian capability bit.
template <typename Pot>
concept AnalyticHessian = requires(const Pot &pot) {
  { pot.caps().analyticHessian } -> std::convertible_to<bool>;
};

/// A kernel that implements the analytic dense-Hessian hook.
template <typename Pot>
concept AnalyticHessianKernel = requires(const Pot &pot, const ForceInput &in,
                                         double *h) { pot.hessianImpl(in, h); };

/// A kernel that reports a dual-number dense-Hessian capability bit.
template <typename Pot>
concept AdHessian = requires(const Pot &pot) {
  { pot.caps().adHessian } -> std::convertible_to<bool>;
};

/// A kernel that implements the dual-number dense-Hessian hook.
template <typename Pot>
concept AdHessianKernel = requires(const Pot &pot, const ForceInput &in,
                                   double *h) { pot.hessianAdImpl(in, h); };

/// The derivative capabilities @a pot advertises. A kernel without a
/// @c caps() bit has only the finite differences.
template <typename Pot>
[[nodiscard]] DerivativeCaps derivativeCaps(const Pot &pot) {
  DerivativeCaps c;
  if constexpr (AnalyticHessianVector<Pot>) {
    c.hvpAnalytic = static_cast<bool>(pot.caps().analyticHvp);
  }
  if constexpr (AdHessianVector<Pot>) {
    c.hvpAd = static_cast<bool>(pot.caps().adHvp);
  }
  if constexpr (AnalyticHessian<Pot>) {
    c.hessianAnalytic = static_cast<bool>(pot.caps().analyticHessian);
  }
  if constexpr (AdHessian<Pot>) {
    c.hessianAd = static_cast<bool>(pot.caps().adHessian);
  }
  return c;
}

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
   * @brief Write @f$H v@f$ to @a hv and return the scheme that computed it.
   *
   * @a v and @a hv hold @c 3*in.nAtoms doubles and must not overlap.
   * @a step 0 selects @c kHvpDefaultStep. A zero direction gives a zero
   * product without a call. @c Auto returns @c Analytic or @c Forward,
   * @c Exact returns @c Analytic or @c Ad, and any other scheme returns
   * itself.
   *
   * @throws std::invalid_argument for a malformed request, an unknown
   *         scheme, and @c Analytic, @c Ad or @c Exact on a potential that
   *         lacks what they need.
   * @throws std::domain_error when the kernel returns a non-finite value.
   */
  template <ForceKernel Pot>
  HvpScheme apply(Pot &pot, const ForceInput &in, const double *v, double *hv,
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
    const double h = checkedStep(step);
    scheme = resolve(derivativeCaps(pot), scheme);
    const double length = norm(v, dim);
    if (length == 0.0) {
      std::fill_n(hv, dim, 0.0);
      countProduct(scheme);
      return scheme;
    }
    auto displaced = [&](double sign) {
      std::vector<double> x(g.positions);
      for (std::size_t i = 0; i < dim; ++i) {
        x[i] += sign * h * (v[i] / length);
      }
      return x;
    };
    if (scheme == HvpScheme::Analytic) {
      if constexpr (AnalyticHessianVectorKernel<Pot>) {
        std::as_const(pot).hessianVectorImpl(in, v, hv);
        requireFinite(hv, dim, "hessianVectorImpl");
        analytic_.fetch_add(1, std::memory_order_relaxed);
      } else {
        throw std::invalid_argument(
            "HvpScheme::Analytic: potential advertises analyticHvp but "
            "provides no hessianVectorImpl");
      }
    } else if (scheme == HvpScheme::Ad) {
      if constexpr (AdHessianVectorKernel<Pot>) {
        std::as_const(pot).hessianVectorAdImpl(in, v, hv);
        requireFinite(hv, dim, "hessianVectorAdImpl");
      } else {
        throw std::invalid_argument(
            "HvpScheme::Ad: potential advertises adHvp but provides no "
            "hessianVectorAdImpl");
      }
    } else if (scheme == HvpScheme::Central) {
      const HvpCentre plus = forcesAt(pot, in, displaced(1.0), forceCalls_);
      const HvpCentre minus = forcesAt(pot, in, displaced(-1.0), forceCalls_);
      const double scale = length / (2.0 * h);
      for (std::size_t i = 0; i < dim; ++i) {
        hv[i] = (minus.forces[i] - plus.forces[i]) * scale;
      }
    } else {
      const auto c = centreOf(pot, in, g);
      const HvpCentre plus = forcesAt(pot, in, displaced(1.0), forceCalls_);
      const double scale = length / h;
      for (std::size_t i = 0; i < dim; ++i) {
        hv[i] = (c->forces[i] - plus.forces[i]) * scale;
      }
    }
    countProduct(scheme);
    return scheme;
  }

  /**
   * @brief Write the dense Hessian at @a in to @a h and return the scheme
   * that computed it.
   *
   * @a h holds @f$(3N)^2@f$ doubles, row-major, coordinate @c 3*i+k for
   * axis @c k of atom @c i. @c Exact returns @c Analytic or @c Ad, any other
   * scheme returns itself. @c Central makes @c 6N force calls, two per
   * coordinate at displacements of @f$\pm h@f$ along it (@a step 0 selects
   * @c kHvpDefaultStep), and writes @f$(C + C^T)/2@f$ of the differenced
   * matrix @f$C@f$, so the result is symmetric to the last bit. Those
   * calls bypass the cached centre and count in
   * @c DerivativeCounts::hessianForceCalls, not in @c HvpCounts. On a
   * failure the content of @a h is unspecified.
   *
   * @throws std::invalid_argument for a malformed request, including an
   *         @a h that overlaps the positions, species or cell of @a in, an
   *         unknown scheme, and @c Analytic, @c Ad or @c Exact on a
   *         potential that lacks what they need.
   * @throws std::domain_error when the kernel or the central difference
   *         gives a non-finite value.
   */
  template <ForceKernel Pot>
  HessianScheme hessian(Pot &pot, const ForceInput &in, double *h,
                        HessianScheme scheme, double step = 0.0) {
    const std::size_t dim = coordinates(in);
    if (dim != 0 && h == nullptr) {
      throw std::invalid_argument("hessian: output is null");
    }
    if (dim != 0) {
      if (dim >
          std::numeric_limits<std::size_t>::max() / sizeof(double) / dim) {
        throw std::invalid_argument("hessian: (3*nAtoms)^2 doubles overflow");
      }
      const auto hb = reinterpret_cast<std::uintptr_t>(h);
      const std::uintptr_t hbytes = dim * dim * sizeof(double);
      const auto overlaps = [hb, hbytes](const void *p, std::size_t bytes) {
        const auto pb = reinterpret_cast<std::uintptr_t>(p);
        return p != nullptr && (pb >= hb ? pb - hb < hbytes : hb - pb < bytes);
      };
      if (overlaps(in.pos, dim * sizeof(double)) ||
          overlaps(in.atmnrs, in.nAtoms * sizeof(int)) ||
          overlaps(in.box, 9 * sizeof(double))) {
        throw std::invalid_argument("hessian: output overlaps the input");
      }
    }
    const Geometry g = readGeometry(in);
    const double hstep = checkedStep(step);
    scheme = resolve(derivativeCaps(pot), scheme);
    if (scheme == HessianScheme::Analytic) {
      if constexpr (AnalyticHessianKernel<Pot>) {
        std::as_const(pot).hessianImpl(in, h);
        requireFinite(h, dim * dim, "hessianImpl");
      } else {
        throw std::invalid_argument(
            "HessianScheme::Analytic: potential advertises analyticHessian "
            "but provides no hessianImpl");
      }
      hessianAnalytic_.fetch_add(1, std::memory_order_relaxed);
    } else if (scheme == HessianScheme::Ad) {
      if constexpr (AdHessianKernel<Pot>) {
        std::as_const(pot).hessianAdImpl(in, h);
        requireFinite(h, dim * dim, "hessianAdImpl");
      } else {
        throw std::invalid_argument(
            "HessianScheme::Ad: potential advertises adHessian but provides "
            "no hessianAdImpl");
      }
      hessianAd_.fetch_add(1, std::memory_order_relaxed);
    } else {
      std::vector<double> x(g.positions);
      const double scale = 1.0 / (2.0 * hstep);
      for (std::size_t c = 0; c < dim; ++c) {
        x[c] = g.positions[c] + hstep;
        const HvpCentre plus = forcesAt(pot, in, x, hessianForceCalls_);
        x[c] = g.positions[c] - hstep;
        const HvpCentre minus = forcesAt(pot, in, x, hessianForceCalls_);
        x[c] = g.positions[c];
        for (std::size_t r = 0; r < dim; ++r) {
          h[r * dim + c] = (minus.forces[r] - plus.forces[r]) * scale;
        }
      }
      for (std::size_t r = 0; r < dim; ++r) {
        for (std::size_t c = r + 1; c < dim; ++c) {
          const double mean = 0.5 * (h[r * dim + c] + h[c * dim + r]);
          h[r * dim + c] = mean;
          h[c * dim + r] = mean;
        }
      }
      requireFinite(h, dim * dim, "central Hessian");
      hessianCentral_.fetch_add(1, std::memory_order_relaxed);
    }
    return scheme;
  }

  /// Drop the cached centre. Call it when the potential changes while the
  /// geometry does not.
  void invalidate() { centre_.invalidate(); }

  [[nodiscard]] HvpCounts counts() const noexcept {
    return {products_.load(std::memory_order_relaxed),
            forceCalls_.load(std::memory_order_relaxed),
            centre_.served(),
            analytic_.load(std::memory_order_relaxed)};
  }

  /// Products and Hessians so far, by the scheme that computed them.
  [[nodiscard]] DerivativeCounts derivativeCounts() const noexcept {
    return {hvpForward_.load(std::memory_order_relaxed),
            hvpCentral_.load(std::memory_order_relaxed),
            hvpAnalytic_.load(std::memory_order_relaxed),
            hvpAd_.load(std::memory_order_relaxed),
            hessianCentral_.load(std::memory_order_relaxed),
            hessianAnalytic_.load(std::memory_order_relaxed),
            hessianAd_.load(std::memory_order_relaxed),
            hessianForceCalls_.load(std::memory_order_relaxed)};
  }

  /// Zero every counter, those of @c derivativeCounts() included. The
  /// cached centre stays.
  void resetCounts() noexcept {
    products_.store(0, std::memory_order_relaxed);
    forceCalls_.store(0, std::memory_order_relaxed);
    centre_.resetCounts();
    analytic_.store(0, std::memory_order_relaxed);
    for (auto *counter :
         {&hvpForward_, &hvpCentral_, &hvpAnalytic_, &hvpAd_, &hessianCentral_,
          &hessianAnalytic_, &hessianAd_, &hessianForceCalls_}) {
      counter->store(0, std::memory_order_relaxed);
    }
  }

private:
  struct Geometry {
    std::vector<double> positions;
  };

  static double checkedStep(double step) {
    const double h = step == 0.0 ? kHvpDefaultStep : step;
    if (!(std::isfinite(h) && h > 0.0)) {
      throw std::invalid_argument("step must be positive and finite, got " +
                                  std::to_string(step));
    }
    return h;
  }

  static HvpScheme resolve(const DerivativeCaps &have, HvpScheme scheme) {
    switch (scheme) {
    case HvpScheme::Auto:
      return have.hvpAnalytic ? HvpScheme::Analytic : HvpScheme::Forward;
    case HvpScheme::Forward:
    case HvpScheme::Central:
      return scheme;
    case HvpScheme::Analytic:
      if (!have.hvpAnalytic) {
        throw std::invalid_argument("HvpScheme::Analytic: the potential has no "
                                    "analytic Hessian-vector product");
      }
      return scheme;
    case HvpScheme::Ad:
      if (!have.hvpAd) {
        throw std::invalid_argument("HvpScheme::Ad: the potential has no "
                                    "dual-number Hessian-vector product");
      }
      return scheme;
    case HvpScheme::Exact:
      if (have.hvpAnalytic) {
        return HvpScheme::Analytic;
      }
      if (have.hvpAd) {
        return HvpScheme::Ad;
      }
      throw std::invalid_argument(
          "HvpScheme::Exact: the potential has neither an analytic nor a "
          "dual-number Hessian-vector product");
    }
    throw std::invalid_argument("unknown scheme " +
                                std::to_string(static_cast<int>(scheme)));
  }

  static HessianScheme resolve(const DerivativeCaps &have,
                               HessianScheme scheme) {
    switch (scheme) {
    case HessianScheme::Central:
      return scheme;
    case HessianScheme::Analytic:
      if (!have.hessianAnalytic) {
        throw std::invalid_argument(
            "HessianScheme::Analytic: the potential has no analytic Hessian");
      }
      return scheme;
    case HessianScheme::Ad:
      if (!have.hessianAd) {
        throw std::invalid_argument(
            "HessianScheme::Ad: the potential has no dual-number Hessian");
      }
      return scheme;
    case HessianScheme::Exact:
      if (have.hessianAnalytic) {
        return HessianScheme::Analytic;
      }
      if (have.hessianAd) {
        return HessianScheme::Ad;
      }
      throw std::invalid_argument(
          "HessianScheme::Exact: the potential has neither an analytic nor a "
          "dual-number Hessian");
    }
    throw std::invalid_argument("unknown Hessian scheme " +
                                std::to_string(static_cast<int>(scheme)));
  }

  void countProduct(HvpScheme scheme) noexcept {
    products_.fetch_add(1, std::memory_order_relaxed);
    switch (scheme) {
    case HvpScheme::Forward:
      hvpForward_.fetch_add(1, std::memory_order_relaxed);
      break;
    case HvpScheme::Central:
      hvpCentral_.fetch_add(1, std::memory_order_relaxed);
      break;
    case HvpScheme::Analytic:
      hvpAnalytic_.fetch_add(1, std::memory_order_relaxed);
      break;
    case HvpScheme::Ad:
      hvpAd_.fetch_add(1, std::memory_order_relaxed);
      break;
    default:
      break;
    }
  }

  /// @c 3*in.nAtoms, refused when that many doubles exceed
  /// @c PTRDIFF_MAX bytes, past which pointer arithmetic over them is
  /// undefined.
  static std::size_t coordinates(const ForceInput &in) {
    constexpr auto most =
        static_cast<std::size_t>(std::numeric_limits<std::ptrdiff_t>::max());
    if (in.nAtoms > most / 3 / sizeof(double)) {
      throw std::invalid_argument("positions: 3*nAtoms doubles overflow");
    }
    return 3 * in.nAtoms;
  }

  static Geometry readGeometry(const ForceInput &in) {
    const std::size_t dim = coordinates(in);
    if (dim != 0 && in.pos == nullptr) {
      throw std::invalid_argument("positions: data pointer is null");
    }
    Geometry g;
    g.positions.assign(in.pos, in.pos + dim);
    for (std::size_t i = 0; i < dim; ++i) {
      if (!std::isfinite(in.pos[i])) {
        throw std::invalid_argument("positions: non-finite value at element " +
                                    std::to_string(i));
      }
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
  static FusedSettingsSpec settingsOf(const Pot &pot) {
    if constexpr (ProvidesFusedSettings<Pot>) {
      return pot.fusedSettings();
    } else {
      return {};
    }
  }

  template <ForceKernel Pot>
  std::shared_ptr<const HvpCentre> centreOf(Pot &pot, const ForceInput &in,
                                            const Geometry &g) {
    const FusedKey key = centre_.key(in, settingsOf(pot));
    return centre_
        .evaluate(key,
                  [&] {
                    return std::make_shared<const HvpCentre>(
                        forcesAt(pot, in, g.positions, forceCalls_));
                  })
        .first;
  }

  /// Forces at @a positions, with the species and cell of @a in, counted
  /// in @a calls.
  template <ForceKernel Pot>
  static HvpCentre forcesAt(Pot &pot, const ForceInput &in,
                            const std::vector<double> &positions,
                            std::atomic<std::uint64_t> &calls) {
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
    guardForceCall(moved, &out, kernelPeriodic(pot));
    calls.fetch_add(1, std::memory_order_relaxed);
    pot.forceImpl(moved, &out);
    if (!std::isfinite(out.energy)) {
      throw std::domain_error("forceImpl: energy is not finite");
    }
    requireFinite(r.forces.data(), r.forces.size(), "forceImpl forces");
    r.energy = out.energy;
    r.variance = out.variance;
    return r;
  }

  FusedCache<std::shared_ptr<const HvpCentre>> centre_;
  std::atomic<std::uint64_t> products_{0};
  std::atomic<std::uint64_t> forceCalls_{0};
  std::atomic<std::uint64_t> analytic_{0};
  std::atomic<std::uint64_t> hvpForward_{0};
  std::atomic<std::uint64_t> hvpCentral_{0};
  std::atomic<std::uint64_t> hvpAnalytic_{0};
  std::atomic<std::uint64_t> hvpAd_{0};
  std::atomic<std::uint64_t> hessianCentral_{0};
  std::atomic<std::uint64_t> hessianAnalytic_{0};
  std::atomic<std::uint64_t> hessianAd_{0};
  std::atomic<std::uint64_t> hessianForceCalls_{0};
};

} // namespace rgpot
