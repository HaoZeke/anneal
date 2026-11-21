#pragma once
// MIT License
// Copyright 2023--present rgpot developers

/**
 * @file Dual.hpp
 * @brief Forward-mode dual numbers for kernels written over a scalar type.
 *
 * A @c Dual<T> carries a value and one directional derivative. Evaluating
 * @f$f@f$ at @f$x + \varepsilon@f$ with @f$\varepsilon^2 = 0@f$ gives
 * @f$f(x) + f'(x)\,\varepsilon@f$: the derivative is exact up to the
 * rounding of the arithmetic, with no step and no truncation error.
 * Nesting gives second derivatives. @c seed2(x) is
 * @f$x + \varepsilon_1 + \varepsilon_2@f$, and for @c y = f(seed2(x)),
 * @c y.v.v is @f$f@f$, @c y.v.d and @c y.d.v are @f$f'@f$ and @c y.d.d is
 * @f$f''@f$.
 *
 * A kernel takes part by being a template over its scalar type, so the
 * @c double instantiation and the dual one are the same source. A dual
 * has no ordering operators: a branch compares @c value(x), takes the
 * branch the @c double evaluation takes, and the derivative is that of the
 * branch taken. A kernel that branches on a value its derivative depends on
 * (a clamp, a cutoff) is differentiated piecewise, as a closed-form
 * derivative of the same code would be.
 *
 * @code
 * const auto d = rgpot::ad::secondDerivative(
 *     [](auto x) { return x * x * x; }, 2.0);
 * // d.value 8, d.first 12, d.second 12
 * @endcode
 *
 * @ingroup rgpot_cpp
 */

#include <cmath>

namespace rgpot::ad {

/// Value @c v and derivative @c d along the seeded direction.
template <typename T> struct Dual {
  T v{};
  T d{};

  constexpr Dual() = default;
  /// A constant: derivative zero. Implicit, so constants mix with duals.
  constexpr Dual(double c)
      : v(c), d(0.0) {} // NOLINT(google-explicit-constructor)
  constexpr Dual(const T &value, const T &tangent) : v(value), d(tangent) {}

  friend constexpr Dual operator-(const Dual &a) { return {-a.v, -a.d}; }

  friend constexpr Dual operator+(const Dual &a, const Dual &b) {
    return {a.v + b.v, a.d + b.d};
  }
  friend constexpr Dual operator-(const Dual &a, const Dual &b) {
    return {a.v - b.v, a.d - b.d};
  }
  friend constexpr Dual operator*(const Dual &a, const Dual &b) {
    return {a.v * b.v, a.v * b.d + a.d * b.v};
  }
  friend constexpr Dual operator/(const Dual &a, const Dual &b) {
    const T inv = 1.0 / b.v;
    const T q = a.v * inv;
    return {q, (a.d - q * b.d) * inv};
  }

  friend constexpr Dual operator+(const Dual &a, double c) {
    return {a.v + c, a.d};
  }
  friend constexpr Dual operator+(double c, const Dual &a) {
    return {c + a.v, a.d};
  }
  friend constexpr Dual operator-(const Dual &a, double c) {
    return {a.v - c, a.d};
  }
  friend constexpr Dual operator-(double c, const Dual &a) {
    return {c - a.v, -a.d};
  }
  friend constexpr Dual operator*(const Dual &a, double c) {
    return {a.v * c, a.d * c};
  }
  friend constexpr Dual operator*(double c, const Dual &a) {
    return {c * a.v, c * a.d};
  }
  friend constexpr Dual operator/(const Dual &a, double c) {
    return {a.v / c, a.d / c};
  }
  friend constexpr Dual operator/(double c, const Dual &a) {
    const T inv = 1.0 / a.v;
    const T q = c * inv;
    return {q, -(q * inv) * a.d};
  }
};

/// Dual with second derivatives: @c Dual<Dual<double>>.
using Dual2 = Dual<Dual<double>>;

/// The @c double a scalar or a dual carries, for branches and output.
constexpr double value(double x) noexcept { return x; }
template <typename T> constexpr double value(const Dual<T> &x) noexcept {
  return value(x.v);
}

template <typename T> Dual<T> sqrt(const Dual<T> &x) {
  using std::sqrt;
  const T s = sqrt(x.v);
  return {s, x.d / (2.0 * s)};
}

template <typename T> Dual<T> exp(const Dual<T> &x) {
  using std::exp;
  const T e = exp(x.v);
  return {e, e * x.d};
}

template <typename T> Dual<T> log(const Dual<T> &x) {
  using std::log;
  return {log(x.v), x.d / x.v};
}

/// @f$x + \varepsilon_1 + \varepsilon_2@f$, the seed of a second derivative.
constexpr Dual2 seed2(double x) noexcept {
  return {Dual<double>{x, 1.0}, Dual<double>{1.0, 0.0}};
}

/// A value and its first two derivatives.
struct Derivatives2 {
  double value;
  double first;
  double second;
};

/// @f$f(x)@f$, @f$f'(x)@f$ and @f$f''(x)@f$ from one evaluation of @a f on
/// @c Dual2. @a f is a template over its scalar type, a generic lambda or a
/// function template.
template <typename F> Derivatives2 secondDerivative(F &&f, double x) {
  const Dual2 y = f(seed2(x));
  return {y.v.v, y.v.d, y.d.d};
}

} // namespace rgpot::ad
