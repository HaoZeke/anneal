#pragma once
// MIT License
// Copyright 2023--present rgpot developers

#include <algorithm>
#include <array>
#include <cstddef>
#include <stdexcept>
#include <string>

namespace rgpot {

/// The six parameter arrays of the Muller-Brown surface.
/// A default-constructed value holds the published parameters.
struct MullerBrownConfig {
  std::array<double, 4> amplitude{-200.0, -100.0, -170.0, 15.0};
  std::array<double, 4> a{-1.0, -1.0, -6.5, 0.7};
  std::array<double, 4> b{0.0, 0.0, 11.0, 0.6};
  std::array<double, 4> c{-10.0, -10.0, -6.5, 0.7};
  std::array<double, 4> x0{1.0, 0.0, -0.5, -1.0};
  std::array<double, 4> y0{0.0, 0.5, 1.5, 1.0};
};

namespace detail {

template <class List>
void readMullerBrownList(const List &source, std::array<double, 4> &target,
                         const char *name) {
  if (source.size() == 0)
    return;
  if (source.size() != target.size())
    throw std::invalid_argument(std::string("Muller-Brown ") + name +
                                " must contain four values");
  std::copy(source.begin(), source.end(), target.begin());
}

} // namespace detail

/// The one reader of Muller-Brown parameter lists, shared by the RPC configure
/// path, the generic engine and the Python constructor.
/// It starts from the published defaults and never from a live potential, so
/// the result depends on these six lists alone: an empty list leaves the
/// published default, four values replace it, any other length throws
/// std::invalid_argument.
/// The List type needs size(), begin() and end().
template <class List>
[[nodiscard]] MullerBrownConfig
mullerBrownConfigFromLists(const List &amplitude, const List &a, const List &b,
                           const List &c, const List &x0, const List &y0) {
  MullerBrownConfig config;
  detail::readMullerBrownList(amplitude, config.amplitude, "amplitude");
  detail::readMullerBrownList(a, config.a, "a");
  detail::readMullerBrownList(b, config.b, "b");
  detail::readMullerBrownList(c, config.c, "c");
  detail::readMullerBrownList(x0, config.x0, "x0");
  detail::readMullerBrownList(y0, config.y0, "y0");
  return config;
}

} // namespace rgpot
