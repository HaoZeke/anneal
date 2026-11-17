#pragma once
// MIT License
// Copyright 2023--present rgpot developers

// Neumaier compensated sum. Pair and reciprocal totals run through this
// when many terms of very different magnitude are added.

#include <cmath>

namespace rgpot::lr {

struct Neumaier {
  double sum = 0.0;
  double c = 0.0;

  void add(double x) noexcept {
    const double t = sum + x;
    if (std::abs(sum) >= std::abs(x)) {
      c += (sum - t) + x;
    } else {
      c += (x - t) + sum;
    }
    sum = t;
  }

  void add(const Neumaier &o) noexcept {
    add(o.sum);
    add(o.c);
  }

  [[nodiscard]] double total() const noexcept { return sum + c; }
};

} // namespace rgpot::lr
