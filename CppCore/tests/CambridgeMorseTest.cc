// MIT License
// Copyright 2023--present rgpot developers

#include <catch2/catch_all.hpp>

#include "rgpot/Morse/cambridge_morse.h"

#include <cmath>
#include <cstdint>
#include <cstring>
#include <vector>

namespace {

std::uint64_t bits_of(double value) {
  std::uint64_t out = 0;
  std::memcpy(&out, &value, sizeof(out));
  return out;
}

double reference_morse_energy(const std::vector<double> &x, double rho) {
  const std::size_t n = x.size() / 3;
  double e = 0.0;
  for (std::size_t i = 0; i < n; ++i) {
    for (std::size_t j = i + 1; j < n; ++j) {
      const double d0 = x[3 * i] - x[3 * j];
      const double d1 = x[3 * i + 1] - x[3 * j + 1];
      const double d2 = x[3 * i + 2] - x[3 * j + 2];
      const double r2 = d0 * d0 + d1 * d1 + d2 * d2;
      if (!(r2 > 0.0)) {
        continue;
      }
      const double a = std::exp(rho * (1.0 - std::sqrt(r2)));
      e += a * (a - 2.0);
    }
  }
  return e;
}

} // namespace

TEST_CASE("Cambridge Morse dimer sits at unit depth with no force", "[morse][cambridge]") {
  const double x[6] = {0.0, 0.0, 0.0, 1.0, 0.0, 0.0};
  double energy = 0.0;
  double gradient[6] = {1.0, 1.0, 1.0, 1.0, 1.0, 1.0};
  rgpot_cambridge_morse(x, 2, 6.0, &energy, gradient);
  REQUIRE(bits_of(energy) == bits_of(-1.0));
  for (double component : gradient) {
    REQUIRE(component == 0.0);
  }
}

TEST_CASE("Cambridge Morse keeps the distant pair", "[morse][cambridge]") {
  // Atoms at 0, 1 and 6. The long pair is outside any contact cutoff and
  // still moves the all-pairs sum.
  const double x[9] = {0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 6.0, 0.0, 0.0};
  double full = 0.0;
  double gradient[9];
  rgpot_cambridge_morse(x, 3, 3.0, &full, gradient);
  const double near[6] = {0.0, 0.0, 0.0, 1.0, 0.0, 0.0};
  double contact = 0.0;
  double unused[6];
  rgpot_cambridge_morse(near, 2, 3.0, &contact, unused);
  REQUIRE(bits_of(full) != bits_of(contact));
  REQUIRE(bits_of(full) == bits_of(reference_morse_energy({x, x + 9}, 3.0)));
}

TEST_CASE("cluster Lennard-Jones dimer is minus one at the minimum", "[lj][cambridge]") {
  const double r = std::pow(2.0, 1.0 / 6.0);
  const double x[6] = {0.0, 0.0, 0.0, r, 0.0, 0.0};
  double energy = 0.0;
  double gradient[6];
  rgpot_cluster_lj(x, 2, &energy, gradient);
  REQUIRE(energy == Catch::Approx(-1.0).margin(1e-12));
  for (double component : gradient) {
    REQUIRE(std::abs(component) < 1e-9);
  }
}

TEST_CASE("Cambridge Morse at 80 atoms", "[.][benchmark][cambridge]") {
  constexpr std::size_t n = 80;
  std::vector<double> x(3 * n);
  std::uint64_t state = 1;
  for (double &value : x) {
    state = state * 6364136223846793005ULL + 1;
    value = (static_cast<double>(state >> 33) / static_cast<double>(1ULL << 31) - 0.5) * 8.0;
  }
  std::vector<double> gradient(3 * n);
  double energy = 0.0;
  BENCHMARK("all-pairs Morse, 80 atoms") {
    rgpot_cambridge_morse(x.data(), n, 3.0, &energy, gradient.data());
    return energy;
  };
}
