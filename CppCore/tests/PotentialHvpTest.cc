// MIT License
// Copyright 2023--present rgpot developers
//
// Hessian-vector products through PotentialHandle and the Rust core: the
// call counts of a product loop, the analytic product that from_impl
// registers, and agreement with HessianVectorCache.

#include <algorithm>
#include <array>
#include <cmath>
#include <cstdint>
#include <limits>
#include <vector>

#include <catch2/catch_all.hpp>

#include "rgpot/HessianVector.hpp"
#include "rgpot/LennardJones/LJClusterPot.hpp"
#include "rgpot/LennardJones/LJPot.hpp"
#include "rgpot/Models/ModelPots.hpp"
#include "rgpot/Morse/MorsePot.hpp"
#include "rgpot/potential.hpp"

using rgpot::ForceInput;
using rgpot::ForceOut;
using rgpot::InputSpec;
using rgpot::LJClusterPot;
using rgpot::LJPot;
using rgpot::MorsePot;
using rgpot::PotentialHandle;

namespace {

/// Morse seen through forceImpl only, with its force calls counted.
struct CountingMorse {
  MorsePot pot;
  mutable std::uint64_t calls = 0;
  void forceImpl(const ForceInput &in, ForceOut *out) const {
    ++calls;
    pot.forceImpl(in, out);
  }
};

struct System {
  std::vector<double> x;
  std::vector<int> z;
  std::vector<double> box{20.0, 0.0, 0.0, 0.0, 20.0, 0.0, 0.0, 0.0, 20.0};
  System() {
    const double s = 2.897 / std::sqrt(2.0);
    const double sites[][3] = {{0, 0, 0}, {s, s, 0}, {s, 0, s}, {0, s, s},
                               {-s, s, 0}, {s, -s, 0}, {0, -s, s}};
    for (std::size_t i = 0; i < 7; ++i) {
      for (std::size_t k = 0; k < 3; ++k) {
        x.push_back(10.0 + sites[i][k] + 0.07 * std::sin(1.3 * double(3 * i + k)));
      }
      z.push_back(78);
    }
  }
  InputSpec input() { return InputSpec(z.size(), x.data(), z.data(), box.data()); }
  ForceInput legacy() const {
    return {.nAtoms = z.size(), .pos = x.data(), .atmnrs = z.data(), .box = box.data()};
  }
};

std::vector<double> direction(std::size_t n) {
  std::vector<double> v(n);
  for (std::size_t k = 0; k < n; ++k) {
    v[k] = std::sin(0.7 * double(k) + 0.4);
  }
  return v;
}

} // namespace

TEST_CASE("k forward products through a handle cost k + 1 force calls", "[hvp]") {
  System s;
  CountingMorse fd;
  auto pot = PotentialHandle::from_impl(fd);
  auto input = s.input();
  const auto v = direction(s.x.size());
  const std::uint64_t k = 10;
  for (std::uint64_t i = 0; i < k; ++i) {
    pot.hvp(input, v);
  }
  CHECK(fd.calls == k + 1);
  const auto counts = pot.hvp_counts();
  CHECK(counts.products == k);
  CHECK(counts.force_calls == k + 1);
  CHECK(counts.centre_reused == k - 1);
  CHECK(counts.analytic == 0);
  CHECK_THROWS_AS(pot.hvp(input, v, 0.0, RGPOT_HVP_ANALYTIC), rgpot::Error);

  pot.reset_hvp_counts();
  CHECK(pot.hvp_counts().products == 0);
  pot.invalidate_hvp();
  pot.hvp(input, v);
  CHECK(pot.hvp_counts().force_calls == 2);
}

TEST_CASE("from_impl registers the analytic product of MorsePot", "[hvp]") {
  System s;
  MorsePot morse;
  auto pot = PotentialHandle::from_impl(morse);
  auto input = s.input();
  const auto v = direction(s.x.size());
  const auto hv = pot.hvp(input, v);
  CHECK(pot.hvp_counts().analytic == 1);
  CHECK(pot.hvp_counts().force_calls == 0);

  std::vector<double> want(v.size());
  morse.hessianVectorImpl(s.legacy(), v.data(), want.data());
  CHECK(hv == want);

  const auto central = pot.hvp(input, v, 0.0, RGPOT_HVP_CENTRAL);
  CHECK(pot.hvp_counts().force_calls == 2);
  for (std::size_t i = 0; i < v.size(); ++i) {
    CHECK_THAT(central[i], Catch::Matchers::WithinAbs(want[i], 1e-5));
  }

  pot.set_hvp_callback(nullptr);
  pot.hvp(input, v);
  CHECK(pot.hvp_counts().analytic == 1);
}

TEST_CASE("from_impl keeps LJClusterPot Auto on finite differences", "[hvp]") {
  LJClusterPot lj;
  auto pot = PotentialHandle::from_impl(lj);
  std::array<double, 9> pos{
      0.0, 0.0, 0.0, 1.2, 0.1, 0.0, 0.3, 1.1, 0.2};
  std::array<int, 3> atmnrs{1, 1, 1};
  std::array<double, 9> box{
      20.0, 0.0, 0.0, 0.0, 20.0, 0.0, 0.0, 0.0, 20.0};
  InputSpec input(atmnrs.size(), pos.data(), atmnrs.data(), box.data());
  const std::vector<double> direction{
      0.2, -0.1, 0.4, -0.3, 0.5, 0.1, 0.3, -0.2, -0.4};
  const double step = std::cbrt(std::numeric_limits<double>::epsilon()) *
                      (1.0 + std::sqrt(static_cast<double>(direction.size())));

  const auto auto_hv = pot.hvp(input, direction, step, RGPOT_HVP_AUTO);
  const auto forward_hv = pot.hvp(input, direction, step, RGPOT_HVP_FORWARD);
  CHECK(auto_hv == forward_hv);
  const auto fine_forward =
      pot.hvp(input, direction, step / 2.0, RGPOT_HVP_FORWARD);
  const auto central = pot.hvp(input, direction, step / 2.0, RGPOT_HVP_CENTRAL);

  const auto counts = pot.hvp_counts();
  CHECK(counts.analytic == 0);
  CHECK(counts.force_calls > 0);
  for (std::size_t i = 0; i < auto_hv.size(); ++i) {
    const double scale =
        std::max(std::abs(auto_hv[i]),
                 std::max(std::abs(fine_forward[i]), std::abs(central[i])));
    // A first-order rule has F_h - D = a h + O(h^2), so its truncation
    // error is twice the change from halving h.
    const double truncation = 2.0 * std::abs(forward_hv[i] - fine_forward[i]);
    const double roundoff =
        std::sqrt(std::numeric_limits<double>::epsilon()) * (1.0 + scale);
    CHECK(std::abs(auto_hv[i] - central[i]) <= truncation + roundoff);
  }
}

TEST_CASE("from_impl registers the analytic product of LJPot", "[hvp]") {
  LJPot lj;
  auto pot = PotentialHandle::from_impl(lj);
  std::array<double, 9> pos{
      0.0, 0.0, 0.0, 1.2, 0.1, 0.0, 0.3, 1.1, 0.2};
  std::array<int, 3> atmnrs{1, 1, 1};
  std::array<double, 9> box{
      20.0, 0.0, 0.0, 0.0, 20.0, 0.0, 0.0, 0.0, 20.0};
  InputSpec input(atmnrs.size(), pos.data(), atmnrs.data(), box.data());
  const std::vector<double> direction{
      0.2, -0.1, 0.4, -0.3, 0.5, 0.1, 0.3, -0.2, -0.4};
  const auto hv = pot.hvp(input, direction, 0.0, RGPOT_HVP_AUTO);
  const auto counts = pot.hvp_counts();
  CHECK(counts.analytic == 1);
  CHECK(counts.force_calls == 0);

  std::vector<double> want(direction.size());
  ForceInput legacy{.nAtoms = atmnrs.size(),
                    .pos = pos.data(),
                    .atmnrs = atmnrs.data(),
                    .box = box.data()};
  lj.hessianVectorImpl(legacy, direction.data(), want.data());
  CHECK(hv == want);
}

TEST_CASE("from_impl registers the analytic product of DoubleWellPot", "[hvp]") {
  rgpot::DoubleWellPot model(2, 1.0, 2.0, 1.0, 0.25);
  auto pot = PotentialHandle::from_impl(model);
  std::array<double, 3> pos{0.7, -0.2, 0.0};
  std::array<int, 1> atmnrs{1};
  std::array<double, 9> box{
      40.0, 0.0, 0.0, 0.0, 40.0, 0.0, 0.0, 0.0, 40.0};
  InputSpec input(atmnrs.size(), pos.data(), atmnrs.data(), box.data());
  const std::vector<double> direction{0.3, -0.4, 0.2};
  const auto hv = pot.hvp(input, direction, 0.0, RGPOT_HVP_AUTO);
  const auto counts = pot.hvp_counts();
  CHECK(counts.analytic == 1);
  CHECK(counts.force_calls == 0);

  std::array<double, 3> want{};
  ForceInput legacy{.nAtoms = atmnrs.size(),
                    .pos = pos.data(),
                    .atmnrs = atmnrs.data(),
                    .box = box.data()};
  model.hessianVectorImpl(legacy, direction.data(), want.data());
  CHECK(hv == std::vector<double>(want.begin(), want.end()));
}

TEST_CASE("The handle and HessianVectorCache agree", "[hvp]") {
  System s;
  CountingMorse through_core;
  CountingMorse direct;
  auto pot = PotentialHandle::from_impl(through_core);
  auto input = s.input();
  const auto v = direction(s.x.size());

  rgpot::HessianVectorCache cache;
  for (auto scheme : {RGPOT_HVP_FORWARD, RGPOT_HVP_CENTRAL}) {
    const auto got = pot.hvp(input, v, 2e-3, scheme);
    std::vector<double> want(v.size());
    cache.apply(direct, s.legacy(), v.data(), want.data(), 2e-3,
                static_cast<rgpot::HvpScheme>(scheme));
    for (std::size_t i = 0; i < v.size(); ++i) {
      CHECK_THAT(got[i], Catch::Matchers::WithinRel(want[i], 1e-12));
    }
  }

  auto centre = pot.hvp_centre(input);
  const auto reference = cache.centre(direct, s.legacy());
  CHECK(centre.energy() == reference.energy);
  CHECK(pot.hvp_counts().centre_reused == 1);
}

TEST_CASE("A changed cell is a new centre for the handle", "[hvp]") {
  System s;
  CountingMorse fd;
  auto pot = PotentialHandle::from_impl(fd);
  const auto v = direction(s.x.size());
  {
    auto input = s.input();
    pot.hvp(input, v);
    pot.hvp(input, v);
  }
  CHECK(fd.calls == 3);
  s.box[4] = 21.0;
  auto input = s.input();
  pot.hvp(input, v);
  CHECK(fd.calls == 5);
}
