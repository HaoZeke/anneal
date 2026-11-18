// MIT License
// Copyright 2023--present rgpot developers
//
// rgpot::SumPot: weights, signs, switches, the component record, the log,
// the caps intersection and failure propagation.

#include <array>
#include <atomic>
#include <cmath>
#include <limits>
#include <memory>
#include <stdexcept>
#include <string>
#include <vector>

#include <catch2/catch_all.hpp>

#include "rgpot/LennardJones/LJPot.hpp"
#include "rgpot/Morse/MorsePot.hpp"
#include "rgpot/SumPot/SumPot.hpp"

using Catch::Matchers::ContainsSubstring;
using Catch::Matchers::WithinAbs;
using Catch::Matchers::WithinRel;

namespace {

struct Eval {
  double energy = 0.0;
  double variance = 0.0;
  std::vector<double> f;
  bool has_stress = false;
  std::array<double, 9> stress{};
};

Eval run(rgpot::PotentialBase &pot, const std::vector<double> &pos,
         const std::vector<int> &z, const std::array<double, 9> &box) {
  Eval e;
  e.f.assign(pos.size(), 0.0);
  rgpot::ForceInput in{z.size(), pos.data(), z.data(), box.data()};
  rgpot::ForceOut out{};
  out.F = e.f.data();
  pot.forceBatch(rgpot::ForceBatch{.nSystems = 1, .in = &in, .out = &out});
  e.energy = out.energy;
  e.variance = out.variance;
  e.has_stress = out.has_stress != 0;
  std::copy(out.stress, out.stress + 9, e.stress.begin());
  return e;
}

/// Counts its calls; a configurable constant energy, force and stress.
class Probe final : public rgpot::Potential<Probe> {
public:
  explicit Probe(std::shared_ptr<std::atomic<int>> calls, rgpot::PotCaps caps = {},
                 bool fail = false)
      : rgpot::Potential<Probe>(rgpot::PotType::UNKNOWN), calls_(std::move(calls)),
        caps_(caps), fail_(fail) {}
  void forceImpl(const rgpot::ForceInput &in, rgpot::ForceOut *out) const override {
    calls_->fetch_add(1);
    if (fail_) {
      throw std::domain_error("probe failure");
    }
    out->energy = 0.25;
    out->variance = 0.5;
    for (std::size_t i = 0; i < in.nAtoms; ++i) {
      out->F[3 * i + 1] += 0.125;
    }
    if (caps_.stress) {
      out->stress[0] = 0.0625;
      out->has_stress = 1;
    }
  }
  [[nodiscard]] rgpot::PotCaps caps() const noexcept override { return caps_; }

private:
  std::shared_ptr<std::atomic<int>> calls_;
  rgpot::PotCaps caps_;
  bool fail_;
};

const std::vector<double> kPos = {0.0, 0.0, 0.0, 1.1, 0.1, 0.0, 0.3, 1.2, 0.2};
const std::vector<int> kZ = {18, 18, 18};
const std::array<double, 9> kBox = {9.0, 0.0, 0.0, 0.0, 9.5, 0.0, 0.0, 0.0, 10.0};

rgpot::LJConfig lj() { return {.u0 = 1.0, .cutoff = 4.0, .psi = 1.0}; }
rgpot::MorseConfig morse() {
  return {.De = 0.7102, .a = 1.6047, .re = 2.897, .cutoff = 4.0};
}

constexpr double kTight = 1e-12;

} // namespace

TEST_CASE("SumPot adds weighted energies, forces and stress", "[SumPot]") {
  rgpot::LJPot a(lj());
  rgpot::MorsePot b(morse());
  const Eval ea = run(a, kPos, kZ, kBox);
  const Eval eb = run(b, kPos, kZ, kBox);
  REQUIRE(ea.has_stress);
  REQUIRE(eb.has_stress);

  rgpot::SumPot sum;
  sum.add({.name = "lj", .weight = 2.0}, std::make_unique<rgpot::LJPot>(lj()));
  sum.add({.name = "morse", .weight = -0.5}, std::make_unique<rgpot::MorsePot>(morse()));
  const Eval es = run(sum, kPos, kZ, kBox);
  REQUIRE_THAT(es.energy, WithinAbs(2.0 * ea.energy - 0.5 * eb.energy, kTight));
  for (std::size_t i = 0; i < kPos.size(); ++i) {
    REQUIRE_THAT(es.f[i], WithinAbs(2.0 * ea.f[i] - 0.5 * eb.f[i], kTight));
  }
  REQUIRE(es.has_stress);
  for (int k = 0; k < 9; ++k) {
    REQUIRE_THAT(es.stress[k], WithinAbs(2.0 * ea.stress[k] - 0.5 * eb.stress[k], kTight));
  }

  const auto &c = sum.lastComponents();
  REQUIRE(c.size() == 2);
  REQUIRE(c[0].name == "lj");
  REQUIRE(c[1].name == "morse");
  REQUIRE(c[0].evaluated);
  REQUIRE(c[1].weight == -0.5);
  REQUIRE_THAT(c[0].energy, WithinAbs(ea.energy, kTight));
  REQUIRE_THAT(c[1].energy, WithinAbs(eb.energy, kTight));
  REQUIRE(c[1].forces.size() == kPos.size());
  for (std::size_t i = 0; i < kPos.size(); ++i) {
    REQUIRE_THAT(c[0].forces[i], WithinAbs(ea.f[i], kTight));
    REQUIRE_THAT(c[1].forces[i], WithinAbs(eb.f[i], kTight));
  }
  REQUIRE(c[1].has_stress);
}

TEST_CASE("SumPot weight -1 cancels the same term", "[SumPot]") {
  rgpot::SumPot sum;
  sum.add({.name = "plus"}, std::make_unique<rgpot::LJPot>(lj()));
  sum.add({.name = "minus", .weight = -1.0}, std::make_unique<rgpot::LJPot>(lj()));
  const Eval es = run(sum, kPos, kZ, kBox);
  REQUIRE(es.energy == 0.0);
  for (double f : es.f) {
    REQUIRE(f == 0.0);
  }
  for (double s : es.stress) {
    REQUIRE(s == 0.0);
  }
}

TEST_CASE("SumPot never evaluates a disabled term", "[SumPot]") {
  auto calls = std::make_shared<std::atomic<int>>(0);
  auto failing = std::make_shared<std::atomic<int>>(0);
  rgpot::SumPot sum;
  sum.add({.name = "probe"}, std::make_unique<Probe>(calls, rgpot::PotCaps{}));
  sum.add({.name = "off", .enabled = false},
          std::make_unique<Probe>(failing, rgpot::PotCaps{}, true));
  const Eval es = run(sum, kPos, kZ, kBox);
  REQUIRE(calls->load() == 1);
  REQUIRE(failing->load() == 0);
  REQUIRE_THAT(es.energy, WithinAbs(0.25, kTight));
  const auto &c = sum.lastComponents();
  REQUIRE(c.size() == 2);
  REQUIRE_FALSE(c[1].enabled);
  REQUIRE_FALSE(c[1].evaluated);
  REQUIRE(c[1].forces.empty());

  sum.setEnabled(0, false);
  const Eval none = run(sum, kPos, kZ, kBox);
  REQUIRE(calls->load() == 1);
  REQUIRE(none.energy == 0.0);
  REQUIRE_FALSE(none.has_stress);
  REQUIRE_FALSE(sum.caps().stress);
}

TEST_CASE("an empty or fully disabled sum does not advertise stress", "[SumPot]") {
  rgpot::SumPot empty;
  REQUIRE_FALSE(empty.caps().stress);
  const Eval bare = run(empty, kPos, kZ, kBox);
  REQUIRE(bare.energy == 0.0);
  REQUIRE_FALSE(bare.has_stress);

  auto calls = std::make_shared<std::atomic<int>>(0);
  rgpot::SumPot off;
  off.add({.name = "off", .enabled = false}, std::make_unique<Probe>(calls));
  REQUIRE_FALSE(off.caps().stress);
  const Eval none = run(off, kPos, kZ, kBox);
  REQUIRE(calls->load() == 0);
  REQUIRE(none.energy == 0.0);
  REQUIRE_FALSE(none.has_stress);
}

TEST_CASE("SumPot variance follows the squared weights", "[SumPot]") {
  auto calls = std::make_shared<std::atomic<int>>(0);
  rgpot::SumPot sum;
  sum.add({.name = "a", .weight = 3.0}, std::make_unique<Probe>(calls));
  sum.add({.name = "b", .weight = -2.0}, std::make_unique<Probe>(calls));
  const Eval es = run(sum, kPos, kZ, kBox);
  REQUIRE_THAT(es.variance, WithinAbs((9.0 + 4.0) * 0.5, kTight));
  REQUIRE_THAT(es.energy, WithinAbs(0.25, kTight));
  REQUIRE_THAT(es.f[1], WithinAbs(0.125, kTight));
}

TEST_CASE("SumPot caps are the intersection of the enabled terms", "[SumPot]") {
  auto calls = std::make_shared<std::atomic<int>>(0);
  rgpot::SumPot sum;
  sum.add(std::make_unique<rgpot::LJPot>(lj()));
  {
    const rgpot::PotCaps c = sum.caps();
    REQUIRE(c.stress);
    REQUIRE(c.periodic);
    REQUIRE(c.reentrancy == rgpot::Reentrancy::PerInstance);
  }
  sum.add({.name = "serial"},
          std::make_unique<Probe>(calls, rgpot::PotCaps{.reentrancy = rgpot::Reentrancy::ProcessSerial,
                                                        .perImageInstances = true,
                                                        .periodic = false,
                                                        .groupCollective = true,
                                                        .stress = false}));
  {
    const rgpot::PotCaps c = sum.caps();
    REQUIRE_FALSE(c.stress);
    REQUIRE_FALSE(c.periodic);
    REQUIRE_FALSE(c.batched);
    REQUIRE(c.perImageInstances);
    REQUIRE(c.groupCollective);
    REQUIRE(c.reentrancy == rgpot::Reentrancy::ProcessSerial);
  }
  const Eval mixed = run(sum, kPos, kZ, kBox);
  REQUIRE_FALSE(mixed.has_stress);
  for (double s : mixed.stress) {
    REQUIRE(s == 0.0);
  }

  sum.setEnabled(1, false);
  const rgpot::PotCaps c = sum.caps();
  REQUIRE(c.stress);
  REQUIRE(c.periodic);
  REQUIRE(c.reentrancy == rgpot::Reentrancy::PerInstance);
  REQUIRE(run(sum, kPos, kZ, kBox).has_stress);
}

TEST_CASE("SumPot passes a term failure on and logs it", "[SumPot]") {
  auto calls = std::make_shared<std::atomic<int>>(0);
  rgpot::SumPot sum;
  std::vector<std::string> lines;
  sum.setLogSink([&](std::string_view l) { lines.emplace_back(l); });
  sum.add({.name = "good"}, std::make_unique<Probe>(calls));
  sum.add({.name = "bad"}, std::make_unique<Probe>(calls, rgpot::PotCaps{}, true));
  REQUIRE_THROWS_AS(run(sum, kPos, kZ, kBox), std::domain_error);
  REQUIRE(calls->load() == 2);
  REQUIRE_FALSE(lines.empty());
  REQUIRE_THAT(lines.back(), ContainsSubstring("bad failed: probe failure"));
  REQUIRE_FALSE(sum.lastComponents()[1].evaluated);
}

TEST_CASE("SumPot logs each component", "[SumPot]") {
  auto calls = std::make_shared<std::atomic<int>>(0);
  rgpot::SumPot sum;
  std::vector<std::string> lines;
  sum.setLogSink([&](std::string_view l) { lines.emplace_back(l); });
  sum.add({.name = "dft", .weight = 1.0}, std::make_unique<Probe>(calls));
  sum.add({.name = "disp", .weight = -1.0, .enabled = false}, std::make_unique<Probe>(calls));
  run(sum, kPos, kZ, kBox);
  REQUIRE(lines.size() == 3);
  REQUIRE_THAT(lines[0], ContainsSubstring("rgpot sum: dft weight=1 energy=0.25 eV"));
  REQUIRE_THAT(lines[1], ContainsSubstring("rgpot sum: disp disabled"));
  REQUIRE_THAT(lines[2], ContainsSubstring("total energy=0.25 eV"));
}

TEST_CASE("SumPot rejects bad terms and indices", "[SumPot]") {
  rgpot::SumPot sum;
  REQUIRE_THROWS_AS(sum.add(nullptr), std::invalid_argument);
  REQUIRE_THROWS_AS(sum.add({.name = "nan", .weight = std::nan("")},
                            std::make_unique<rgpot::LJPot>(lj())),
                    std::invalid_argument);
  sum.add(std::make_unique<rgpot::LJPot>(lj()));
  REQUIRE(sum.term(0).name == "term0");
  REQUIRE_THROWS_AS(sum.setWeight(1, 1.0), std::out_of_range);
  REQUIRE_THROWS_AS(sum.setEnabled(1, true), std::out_of_range);
  REQUIRE_THROWS_AS(sum.term(1), std::out_of_range);
  REQUIRE_THROWS_AS(sum.setWeight(0, std::numeric_limits<double>::infinity()),
                    std::invalid_argument);
  const uint64_t k0 = sum.paramsKey();
  sum.setWeight(0, 0.5);
  REQUIRE(sum.paramsKey() != k0);
}
