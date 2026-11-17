// MIT License
// Copyright 2023--present rgpot developers
//
// rgpot::Kept keeps one object per thread and type, for the key it was
// built for. A probe type stands in for a potential.

#include <atomic>
#include <cstdint>
#include <limits>
#include <memory>
#include <stdexcept>
#include <thread>

#include <catch2/catch_all.hpp>

#include "rgpot/KeptPotential.hpp"

namespace {

struct Probe {
  explicit Probe(int built) : id(built) {}
  ~Probe() { ++dropped; }
  int id;
  static inline std::atomic<int> dropped{0};
};

using KeptProbe = rgpot::Kept<Probe>;

rgpot::KeptKey key(double param, std::size_t n, const int *species) {
  rgpot::KeptKey k;
  k.f64(param).i64(7).atoms(n, species);
  return k;
}

int id(const Probe &probe) { return probe.id; }

} // namespace

TEST_CASE("a kept object serves only the key it was built for", "[kept]") {
  KeptProbe::release();
  const rgpot::KeptCounts before = KeptProbe::counts();
  int built = 0;
  auto make = [&] { return std::make_unique<Probe>(++built); };

  const int z[3] = {1, 2, 3};
  int other[3] = {1, 2, 3};
  REQUIRE(KeptProbe::with(key(1.0, 3, z), make, id) == 1);
  REQUIRE(KeptProbe::with(key(1.0, 3, z), make, id) == 1);
  // Species compare by value, not by address.
  REQUIRE(KeptProbe::with(key(1.0, 3, other), make, id) == 1);
  other[1] = 4;
  REQUIRE(KeptProbe::with(key(1.0, 3, other), make, id) == 2);
  REQUIRE(KeptProbe::with(key(1.0, 2, other), make, id) == 3);
  REQUIRE(KeptProbe::with(key(2.0, 2, other), make, id) == 4);
  REQUIRE(KeptProbe::with(key(2.0, 2, nullptr), make, id) == 5);
  REQUIRE(KeptProbe::with(key(2.0, 2, nullptr), make, id) == 5);
  // One object is kept, so an earlier key builds again.
  REQUIRE(KeptProbe::with(key(1.0, 3, z), make, id) == 6);

  // Another field, or the same fields in another order, is another key.
  rgpot::KeptKey longer = key(1.0, 3, z);
  longer.f64(0.0);
  REQUIRE(KeptProbe::with(longer, make, id) == 7);
  rgpot::KeptKey swapped;
  swapped.i64(7).f64(1.0).atoms(3, z);
  REQUIRE(KeptProbe::with(swapped, make, id) == 8);

  // Bits, not values: -0.0 is not 0.0, and a NaN matches itself.
  REQUIRE(KeptProbe::with(key(0.0, 3, z), make, id) == 9);
  REQUIRE(KeptProbe::with(key(-0.0, 3, z), make, id) == 10);
  const double nan = std::numeric_limits<double>::quiet_NaN();
  REQUIRE(KeptProbe::with(key(nan, 3, z), make, id) == 11);
  REQUIRE(KeptProbe::with(key(nan, 3, z), make, id) == 11);

  const rgpot::KeptCounts after = KeptProbe::counts();
  REQUIRE(after.builds - before.builds == 11);
  REQUIRE(after.reuses - before.reuses == 4);
}

TEST_CASE("a key past its room never matches", "[kept]") {
  KeptProbe::release();
  int built = 0;
  auto make = [&] { return std::make_unique<Probe>(++built); };
  rgpot::KeptKey full;
  for (std::size_t k = 0; k < rgpot::KeptKey::kBytes / sizeof(double); ++k) {
    full.f64(static_cast<double>(k));
  }
  REQUIRE(KeptProbe::with(full, make, id) == 1);
  REQUIRE(KeptProbe::with(full, make, id) == 1);
  rgpot::KeptKey over = full;
  over.f64(1.0);
  REQUIRE(KeptProbe::with(over, make, id) == 2);
  REQUIRE(KeptProbe::with(over, make, id) == 3);
  // The fields that fit are those of `full`, which still builds its own.
  REQUIRE(KeptProbe::with(full, make, id) == 4);
  REQUIRE(KeptProbe::with(full, make, id) == 4);
}

TEST_CASE("a throw leaves no kept object behind", "[kept]") {
  KeptProbe::release();
  int built = 0;
  auto make = [&] { return std::make_unique<Probe>(++built); };
  const rgpot::KeptKey k = key(1.0, 0, nullptr);
  REQUIRE(KeptProbe::with(k, make, id) == 1);

  // The run may have left the object half written, so it is dropped.
  const int dropped = Probe::dropped;
  REQUIRE_THROWS_AS(
      KeptProbe::with(k, make, [](const Probe &) -> int { throw std::runtime_error("run"); }),
      std::runtime_error);
  REQUIRE(Probe::dropped == dropped + 1);
  REQUIRE(KeptProbe::with(k, make, id) == 2);

  // A throw from make keeps nothing, not even the object before it.
  const rgpot::KeptCounts before = KeptProbe::counts();
  REQUIRE_THROWS_AS(KeptProbe::with(
                        key(2.0, 0, nullptr),
                        []() -> std::unique_ptr<Probe> { throw std::invalid_argument("make"); },
                        id),
                    std::invalid_argument);
  REQUIRE(KeptProbe::counts().builds == before.builds);
  REQUIRE(KeptProbe::with(k, make, id) == 3);
}

TEST_CASE("each thread keeps its own object", "[kept]") {
  KeptProbe::release();
  std::atomic<int> built{0};
  auto make = [&] { return std::make_unique<Probe>(++built); };
  const rgpot::KeptKey k = key(1.0, 0, nullptr);
  const int here = KeptProbe::with(k, make, id);

  int there = 0;
  int again = 0;
  rgpot::KeptCounts counts;
  std::thread worker([&] {
    there = KeptProbe::with(k, make, id);
    again = KeptProbe::with(k, make, id);
    counts = KeptProbe::counts();
    KeptProbe::release();
  });
  worker.join();
  REQUIRE(there != here);
  REQUIRE(again == there);
  REQUIRE(counts.builds == 1);
  REQUIRE(counts.reuses == 1);
  REQUIRE(KeptProbe::with(k, make, id) == here);

  const int dropped = Probe::dropped;
  KeptProbe::release();
  REQUIRE(Probe::dropped == dropped + 1);
  REQUIRE(KeptProbe::with(k, make, id) != here);
}
