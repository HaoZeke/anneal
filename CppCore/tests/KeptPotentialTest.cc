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
  explicit Probe(int built) : id(built), heap_id(std::make_unique<int>(built)) {}
  ~Probe() {
    if (heap_id) {
      *heap_id = -1;
    }
    ++dropped;
  }
  int id;
  std::unique_ptr<int> heap_id;
  static inline std::atomic<int> dropped{0};
};

using KeptProbe = rgpot::Kept<Probe>;

/// A kept object whose destructor uses another table of this thread, as
/// LJPot does with its neighbour scratch. The destructor leaves what it
/// builds in that table alone, so only a retired table frees it.
struct Owner {
  explicit Owner(const rgpot::KeptKey &inner) : inner_key(inner) {}
  ~Owner();
  rgpot::KeptKey inner_key;
  static inline std::atomic<int> destroyed{0};
  static inline std::atomic<int> dropped_before{0};
  static inline std::atomic<int> dropped_at_entry{0};
  static inline std::atomic<bool> found_none{false};
  static inline std::atomic<std::uint64_t> builds_at_entry{0};
  static inline std::atomic<std::uint64_t> reuses_at_entry{0};
  static inline std::atomic<int> rebuilt_id{0};
  static inline std::atomic<int> rebuilt_ref_id{0};
};

using KeptInner = rgpot::Kept<Probe, 2>;
using KeptOwner = rgpot::Kept<Owner>;

rgpot::KeptKey key(double param, std::size_t n, const int *species) {
  rgpot::KeptKey k;
  k.f64(param).i64(7).atoms(n, species);
  return k;
}

int id(const Probe &probe) { return probe.id; }
const int &ref_id(const Probe &probe) { return *probe.heap_id; }

Owner::~Owner() {
  dropped_at_entry = Probe::dropped.load() - dropped_before.load();
  found_none = KeptInner::peek(inner_key) == nullptr;
  builds_at_entry = KeptInner::counts().builds;
  reuses_at_entry = KeptInner::counts().reuses;
  rebuilt_id = KeptInner::with(
      inner_key, [] { return std::make_unique<Probe>(41); }, id);
  rebuilt_ref_id = KeptInner::with(
      inner_key, [] { return std::make_unique<Probe>(53); }, ref_id);
  ++destroyed;
}

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

TEST_CASE("multi-slot Kept evicts the least recently used slot and supports peek",
          "[kept]") {
  using KeptTwo = rgpot::Kept<Probe, 2>;
  KeptTwo::release();
  int built = 0;
  auto make = [&] { return std::make_unique<Probe>(++built); };

  const int tag_a = 1;
  const int tag_b = 2;
  const int tag_c = 3;
  rgpot::KeptKey ka, kb, kc;
  ka.ptr(&tag_a).f64(1.0);
  kb.ptr(&tag_b).f64(1.0);
  kc.ptr(&tag_c).f64(1.0);

  REQUIRE(KeptTwo::peek(ka) == nullptr);
  REQUIRE(KeptTwo::with(ka, make, id) == 1);
  REQUIRE(KeptTwo::with(kb, make, id) == 2);
  REQUIRE(KeptTwo::peek(ka) != nullptr);
  REQUIRE(KeptTwo::peek(ka)->id == 1);
  REQUIRE(KeptTwo::peek(kb) != nullptr);
  REQUIRE(KeptTwo::peek(kb)->id == 2);

  // Touch ka so kb becomes least recently used; kc then evicts kb.
  REQUIRE(KeptTwo::with(ka, make, id) == 1);
  REQUIRE(KeptTwo::with(kc, make, id) == 3);
  REQUIRE(KeptTwo::peek(ka) != nullptr);
  REQUIRE(KeptTwo::peek(ka)->id == 1);
  REQUIRE(KeptTwo::peek(kb) == nullptr);
  REQUIRE(KeptTwo::peek(kc) != nullptr);
  REQUIRE(KeptTwo::peek(kc)->id == 3);

  KeptTwo::release(ka);
  REQUIRE(KeptTwo::peek(ka) == nullptr);
  REQUIRE(KeptTwo::peek(kc) != nullptr);
  KeptTwo::release();
  REQUIRE(KeptTwo::peek(kc) == nullptr);
}

TEST_CASE("a kept object may use another kept table while its thread exits",
          "[kept][threads]") {
  const int tag_owner = 1;
  const int tag_inner = 2;
  rgpot::KeptKey owner_key;
  rgpot::KeptKey inner_key;
  owner_key.ptr(&tag_owner);
  inner_key.ptr(&tag_inner);

  const int destroyed = Owner::destroyed;
  const int dropped = Probe::dropped;
  Owner::dropped_before = dropped;
  std::thread worker([&] {
    // The owner's table is first used before the inner one, so it is
    // destroyed after it.
    KeptOwner::with(
        owner_key, [&] { return std::make_unique<Owner>(inner_key); },
        [](Owner &) {});
    KeptInner::with(
        inner_key, [] { return std::make_unique<Probe>(7); }, id);
  });
  worker.join();

  REQUIRE(Owner::destroyed == destroyed + 1);
  // The inner table was destroyed first, with its probe, so the owner's
  // destructor runs against a table that is gone.
  REQUIRE(Owner::dropped_at_entry.load() == 1);
  CHECK(Owner::found_none.load());
  CHECK(Owner::builds_at_entry.load() == 0);
  CHECK(Owner::reuses_at_entry.load() == 0);
  CHECK(Owner::rebuilt_id.load() == 41);
  CHECK(Owner::rebuilt_ref_id.load() == 53);
  // The probes built there are dropped when the owner's destructor ends. A
  // table that is gone would keep them, and nothing would ever free them.
  CHECK(Probe::dropped.load() == dropped + 3);
}
