// MIT License
// Copyright 2023--present rgpot developers
#include <catch2/catch_all.hpp>
#include <cmath>
#include <vector>

#include "TestTempDir.hpp"
#include "rgpot/LennardJones/LJPot.hpp"
#include "rgpot/PotHelpers.hpp"
#include "rgpot/PotentialCache.hpp"

using namespace Catch::Matchers;

// Simple helper to apply rotation
void rotate_z(rgpot::types::AtomMatrix &pos, double angle_rad) {
  double c = std::cos(angle_rad);
  double s = std::sin(angle_rad);
  for (size_t i = 0; i < pos.rows(); ++i) {
    double x = pos(i, 0);
    double y = pos(i, 1);
    pos(i, 0) = x * c - y * s;
    pos(i, 1) = x * s + y * c;
  }
}

TEST_CASE("Invariance and Caching Behavior", "[Invariance]") {
#ifdef RGPOT_HAS_CACHE
  //  Setup Cache
  rgpot::tests::TestTempDir dir("rgpot-test-invariance");
  std::string db_path = (dir.path() / "cache-db").string();
  rocksdb::Options opts;
  rocksdb::DestroyDB(db_path, opts);
  auto pcache = rgpot::cache::PotentialCache(db_path);
  auto pot = std::make_shared<rgpot::LJPot>();
  pot->setCache(&pcache);

  // 2. Define System (Dimer)
  rgpot::types::AtomMatrix pos(2, 3);
  pos(0, 0) = 0.0;
  pos(0, 1) = 0.0;
  pos(0, 2) = 0.0;
  pos(1, 0) = 1.5;
  pos(1, 1) = 0.0;
  pos(1, 2) = 0.0; // r = 1.5
  std::vector<int> types = {1, 1};
  std::array<std::array<double, 3>, 3> box = {
      {{10, 0, 0}, {0, 10, 0}, {0, 0, 10}}};

  // Reset counters
  rgpot::registry<rgpot::LJPot>::forceCalls = 0;

  // --- Baseline ---
  auto [e_base, f_base, v_base] = (*pot)(pos, types, box);
  size_t calls_after_base = rgpot::registry<rgpot::LJPot>::forceCalls;
  REQUIRE(calls_after_base == 1);

  SECTION("Global Translation") {
    // Shift entire system by (5, 5, 5)
    for (size_t i = 0; i < pos.rows(); ++i) {
      pos(i, 0) += 5.0;
      pos(i, 1) += 5.0;
      pos(i, 2) += 5.0;
    }

    auto [e_trans, f_trans, v_trans] = (*pot)(pos, types, box);
    size_t calls_after_trans = rgpot::registry<rgpot::LJPot>::forceCalls;

    // Physics check: Energy should be invariant
    REQUIRE_THAT(e_trans, WithinAbs(e_base, 1e-12));

    // Cache check:
    // TODO(rg): Ideally, this SHOULD be 1 (Hit) if we used relative
    // coordinates/descriptors. Currently, it relies on raw positions, so we
    // expect it to be 2 (Miss). This assertion documents current behavior.
    CHECK(calls_after_trans == 2);
  }

  SECTION("Global Rotation") {
    // Rotate 90 degrees around Z
    rotate_z(pos, M_PI / 2.0);

    auto [e_rot, f_rot, v_rot] = (*pot)(pos, types, box);
    size_t calls_after_rot = rgpot::registry<rgpot::LJPot>::forceCalls;

    // Physics check: Energy should be invariant
    REQUIRE_THAT(e_rot, WithinAbs(e_base, 1e-12));

    // Cache check:
    // Currently expects a miss (2 calls total)
    CHECK(calls_after_rot == 2);
  }
#else
  SKIP("Caching disabled; skipping invariance cache checks");
#endif
}
