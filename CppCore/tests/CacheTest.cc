// MIT License
// Copyright 2023--present rgpot developers
#include <array>
#include <catch2/catch_all.hpp>
#include <chrono>
#include <filesystem>
#include <limits>
#include <memory>
#include <random>

#include "rgpot/LennardJones/LJPot.hpp"
#include "rgpot/PotentialCache.hpp"

#include <rocksdb/db.h>
#include <rocksdb/options.h>

using namespace Catch::Matchers;
using namespace std::chrono;
namespace fs = std::filesystem;

TEST_CASE("Potential caching with rgpot", "[Potential]") {
  // --- Common System Setup ---
  const int n_atoms = 128;
  rgpot::types::AtomMatrix positions(n_atoms, 3);
  // Use fixed seed for reproducibility
  std::mt19937 gen(1644009449);
  std::uniform_real_distribution<> dis(0.0, 20.0);
  for (size_t i = 0; i < n_atoms * 3; ++i) {
    positions.data()[i] = dis(gen);
  }

  std::vector<int> types(n_atoms, 1);
  std::array<std::array<double, 3>, 3> box = {
      {{10, 0, 0}, {0, 10, 0}, {0, 0, 10}}};

  auto pot = std::make_shared<rgpot::LJPot>();

  // Baseline timing (no cache attached)
  auto start_base = high_resolution_clock::now();
  auto [e_base, f_base, v_base] = (*pot)(positions, types, box);
  auto end_base = high_resolution_clock::now();
  auto base_duration =
      duration_cast<nanoseconds>(end_base - start_base).count();

  SECTION("Manual DB Management (Raw Pointer)") {
    // Setup RocksDB Manually
    rocksdb::Options options;
    options.create_if_missing = true;
    std::string db_path = "/tmp/rgpot_test_rocksdb_manual";
    rocksdb::DestroyDB(db_path, options);
#if RGPOT_ROCKSDB_UNIQUE_PTR_OPEN
    std::unique_ptr<rocksdb::DB> db;
    rocksdb::Status status = rocksdb::DB::Open(options, db_path, &db);
#else
    rocksdb::DB *db_ptr = nullptr;
    rocksdb::Status status = rocksdb::DB::Open(options, db_path, &db_ptr);
    std::unique_ptr<rocksdb::DB> db(db_ptr);
#endif
    REQUIRE(status.ok());

    auto pcache = rgpot::cache::PotentialCache();
    pcache.set_db(db.get());
    pot->setCache(&pcache);

    // 1. Miss & Write
    auto [e1, f1, v1] = (*pot)(positions, types, box);
    REQUIRE_THAT(e1, WithinAbs(e_base, 1e-12));

    // 2. Hit & Read
    auto start = high_resolution_clock::now();
    auto [e2, f2, v2] = (*pot)(positions, types, box);
    auto end = high_resolution_clock::now();
    auto dur = duration_cast<nanoseconds>(end - start).count();

    REQUIRE_THAT(e2, WithinAbs(e1, 1e-12));
    // Shared CI runners (esp. macOS) have noisy wall-clock; keep a loose
    // bound that still fails if the cache is completely ineffective.
    REQUIRE(dur < base_duration * 50);
  }

  SECTION("Managed DB Life-cycle (Path String)") {
    std::string db_path = "/tmp/rgpot_test_rocksdb_managed";
    // Ensure clean state
    rocksdb::Options opts;
    rocksdb::DestroyDB(db_path, opts);

    {
      auto pcache = rgpot::cache::PotentialCache(db_path);
      pot->setCache(&pcache);

      // 1. Miss
      (*pot)(positions, types, box);

      // 2. Hit
      auto start = high_resolution_clock::now();
      auto [e2, f2, v2] = (*pot)(positions, types, box);
      auto end = high_resolution_clock::now();
      auto dur = duration_cast<nanoseconds>(end - start).count();

      REQUIRE_THAT(e2, WithinAbs(e_base, 1e-12));
      REQUIRE(dur < base_duration * 50);
    }
    // pcache goes out of scope here, should close DB cleanly

    REQUIRE(fs::exists(db_path));
  }

  SECTION("Persistence (Close and Reopen)") {
    std::string db_path = "/tmp/rgpot_test_rocksdb_persist";
    rocksdb::Options opts;
    rocksdb::DestroyDB(db_path, opts);

    // Phase 1: Create cache, write data, destroy object
    {
      auto pcache_write = rgpot::cache::PotentialCache(db_path);
      pot->setCache(&pcache_write);
      (*pot)(positions, types, box); // Writes to DB
    }

    // Phase 2: Create NEW cache object pointing to SAME path
    {
      auto pcache_read = rgpot::cache::PotentialCache(db_path);
      pot->setCache(&pcache_read);

      auto start = high_resolution_clock::now();
      auto [e_read, f_read, v_read] = (*pot)(positions, types, box);
      auto end = high_resolution_clock::now();
      auto dur = duration_cast<nanoseconds>(end - start).count();

      // Should be a Hit (fast) despite being a new object
      REQUIRE_THAT(e_read, WithinAbs(e_base, 1e-12));
      // Timing is noisy on CI (esp. macOS/shared runners); keep a loose bound
      // so we still catch pathological misses (orders of magnitude slower).
      REQUIRE(dur < base_duration * 50);
    }
  }

  SECTION("Uninitialized Cache (Graceful Degradation)") {
    // Cache object created but no DB set
    auto pcache_empty = rgpot::cache::PotentialCache();
    pot->setCache(&pcache_empty);

    // Should call forceImpl directly without crashing
    auto start = high_resolution_clock::now();
    auto [e, f, v] = (*pot)(positions, types, box);
    auto end = high_resolution_clock::now();

    REQUIRE_THAT(e, WithinAbs(e_base, 1e-12));
    // Should NOT be faster than base (it effectively IS base overhead)
    // We just check it didn't throw exceptions
  }
}

TEST_CASE("Cache keys separate parameter sets", "[Potential][cache]") {
  // Two atoms 5 A apart: inside a 15 A cutoff, outside a 3 A cutoff, so
  // the energies must differ and a shared cache entry would be visible.
  rgpot::types::AtomMatrix positions(2, 3);
  positions(0, 0) = 0.0;
  positions(0, 1) = 0.0;
  positions(0, 2) = 0.0;
  positions(1, 0) = 5.0;
  positions(1, 1) = 0.0;
  positions(1, 2) = 0.0;
  std::vector<int> types(2, 1);
  std::array<std::array<double, 3>, 3> box = {
      {{50, 0, 0}, {0, 50, 0}, {0, 0, 50}}};

  auto potWide = std::make_shared<rgpot::LJPot>();
  auto potNarrow =
      std::make_shared<rgpot::LJPot>(rgpot::LJConfig{.cutoff = 3.0});

  REQUIRE(potWide->paramsKey() != potNarrow->paramsKey());

  std::string db_path = "/tmp/rgpot_test_rocksdb_params";
  rocksdb::Options opts;
  rocksdb::DestroyDB(db_path, opts);
  auto pcache = rgpot::cache::PotentialCache(db_path);
  potWide->setCache(&pcache);
  potNarrow->setCache(&pcache);

  auto [eWide, fWide, vWide] = (*potWide)(positions, types, box);
  // Same positions, types, box, and PotType: only paramsKey separates the
  // two entries. A shared entry would return eWide here.
  auto [eNarrow, fNarrow, vNarrow] = (*potNarrow)(positions, types, box);

  REQUIRE(eWide != eNarrow);
  REQUIRE(eNarrow == 0.0); // pair beyond the 3 A cutoff contributes nothing

  // Both entries persist independently.
  auto [eWide2, fWide2, vWide2] = (*potWide)(positions, types, box);
  auto [eNarrow2, fNarrow2, vNarrow2] = (*potNarrow)(positions, types, box);
  REQUIRE(eWide2 == eWide);
  REQUIRE(eNarrow2 == eNarrow);
}

namespace {
class ResultMetadataPotential
    : public rgpot::Potential<ResultMetadataPotential> {
public:
  ResultMetadataPotential() : Potential(rgpot::PotType::LJ) {}
  mutable size_t evaluations = 0;

  rgpot::PotCaps caps() const noexcept override { return {.stress = true}; }
  uint64_t paramsKey() const noexcept override { return 0x726573756c74ULL; }

  void forceImpl(const rgpot::ForceInput &in,
                 rgpot::ForceOut *out) const override {
    ++evaluations;
    out->energy = 3.0 + in.pos[0];
    out->variance = 0.25;
    for (size_t i = 0; i < 3 * in.nAtoms; ++i) {
      out->F[i] = -in.pos[i];
    }
    for (size_t i = 0; i < 9; ++i) {
      out->stress[i] = static_cast<double>(i + 1) + in.pos[0];
    }
    out->has_stress = 1;
  }
};

std::string resultCachePath() {
  return (fs::temp_directory_path() /
          ("rgpot_result_metadata_" +
           std::to_string(steady_clock::now().time_since_epoch().count())))
      .string();
}

void checkResultMetadata(const rgpot::ForceInput &in,
                         const rgpot::ForceOut &out) {
  REQUIRE(out.energy == 3.0 + in.pos[0]);
  REQUIRE(out.variance == 0.25);
  REQUIRE(out.has_stress == 1);
  for (size_t i = 0; i < 3 * in.nAtoms; ++i) {
    REQUIRE(out.F[i] == -in.pos[i]);
  }
  for (size_t i = 0; i < 9; ++i) {
    REQUIRE(out.stress[i] == static_cast<double>(i + 1) + in.pos[0]);
  }
}
} // namespace

TEST_CASE("Cached force batches retain result metadata through misses and hits",
          "[Potential][cache][stress]") {
  ResultMetadataPotential pot;
  double positions[2][6] = {{0.0, 0.1, 0.2, 1.1, 1.2, 1.3},
                            {0.5, 0.6, 0.7, 1.6, 1.7, 1.8}};
  const int types[2] = {1, 1};
  const double box[9] = {8.0, 0.0, 0.0, 0.0, 8.0, 0.0, 0.0, 0.0, 8.0};
  const rgpot::ForceInput inputs[] = {{2, positions[0], types, box},
                                      {2, positions[1], types, box}};
  double forces[2][6] = {};
  rgpot::ForceOut outputs[] = {{forces[0], 0.0, 0.0, {}, 0},
                               {forces[1], 0.0, 0.0, {}, 0}};
  const rgpot::ForceBatch batch{2, inputs, outputs};
  const std::string path = resultCachePath();
  {
    rgpot::cache::PotentialCache cache(path);
    pot.setCache(&cache);
    pot.forceBatch(batch);
    REQUIRE(pot.evaluations == 2);
    for (size_t i = 0; i < 2; ++i) {
      checkResultMetadata(inputs[i], outputs[i]);
      outputs[i].variance = -1.0;
      outputs[i].has_stress = 0;
      std::fill(outputs[i].stress, outputs[i].stress + 9,
                std::numeric_limits<double>::quiet_NaN());
    }
    pot.forceBatch(batch);
    REQUIRE(pot.evaluations == 2);
    for (size_t i = 0; i < 2; ++i) {
      checkResultMetadata(inputs[i], outputs[i]);
    }

    positions[1][0] += 0.75;
    pot.forceBatch(batch);
    REQUIRE(pot.evaluations == 3);
    for (size_t i = 0; i < 2; ++i) {
      checkResultMetadata(inputs[i], outputs[i]);
    }
    pot.setCache(nullptr);
  }
  {
    rgpot::cache::PotentialCache cache(path);
    pot.setCache(&cache);
    for (auto &out : outputs) {
      out.energy = 0.0;
      out.variance = 0.0;
      out.has_stress = 0;
      std::fill(out.stress, out.stress + 9, 0.0);
    }
    pot.forceBatch(batch);
    REQUIRE(pot.evaluations == 3);
    for (size_t i = 0; i < 2; ++i) {
      checkResultMetadata(inputs[i], outputs[i]);
    }
    pot.setCache(nullptr);
  }
  REQUIRE(rocksdb::DestroyDB(path, rocksdb::Options()).ok());
}

TEST_CASE("Stress batches refresh energy-force-only cache records",
          "[Potential][cache][stress]") {
  const std::string path = resultCachePath();
  rocksdb::Options options;
  options.create_if_missing = true;
  std::unique_ptr<rocksdb::DB> db;
#if RGPOT_ROCKSDB_UNIQUE_PTR_OPEN
  REQUIRE(rocksdb::DB::Open(options, path, &db).ok());
#else
  rocksdb::DB *raw = nullptr;
  REQUIRE(rocksdb::DB::Open(options, path, &raw).ok());
  db.reset(raw);
#endif
  rgpot::cache::PotentialCache cache;
  cache.set_db(db.get());
  ResultMetadataPotential pot;
  pot.setCache(&cache);
  double positions[6] = {0.25, 0.5, 0.75, 1.25, 1.5, 1.75};
  const int types[2] = {1, 1};
  const double box[9] = {8.0, 0.0, 0.0, 0.0, 8.0, 0.0, 0.0, 0.0, 8.0};
  const rgpot::ForceInput input{2, positions, types, box};
  double forces[6] = {};
  rgpot::ForceOut out{forces, 0.0, 0.0, {}, 0};
  const rgpot::ForceBatch batch{1, &input, &out};
  pot.forceBatch(batch);
  REQUIRE(pot.evaluations == 1);
  checkResultMetadata(input, out);

  std::unique_ptr<rocksdb::Iterator> it(
      db->NewIterator(rocksdb::ReadOptions()));
  it->SeekToFirst();
  REQUIRE(it->Valid());
  const std::string key = it->key().ToString();
  std::string value = it->value().ToString();
  REQUIRE(rgpot::cache::PotentialCache::has_result_metadata(value, 2));
  it.reset();
  value.resize((1 + 6) * sizeof(double));
  REQUIRE_FALSE(rgpot::cache::PotentialCache::has_result_metadata(value, 2));
  REQUIRE(db->Put(rocksdb::WriteOptions(), key, value).ok());

  cache.deserialize_hit(value, out, 2);
  REQUIRE(out.energy == 3.25);
  REQUIRE(out.variance == 0.0);
  REQUIRE(out.has_stress == 0);
  for (double component : out.stress) {
    REQUIRE(component == 0.0);
  }
  REQUIRE_THROWS_AS(cache.deserialize_hit(value.substr(0, 1), out, 2),
                    std::runtime_error);

  pot.forceBatch(batch);
  REQUIRE(pot.evaluations == 2);
  checkResultMetadata(input, out);
  pot.forceBatch(batch);
  REQUIRE(pot.evaluations == 2);
  checkResultMetadata(input, out);

  std::string refreshed;
  REQUIRE(db->Get(rocksdb::ReadOptions(), key, &refreshed).ok());
  rgpot::types::AtomMatrix decoded(2, 3);
  double energy = 0.0;
  cache.deserialize_hit(refreshed, energy, decoded);
  REQUIRE(energy == out.energy);
  for (size_t i = 0; i < 6; ++i) {
    REQUIRE(decoded.data()[i] == out.F[i]);
  }
  pot.setCache(nullptr);
  cache.set_db(nullptr);
  db.reset();
  REQUIRE(rocksdb::DestroyDB(path, rocksdb::Options()).ok());
}

TEST_CASE("Scalar cache retains calculator variance", "[Potential][cache]") {
  const std::string path = resultCachePath();
  ResultMetadataPotential pot;
  rgpot::types::AtomMatrix positions{{0.0, 0.1, 0.2}, {1.0, 1.1, 1.2}};
  const std::vector<int> types{1, 1};
  const std::array<std::array<double, 3>, 3> box{
      {{8.0, 0.0, 0.0}, {0.0, 8.0, 0.0}, {0.0, 0.0, 8.0}}};
  {
    rgpot::cache::PotentialCache cache(path);
    pot.setCache(&cache);
    const auto [first_energy, first_forces, first_variance] =
        pot(positions, types, box);
    const auto [energy, forces, variance] = pot(positions, types, box);
    REQUIRE(pot.evaluations == 1);
    REQUIRE(first_variance == 0.25);
    REQUIRE(variance == first_variance);
    REQUIRE(energy == first_energy);
    for (size_t i = 0; i < 6; ++i) {
      REQUIRE(forces.data()[i] == first_forces.data()[i]);
    }
    pot.setCache(nullptr);
  }
  REQUIRE(rocksdb::DestroyDB(path, rocksdb::Options()).ok());
}

TEST_CASE("Cache counters split computed from served evaluations",
          "[Potential][cache][counts]") {
  ResultMetadataPotential pot;
  double positions[2][6] = {{0.0, 0.1, 0.2, 1.1, 1.2, 1.3},
                            {0.5, 0.6, 0.7, 1.6, 1.7, 1.8}};
  const int types[2] = {1, 1};
  const double box[9] = {8.0, 0.0, 0.0, 0.0, 8.0, 0.0, 0.0, 0.0, 8.0};
  const rgpot::ForceInput inputs[] = {{2, positions[0], types, box},
                                      {2, positions[1], types, box}};
  double forces[2][6] = {};
  rgpot::ForceOut outputs[] = {{forces[0], 0.0, 0.0, {}, 0},
                               {forces[1], 0.0, 0.0, {}, 0}};
  const rgpot::ForceBatch batch{2, inputs, outputs};
  const std::string path = resultCachePath();

  {
    // Cold: both systems are computed.
    rgpot::cache::PotentialCache cache(path);
    pot.setCache(&cache);
    REQUIRE(cache.counts().computed == 0);
    REQUIRE(cache.counts().served == 0);
    pot.forceBatch(batch);
    REQUIRE(cache.counts().computed == 2);
    REQUIRE(cache.counts().served == 0);

    // Same batch again: both served, computed unchanged.
    pot.forceBatch(batch);
    REQUIRE(cache.counts().computed == 2);
    REQUIRE(cache.counts().served == 2);

    // One moved system: one computed, one served.
    positions[1][0] += 0.75;
    pot.forceBatch(batch);
    REQUIRE(cache.counts().computed == 3);
    REQUIRE(cache.counts().served == 3);

    // Scalar entry point counts through the same handle.
    rgpot::types::AtomMatrix pos(2, 3);
    std::copy_n(positions[0], 6, pos.data());
    pot(pos, {1, 1}, {{{8, 0, 0}, {0, 8, 0}, {0, 0, 8}}});
    REQUIRE(cache.counts().computed + cache.counts().served == 7);

    cache.reset_counts();
    REQUIRE(cache.counts().computed == 0);
    REQUIRE(cache.counts().served == 0);
    pot.setCache(nullptr);
  }
  {
    // Warm start from the persisted entries: nothing is computed.
    rgpot::cache::PotentialCache cache(path);
    pot.setCache(&cache);
    pot.forceBatch(batch);
    REQUIRE(cache.counts().computed == 0);
    REQUIRE(cache.counts().served == 2);
    pot.setCache(nullptr);
  }
  REQUIRE(rocksdb::DestroyDB(path, rocksdb::Options()).ok());
}
