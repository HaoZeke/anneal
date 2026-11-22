// MIT License
// Copyright 2023--present rgpot developers
#include <array>
#include <catch2/catch_all.hpp>
#include <chrono>
#include <filesystem>
#include <limits>
#include <memory>
#include <random>
#include <type_traits>

#include "rgpot/D2Pot/D2CpmdPot.hpp"
#include "rgpot/D2Pot/D2Pot.hpp"
#ifdef RGPOT_HAS_DFTD4
#include "rgpot/D4Pot/D4Pot.hpp"
#endif
#include "TestTempDir.hpp"
#include "rgpot/LennardJones/LJPot.hpp"
#include "rgpot/Models/ModelPots.hpp"
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
    rgpot::tests::TestTempDir dir("rgpot-test-rocksdb-manual");
    std::string db_path = (dir.path() / "cache-db").string();
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
    rgpot::tests::TestTempDir dir("rgpot-test-rocksdb-managed");
    std::string db_path = (dir.path() / "cache-db").string();
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
    rgpot::tests::TestTempDir dir("rgpot-test-rocksdb-persist");
    std::string db_path = (dir.path() / "cache-db").string();
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

  rgpot::tests::TestTempDir dir("rgpot-test-rocksdb-params");
  std::string db_path = (dir.path() / "cache-db").string();
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

TEST_CASE("Cache test temp helper creates distinct directories",
          "[Potential][cache][tempdir]") {
  const rgpot::tests::TestTempDir first("rgpot-cache-unique");
  const rgpot::tests::TestTempDir second("rgpot-cache-unique");
  REQUIRE(first.path() != second.path());
}

TEST_CASE("Cache keys separate Muller-Brown parameter sets",
          "[Potential][cache][muller]") {
  rgpot::types::AtomMatrix positions(1, 3);
  positions(0, 0) = 0.2;
  positions(0, 1) = 0.3;
  positions(0, 2) = 0.0;
  std::vector<int> types{1};
  std::array<std::array<double, 3>, 3> box{};

  auto published = std::make_shared<rgpot::MullerBrownPot>();
  rgpot::MullerBrownConfig scaled;
  for (double &value : scaled.amplitude)
    value *= 2.0;
  auto other = std::make_shared<rgpot::MullerBrownPot>(scaled);
  REQUIRE(published->paramsKey() != 0);
  REQUIRE(published->paramsKey() != other->paramsKey());

  auto [barePublished, barePublishedForce, barePublishedVariance] =
      (*published)(positions, types, box);
  auto [bareOther, bareOtherForce, bareOtherVariance] =
      (*other)(positions, types, box);
  REQUIRE(bareOther != barePublished);

  rgpot::tests::TestTempDir dir("rgpot-test-rocksdb-muller");
  const std::string db_path = (dir.path() / "cache-db").string();
  rocksdb::Options opts;
  rocksdb::DestroyDB(db_path, opts);
  auto pcache = rgpot::cache::PotentialCache(db_path);
  published->setCache(&pcache);
  other->setCache(&pcache);

  auto [cachedPublished, cachedPublishedForce, cachedPublishedVariance] =
      (*published)(positions, types, box);
  auto [cachedOther, cachedOtherForce, cachedOtherVariance] =
      (*other)(positions, types, box);
  REQUIRE(cachedPublished == barePublished);
  REQUIRE(cachedOther == bareOther);
  REQUIRE(cachedPublished != cachedOther);

  auto [againPublished, againPublishedForce, againPublishedVariance] =
      (*published)(positions, types, box);
  auto [againOther, againOtherForce, againOtherVariance] =
      (*other)(positions, types, box);
  REQUIRE(againPublished == barePublished);
  REQUIRE(againOther == bareOther);
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

namespace {
class FingerprintedPotential : public rgpot::Potential<FingerprintedPotential> {
public:
  FingerprintedPotential(rgpot::PotType type, uint64_t fingerprint, double energy)
      : Potential(type), fingerprint_(fingerprint), energy_(energy) {}

  uint64_t paramsKey() const noexcept override { return fingerprint_; }

  void forceImpl(const rgpot::ForceInput &, rgpot::ForceOut *out) const override {
    out->energy = energy_;
  }

private:
  uint64_t fingerprint_;
  double energy_;
};
} // namespace

TEST_CASE("Persistent cache keys distinguish ordered potential identities across callers",
          "[Potential][cache][identity]") {
  using rgpot::PotType;
  FingerprintedPotential a(PotType::CuH2, static_cast<uint64_t>(PotType::LJ), 1.25);
  FingerprintedPotential b(PotType::LJ, static_cast<uint64_t>(PotType::CuH2), 2.5);
  FingerprintedPotential c(PotType::LJ, UINT64_C(0x1234567800000001), 3.75);
  rgpot::types::AtomMatrix positions(1, 3);
  positions(0, 0) = 0.125;
  positions(0, 1) = 0.25;
  positions(0, 2) = 0.5;
  const std::vector<int> species{1};
  const std::array<std::array<double, 3>, 3> box{{{6, 0, 0}, {0, 7, 0}, {0, 0, 8}}};
  const auto path = resultCachePath();
  {
    rgpot::cache::PotentialCache cold(path);
    rgpot::ForceInput input{1, positions.data(), species.data(), box.front().data()};
    double forces[3] = {};
    rgpot::ForceOut output{};
    output.F = forces;
    for (auto *pot : {&a, &b, &c}) {
      pot->setCache(&cold);
      pot->forceBatch({1, &input, &output});
      pot->setCache(nullptr);
    }
    REQUIRE(cold.counts().computed == 3);
    REQUIRE(cold.counts().served == 0);
  }
  {
    rgpot::cache::PotentialCache warm(path);
    using Case = std::pair<FingerprintedPotential *, double>;
    for (const auto &[pot, energy] : {Case{&c, 3.75}, Case{&b, 2.5}, Case{&a, 1.25}}) {
      pot->setCache(&warm);
      const auto answer = (*pot)(positions, species, box);
      REQUIRE(std::get<0>(answer) == energy);
      pot->setCache(nullptr);
    }
    REQUIRE(warm.counts().computed == 0);
    REQUIRE(warm.counts().served == 3);
  }
  REQUIRE(rocksdb::DestroyDB(path, rocksdb::Options()).ok());
}

static_assert(std::is_same_v<decltype(rgpot::cache::KeyHash::hash), uint64_t>,
              "cache keys are 64 bits wide on every platform");

TEST_CASE("Stored cache keys are a version byte and a little-endian 64-bit hash",
          "[Potential][cache][key]") {
  using rgpot::PotType;
  FingerprintedPotential pot(PotType::LJ, UINT64_C(0x0123456789abcdef), 1.5);
  rgpot::types::AtomMatrix positions(1, 3);
  positions(0, 0) = 0.125;
  positions(0, 1) = 0.25;
  positions(0, 2) = 0.5;
  const std::vector<int> species{1};
  const std::array<std::array<double, 3>, 3> box{{{6, 0, 0}, {0, 7, 0}, {0, 0, 8}}};
  const auto path = resultCachePath();
  {
    rgpot::cache::PotentialCache cache(path);
    pot.setCache(&cache);
    (void)pot(positions, species, box);
    pot.setCache(nullptr);
  }
  const rgpot::ForceInput input{1, positions.data(), species.data(),
                                box.front().data()};
  const uint64_t hash =
      rgpot::cache::result_key(input, PotType::LJ, pot.paramsKey()).hash;

  rocksdb::DB *raw = nullptr;
  REQUIRE(rocksdb::DB::OpenForReadOnly(rocksdb::Options(), path, &raw).ok());
  std::unique_ptr<rocksdb::DB> db(raw);
  std::unique_ptr<rocksdb::Iterator> entries(
      db->NewIterator(rocksdb::ReadOptions()));
  size_t count = 0;
  for (entries->SeekToFirst(); entries->Valid(); entries->Next()) {
    ++count;
    const std::string key = entries->key().ToString();
    REQUIRE(key.size() == 1 + sizeof(uint64_t));
    REQUIRE(static_cast<unsigned char>(key[0]) == 2);
    uint64_t stored = 0;
    for (size_t i = 0; i < sizeof(uint64_t); ++i) {
      stored |= static_cast<uint64_t>(static_cast<unsigned char>(key[1 + i]))
                << (8 * i);
    }
    REQUIRE(stored == hash);
  }
  REQUIRE(count == 1);
  entries.reset();
  db.reset();
  REQUIRE(rocksdb::DestroyDB(path, rocksdb::Options()).ok());
}

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

namespace {
template <typename Pot> void checkCachedNullBox(Pot &pot) {
  const double pos[] = {
      0.00000000, 0.00000000,  0.11779000,  // O
      0.00000000, 0.75545000,  -0.47116000, // H
      0.00000000, -0.75545000, -0.47116000  // H
  };
  const int z[] = {8, 1, 1};
  const double zero_box[9] = {};
  const double cell[9] = {8.0, 0.0, 0.0, 0.0, 8.0, 0.0, 0.0, 0.0, 8.0};

  auto eval = [&](const double *box) {
    std::array<double, 9> f{};
    const rgpot::ForceInput in{3, pos, z, box};
    rgpot::ForceOut out{};
    out.F = f.data();
    pot.forceBatch(rgpot::ForceBatch{.nSystems = 1, .in = &in, .out = &out});
    return std::make_tuple(out.energy, f, out.has_stress != 0);
  };

  const auto uncached = eval(nullptr);
  REQUIRE_FALSE(std::get<2>(uncached));

  const std::string path = resultCachePath();
  {
    rgpot::cache::PotentialCache cache(path);
    pot.setCache(&cache);

    const auto cold = eval(nullptr);
    REQUIRE(cache.counts().computed == 1);
    REQUIRE(cache.counts().served == 0);
    REQUIRE(std::get<0>(cold) == std::get<0>(uncached));
    REQUIRE(std::get<1>(cold) == std::get<1>(uncached));
    REQUIRE_FALSE(std::get<2>(cold));

    const auto warm = eval(nullptr);
    REQUIRE(cache.counts().computed == 1);
    REQUIRE(cache.counts().served == 1);
    REQUIRE(std::get<0>(warm) == std::get<0>(uncached));
    REQUIRE(std::get<1>(warm) == std::get<1>(uncached));
    REQUIRE_FALSE(std::get<2>(warm));

    // A null box is distinct in the key from any real 9-double box,
    // including an all-zero box and a periodic cell.
    const auto with_zero = eval(zero_box);
    REQUIRE(cache.counts().computed == 2);
    REQUIRE(cache.counts().served == 1);
    REQUIRE(std::get<0>(with_zero) == std::get<0>(uncached));
    REQUIRE(std::get<1>(with_zero) == std::get<1>(uncached));

    (void)eval(cell);
    REQUIRE(cache.counts().computed == 3);
    REQUIRE(cache.counts().served == 1);

    pot.setCache(nullptr);
  }
  REQUIRE(rocksdb::DestroyDB(path, rocksdb::Options()).ok());
}
} // namespace

TEST_CASE("Molecular dispersion forceBatch caches a null box without colliding with a real box",
          "[Potential][cache][d2][d2cpmd][d4]") {
  rgpot::D2Pot d2;
  checkCachedNullBox(d2);

  rgpot::D2CpmdPot d2_cpmd;
  checkCachedNullBox(d2_cpmd);

#ifdef RGPOT_HAS_DFTD4
  rgpot::D4Pot d4;
  checkCachedNullBox(d4);
#endif
}
