// MIT License
// Copyright 2023--present rgpot developers

/**
 * @brief Implementation of the RocksDB-based potential cache.
 *
 * This file implements the methods for the PotentialCache class, handling the
 * low-level interactions with the RocksDB library for persistent storage of
 * potential evaluations.
 */

#include "rgpot/PotentialCache.hpp"
#include <algorithm>
#include <array>
#include <cstring>
#include <iostream>
#include <memory>
#include <rocksdb/options.h>
#include <stdexcept>
#include <vector>

// memcpy-based word loads preserve C++ aliasing rules for typed input arrays.
#ifndef XXH_FORCE_MEMORY_ACCESS
#define XXH_FORCE_MEMORY_ACCESS 0
#endif
#define XXH_INLINE_ALL
#include <xxhash.h>

namespace rgpot::cache {

namespace {
// Versioned storage keys keep entries formed with a different hash protocol
// in a separate RocksDB namespace. The key is one version byte and the 64-bit
// hash in little-endian order, so its width and byte order do not depend on
// the platform.
constexpr size_t storage_key_bytes = 1 + sizeof(uint64_t);

std::array<char, storage_key_bytes> storage_key(const KeyHash &key) {
  std::array<char, storage_key_bytes> bytes{};
  bytes[0] = 2;
  for (size_t i = 0; i < sizeof(uint64_t); ++i) {
    bytes[1 + i] = static_cast<char>((key.hash >> (8 * i)) & 0xffU);
  }
  return bytes;
}

constexpr char result_marker[] = "RGPOTR1";
constexpr size_t result_metadata_bytes =
    sizeof(result_marker) + 10 * sizeof(double) + 1;

size_t energy_force_bytes(size_t n_atoms) {
  return (1 + 3 * n_atoms) * sizeof(double);
}
} // namespace

KeyHash result_key(const ForceInput &input, PotType type, uint64_t parameters) {
  const std::array<uint64_t, 4> identity{
      input.nAtoms, static_cast<uint64_t>(type), parameters, input.box != nullptr};
  XXH3_state_t state;
  XXH3_INITSTATE(&state);
  XXH3_64bits_reset(&state);
  XXH3_64bits_update(&state, identity.data(), sizeof(identity));
  XXH3_64bits_update(&state, input.pos, input.nAtoms * 3 * sizeof(double));
  XXH3_64bits_update(&state, input.atmnrs, input.nAtoms * sizeof(int));
  if (input.box) {
    XXH3_64bits_update(&state, input.box, 9 * sizeof(double));
  }
  return KeyHash(XXH3_64bits_digest(&state));
}

bool PotentialCache::has_result_metadata(const std::string &value,
                                         size_t n_atoms) {
  const size_t prefix = energy_force_bytes(n_atoms);
  return value.size() == prefix + result_metadata_bytes &&
         std::memcmp(value.data() + prefix, result_marker,
                     sizeof(result_marker)) == 0 &&
         static_cast<unsigned char>(value.back()) <= 1;
}

void PotentialCache::deserialize_hit(const std::string &value, ForceOut &out,
                                     size_t n_atoms) {
  const size_t prefix = energy_force_bytes(n_atoms);
  if (value.size() < prefix) {
    throw std::runtime_error("PotentialCache: incomplete energy/force record");
  }
  std::memcpy(&out.energy, value.data(), sizeof(double));
  std::memcpy(out.F, value.data() + sizeof(double),
              3 * n_atoms * sizeof(double));
  out.variance = 0.0;
  out.has_stress = 0;
  std::fill(out.stress, out.stress + 9, 0.0);
  if (has_result_metadata(value, n_atoms)) {
    const char *metadata = value.data() + prefix + sizeof(result_marker);
    std::memcpy(&out.variance, metadata, sizeof(double));
    std::memcpy(out.stress, metadata + sizeof(double), sizeof(out.stress));
    out.has_stress = static_cast<unsigned char>(value.back());
  }
}

void PotentialCache::add_serialized(const KeyHash &key, const ForceOut &out,
                                    size_t n_atoms) {
  std::lock_guard<std::mutex> lock(db_mu_);
  if (!db_) {
    return;
  }
  const size_t prefix = energy_force_bytes(n_atoms);
  std::vector<char> buffer(prefix + result_metadata_bytes, 0);
  std::memcpy(buffer.data(), &out.energy, sizeof(double));
  std::memcpy(buffer.data() + sizeof(double), out.F,
              3 * n_atoms * sizeof(double));
  std::memcpy(buffer.data() + prefix, result_marker, sizeof(result_marker));
  char *metadata = buffer.data() + prefix + sizeof(result_marker);
  std::memcpy(metadata, &out.variance, sizeof(double));
  if (out.has_stress) {
    std::memcpy(metadata + sizeof(double), out.stress, sizeof(out.stress));
    buffer.back() = 1;
  }
  const auto bytes = storage_key(key);
  db_->Put(rocksdb::WriteOptions(), rocksdb::Slice(bytes.data(), bytes.size()),
           rocksdb::Slice(buffer.data(), buffer.size()));
}

/**
 * @details
 * Initializes the RocksDB options, specifically setting `create_if_missing`.
 * It attempts to open the database at the specified path. If the open fails,
 * an error is printed to stderr and the internal database pointer remains null.
 * If successful, the `own_db_` flag is set to true.
 */
PotentialCache::PotentialCache(const std::string &db_path,
                               bool create_if_missing) {
  rocksdb::Options options;
  options.create_if_missing = create_if_missing;
#if RGPOT_ROCKSDB_UNIQUE_PTR_OPEN
  std::unique_ptr<rocksdb::DB> db;
  rocksdb::Status status = rocksdb::DB::Open(options, db_path, &db);
  db_ = db.release();
#else
  rocksdb::Status status = rocksdb::DB::Open(options, db_path, &db_);
#endif
  if (!status.ok()) {
    std::cerr << "Unable to open RocksDB at " << db_path << ": "
              << status.ToString() << std::endl;
    db_ = nullptr;
  } else {
    own_db_ = true;
  }
}

/**
 * @details
 * Checks if the class owns the database pointer. If so, it deletes the
 * `rocksdb::DB` instance to prevent memory leaks.
 */
PotentialCache::~PotentialCache() {
  std::lock_guard<std::mutex> lock(db_mu_);
  if (own_db_ && db_) {
    delete db_;
    db_ = nullptr;
  }
}

/**
 * @details
 * This function allows for manually injecting a RocksDB pointer.
 * If the current instance already owns a database, it is deleted before
 * accepting the new pointer. Ownership is transferred away from this class
 * (own_db_ set to false), implying the caller manages the new pointer's
 * lifetime or it is a shared resource.
 *
 * @warning Use with caution to avoid double-free or memory leaks if the
 * passed pointer is managed elsewhere.
 */
void PotentialCache::set_db(rocksdb::DB *db) {
  std::lock_guard<std::mutex> lock(db_mu_);
  if (own_db_ && db_)
    delete db_;
  db_ = db;
  own_db_ = false;
}

/**
 * @details
 * Performs a binary copy (`std::memcpy`) from the serialized string buffer
 * back into the energy variable and force matrix.
 *
 * The layout is assumed to be:
 * `[double energy] [double force_0] ... [double force_N]`
 */
void PotentialCache::deserialize_hit(const std::string &hit, double &energy,
                                     rgpot::types::AtomMatrix &forces) {
  ForceOut out{.F = forces.data()};
  deserialize_hit(hit, out, static_cast<size_t>(forces.rows()));
  energy = out.energy;
}

/**
 * @details
 * Serializes the energy and forces into a contiguous binary buffer.
 * The buffer size is calculated as `sizeof(double) + N * sizeof(double)`.
 * This buffer is then stored in RocksDB using the provided key.
 *
 * @note If the database is not initialized, this function returns immediately.
 */
void PotentialCache::add_serialized(const KeyHash &kv, double energy,
                                    const rgpot::types::AtomMatrix &forces) {
  std::lock_guard<std::mutex> lock(db_mu_);
  if (!db_)
    return;
  // Calculate total size needed
  size_t value_size = sizeof(double) + forces.size() * sizeof(double);
  std::vector<char> buffer(value_size);

  // Copy energy to buffer
  std::memcpy(buffer.data(), &energy, sizeof(double));
  // Copy forces to buffer offset by sizeof(double)
  std::memcpy(buffer.data() + sizeof(double), forces.data(),
              forces.size() * sizeof(double));

  rocksdb::Slice value(buffer.data(), buffer.size());
  const auto bytes = storage_key(kv);
  db_->Put(rocksdb::WriteOptions(), rocksdb::Slice(bytes.data(), bytes.size()), value);
}

/**
 * @details
 * Queries the RocksDB instance for the given key.
 *
 * @return std::optional containing the serialized string if found,
 * or std::nullopt if the key does not exist or the DB is closed.
 */
std::optional<std::string> PotentialCache::find(const KeyHash &kv) {
  std::lock_guard<std::mutex> lock(db_mu_);
  if (!db_)
    return std::nullopt;
  std::string value;
  const auto bytes = storage_key(kv);
  rocksdb::Status s = db_->Get(rocksdb::ReadOptions(), rocksdb::Slice(bytes.data(), bytes.size()), &value);
  if (s.ok()) {
    return value;
  }
  return std::nullopt;
}
} // namespace rgpot::cache
