#pragma once
// MIT License
// Copyright 2023--present rgpot developers

/**
 * @brief Header file for the PotentialCache class.
 *
 * This file defines the caching mechanism for the rgpot library, utilizing
 * RocksDB to store and retrieve potential energy and force calculations.
 */

#include "rgpot/ForceStructs.hpp"
#include "rgpot/pot_types.hpp"
#include "rgpot/types/AtomMatrix.hpp"
#include <atomic>
#include <cstdint>
#include <mutex>
#include <optional>
#include <rocksdb/db.h>
#include <rocksdb/version.h>
#include <string>

// RocksDB 10.4 added the std::unique_ptr<DB> DB::Open overload and 11.0
// removed the raw-pointer one.
#if ROCKSDB_MAJOR > 10 || (ROCKSDB_MAJOR == 10 && ROCKSDB_MINOR >= 4)
#define RGPOT_ROCKSDB_UNIQUE_PTR_OPEN 1
#else
#define RGPOT_ROCKSDB_UNIQUE_PTR_OPEN 0
#endif

namespace rgpot::cache {

/**
 * @class KeyHash
 * @brief Struct to hold the hash and string key for caching.
 * @ingroup rgpot_cache
 */
struct KeyHash {
  uint64_t hash; //!< The numeric hash value, 64 bits on every platform.

  /**
   * @brief Constructor for KeyHash.
   * @param _hash The numeric hash to wrap.
   */
  KeyHash(uint64_t _hash) : hash{_hash} {}
};

/// Hash the ordered geometry, species, cell and potential identity.
KeyHash result_key(const ForceInput &input, PotType type, uint64_t parameters);

/**
 * @brief Evaluations a cache handle has answered, split by origin.
 *
 * `computed` counts results the kernel produced while the cache was
 * attached; `served` counts results returned from the cache without
 * running the kernel. A caller that charges evaluations picks which field
 * it bills, so two runs under different cache states stay comparable.
 * @ingroup rgpot_cache
 */
struct EvalCounts {
  uint64_t computed = 0; //!< Results produced by the kernel.
  uint64_t served = 0;   //!< Results answered from the cache.
};

/**
 * @class PotentialCache
 * @brief Caches potential energy and force calculations using RocksDB.
 * @ingroup rgpot_cache
 */
class PotentialCache {
private:
  rocksdb::DB *db_ = nullptr; //!< Pointer to the RocksDB instance.
  bool own_db_ = false;       //!< Ownership flag for the DB pointer.
  /// F2: covers db_ only. Callers detach before taking it. The counters
  /// below stay atomic and are not under this mutex.
  mutable std::mutex db_mu_;
  std::atomic<uint64_t> computed_{0}; //!< Kernel-produced result count.
  std::atomic<uint64_t> served_{0};   //!< Cache-served result count.

public:
  /**
   * @brief Constructor opens the DB at the given path.
   * @param db_path Path to the RocksDB database.
   * @param create_if_missing Toggle creation of DB if absent.
   */
  explicit PotentialCache(const std::string &db_path,
                          bool create_if_missing = true);

  /**
   * @brief Default constructor.
   */
  PotentialCache() = default;

  /**
   * @brief Destructor.
   */
  ~PotentialCache();

  /**
   * @brief Helper for manual pointer setting.
   * @param db Pointer to an existing RocksDB instance.
   * @return Void.
   */
  void set_db(rocksdb::DB *db);

  /**
   * @brief Deserializes a cache hit into output containers.
   * @param value Serialized string from the cache.
   * @param energy Reference to store the energy.
   * @param forces Reference to store the forces.
   * @return Void.
   */
  void deserialize_hit(const std::string &value, double &energy,
                       rgpot::types::AtomMatrix &forces);

  /**
   * @brief Adds a serialized calculation to the cache.
   * @param key Unique hash key for the configuration.
   * @param energy Calculated energy.
   * @param forces Calculated forces.
   * @return Void.
   */
  void add_serialized(const KeyHash &key, double energy,
                      const rgpot::types::AtomMatrix &forces);

  /// Whether a database is attached.
  [[nodiscard]] bool is_open() const {
    std::lock_guard<std::mutex> lock(db_mu_);
    return db_ != nullptr;
  }

  /// Result records retain the energy/force prefix and append versioned
  /// variance and stress metadata. Energy/force-only records remain readable.
  static bool has_result_metadata(const std::string &value, size_t n_atoms);
  void deserialize_hit(const std::string &value, ForceOut &out, size_t n_atoms);
  void add_serialized(const KeyHash &key, const ForceOut &out, size_t n_atoms);

  /// Records @p n results produced by the kernel.
  void note_computed(uint64_t n = 1) {
    computed_.fetch_add(n, std::memory_order_relaxed);
  }
  /// Records @p n results answered from the cache.
  void note_served(uint64_t n = 1) {
    served_.fetch_add(n, std::memory_order_relaxed);
  }
  /// Snapshot of the computed and served counters of this handle.
  [[nodiscard]] EvalCounts counts() const {
    return {computed_.load(std::memory_order_relaxed),
            served_.load(std::memory_order_relaxed)};
  }
  /// Zeroes both counters; stored entries are untouched.
  void reset_counts() {
    computed_.store(0, std::memory_order_relaxed);
    served_.store(0, std::memory_order_relaxed);
  }

  /**
   * @brief Searches the cache for a specific key.
   * @param key Unique hash key.
   * @return Optional string containing the serialized data.
   */
  std::optional<std::string> find(const KeyHash &key);
};

} // namespace rgpot::cache
