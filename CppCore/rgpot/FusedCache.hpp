#pragma once
// MIT License
// Copyright 2023--present rgpot developers

/**
 * @file FusedCache.hpp
 * @brief One cache for every fused evaluation of a C++ potential.
 *
 * The C++ form of the Rust @c FusedCache that backs @c rgpot_potential_t,
 * for potentials used without the Rust core (the Python module among them).
 * It has the same contract:
 *
 * - The key is the bits of the positions (@c std::bit_cast, never a
 *   tolerance) and a fingerprint of everything else that changes the answer:
 *   the atomic numbers, the cell, and the @c FusedSettingsSpec of the
 *   potential (periodicity, charge, spin, accuracy, electronic temperature,
 *   SCC iteration budget and mixer, engine build).
 * - The value is the outcome of the engine call: the result, or the
 *   exception it threw. The exception is stored as a @c std::exception_ptr
 *   and rethrown as the same object, so a typed failure such as
 *   @c SccNotConverged stays typed.
 * - A request with the stored key returns the stored outcome with no engine
 *   call. A request with another key runs the engine and replaces the entry.
 * - A retry reruns the engine only through a changed setting, which is a
 *   changed key. A plain retry of a failed geometry rethrows the stored
 *   failure.
 *
 * @ingroup rgpot_cpp
 */

#include <array>
#include <atomic>
#include <bit>
#include <cstddef>
#include <cstdint>
#include <exception>
#include <mutex>
#include <optional>
#include <string>
#include <utility>
#include <variant>
#include <vector>

#include "rgpot/ForceStructs.hpp"
#include "rgpot/FusedSettings.hpp"

namespace rgpot {

/// Everything but the positions that decides the answer of an engine call.
struct FusedFingerprint {
  std::optional<std::vector<int>> atomicNumbers;
  std::optional<std::array<std::uint64_t, 9>> cell;
  unsigned periodic = 0;
  std::uint64_t charge = 0;
  int spin = 0;
  std::uint64_t accuracy = 0;
  std::uint64_t electronicTemperature = 0;
  std::uint64_t warmMove = 0;
  std::uint64_t warmMinGap = 0;
  unsigned sccMaxIterations = 0;
  std::string sccMixer;
  std::string engineBuild;
  std::uint64_t epoch = 0;
  bool operator==(const FusedFingerprint &) const = default;
};

/// Positions by bit pattern, plus the fingerprint. Equality is exact.
struct FusedKey {
  std::vector<std::uint64_t> positions;
  FusedFingerprint fingerprint;
  bool operator==(const FusedKey &) const = default;
};

/// Whether an answer came from the engine or from the stored entry.
enum class FusedSource { Computed, Stored };

/// Single-entry cache of the outcome of a fused evaluation.
template <typename V> class FusedCache {
public:
  FusedCache() = default;
  FusedCache(const FusedCache &) = delete;
  FusedCache &operator=(const FusedCache &) = delete;

  /// The key of a request at @a in under @a settings. Positions are read
  /// as bits, so non-finite values key like any other.
  [[nodiscard]] FusedKey key(const ForceInput &in,
                             const FusedSettingsSpec &settings) const {
    FusedKey k;
    const std::size_t dim = 3 * in.nAtoms;
    k.positions.reserve(dim);
    for (std::size_t i = 0; i < dim; ++i) {
      k.positions.push_back(std::bit_cast<std::uint64_t>(in.pos[i]));
    }
    FusedFingerprint &f = k.fingerprint;
    if (in.atmnrs != nullptr) {
      f.atomicNumbers.emplace(in.atmnrs, in.atmnrs + in.nAtoms);
    }
    if (in.box != nullptr) {
      std::array<std::uint64_t, 9> bits{};
      for (std::size_t i = 0; i < 9; ++i) {
        bits[i] = std::bit_cast<std::uint64_t>(in.box[i]);
      }
      f.cell = bits;
    }
    f.periodic = settings.periodic;
    f.charge = std::bit_cast<std::uint64_t>(settings.charge);
    f.spin = settings.spin;
    f.accuracy = std::bit_cast<std::uint64_t>(settings.accuracy);
    f.electronicTemperature =
        std::bit_cast<std::uint64_t>(settings.electronicTemperature);
  f.warmMove = std::bit_cast<std::uint64_t>(settings.warmMove);
  f.warmMinGap = std::bit_cast<std::uint64_t>(settings.warmMinGap);
    f.sccMaxIterations = settings.sccMaxIterations;
    f.sccMixer = settings.sccMixer;
    f.engineBuild = settings.engineBuild;
    f.epoch = epoch_.load(std::memory_order_acquire);
    return k;
  }

  /**
   * @brief Answer a request: the stored outcome when @a key matches the
   *        entry, otherwise the outcome of @a engine, which replaces it.
   *
   * @a engine returns a @c V or throws a @c std::exception. A stored
   * failure is rethrown as the exception object the engine threw. The entry
   * stays locked while @a engine runs, so concurrent requests for one key
   * run the engine once.
   */
  template <typename Engine>
  std::pair<V, FusedSource> evaluate(const FusedKey &key, Engine &&engine) {
    std::lock_guard<std::mutex> lock(mu_);
    if (entry_ && entry_->key == key) {
      served_.fetch_add(1, std::memory_order_relaxed);
      return {take(entry_->outcome), FusedSource::Stored};
    }
    computed_.fetch_add(1, std::memory_order_relaxed);
    Outcome outcome;
    try {
      outcome = V(engine());
    } catch (const std::exception &) {
      outcome = std::current_exception();
    }
    entry_ = Entry{key, std::move(outcome)};
    return {take(entry_->outcome), FusedSource::Computed};
  }

  /// Declare that the potential changed in place. The next request has
  /// another key, so it runs the engine even at the stored positions.
  void invalidate() {
    std::lock_guard<std::mutex> lock(mu_);
    epoch_.fetch_add(1, std::memory_order_acq_rel);
    entry_.reset();
  }

  /// Engine calls made through the cache.
  [[nodiscard]] std::uint64_t computed() const noexcept {
    return computed_.load(std::memory_order_relaxed);
  }

  /// Requests answered from the stored entry.
  [[nodiscard]] std::uint64_t served() const noexcept {
    return served_.load(std::memory_order_relaxed);
  }

  void resetCounts() noexcept {
    computed_.store(0, std::memory_order_relaxed);
    served_.store(0, std::memory_order_relaxed);
  }

private:
  using Outcome = std::variant<std::monostate, V, std::exception_ptr>;

  struct Entry {
    FusedKey key;
    Outcome outcome;
  };

  static V take(const Outcome &outcome) {
    if (const auto *failure = std::get_if<std::exception_ptr>(&outcome)) {
      std::rethrow_exception(*failure);
    }
    return std::get<V>(outcome);
  }

  mutable std::mutex mu_;
  std::optional<Entry> entry_;
  std::atomic<std::uint64_t> epoch_{0};
  std::atomic<std::uint64_t> computed_{0};
  std::atomic<std::uint64_t> served_{0};
};

} // namespace rgpot
