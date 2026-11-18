#pragma once
// MIT License
// Copyright 2023--present rgpot developers

/**
 * @brief A potential that a C entry point keeps between calls.
 *
 * An entry point that takes its parameters with every call, such as
 * rgpot_sutton_chen_energy_forces, would build a potential each time and
 * drop it, and with it the neighbour update and the scratch the potential
 * keeps for its next evaluation. @c Kept holds one potential per thread
 * for each potential type, with the key it was built for: the parameters,
 * the atom count and, when the call has them, the species. A call with
 * the same key evaluates the potential held. Any other key drops it and
 * builds a new one, so no state crosses a change of parameters, atom
 * count or species.
 *
 * The key compares bits: 0.0 and -0.0 are different parameters and a NaN
 * matches itself. A potential whose result depends on anything else, such
 * as a file it reads, needs that in the key as well.
 */

#include <array>
#include <cstddef>
#include <cstdint>
#include <cstring>
#include <memory>
#include <vector>

// Hidden, so each shared object keeps its own slots: objects built from
// two versions of this header never share one slot layout.
#if (defined(__GNUC__) || defined(__clang__)) && !defined(_WIN32) && !defined(__CYGWIN__)
#define RGPOT_KEPT_HIDDEN __attribute__((visibility("hidden")))
#else
#define RGPOT_KEPT_HIDDEN
#endif

namespace rgpot {

template <typename Pot, std::size_t Slots = 1>
class RGPOT_KEPT_HIDDEN Kept;

/// What a kept potential was built for.
class KeptKey {
public:
  /// Room for 16 eight-byte fields. A key that outgrows it never matches,
  /// so every call builds.
  static constexpr std::size_t kBytes = 128;

  KeptKey &f64(double v) noexcept { return raw(&v, sizeof(v)); }
  KeptKey &i64(std::int64_t v) noexcept { return raw(&v, sizeof(v)); }
  KeptKey &ptr(const void *v) noexcept { return raw(&v, sizeof(v)); }

  /// The atom count, and the species of each atom, or null for a call
  /// without species. The species are read, not copied, until the key is
  /// kept.
  KeptKey &atoms(std::size_t n, const int *species) noexcept {
    n_ = n;
    species_ = species;
    return *this;
  }

private:
  template <typename Pot, std::size_t Slots>
  friend class Kept;

  KeptKey &raw(const void *v, std::size_t size) noexcept {
    if (len_ + size > kBytes) {
      overflow_ = true;
      return *this;
    }
    std::memcpy(params_ + len_, v, size);
    len_ += size;
    return *this;
  }

  unsigned char params_[kBytes] = {};
  std::size_t len_ = 0;
  bool overflow_ = false;
  std::size_t n_ = 0;
  const int *species_ = nullptr;
};

/// Builds and reuses on one thread, for one potential type.
struct KeptCounts {
  std::uint64_t builds = 0;
  std::uint64_t reuses = 0;
};

/// The object(s) of type @c Pot this thread keeps (up to @c Slots, LRU).
template <typename Pot, std::size_t Slots>
class RGPOT_KEPT_HIDDEN Kept {
  static_assert(Slots >= 1, "Kept requires at least one slot");

public:
  /// `run(pot)` on the object kept for @a key. When no slot matches, the
  /// least recently used slot (or an empty one) is cleared and `make()`,
  /// which returns a `std::unique_ptr<Pot>`, builds the next one. A throw
  /// from `make` leaves that slot empty. A throw from `run` drops the
  /// object, whose scratch it may have left half written, and passes on.
  template <typename Make, typename Run>
  static decltype(auto) with(const KeptKey &key, Make &&make, Run &&run) {
    if (retired()) {
      const std::unique_ptr<Pot> pot = make();
      return run(*pot);
    }
    Table &table = local();
    Slot *slot = table.find(key);
    if (slot != nullptr) {
      slot->stamp = ++table.clock;
      ++table.counts.reuses;
    } else {
      slot = &table.victim();
      slot->clear();
      std::unique_ptr<Pot> pot = make();
      slot->keep(key);
      slot->pot = std::move(pot);
      slot->stamp = ++table.clock;
      ++table.counts.builds;
    }
    try {
      return run(*slot->pot);
    } catch (...) {
      slot->clear();
      throw;
    }
  }

  /// The object this thread currently keeps for @a key, or null when no
  /// slot matches. Leaves the slots and counts unchanged.
  [[nodiscard]] static const Pot *peek(const KeptKey &key) noexcept {
    if (retired()) {
      return nullptr;
    }
    const Slot *slot = local().find(key);
    return slot != nullptr ? slot->pot.get() : nullptr;
  }

  [[nodiscard]] static KeptCounts counts() noexcept {
    return retired() ? KeptCounts{} : local().counts;
  }

  /// Drops every object this thread keeps.
  static void release() noexcept {
    if (retired()) {
      return;
    }
    Table &table = local();
    for (Slot &slot : table.slots) {
      slot.clear();
    }
  }

  /// Drops the object this thread keeps for @a key, if any.
  static void release(const KeptKey &key) noexcept {
    if (retired()) {
      return;
    }
    Slot *slot = local().find(key);
    if (slot != nullptr) {
      slot->clear();
    }
  }

private:
  struct Slot {
    std::unique_ptr<Pot> pot;
    unsigned char params[KeptKey::kBytes] = {};
    std::size_t len = 0;
    std::size_t n = 0;
    bool has_species = false;
    std::vector<int> species;
    std::uint64_t stamp = 0;

    [[nodiscard]] bool matches(const KeptKey &key) const noexcept {
      if (key.overflow_ || key.len_ != len || key.n_ != n ||
          (key.species_ != nullptr) != has_species) {
        return false;
      }
      if (std::memcmp(key.params_, params, len) != 0) {
        return false;
      }
      return !has_species || n == 0 ||
             std::memcmp(key.species_, species.data(), n * sizeof(int)) == 0;
    }

    void keep(const KeptKey &key) {
      if (key.species_ != nullptr) {
        species.assign(key.species_, key.species_ + key.n_);
      } else {
        species.clear();
      }
      std::memcpy(params, key.params_, key.len_);
      // An overflowed key lost its last fields, so what it built matches
      // no key, not even one with the fields it kept.
      len = key.overflow_ ? KeptKey::kBytes + 1 : key.len_;
      n = key.n_;
      has_species = key.species_ != nullptr;
    }

    void clear() noexcept {
      pot.reset();
      std::vector<int>().swap(species);
      len = 0;
      n = 0;
      has_species = false;
      stamp = 0;
    }
  };

  struct Table {
    std::array<Slot, Slots> slots{};
    KeptCounts counts{};
    std::uint64_t clock = 0;

    Table() = default;
    Table(const Table &) = delete;
    Table &operator=(const Table &) = delete;
    ~Table() { retired() = true; }

    [[nodiscard]] Slot *find(const KeptKey &key) noexcept {
      for (Slot &slot : slots) {
        if (slot.pot != nullptr && slot.matches(key)) {
          return &slot;
        }
      }
      return nullptr;
    }

    [[nodiscard]] const Slot *find(const KeptKey &key) const noexcept {
      for (const Slot &slot : slots) {
        if (slot.pot != nullptr && slot.matches(key)) {
          return &slot;
        }
      }
      return nullptr;
    }

    [[nodiscard]] Slot &victim() noexcept {
      Slot *oldest = &slots[0];
      for (Slot &slot : slots) {
        if (slot.pot == nullptr) {
          return slot;
        }
        if (slot.stamp < oldest->stamp) {
          oldest = &slot;
        }
      }
      return *oldest;
    }
  };

  /// True once this thread's table is gone. Thread-local objects are
  /// destroyed in the reverse of the order they were first used, so an
  /// object kept in one table can be destroyed after another table has
  /// been, and its destructor may then release a slot of that table. The
  /// flag needs no destructor, so it can be read all through thread exit.
  static bool &retired() noexcept {
    thread_local bool gone = false;
    return gone;
  }

  static Table &local() {
    thread_local Table table;
    return table;
  }
};

} // namespace rgpot
