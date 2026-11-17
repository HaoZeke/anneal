#pragma once
// MIT License
// Copyright 2023--present rgpot developers

/**
 * @brief Pair terms a two-pass potential computes once.
 *
 * An embedded-atom potential walks its pairs twice: the first pass sums
 * each atom's density, and the second needs the embedding of both ends of
 * a pair before it can write that pair's force. The powers and quotients
 * of the distance that the force takes are mostly ones the first pass
 * already had. @c PairTerms keeps a few doubles for each pair the first
 * pass accepts, and the second pass reads them back.
 *
 * Two walks over one neighbour update visit the same pairs in the same
 * order with the same separations, so when both passes accept a pair by
 * the same test, the k-th pair the second pass accepts is the k-th one the
 * first pass kept. A team walk has the same order within each thread, and
 * takes one @c PairTerms per thread.
 */

#include <algorithm>
#include <cstddef>
#include <limits>
#include <memory>

namespace rgpot {

/**
 * @brief @c Width doubles for each pair of a first pass, in visit order.
 *
 * A pass keeps pairs until the room that @c record made is full. Past it
 * @c push and @c next return null and the second pass computes its terms
 * again. Nothing is allocated during a pass, so a pair loop holds no
 * call; the next @c record makes room for every pair the last pass saw,
 * up to @c limit(). The block is kept from one evaluation to the next, so
 * a potential that outlives a call refills the same memory.
 *
 * A team keeps one per thread side by side, and every pair a thread
 * pushes writes a count, so each takes a cache line of its own.
 */
template <std::size_t Width>
class alignas(64) PairTerms {
public:
  static_assert(Width > 0, "a pair keeps at least one term");

  /// Pairs kept by default: 32 MiB of terms.
  static constexpr std::size_t kDefaultLimit = (std::size_t{32} << 20) / (Width * sizeof(double));

  /// Starts a first pass with room for @a expected pairs or for every
  /// pair the last pass saw, whichever is more, capped at the limit.
  void record(std::size_t expected) {
    const std::size_t want = std::min(std::max(expected, size_ + dropped_), limit_);
    if (want > capacity_ || capacity_ > limit_) {
      data_.reset();
      capacity_ = 0;
      if (want > 0) {
        data_ = std::make_unique_for_overwrite<double[]>(want * Width);
        capacity_ = want;
      }
    }
    size_ = 0;
    read_ = 0;
    dropped_ = 0;
  }

  /// Room for the terms of the next pair, or null once the pass is out of
  /// room. The pointer is good until the next record.
  [[nodiscard]] double *push() noexcept {
    // Counts, not pointers: a pointer store may alias every reference a
    // pair loop captured, and those would be loaded again for each pair.
    if (size_ == capacity_) {
      ++dropped_;
      return nullptr;
    }
    return data_.get() + Width * size_++;
  }

  /// Starts the second pass.
  void replay() noexcept { read_ = 0; }

  /// The terms of the next pair, or null past the pairs kept.
  [[nodiscard]] const double *next() noexcept {
    if (read_ == size_) {
      return nullptr;
    }
    return data_.get() + Width * read_++;
  }

  /// Pairs the last first pass kept.
  [[nodiscard]] std::size_t size() const noexcept { return size_; }

  [[nodiscard]] std::size_t limit() const noexcept { return limit_; }

  /// Pairs to keep from the next record on. 0 keeps none.
  void setLimit(std::size_t pairs) noexcept { limit_ = std::min(pairs, kMaxPairs); }

  /// Frees the block.
  void release() noexcept {
    data_.reset();
    capacity_ = 0;
    size_ = 0;
    read_ = 0;
    dropped_ = 0;
  }

private:
  static constexpr std::size_t kMaxPairs = std::numeric_limits<std::size_t>::max() /
                                           (Width * sizeof(double));

  std::unique_ptr<double[]> data_;
  std::size_t capacity_ = 0;
  std::size_t size_ = 0;
  std::size_t read_ = 0;
  std::size_t dropped_ = 0;
  std::size_t limit_ = kDefaultLimit;
};

} // namespace rgpot
