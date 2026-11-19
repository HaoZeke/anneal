#pragma once
// MIT License
// Copyright 2023--present rgpot developers
//
// Private snapshot of the cached Morse stored-list slice splits. Kept out of
// the installed MorsePot.hpp header and excluded from Doxygen and meson/CMake
// header installation so public consumers only see the MorsePot interface.

#include <cstddef>
#include <cstdint>
#include <vector>

#if (defined(__GNUC__) || defined(__clang__)) && !defined(_WIN32) &&           \
    !defined(__CYGWIN__)
#define RGPOT_MORSE_TEST_HIDDEN __attribute__((visibility("hidden")))
#else
#define RGPOT_MORSE_TEST_HIDDEN
#endif

namespace rgpot {

namespace nlist {
class Neighbours;
} // namespace nlist

namespace detail {

struct MorseSplitSnapshot {
  const nlist::Neighbours *owner = nullptr;
  std::uint64_t build_id = 0;
  std::size_t atoms = 0;
  int slices = 0;
  std::size_t total_up = 0;
  std::size_t total_inc = 0;
  std::size_t up_off_cap = 0;
  std::size_t lo_off_cap = 0;
  std::size_t lo_u_cap = 0;
  std::size_t lo_count_cap = 0;
  std::size_t active_bits_cap = 0;
  std::size_t pair_forces_cap = 0;
  std::size_t pair_he_cap = 0;
  std::size_t pair_hv_cap = 0;
  std::size_t row_e_cap = 0;
  std::size_t row_s_cap = 0;
  std::vector<std::size_t> p0_splits;
  std::vector<std::size_t> p1_splits;
};

[[nodiscard]] RGPOT_MORSE_TEST_HIDDEN MorseSplitSnapshot
morseStoredSplitSnapshot();

RGPOT_MORSE_TEST_HIDDEN void
morseAtomicFetchOrWord(std::uint64_t &word, std::uint64_t mask) noexcept;

/// Element capacity above which a smaller request releases scratch.
/// `resizeBuffer` shrinks when the retained capacity is above this cap and
/// the new request is under a quarter of that capacity.
[[nodiscard]] RGPOT_MORSE_TEST_HIDDEN std::size_t
morseWorkspaceRetainCap() noexcept;

} // namespace detail

} // namespace rgpot

#undef RGPOT_MORSE_TEST_HIDDEN
