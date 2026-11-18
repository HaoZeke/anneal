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
  std::vector<std::size_t> p0_splits;
  std::vector<std::size_t> p1_splits;
};

[[nodiscard]] RGPOT_MORSE_TEST_HIDDEN MorseSplitSnapshot
morseStoredSplitSnapshot();

} // namespace detail

} // namespace rgpot

#undef RGPOT_MORSE_TEST_HIDDEN
