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

[[nodiscard]] MorseSplitSnapshot morseStoredSplitSnapshot();

} // namespace detail

} // namespace rgpot
