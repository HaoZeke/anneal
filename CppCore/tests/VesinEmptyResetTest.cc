// MIT License
// Copyright 2023--present rgpot developers
//
// A neighbour list that stored no pairs still has a null pairs pointer.
// Debug builds poison that pointer when the list is reset. Two atoms
// outside the cutoff, then a second build, take that path.

#include <cstdlib>

#include <catch2/catch_all.hpp>

#include "vesin.h"

namespace {

VesinOptions options_for(VesinAlgorithm algorithm, double skin) {
  VesinOptions options{};
  options.cutoff = 2.0;
  options.full = false;
  options.sorted = false;
  options.algorithm = algorithm;
  options.skin = skin;
  options.n_threads = 1;
  options.return_shifts = true;
  options.return_distances = true;
  options.return_vectors = true;
  return options;
}

void build(const double (&points)[2][3], VesinOptions options, VesinNeighborList &neighbors) {
  const double box[3][3] = {
      {30.0, 0.0, 0.0},
      {0.0, 30.0, 0.0},
      {0.0, 0.0, 30.0},
  };
  const bool periodic[3] = {false, false, false};
  const char *error = nullptr;
  const VesinDevice device{VesinCPU, 0};
  const int status =
      vesin_neighbors(points, 2, box, periodic, device, options, &neighbors, &error);
  if (status != EXIT_SUCCESS) {
    if (error != nullptr) {
      FAIL(error);
    }
    FAIL("vesin_neighbors failed");
  }
}

} // namespace

TEST_CASE("resetting a vesin neighbour list with zero pairs", "[vesin][nlist]") {
  // Ten angstroms apart, cutoff of two: no pair in either direction.
  const double apart[2][3] = {
      {0.0, 0.0, 0.0},
      {10.0, 0.0, 0.0},
  };
  const double close[2][3] = {
      {0.0, 0.0, 0.0},
      {1.0, 0.0, 0.0},
  };

  SECTION("cell list recompute") {
    VesinNeighborList neighbors;
    const auto options = options_for(VesinCellList, 0.0);
    build(apart, options, neighbors);
    REQUIRE(neighbors.length == 0);
    build(apart, options, neighbors);
    REQUIRE(neighbors.length == 0);
    vesin_free(&neighbors);
  }

  SECTION("brute force recompute") {
    VesinNeighborList neighbors;
    const auto options = options_for(VesinBruteForce, 0.0);
    build(apart, options, neighbors);
    REQUIRE(neighbors.length == 0);
    build(apart, options, neighbors);
    REQUIRE(neighbors.length == 0);
    vesin_free(&neighbors);
  }

  SECTION("skin filter recompute") {
    VesinNeighborList neighbors;
    const auto options = options_for(VesinCellList, 1.0);
    build(apart, options, neighbors);
    REQUIRE(neighbors.length == 0);
    build(apart, options, neighbors);
    REQUIRE(neighbors.length == 0);
    vesin_free(&neighbors);
  }

  SECTION("a stored pair then none") {
    VesinNeighborList neighbors;
    const auto options = options_for(VesinBruteForce, 0.0);
    build(close, options, neighbors);
    REQUIRE(neighbors.length == 1);
    build(apart, options, neighbors);
    REQUIRE(neighbors.length == 0);
    build(apart, options, neighbors);
    REQUIRE(neighbors.length == 0);
    vesin_free(&neighbors);
  }
}
