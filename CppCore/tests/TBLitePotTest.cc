// MIT License
// Copyright 2023--present rgpot developers

#include <catch2/catch_test_macros.hpp>
#include <catch2/matchers/catch_matchers_floating_point.hpp>

#include <array>
#include <exception>
#include <vector>

#include "rgpot/TBLitePot/TBLitePot.hpp"
#include "rgpot/types/AtomMatrix.hpp"

#include "XtbHotClusters.hpp"

using rgpot::types::AtomMatrix;

// Water molecule geometry (Angstrom)
static const double water_pos[] = {
    0.00000000, 0.00000000,  0.11779000,  // O
    0.00000000, 0.75545000,  -0.47116000, // H
    0.00000000, -0.75545000, -0.47116000  // H
};
static const int water_atmnrs[] = {8, 1, 1};

TEST_CASE("TBLitePot GFN2 water energy", "[tblite]") {
  rgpot::TBLitePot pot;

  AtomMatrix positions(3, 3);
  for (int i = 0; i < 3; ++i)
    for (int j = 0; j < 3; ++j)
      positions(i, j) = water_pos[i * 3 + j];
  std::vector<int> atmtypes(water_atmnrs, water_atmnrs + 3);
  std::array<std::array<double, 3>, 3> box = {
      {{100.0, 0.0, 0.0}, {0.0, 100.0, 0.0}, {0.0, 0.0, 100.0}}};

  auto [energy, forces, variance] = pot(positions, atmtypes, box);
  (void)variance;

  // GFN2-xTB water energy is approximately -5.07 Hartree = -137.9 eV
  REQUIRE(energy < 0.0);
  REQUIRE_THAT(energy, Catch::Matchers::WithinAbs(-137.9, 1.0));

  // Forces should sum to approximately zero (translational invariance)
  double fx_sum = 0.0, fy_sum = 0.0, fz_sum = 0.0;
  for (size_t i = 0; i < 3; ++i) {
    fx_sum += forces(i, 0);
    fy_sum += forces(i, 1);
    fz_sum += forces(i, 2);
  }
  REQUIRE_THAT(fx_sum, Catch::Matchers::WithinAbs(0.0, 1e-6));
  REQUIRE_THAT(fy_sum, Catch::Matchers::WithinAbs(0.0, 1e-6));
  REQUIRE_THAT(fz_sum, Catch::Matchers::WithinAbs(0.0, 1e-6));
}

TEST_CASE("TBLitePot GFN1 water energy", "[tblite]") {
  rgpot::TBLiteConfig cfg;
  cfg.method = rgpot::TBLiteMethod::GFN1;
  rgpot::TBLitePot pot(cfg);

  AtomMatrix positions(3, 3);
  for (int i = 0; i < 3; ++i)
    for (int j = 0; j < 3; ++j)
      positions(i, j) = water_pos[i * 3 + j];
  std::vector<int> atmtypes(water_atmnrs, water_atmnrs + 3);
  std::array<std::array<double, 3>, 3> box = {
      {{100.0, 0.0, 0.0}, {0.0, 100.0, 0.0}, {0.0, 0.0, 100.0}}};

  auto [energy, forces, variance] = pot(positions, atmtypes, box);
  (void)variance;

  // GFN1-xTB energy should be negative for a bound molecule
  REQUIRE(energy < 0.0);
}

// A failed SCF leaves its last iterate in the result object. The next call
// has to start from tblite's own guess, as a new instance does. The SCF of
// Si13 frames 2 and 3 does not converge at 300 K. Seeded from frame 2,
// frame 3 used to converge, to -588.78 eV.
TEST_CASE("TBLitePot does not seed an SCF from a failed one", "[tblite][hot]") {
  using namespace rgpot_test_hot;
  auto frame = [](int f) {
    AtomMatrix positions(kSi13Atoms, 3);
    for (int i = 0; i < kSi13Atoms; ++i)
      for (int j = 0; j < 3; ++j)
        positions(i, j) = kSi13Pos[f][i][j];
    return positions;
  };
  const std::vector<int> atmtypes(kSi13Atoms, kSi13Z);
  std::array<std::array<double, 3>, 3> box{};
  box[0][0] = box[1][1] = box[2][2] = 60.0;
  auto evaluate = [&](rgpot::TBLitePot &pot, int f, double *energy) {
    try {
      auto [e, forces, variance] = pot(frame(f), atmtypes, box);
      (void)forces;
      (void)variance;
      *energy = e;
      return true;
    } catch (const std::exception &) {
      return false;
    }
  };

  rgpot::TBLitePot session;
  int checked = 0;
  bool previous_failed = false;
  for (int f = 2; f <= 3; ++f) {
    double e_session = 0.0;
    const bool ok_session = evaluate(session, f, &e_session);
    rgpot::TBLitePot fresh;
    double e_fresh = 0.0;
    const bool ok_fresh = evaluate(fresh, f, &e_fresh);
    UNSCOPED_INFO("frame " << f << " session " << ok_session << " " << e_session
                           << " fresh " << ok_fresh << " " << e_fresh);
    if (previous_failed) {
      ++checked;
      REQUIRE(ok_session == ok_fresh);
      if (ok_session)
        REQUIRE_THAT(e_session, Catch::Matchers::WithinAbs(e_fresh, 1e-6));
    }
    previous_failed = !ok_session;
  }
  REQUIRE(checked > 0);
}
