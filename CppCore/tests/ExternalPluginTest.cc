// MIT License
// Copyright 2023--present rgpot developers
//
// Integration tests for external plugin .so files.
// Each test loads a plugin via PluginRegistry::load_plugin(), creates
// an instance via the bridge, and verifies energy/forces.

#include <catch2/catch_test_macros.hpp>
#include <catch2/matchers/catch_matchers_floating_point.hpp>

#include <array>
#include <cmath>
#include <cstdlib>
#include <iostream>
#include <sstream>
#include <string>
#include <vector>

#include "rgpot/plugin/PluginBridge.hpp"
#include "rgpot/plugin/PluginRegistry.hpp"
#include "rgpot/types/AtomMatrix.hpp"

using Catch::Matchers::WithinAbs;
using Catch::Matchers::WithinRel;
using rgpot::types::AtomMatrix;

// Helper: get plugin .so path from RGPOT_TEST_PLUGIN_DIR env var
static std::string plugin_path(const char *filename) {
  const char *dir = std::getenv("RGPOT_TEST_PLUGIN_DIR");
  if (!dir)
    dir = ".";
  std::string library(filename);
#ifdef _WIN32
  library.replace(library.size() - 3, 3, ".dll");
#elif defined(__APPLE__)
  library.replace(library.size() - 3, 3, ".dylib");
#endif
  return std::string(dir) + "/" + library;
}

// Two-atom test system
static const double two_pos[] = {0.0, 0.0, 0.0, 1.5, 0.0, 0.0};
static const int two_atm[] = {1, 1};
static const std::array<std::array<double, 3>, 3> big_box = {
    {{20.0, 0.0, 0.0}, {0.0, 20.0, 0.0}, {0.0, 0.0, 20.0}}};

static AtomMatrix make_pos(const double *data, int n) {
  AtomMatrix m(n, 3);
  for (int i = 0; i < n; ++i)
    for (int j = 0; j < 3; ++j)
      m(i, j) = data[i * 3 + j];
  return m;
}

// ---- Null plugin ----

TEST_CASE("External null plugin: zero energy and forces",
          "[plugin][external]") {
  auto &reg = rgpot::plugin::PluginRegistry::instance();
  reg.load_plugin(plugin_path("librgpot_null.so"));
  const auto *p = reg.find("null");
  REQUIRE(p != nullptr);

  auto pot = rgpot::plugin::create_from_plugin(p->desc);
  auto positions = make_pos(two_pos, 2);
  std::vector<int> types(two_atm, two_atm + 2);

  auto [energy, forces, unused_variance] = (*pot)(positions, types, big_box);

  REQUIRE(energy == 0.0);
  for (int i = 0; i < 2; ++i)
    for (int j = 0; j < 3; ++j)
      REQUIRE(forces(i, j) == 0.0);
}

// ---- Morse plugin ----

TEST_CASE("External Morse plugin: bound pair", "[plugin][external]") {
  auto &reg = rgpot::plugin::PluginRegistry::instance();
  reg.load_plugin(plugin_path("librgpot_morse.so"));
  const auto *p = reg.find("morse");
  REQUIRE(p != nullptr);

  // Default Morse: D_e=1, alpha=1, r_e=1, cutoff=10
  auto pot = rgpot::plugin::create_from_plugin(p->desc);
  auto positions = make_pos(two_pos, 2);
  std::vector<int> types(two_atm, two_atm + 2);

  auto [energy, forces, unused_variance] = (*pot)(positions, types, big_box);

  // At r=1.5, E = D_e*(1-exp(-alpha*(1.5-1)))^2 = (1-exp(-0.5))^2
  double exp_term = std::exp(-0.5);
  double expected_e = (1.0 - exp_term) * (1.0 - exp_term);
  REQUIRE_THAT(energy, WithinRel(expected_e, 1e-10));

  // Forces should be equal and opposite
  REQUIRE_THAT(forces(0, 0) + forces(1, 0), WithinAbs(0.0, 1e-12));
  // Force on atom 0 should be positive (attractive, pulling toward atom 1 at
  // +x)
  REQUIRE(forces(0, 0) > 0.0);
}

TEST_CASE("External Morse plugin: forces sum to zero", "[plugin][external]") {
  auto &reg = rgpot::plugin::PluginRegistry::instance();
  reg.load_plugin(plugin_path("librgpot_morse.so")); // may already be loaded
  const auto *p = reg.find("morse");
  REQUIRE(p != nullptr);

  auto pot = rgpot::plugin::create_from_plugin(p->desc);

  // Three-atom system
  double pos3[] = {0.0, 0.0, 0.0, 1.2, 0.0, 0.0, 0.6, 1.0, 0.0};
  int atm3[] = {1, 1, 1};
  auto positions = make_pos(pos3, 3);
  std::vector<int> types(atm3, atm3 + 3);

  auto [energy, forces, unused_variance] = (*pot)(positions, types, big_box);

  double fx = 0, fy = 0, fz = 0;
  for (int i = 0; i < 3; ++i) {
    fx += forces(i, 0);
    fy += forces(i, 1);
    fz += forces(i, 2);
  }
  REQUIRE_THAT(fx, WithinAbs(0.0, 1e-10));
  REQUIRE_THAT(fy, WithinAbs(0.0, 1e-10));
  REQUIRE_THAT(fz, WithinAbs(0.0, 1e-10));
}

// ---- Discovery test ----

TEST_CASE("Plugin registry tracks all loaded plugins", "[plugin][external]") {
  auto &reg = rgpot::plugin::PluginRegistry::instance();
  // Ensure all three are loaded (idempotent -- returns false if already loaded)
  reg.load_plugin(plugin_path("librgpot_null.so"));
  reg.load_plugin(plugin_path("librgpot_morse.so"));
  if (std::getenv("RGPOT_TEST_PLUGIN_DIR")) {
    reg.load_plugin(plugin_path("librgpot_coulomb.so"));
  }

  auto names = reg.list_plugins();
  bool has_lj = false, has_null = false, has_morse = false;
  for (const auto &n : names) {
    if (n == "lennard-jones")
      has_lj = true;
    if (n == "null")
      has_null = true;
    if (n == "morse")
      has_morse = true;
  }
  REQUIRE(has_lj);    // builtin
  REQUIRE(has_null);  // external C
  REQUIRE(has_morse); // external C
}

TEST_CASE("an external plugin cannot replace a builtin name",
          "[plugin][external]") {
  auto &reg = rgpot::plugin::PluginRegistry::instance();
  std::stringstream captured;
  const auto old = std::cerr.rdbuf(captured.rdbuf());
  const bool loaded = reg.load_plugin(plugin_path("librgpot_lj_collision.so"));
  std::cerr.rdbuf(old);
  REQUIRE_FALSE(loaded);
  const auto *lj = reg.find("lennard-jones");
  REQUIRE(lj != nullptr);
  REQUIRE(std::string(lj->desc->version) == "1.0.0");
  REQUIRE(captured.str().find("[rgpot] plugin name collision ") !=
          std::string::npos);
  REQUIRE(captured.str().find(": lennard-jones is already registered") !=
          std::string::npos);
}
