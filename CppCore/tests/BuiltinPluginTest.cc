// MIT License
// Copyright 2023--present rgpot developers

#include <catch2/catch_test_macros.hpp>

#include <string>
#include <vector>

#include "rgpot/LennardJones/LJPot.hpp"
#include "rgpot/Morse/MorsePot.hpp"
#include "rgpot/PotHelpers.hpp"
#include "rgpot/SuttonChen/SuttonChenPot.hpp"
#include "rgpot/plugin/PluginRegistry.hpp"
#ifdef RGPOT_HAS_EAM_TABLE
#include "rgpot/EAMTable/EAMTablePot.hpp"
#endif

namespace {

const rgpot_plugin_descriptor_t *pluginNamed(const char *name) {
  auto &registry = rgpot::plugin::PluginRegistry::instance();
  const auto *found = registry.find(name);
  REQUIRE(found != nullptr);
  return found->desc;
}

double pluginEnergy(const rgpot_plugin_descriptor_t *desc, const char *config,
                    const double *pos, const int *species, const double *box,
                    std::size_t n) {
  rgpot_plugin_instance *instance = nullptr;
  REQUIRE(desc->create(config, &instance) == RGPOT_PLUGIN_OK);
  std::vector<double> forces(3 * n);
  rgpot_plugin_input_t input{};
  input.n_atoms = n;
  input.positions = pos;
  input.atomic_numbers = species;
  input.cell = box;
  rgpot_plugin_output_t output{};
  output.forces = forces.data();
  REQUIRE(desc->calculate(instance, &input, &output) == RGPOT_PLUGIN_OK);
  const double energy = output.energy;
  desc->destroy(instance);
  return energy;
}

double directEnergy(const rgpot::ForceInput &in, auto &&pot) {
  std::vector<double> forces(3 * in.nAtoms);
  rgpot::ForceOut out{};
  out.F = forces.data();
  pot.forceImpl(in, &out);
  return out.energy;
}

void fillCell(double *box) {
  const double values[9] = {20.0, 0.0, 0.0, 0.0, 20.0, 0.0, 0.0, 0.0, 20.0};
  for (int i = 0; i < 9; ++i) {
    box[i] = values[i];
  }
}

} // namespace

TEST_CASE("Lennard-Jones ignores its configuration string", "[builtin]") {
  const auto *desc = pluginNamed("lennard-jones");
  double pos[6] = {0.0, 0.0, 0.0, 1.5, 0.0, 0.0};
  int species[2] = {1, 1};
  double box[9];
  fillCell(box);
  rgpot::LJPot direct;
  const rgpot::ForceInput in{2, pos, species, box};
  REQUIRE(pluginEnergy(desc, "{\"De\":4}", pos, species, box, 2) ==
          directEnergy(in, direct));
}

TEST_CASE("classical Morse reads its configuration", "[builtin]") {
  const auto *desc = pluginNamed("classical-morse");
  const char *config =
      "{\"De\":0.4,\"a\":1.5,\"re\":2,\"cutoff\":5,\"switch_width\":0.5}";
  rgpot::MorseConfig parsed;
  parsed.De = 0.4;
  parsed.a = 1.5;
  parsed.re = 2.0;
  parsed.cutoff = 5.0;
  parsed.switch_width = 0.5;
  rgpot::MorsePot direct(parsed);
  double pos[6] = {0.0, 0.0, 0.0, 2.2, 0.0, 0.0};
  int species[2] = {1, 1};
  double box[9];
  fillCell(box);
  const rgpot::ForceInput in{2, pos, species, box};
  REQUIRE(pluginEnergy(desc, config, pos, species, box, 2) ==
          directEnergy(in, direct));
  REQUIRE(pluginEnergy(desc, "{}", pos, species, box, 2) ==
          directEnergy(in, rgpot::MorsePot{}));

  rgpot_plugin_instance *instance = nullptr;
  REQUIRE(desc->create("{\"nope\":1}", &instance) ==
          RGPOT_PLUGIN_INVALID_PARAM);
  REQUIRE(instance == nullptr);
  REQUIRE(std::string(rgpot::forceLastError()) ==
          "invalid parameter: unknown config key");
  REQUIRE(desc->create("{\"De\":0.2,\"De\":0.3}", &instance) ==
          RGPOT_PLUGIN_INVALID_PARAM);
  REQUIRE(std::string(rgpot::forceLastError()) ==
          "invalid parameter: duplicate config key");
}

TEST_CASE("Sutton-Chen reads periodic from its configuration", "[builtin]") {
  const auto *desc = pluginNamed("sutton-chen");
  rgpot_plugin_instance *defaults = nullptr;
  REQUIRE(desc->create("{}", &defaults) == RGPOT_PLUGIN_OK);
  REQUIRE((desc->capabilities(defaults) & RGPOT_CAP_PERIODIC) != 0);
  desc->destroy(defaults);

  const char *config = "{\"periodic\":false,\"epsilon\":2}";
  rgpot_plugin_instance *free = nullptr;
  REQUIRE(desc->create(config, &free) == RGPOT_PLUGIN_OK);
  REQUIRE((desc->capabilities(free) & RGPOT_CAP_PERIODIC) == 0);
  desc->destroy(free);

  rgpot::SuttonChenConfig parsed;
  parsed.periodic = false;
  parsed.epsilon = 2.0;
  rgpot::SuttonChenPot direct(parsed);
  double pos[6] = {0.0, 0.0, 0.0, 1.2, 0.0, 0.0};
  int species[2] = {1, 1};
  const rgpot::ForceInput in{2, pos, species, nullptr};
  REQUIRE(pluginEnergy(desc, config, pos, species, nullptr, 2) ==
          directEnergy(in, direct));
}

#ifdef RGPOT_HAS_EAM_TABLE

TEST_CASE("tabulated EAM reads a path and a format", "[builtin]") {
  const char *path = "CppCore/tests/data/eam_table/Cu_u3.eam";
  const std::string object = std::string("{\"path\":\"") + path +
                             "\",\"format\":\"funcfl\",\"periodic\":false}";
  const auto *desc = pluginNamed("eam-funcfl");
  rgpot_plugin_instance *instance = nullptr;
  REQUIRE(desc->create(object.c_str(), &instance) == RGPOT_PLUGIN_OK);
  REQUIRE((desc->capabilities(instance) & RGPOT_CAP_PERIODIC) == 0);
  desc->destroy(instance);

  rgpot::EAMTablePot direct({.path = path,
                             .format = rgpot::EAMTableFormat::Funcfl,
                             .periodic = false});
  double pos[6] = {0.0, 0.0, 0.0, 2.5, 0.0, 0.0};
  int species[2] = {29, 29};
  double box[9];
  fillCell(box);
  const rgpot::ForceInput in{2, pos, species, box};
  REQUIRE(pluginEnergy(desc, path, pos, species, box, 2) ==
          directEnergy(in, direct));

  const auto *setfl = pluginNamed("eam-setfl");
  REQUIRE(setfl->create("{}", &instance) == RGPOT_PLUGIN_INVALID_PARAM);
  REQUIRE(std::string(rgpot::forceLastError()) ==
          "invalid parameter: eam table path is missing");
}

#endif
