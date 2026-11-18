// MIT License
// Copyright 2023--present rgpot developers

#include <rgpot/plugin.h>

#include "rgpot/LennardJones/LJPot.hpp"
#include "rgpot/PotHelpers.hpp"

#include <exception>
#include <new>

// C linkage so the constexpr descriptor can hold the pointers. The
// functions stay internal to this translation unit.
extern "C" {

static rgpot_plugin_status_t
rgpot_builtin_lj_create(const char *, rgpot_plugin_instance **output) {
  if (output == nullptr) {
    return RGPOT_PLUGIN_INVALID_PARAM;
  }
  *output = nullptr;
  try {
    *output = reinterpret_cast<rgpot_plugin_instance *>(new rgpot::LJPot());
    return RGPOT_PLUGIN_OK;
  } catch (...) {
    return RGPOT_PLUGIN_ERROR;
  }
}

static void rgpot_builtin_lj_destroy(rgpot_plugin_instance *instance) {
  delete reinterpret_cast<rgpot::LJPot *>(instance);
}

static rgpot_plugin_status_t
rgpot_builtin_lj_calculate(rgpot_plugin_instance *instance,
                           const rgpot_plugin_input_t *input,
                           rgpot_plugin_output_t *output) {
  if (instance == nullptr || input == nullptr || output == nullptr) {
    rgpot::setForceLastError("invalid parameter: input is null");
    return RGPOT_PLUGIN_INVALID_PARAM;
  }
  auto *pot = reinterpret_cast<rgpot::LJPot *>(instance);
  rgpot::ForceOut result{};
  result.F = output->forces;
  const rgpot::ForceInput request{input->n_atoms, input->positions,
                                  input->atomic_numbers, input->cell};
  if (rgpot::rejectForceCall(request, &result, pot->caps().periodic) != 0) {
    return RGPOT_PLUGIN_INVALID_PARAM;
  }
  try {
    pot->forceImpl(request, &result);
    output->energy = result.energy;
    output->variance = result.variance;
    return RGPOT_PLUGIN_OK;
  } catch (const std::exception &ex) {
    rgpot::setForceLastError(ex.what());
    return RGPOT_PLUGIN_ERROR;
  } catch (...) {
    rgpot::setForceLastError("unknown exception in the potential");
    return RGPOT_PLUGIN_ERROR;
  }
}

static uint32_t rgpot_builtin_lj_capabilities(const rgpot_plugin_instance *) {
  return RGPOT_CAP_ENERGY | RGPOT_CAP_FORCES | RGPOT_CAP_INSTANCE_SAFE |
         RGPOT_CAP_PERIODIC;
}

static const char *rgpot_builtin_lj_last_error(const rgpot_plugin_instance *) {
  return rgpot::forceLastError();
}

} // extern "C"

namespace rgpot::plugin {
namespace {

constexpr rgpot_plugin_descriptor_t kLennardJones{
    sizeof(rgpot_plugin_descriptor_t),
    RGPOT_PLUGIN_API_VERSION,
    "lennard-jones",
    "1.0.0",
    nullptr,
    nullptr,
    rgpot_builtin_lj_create,
    rgpot_builtin_lj_destroy,
    rgpot_builtin_lj_calculate,
    rgpot_builtin_lj_capabilities,
    rgpot_builtin_lj_last_error,
};

static_assert(kLennardJones.api_version == RGPOT_PLUGIN_API_VERSION);
static_assert(kLennardJones.name[0] == 'l');

} // namespace

const rgpot_plugin_descriptor_t *builtinLennardJones() {
  return &kLennardJones;
}

} // namespace rgpot::plugin
