// MIT License
// Copyright 2023--present rgpot developers

/**
 * @brief Implementation of utility functions for potential management.
 *
 * Contains the implementations for global helper functions used to manage and
 * validate the core force and energy data structures.
 */

#include "PotHelpers.hpp"
#include <rgpot/plugin.h>
#include <stdexcept>
#include <string>

#ifdef RGPOT_HAS_RUST_CORE
#include "rgpot/eindir_abi.hpp"
extern "C" {
#include <rgpot.h>
}
#endif

namespace rgpot {

/**
 * @details
 * This function performs a manual reset of the @c ForceOut structure.
 * It ensures the energy and variance are set to zero and iterates
 * through the force array to clear components for each atom.
 */
void zeroForceOut(const size_t &nAtoms, ForceOut *efvd) {
  efvd->energy = 0;
  efvd->variance = 0;
  efvd->has_stress = 0;
  for (int s = 0; s < 9; ++s)
    efvd->stress[s] = 0;
  for (size_t idx{0}; idx < nAtoms; idx++) {
    efvd->F[3 * idx] = 0;
    efvd->F[3 * idx + 1] = 0;
    efvd->F[3 * idx + 2] = 0;
  }
}

/**
 * @details
 * Verifies that the input parameters represent a physically valid
 * configuration. Currently, it strictly checks that the system
 * contains at least one atom.
 *
 * @warning Throws a @c std::runtime_error if @a nAtoms is zero or less.
 */
void checkParams(const ForceInput &params) {
  // Simple sanity check
  if (params.nAtoms <= 0) {
    throw std::runtime_error("Can't work with zero atoms in force call");
  }
}

namespace {
thread_local std::string g_force_error;

int fail(const char *reason) {
  g_force_error = reason;
  return kInvalidParameter;
}
} // namespace

static_assert(kInvalidParameter ==
              static_cast<int>(RGPOT_PLUGIN_INVALID_PARAM));
#ifdef RGPOT_HAS_RUST_CORE
static_assert(kInvalidParameter == static_cast<int>(RGPOT_INVALID_PARAMETER));
#endif

void setForceLastError(const char *reason) noexcept {
  if (reason == nullptr || reason[0] == '\0') {
    return;
  }
  g_force_error = reason;
}

const char *forceLastError() noexcept { return g_force_error.c_str(); }

int rejectForceCall(const ForceInput &in, const ForceOut *out, bool periodic) {
  // Before the force pointer. A zero-atom call often has no force
  // allocation, and that null must not become the reported error.
  if (in.nAtoms == 0) {
    return fail("invalid parameter: nAtoms must be positive");
  }
  if (out == nullptr || out->F == nullptr) {
    return fail("invalid parameter: forces are null");
  }
  if (in.pos == nullptr) {
    return fail("invalid parameter: positions is null");
  }
  if (in.atmnrs == nullptr) {
    return fail("invalid parameter: atomic numbers are null");
  }
  if (periodic && in.box == nullptr) {
    return fail("invalid parameter: box is null");
  }
  return 0;
}

void guardForceCall(const ForceInput &in, const ForceOut *out, bool periodic) {
  if (rejectForceCall(in, out, periodic) != 0) {
    throw std::invalid_argument(forceLastError());
  }
}

} // namespace rgpot
