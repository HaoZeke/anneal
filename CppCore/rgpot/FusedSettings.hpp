#pragma once
// MIT License
// Copyright 2023--present rgpot developers

/**
 * @file FusedSettings.hpp
 * @brief What a potential tells the fused-evaluation cache about itself.
 *
 * A fused cache (@c FusedCache here, the fused cache of @c rgpot_potential_t
 * in the Rust core) answers a request at stored positions without an engine
 * call. It may do so only when nothing else that changes the answer has
 * changed, so a potential whose answer depends on settings other than the
 * positions, the species and the cell reports them through
 * @c fusedSettings(). Changing one of them is a changed cache key, and the
 * way a failed geometry is evaluated again.
 *
 * @c SccNotConverged is the typed failure of an electronic-structure
 * engine whose self-consistent field ran out of iterations. A potential
 * throws it instead of @c std::runtime_error so that the cache, and the C
 * ABI behind @c PotentialHandle, can tell it from every other failure.
 *
 * @ingroup rgpot_cpp
 */

#include <concepts>
#include <stdexcept>
#include <string>

namespace rgpot {

/// The self-consistent field of an engine did not converge.
class SccNotConverged : public std::runtime_error {
public:
  using std::runtime_error::runtime_error;
};

/// Engine settings that change the answer at a fixed geometry. The fields
/// match @c rgpot_fused_settings_t.
struct FusedSettingsSpec {
  unsigned periodic = 0;         //!< Bit i set: periodic along cell vector i.
  int spin = 0;                  //!< Number of unpaired electrons.
  unsigned sccMaxIterations = 0; //!< SCC iteration budget.
  double charge = 0.0;           //!< Total charge, in elementary charges.
  double accuracy = 0.0;         //!< Numerical accuracy level.
  double electronicTemperature = 0.0; //!< Kelvin.
  std::string sccMixer;          //!< Name of the SCC mixer.
  std::string engineBuild;       //!< Build or version string of the engine.
};

/// A potential that reports the settings its answer depends on.
template <typename Pot>
concept ProvidesFusedSettings = requires(const Pot &pot) {
  { pot.fusedSettings() } -> std::convertible_to<FusedSettingsSpec>;
};

} // namespace rgpot
