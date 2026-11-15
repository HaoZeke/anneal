#pragma once
// MIT License
// Copyright 2023--present rgpot developers
//
// The Asap effective-medium kernel is the eOn vendored tree
// (client/potentials/EMT/Asap), BSD-3-Clause, copyright the eOn
// Development Team.

#include <cstdint>
#include <vector>

#include "rgpot/ParamHash.hpp"
#include "rgpot/Potential.hpp"

namespace rgpot {

/// Parameters for the Asap effective-medium potential.
/// `rasmussen` selects EMTRasmussenParameterProvider. The default is the
/// Asap copper parameter set, which is what `emt_rasmussen = false` means.
struct EMTConfig {
  bool rasmussen = false;
};

class EMTPot : public Potential<EMTPot> {
public:
  EMTPot() : EMTPot(EMTConfig{}) {}
  explicit EMTPot(const EMTConfig &config);
  ~EMTPot() override;

  EMTPot(const EMTPot &) = delete;
  EMTPot &operator=(const EMTPot &) = delete;

  [[nodiscard]] const EMTConfig &config() const noexcept { return m_config; }

  [[nodiscard]] PotCaps caps() const noexcept override {
    return {.reentrancy = Reentrancy::PerInstance, .perImageInstances = true};
  }

  [[nodiscard]] uint64_t paramsKey() const noexcept override {
    return m_paramsKey;
  }

  void forceImpl(const ForceInput &in, ForceOut *out) const override;

private:
  void reset() const;

  EMTConfig m_config;
  uint64_t m_paramsKey = 0;
  mutable long m_nAtoms = 0;
  mutable std::vector<int> m_numbers;
  mutable void *m_cell = nullptr;
  mutable void *m_atoms = nullptr;
  mutable void *m_provider = nullptr;
  mutable void *m_emt = nullptr;
};

} // namespace rgpot
