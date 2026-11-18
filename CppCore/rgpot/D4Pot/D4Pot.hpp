#pragma once
// MIT License
// Copyright 2023--present rgpot developers

#include <vector>

#include <dftd4.h>

#include "rgpot/D4Pot/D4Config.hpp"
#include "rgpot/Potential.hpp"

namespace rgpot {

/**
 * Grimme DFT-D4 dispersion via the dftd4 ISO C API (``dftd4.h``).
 *
 * In-process Potential summand: ForceInput in, energy and forces out,
 * and the stress when ``periodic`` is set. Do not fold this into
 * XcKernel::contract or applyFxc. PotentialConfig.d4 @13, whose D4Params
 * mirror D4Config, configures potserv's D4 backend and a ``d4`` composite
 * term, so a DFT engine with its own dispersion off can carry D4 beside it.
 *
 * Each instance owns ``dftd4_error`` / ``dftd4_structure`` / ``dftd4_model``
 * / ``dftd4_param`` handles. First force builds the structure and model;
 * later forces call ``dftd4_update_structure``. Do not share one instance
 * across threads.
 */
class D4Pot : public Potential<D4Pot> {
public:
  D4Pot();
  explicit D4Pot(const D4Config &config);
  ~D4Pot();

  D4Pot(const D4Pot &) = delete;
  D4Pot &operator=(const D4Pot &) = delete;

  void forceImpl(const ForceInput &in, ForceOut *out) const override;

  [[nodiscard]] PotCaps caps() const noexcept override {
    return {.reentrancy = Reentrancy::PerInstance,
            .periodic = m_config.periodic,
            .stress = m_config.periodic};
  }

  [[nodiscard]] const D4Config &config() const noexcept { return m_config; }

private:
  D4Config m_config;

  mutable dftd4_error m_err = nullptr;
  mutable dftd4_structure m_mol = nullptr;
  mutable dftd4_model m_model = nullptr;
  mutable dftd4_param m_param = nullptr;
  mutable bool m_initialized = false;
  mutable int m_natoms = 0;
  mutable std::vector<int> m_numbers;
  mutable std::vector<double> m_pos_bohr;

  void initHandles();
  void loadParam() const;
  void throwIfError(const char *what) const;
};

} // namespace rgpot
