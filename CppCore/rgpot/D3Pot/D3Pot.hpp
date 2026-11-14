#pragma once
// MIT License
// Copyright 2023--present rgpot developers

#include <string>
#include <vector>

#include <dftd3.h>

#include "rgpot/Potential.hpp"

namespace rgpot {

/// Damping form for DFT-D3. BJ is rational (Becke-Johnson) damping.
enum class D3Damping { BJ, Zero };

/**
 * Linked s-dftd3 pot (NEEDED libs-dftd3).
 *
 * ``atm`` is the Axilrod-Teller-Muto E(3) three-body term. The literature
 * D3-ATM default is on; the flag is an explicit constructor field so tests
 * can pin both states. ``functional`` is the s-dftd3 method key for C6
 * parameters (for example ``"pbe"``).
 */
struct D3Config {
  D3Damping damping = D3Damping::BJ;
  std::string functional = "pbe";
  bool atm = true; //!< Axilrod-Teller-Muto E(3); default on (D3-ATM).
};

/**
 * Grimme DFT-D3 dispersion via the s-dftd3 ISO C API (``dftd3.h``).
 *
 * In-process Potential summand: ForceInput in, energy+forces out. Do not
 * fold this into XcKernel::contract or applyFxc. The schema carries
 * PotentialConfig.d3 @12, whose D3Params mirror D3Config, but nothing
 * applies it: potserv's D3 backend runs the default D3Config, and
 * configure() refuses the d3 arm on every backend. Wiring it waits on a DFT
 * host summing XcKernel + VV10 + D4 on the wire (rgpot-8mse; rgpot-uw9w /
 * rgpot-eipa add this as Edisp from rgpot-2ftk).
 *
 * Each instance owns ``dftd3_error`` / ``dftd3_structure`` / ``dftd3_model``
 * / ``dftd3_param`` handles. First force builds the structure and model;
 * later forces call ``dftd3_update_structure``. Do not share one instance
 * across threads.
 */
class D3Pot : public Potential<D3Pot> {
public:
  D3Pot();
  explicit D3Pot(const D3Config &config);
  ~D3Pot();

  D3Pot(const D3Pot &) = delete;
  D3Pot &operator=(const D3Pot &) = delete;

  void forceImpl(const ForceInput &in, ForceOut *out) const override;

  [[nodiscard]] PotCaps caps() const noexcept override {
    return {.reentrancy = Reentrancy::PerInstance};
  }

  [[nodiscard]] const D3Config &config() const noexcept { return m_config; }

private:
  D3Config m_config;

  mutable dftd3_error m_err = nullptr;
  mutable dftd3_structure m_mol = nullptr;
  mutable dftd3_model m_model = nullptr;
  mutable dftd3_param m_param = nullptr;
  mutable bool m_initialized = false;
  mutable int m_natoms = 0;
  mutable std::vector<double> m_pos_bohr;

  void initHandles();
  void loadParam() const;
  void throwIfError(const char *what) const;
};

} // namespace rgpot
