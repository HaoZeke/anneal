#pragma once
// MIT License
// Copyright 2023--present rgpot developers

#include <string>
#include <vector>

#include <dftd4.h>

#include "rgpot/Potential.hpp"

namespace rgpot {

/**
 * Linked dftd4 pot (NEEDED libdftd4).
 *
 * ``atm`` is the many-body (Axilrod-Teller-Muto) term the C API names
 * ``mdb``. The literature D4-ATM default is on; the flag is an explicit
 * constructor field so tests can pin both states. ``functional`` is the
 * dftd4 method key for damping parameters (for example ``"pbe"``).
 * ``charge`` is the total molecular charge passed to
 * ``dftd4_new_structure``. Charge scaling lives inside dftd4; this
 * wrapper does not reimplement it.
 */
struct D4Config {
  std::string functional = "pbe";
  double charge = 0.0;
  bool atm = true; //!< many-body ATM (``mdb``); default on (D4-ATM).
};

/**
 * Grimme DFT-D4 dispersion via the dftd4 ISO C API (``dftd4.h``).
 *
 * In-process Potential summand: ForceInput in, energy+forces out. Do not
 * fold this into XcKernel::contract or applyFxc. The schema carries
 * PotentialConfig.d4 @13, whose D4Params mirror D4Config, but nothing
 * applies it: potserv's D4 backend runs the default D4Config, and
 * configure() refuses the d4 arm on every backend. Wiring it waits on a DFT
 * host summing XcKernel + VV10 + D4 on the wire (rgpot-8mse; rgpot-uw9w /
 * rgpot-eipa add this as Edisp from rgpot-2ftk).
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
    return {.reentrancy = Reentrancy::PerInstance};
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
  mutable std::vector<double> m_pos_bohr;

  void initHandles();
  void loadParam() const;
  void throwIfError(const char *what) const;
};

} // namespace rgpot
