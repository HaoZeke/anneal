#pragma once
// MIT License
// Copyright 2023--present rgpot developers

#include <string>
#include <vector>

#include <xtb.h>

#include "rgpot/Potential.hpp"
#include "rgpot/XTBPot/XTBConfig.hpp"

namespace rgpot {

/**
 * Linked GFN-xTB pot (NEEDED libxtb).
 *
 * libxtb exposes a stable **ISO_C_BINDING** C API (``xtb.h``): opaque
 * ``xtb_TEnvironment`` / ``xtb_TCalculator`` / ``xtb_TMolecule`` /
 * ``xtb_TResults`` handles with explicit create/update/destroy. We test and
 * use that C contract—not free-form Fortran ABI. Each ``XTBPot`` owns its
 * handles. The first force builds the molecule. Later forces call
 * ``xtb_updateMolecule`` and keep the SCC wavefunction in the results
 * object while no atom moves by more than ``XTBConfig::warm_move``
 * (1 Angstrom unless changed). A larger move, a new composition, or
 * ``reset()`` drops that object so the next single point uses xtb's own
 * guess. An SCC that does not converge never seeds another one.
 * ``xtb_releaseOutput`` is called on create (library recommendation for
 * multi-env processes). Do not share one instance across threads;
 * distinct instances may be used from different threads if the loaded
 * libxtb build is reentrant for independent handle sets.
 *
 * For hosts that must not NEEDED libxtb, use ``XTBDlopen`` +
 * ``libxtb_engine.so`` instead (always built; no ``-Dwith_xtb`` required).
 */
class XTBPot : public Potential<XTBPot> {
public:
  XTBPot();
  explicit XTBPot(const XTBConfig &config);
  ~XTBPot();

  XTBPot(const XTBPot &) = delete;
  XTBPot &operator=(const XTBPot &) = delete;

  /// Drop the stored SCC wavefunction. The calculator stays. The next
  /// force starts from xtb's own guess, even when the geometry is unchanged.
  void reset();

  /// Replace the displacement guard. Zero or a negative value disables reuse.
  void setWarmMove(double angstrom);

  [[nodiscard]] double warmMove() const noexcept;

  /// True when the last successful force reused the previous wavefunction.
  [[nodiscard]] bool reusedWavefunction() const noexcept;

  void forceImpl(const ForceInput &in, ForceOut *out) const override;

  /// Native handles are per-instance; concurrent use needs one instance
  /// per thread.
  [[nodiscard]] PotCaps caps() const noexcept override {
    return {.reentrancy = Reentrancy::PerInstance, .stress = true};
  }

private:
  XTBConfig m_config;

  mutable xtb_TEnvironment m_env = nullptr;
  mutable xtb_TCalculator m_calc = nullptr;
  mutable xtb_TMolecule m_mol = nullptr;
  mutable xtb_TResults m_res = nullptr;
  mutable bool m_initialized = false;
  mutable bool m_has_wavefunction = false;
  mutable bool m_last_reused = false;
  mutable double m_warm_move = 1.0;
  mutable std::vector<double> m_pos_bohr; //!< Preallocated position buffer.
  mutable std::vector<double> m_previous; //!< Last host coordinates, Angstrom.
  mutable std::vector<int> m_numbers;

  void initHandles() const;
  void loadParametrisation() const;
  void freshResults() const;
  void discardAfterFailure() const;
  /// One xtb single point. On failure the environment log is read, which
  /// clears it, and the message is stored in ``error`` when given.
  bool singlepoint(std::string *error) const;
  /// A failed SCC on a rearranged cluster is retried with smeared
  /// occupation from xtb's own guess, then evaluated again at the
  /// configured temperature. The hot single point is only a guess. False
  /// leaves the caller to discard the handles.
  bool rescueUnconvergedScc() const;
};

} // namespace rgpot
