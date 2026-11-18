#pragma once
// MIT License
// Copyright 2023--present rgpot developers

#include <limits>
#include <string>
#include <vector>

#include <xtb.h>

#include "rgpot/FusedSettings.hpp"
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
 * (1 Angstrom unless changed) and the HOMO-LUMO gap stays at or above
 * ``XTBConfig::warm_min_gap`` (1 eV unless changed). A larger move, a
 * smaller gap, a new composition, or ``reset()`` drops that object so the
 * next single point uses xtb's own guess. An SCC that does not converge
 * never seeds another one. ``xtb_releaseOutput`` is called on create (library
 * recommendation for multi-env processes). Do not share one instance across
 * threads; distinct instances may be used from different threads if the loaded
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

  /// Replace the frontier gap guard, in eV. Zero or a negative value
  /// reuses the wavefunction at any gap.
  void setWarmMinGap(double ev);

  [[nodiscard]] double warmMinGap() const noexcept;

  /// True when the last successful force reused the previous wavefunction.
  [[nodiscard]] bool reusedWavefunction() const noexcept;

  /// Replace the numerical accuracy level. Applied by the next force.
  void setAccuracy(double accuracy);

  /// Replace the electronic temperature, in kelvin. Applied by the next
  /// force.
  void setElectronicTemperature(double kelvin);

  /// Replace the SCC iteration budget. Applied by the next force.
  void setMaxIterations(int iterations);

  [[nodiscard]] const XTBConfig &config() const noexcept { return m_config; }

  /**
   * Settings that change the answer at a fixed geometry: charge, spin,
   * accuracy, electronic temperature, SCC iteration budget, and the method
   * and libxtb API version as the engine build. @c PotentialHandle installs
   * them as the key of its fused cache. After a setter above, call
   * @c PotentialHandle::sync_fused_settings so a failure stored at the
   * current positions is evaluated again.
   */
  [[nodiscard]] FusedSettingsSpec fusedSettings() const;

  /**
   * @throws SccNotConverged when the SCC does not converge, the rescue
   *         included.
   * @throws std::runtime_error for every other failure.
   */
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
  mutable double m_warm_min_gap = 1.0;
  //! HOMO-LUMO gap of the stored wavefunction, eV.
  mutable double m_gap = std::numeric_limits<double>::infinity();
  mutable std::vector<double> m_pos_bohr; //!< Preallocated position buffer.
  mutable std::vector<double> m_previous; //!< Last host coordinates, Angstrom.
  mutable std::vector<int> m_numbers;
  mutable std::vector<double> m_orbital_buffer;

  void initHandles() const;
  void loadParametrisation() const;
  void freshResults() const;
  void discardAfterFailure() const;
  /// HOMO-LUMO gap of the results object in eV. Infinite when the method
  /// has no orbitals.
  double frontierGap() const;
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
