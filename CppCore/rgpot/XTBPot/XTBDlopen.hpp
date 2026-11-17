#pragma once
// MIT License — thin XTB frontend: dlopen libxtb_engine.so
// No libxtb headers; safe for hosts built without -Dwith_xtb.
#include "rgpot/ForceStructs.hpp"
#include "rgpot/XTBPot/XTBConfig.hpp"
#include "rgpot/XTBPot/xtb_c_abi.h"
#include "rgpot/types/AtomMatrix.hpp"

#include <array>
#include <string>
#include <tuple>
#include <vector>

namespace rgpot {

/**
 * Plugin path: ``dlopen`` ``libxtb_engine.so``, which implements the same
 * ISO_C_BINDING singlepoint as linked ``XTBPot`` behind ``xtb_c_abi.h``.
 *
 * A non-empty ``engine_path`` is the only candidate, and it must contain a
 * directory separator (``/`` or ``\``). A bare soname such as
 * ``libxtb_engine.so`` is rejected: the loader would resolve it through
 * ``LD_LIBRARY_PATH``, ``DYLD_LIBRARY_PATH``, or ``PATH`` and could open a
 * different library. If the path does not open, or the library lacks the C
 * ABI, construction throws. The error names that path and, on POSIX, the
 * ``dlerror`` text. Environment variables, the bare soname, and directory
 * lists are not consulted.
 *
 * When ``engine_path`` is empty, search ``RGPOT_XTB_ENGINE``, ``XTB_ENGINE``,
 * the platform bare soname (``libxtb_engine.so`` on Linux,
 * ``libxtb_engine.dylib`` then ``.so`` on Apple, ``xtb_engine.dll`` then
 * ``libxtb_engine.dll`` on Windows), then ``EON_POTENTIALS_PATH`` /
 * ``RGPOT_ENGINE_PATH``.
 *
 * Always compiled (no NEEDED libxtb). Linked ``XTBPot`` remains optional via
 * ``-Dwith_xtb=true``.
 */
class XTBDlopen {
public:
  explicit XTBDlopen(const XTBDlopenConfig &config);
  explicit XTBDlopen(const XTBConfig &xtb = XTBConfig{});
  ~XTBDlopen();

  XTBDlopen(const XTBDlopen &) = delete;
  XTBDlopen &operator=(const XTBDlopen &) = delete;

  [[nodiscard]] bool available() const noexcept { return m_pot != nullptr; }

  void forceImpl(const ForceInput &in, ForceOut *out) const;

  /// Drop the stored SCC wavefunction. The calculator stays.
  void reset();

  /// Displacement guard in Angstrom. Zero or negative disables reuse.
  void setWarmMove(double angstrom);

  [[nodiscard]] double warmMove() const;

  /// HOMO-LUMO gap guard in eV. Zero or negative reuses at any gap.
  void setWarmMinGap(double ev);

  [[nodiscard]] double warmMinGap() const;

  /// True when the last successful force reused the previous wavefunction.
  [[nodiscard]] bool reusedWavefunction() const;

  /// Convenience operator matching Potential interface shape.
  std::tuple<double, types::AtomMatrix, double>
  operator()(const types::AtomMatrix &positions,
             const std::vector<int> &atmtypes,
             const std::array<std::array<double, 3>, 3> &box) const;

private:
  void *m_lib{nullptr};
  RgpotXtbPot *m_pot{nullptr};
  using create_fn = RgpotXtbPot *(*)(const RgpotXtbConfig *, char *, size_t);
  using destroy_fn = void (*)(RgpotXtbPot *);
  using force_fn = int (*)(RgpotXtbPot *, long, const double *, const int *,
                           double *, double *, double *, const double *);
  create_fn m_create{nullptr};
  destroy_fn m_destroy{nullptr};
  force_fn m_force{nullptr};
  using reset_fn = void (*)(RgpotXtbPot *);
  using set_warm_fn = void (*)(RgpotXtbPot *, double);
  using warm_move_fn = double (*)(const RgpotXtbPot *);
  using reused_fn = int (*)(const RgpotXtbPot *);
  reset_fn m_reset{nullptr};
  set_warm_fn m_set_warm{nullptr};
  warm_move_fn m_warm_move{nullptr};
  set_warm_fn m_set_min_gap{nullptr};
  warm_move_fn m_min_gap{nullptr};
  reused_fn m_reused{nullptr};
};

} // namespace rgpot
