#pragma once
// MIT License
// Copyright 2023--present rgpot developers

/**
 * @brief C++ faces for the Fortran 2018 potential kernels.
 *
 * Each kernel exposes one `bind(c)` entry taking flat buffers and
 * returning a status. Those entries stay inside librgpot: `forceImpl` is
 * defined out of line, and the archive holding the Fortran objects is
 * linked with `--exclude-libs`, so no Fortran symbol reaches the dynamic
 * table and consumers reach the kernels only through these classes.
 *
 * Each instance owns a Fortran workspace (rgpot_workspace.f90): its own
 * neighbour table. The builder keeps that list between calls and rebuilds
 * it only when an atom leaves the Verlet skin. The kernels also keep an
 * error message. The kernels
 * carry no other state, so instances evaluate on separate threads at once
 * (`Reentrancy::PerInstance`), and multi-image callers keep one instance
 * per image (`perImageInstances`) so each table follows one geometry
 * instead of every image rebuilding a shared one.
 */

#include "rgpot/ForceStructs.hpp"
#include "rgpot/Potential.hpp"
#include "rgpot/Tersoff/TersoffSiC.hpp"
#include "rgpot/pot_caps.hpp"
#include "rgpot/pot_types.hpp"

namespace rgpot {
namespace fortranpots {

/// Owner of one Fortran workspace handle. Copies get a workspace of their
/// own, so a cloned potential never shares a neighbour table.
class FortranWorkspace {
public:
  FortranWorkspace();
  ~FortranWorkspace();
  FortranWorkspace(const FortranWorkspace &);
  FortranWorkspace &operator=(const FortranWorkspace &);
  FortranWorkspace(FortranWorkspace &&) = delete;
  FortranWorkspace &operator=(FortranWorkspace &&) = delete;
  [[nodiscard]] void *get() const noexcept { return m_handle; }

private:
  void *m_handle;
};

#define RGPOT_FORTRAN_POT_CLASS(ClassName, PotTypeValue)                       \
  class ClassName : public Potential<ClassName> {                              \
  public:                                                                      \
    ClassName() : Potential(PotType::PotTypeValue) {}                          \
    void forceImpl(const ForceInput &in, ForceOut *out) const override;        \
    [[nodiscard]] PotCaps caps() const noexcept override {                     \
      return {.reentrancy = Reentrancy::PerInstance,                           \
              .perImageInstances = true};                                      \
    }                                                                          \
                                                                               \
  private:                                                                     \
    FortranWorkspace m_ws;                                                     \
  }

RGPOT_FORTRAN_POT_CLASS(EDIPPot, EDIP);
RGPOT_FORTRAN_POT_CLASS(LenoskyPot, LenoskySi);
RGPOT_FORTRAN_POT_CLASS(EAMAlPot, EAMAl);
RGPOT_FORTRAN_POT_CLASS(FeHePot, FeHe);
RGPOT_FORTRAN_POT_CLASS(CuH2Pot, CuH2);
RGPOT_FORTRAN_POT_CLASS(WaterHPot, WaterH);

#undef RGPOT_FORTRAN_POT_CLASS

/// Stillinger-Weber silicon (PotType::SWSi). Epsilon is eOn's
/// 2.16826 * 1.06767385 = 2.3149945 eV, not the published 2.16723920 eV
/// of PotType::SWSi1985. Their ratio is 1.068176741337. The factor
/// 1.06767385 is untraced; see rgpot_sw.f90.
class SWPot : public Potential<SWPot> {
public:
  SWPot() : Potential(PotType::SWSi) {}
  /// Forces, energy and the Cauchy stress sigma = (1/V) dE/dε in
  /// eV/Angstrom^3, the convention of stress.hpp.
  void forceImpl(const ForceInput &in, ForceOut *out) const override;
  /// Forces and energy only. eOn's engine calls arrive here and discard
  /// the stress, so the kernel skips the virial and has_stress stays 0.
  void forceBatchImpl(const ForceBatch &batch) const override;
  [[nodiscard]] PotCaps caps() const noexcept override {
    return {.reentrancy = Reentrancy::PerInstance,
            .perImageInstances = true,
            .stress = true};
  }

private:
  FortranWorkspace m_ws;
};

/// Tersoff silicon (PotType::TersoffSi). All-silicon calls, and a null
/// species pointer, use the Fortran gather. Any other atomic number
/// uses the 1989 silicon-carbon kernel. There is no second PotType.
/// No call writes a stress: has_stress is 0 and the tensor is zero on
/// both paths until the gather computes one.
class TersoffPot : public Potential<TersoffPot> {
public:
  TersoffPot() : Potential(PotType::TersoffSi) {}
  void forceImpl(const ForceInput &in, ForceOut *out) const override;
  [[nodiscard]] PotCaps caps() const noexcept override {
    return {.reentrancy = Reentrancy::PerInstance, .perImageInstances = true};
  }

  /// Held list for the calls that reach the 1989 kernel, as
  /// TersoffSiC::setHeldSkin. The gather keeps its own neighbour table.
  void setHeldSkin(double skin) { m_sic.setHeldSkin(skin); }

private:
  FortranWorkspace m_ws;
  TersoffSiC m_sic;
};

} // namespace fortranpots
} // namespace rgpot
