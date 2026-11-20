#pragma once
// MIT License
// Copyright 2023--present rgpot developers

/**
 * @brief CPMD's built-in empirical D2, reproduced term for term.
 *
 * OpenCPMD (commit 062582b, `vdw` in src/vdw_utils.mod.F90 with the tables
 * of `empvdwin` in src/vdwin_utils.mod.F90) evaluates Grimme's D2 with
 * `EMPIRICAL CORRECTION` / `ALL DFT-D2`:
 *
 * @f[
 *   E = -s_6 \sum_{i<j} \frac{C_6^{ij}}{r_{ij}^6}
 *         \frac{1}{1 + e^{-\alpha (r_{ij} / R_r^{ij} - 1)}}
 * @f]
 *
 * with C6 = sqrt(C6_i C6_j) times CPMD's 17.33979, R_r = (R0_i + R0_j) in
 * CPMD's bohr, the paper's R0 without dftd3's 1.1 split, and alpha = 20.
 * Each pair takes only the image CPMD's `pbc3` picks (src/pbc_utils.mod.F90,
 * called from subroutine vdw in src/vdw_utils.mod.F90): per axis in an
 * orthorhombic cell, the face centred cubic branch for latgen's symmetry 2,
 * the shear branch for symmetry 4 and for symmetry 14 with alpha and beta
 * at 90 degrees, the monoclinic branch for symmetry 12 (the label 20 of
 * pbc3, which only 12 reaches), and the nearest integer fractional shift
 * otherwise. D2CpmdConfig::symmetry names the code; left at its default it
 * is read from the box. An all-zero box takes no image. A pair is dropped once the magnitude of
 * alpha (r / R_r - 1) reaches 1 / vdweps, that is r >= 6 R_r at the default
 * vdweps, and below CPMD's 0.01 bohr.
 *
 * Positions, the cell, the forces and the stress all use CPMD's own
 * 0.529177210859 (kCpmdBohr, the reciprocal of fbohr in src/cnst.mod.F90),
 * so the forces and stress are derivatives of the returned energy and the
 * hartree values match OpenCPMD.
 *
 * The minimum image makes the energy continuous but its gradient jumps when
 * a pair crosses half a cell: the image on the other side takes over. Use
 * rgpot::D2Pot, which sums every image, for anything but undoing CPMD's term.
 */

#include <cstdint>
#include <string_view>

#include "rgpot/Potential.hpp"

namespace rgpot {

/// Angstrom per bohr as OpenCPMD types it: the reciprocal of fbohr in
/// src/cnst.mod.F90. Positions, the cell, the forces and the stress of
/// D2CpmdPot all use this one factor.
inline constexpr double kCpmdBohr = 0.529177210859;

/// D2CpmdConfig::symmetry value that picks pbc3's branch from the box.
inline constexpr int kCpmdInferSymmetry = -1;

/// Parameters of rgpot::D2CpmdPot; the defaults are CPMD's for BLYP.
struct D2CpmdConfig {
  double s6 = 1.2;      //!< CPMD's S6GRIMME, from set_s6 for the functional.
  double alpha = 20.0;  //!< CPMD's damping exponent (a6grim).
  double vdweps = 0.01; //!< CPMD's vdweps; pairs with abs(alpha (r/R_r - 1)) >= 1/vdweps drop.
  /// CPMD's SYMMETRY code (ibrav) of the run, which selects pbc3's branch:
  /// 1 and 8 fold per axis, 2 is face centred cubic, 4 and 14 with alpha and
  /// beta at 90 degrees the shear branch, 12 the monoclinic branch, and 3, 5,
  /// 6 and 7 the nearest integer. A box that does not have the shape of its
  /// code is refused. kCpmdInferSymmetry reads the branch from the box, which
  /// cannot tell 12 from 14.
  int symmetry = kCpmdInferSymmetry;
};

/// The s6 CPMD's set_s6 gives @p functional (BLYP, B3LYP, BP86, TPSS, PBE,
/// REVPBE, PBE0, in any case). Throws std::invalid_argument otherwise.
D2CpmdConfig d2_cpmd_config_for(std::string_view functional);

/**
 * @class D2CpmdPot
 * @brief CPMD's minimum-image D2 energy, forces and stress.
 * @ingroup rgpot_potentials
 */
class D2CpmdPot final : public Potential<D2CpmdPot> {
public:
  explicit D2CpmdPot(D2CpmdConfig cfg = {});

  void forceImpl(const ForceInput &in, ForceOut *out) const override;

  [[nodiscard]] PotCaps caps() const noexcept override {
    return {.reentrancy = Reentrancy::SharedInstance, .periodic = true, .stress = true};
  }
  [[nodiscard]] uint64_t paramsKey() const noexcept override;
  [[nodiscard]] const D2CpmdConfig &config() const noexcept { return cfg_; }

private:
  D2CpmdConfig cfg_;
};

} // namespace rgpot
