#pragma once
// MIT License
// Copyright 2023--present rgpot developers

/**
 * @brief Grimme's DFT-D2 dispersion as a lattice sum.
 *
 * S. Grimme, "Semiempirical GGA-type density functional constructed with a
 * long-range dispersion correction", J. Comput. Chem. 27, 1787 (2006),
 * doi:10.1002/jcc.20495:
 *
 * @f[
 *   E = -s_6 \sum_{i<j} \sum_{\mathbf T}
 *         \frac{C_6^{ij}}{r_{ij,\mathbf T}^6}
 *         \frac{1}{1 + e^{-d (r_{ij,\mathbf T} / (s_R R_0^{ij}) - 1)}}
 * @f]
 *
 * with @f$C_6^{ij} = \sqrt{C_6^i C_6^j}@f$ and @f$R_0^{ij} = R_0^i + R_0^j@f$.
 * The tables, unit conversions and per-functional @f$s_6@f$ are those of
 * the reference dftd3 program (V3.1, `loadoldpar` and `setfuncpar`, version
 * 2), which reproduces the paper for H to Xe and extends it to Rn.
 *
 * In a cell every image within @c cutoff is summed, including the images
 * of an atom with itself (with weight one half), so the result is the
 * periodic energy for any cell, sheared or not. An all-zero box sums the
 * pairs of the molecule.
 */

#include <cstdint>
#include <string_view>

#include "rgpot/Potential.hpp"

namespace rgpot {

/**
 * @brief Parameters of rgpot::D2Pot.
 *
 * The defaults are BLYP (s6 = 1.2, d = 20, sR = 1.1 on dftd3's radii, which
 * gives the paper's van der Waals radii).
 */
struct D2Config {
  double s6 = 1.2;  //!< Global scaling; functional dependent.
  double d = 20.0;  //!< Steepness of the damping function.
  double sr = 1.1;  //!< Scale on the tabulated R0 (dftd3's rs6).
  /// Pair cutoff in Angstrom, applied to every pair and image. At the
  /// default, doubling the cutoff moves the energy of liquid water by less
  /// than sqrt(eps) Hartree per atom (D2PotTest); the force tail is already
  /// below that at 50 Angstrom.
  double cutoff = 150.0;
};

/// The D2 parameters dftd3 uses for @p functional (its `-func` names, and
/// "blyp", "bp86", "b3lyp" as aliases), with the default cutoff.
/// Throws std::invalid_argument for an unknown name.
D2Config d2_config_for(std::string_view functional);

/// C6 in J nm^6 mol^-1 and R0 in Angstrom for atomic number @p z, from
/// dftd3's `loadoldpar`. Throws std::invalid_argument outside 1 to 86.
double d2_c6(int z);
double d2_r0(int z);

/**
 * @class D2Pot
 * @brief DFT-D2 energy, forces and stress, periodic or molecular.
 * @ingroup rgpot_potentials
 */
class D2Pot final : public Potential<D2Pot> {
public:
  explicit D2Pot(D2Config cfg = {});

  void forceImpl(const ForceInput &in, ForceOut *out) const override;

  [[nodiscard]] PotCaps caps() const noexcept override {
    return {.reentrancy = Reentrancy::SharedInstance, .periodic = true, .stress = true};
  }
  [[nodiscard]] uint64_t paramsKey() const noexcept override;
  [[nodiscard]] const D2Config &config() const noexcept { return cfg_; }

private:
  D2Config cfg_;
};

} // namespace rgpot
