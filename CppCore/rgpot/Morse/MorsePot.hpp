#pragma once
// MIT License
// Copyright 2023--present rgpot developers

/**
 * @brief Header file for the Morse potential class.
 *
 * This file defines the @c MorsePot class, which implements the pairwise
 * Morse potential with a shifted cutoff. The kernel is ported from eOn
 * (https://github.com/TheochemUI/eOn, client/potentials/Morse), BSD-3-Clause
 * licensed, copyright the eOn Development Team; the original attribution
 * names A. Pedersen or G. Henkelman, revised by J.-C. C. Berthet
 * (2010, University of Iceland).
 */

// clang-format off
#include <cmath>
#include <cstdint>
// clang-format on
#include "rgpot/Locality.hpp"
#include "rgpot/PairSwitch.hpp"
#include "rgpot/ParamHash.hpp"
#include "rgpot/PerAtom.hpp"
#include "rgpot/Potential.hpp"
#include "rgpot/Staged.hpp"
#include "rgpot/nlist/Neighbours.hpp"
#include "rgpot/simd/exp.hpp"
#include "rgpot/types/AtomMatrix.hpp"

namespace rgpot {

/**
 * @brief Parameters for the shifted Morse potential.
 *
 * Plain aggregate with inline defaults (platinum): construct with
 * designated initializers.
 */
struct MorseConfig {
  double De = 0.7102;  //!< Well depth (eV).
  double a = 1.6047;   //!< Range parameter (1/Angstrom).
  double re = 2.8970;  //!< Equilibrium pair distance (Angstrom).
  double cutoff = 9.5; //!< Truncation distance (Angstrom).
  /// Width (Angstrom) of the C^2 quintic switch (PairSwitch.hpp) that
  /// takes each pair term to zero over [cutoff - switch_width, cutoff].
  /// 0 keeps the shifted truncation, whose force jumps at the cutoff.
  double switch_width = 0.0;
};

/**
 * @class MorsePot
 * @brief Pairwise Morse potential, @f$V(r) = D_e [1 - e^{-a(r - r_e)}]^2 -
 * D_e@f$, shifted so that @f$V(r_\mathrm{cut}) = 0@f$.
 * @ingroup rgpot_potentials
 */
class MorsePot : public Potential<MorsePot> {
public:
  MorsePot() : MorsePot(MorseConfig{}) {}

  explicit MorsePot(const MorseConfig &c)
      : Potential(PotType::Morse), De{c.De}, a{c.a}, re{c.re},
        cuttOffR{c.cutoff}, m_config{c} {
    // Shift so U(cuttOffR) = 0, evaluating the same closed form and the
    // same exp as the pair kernel.
    const double d = 1.0 - simd::exp_fixed(-a * (cuttOffR - re));
    energyCutoff = De * d * d - De;
    if (c.switch_width != 0.0) {
      m_switch = QuinticSwitch::endingAt(c.cutoff, c.switch_width);
    }
    Fnv1a fp;
    fp.u64(kKernelVersion);
    fp.f64(c.De);
    fp.f64(c.a);
    fp.f64(c.re);
    fp.f64(c.cutoff);
    // Hashed only when set, so unswitched configs keep their cache keys.
    if (c.switch_width != 0.0) {
      fp.f64(c.switch_width);
    }
    m_paramsKey = fp.h;
  }

  [[nodiscard]] const MorseConfig &config() const noexcept { return m_config; }

  /// Energy offset subtracted from every pair term (the unshifted pair
  /// energy at the cutoff); 0 when the quintic switch replaces the shift.
  [[nodiscard]] double energyShift() const noexcept {
    return m_config.switch_width != 0.0 ? 0.0 : energyCutoff;
  }

  [[nodiscard]] PotCaps caps() const noexcept override {
    return {.reentrancy = Reentrancy::PerInstance, .stress = true};
  }

  [[nodiscard]] uint64_t paramsKey() const noexcept override {
    return m_paramsKey;
  }

  /// Pairs count out to the cutoff; a cutoff <= 0 has no pairs.
  [[nodiscard]] Locality locality() const noexcept {
    return {.cutoff = cuttOffR > 0.0 ? cuttOffR : 0.0, .pairwise = true};
  }

  /**
   * @brief Computes the forces and energy for a given configuration.
   * @param in Structure containing coordinates and cell info.
   * @param out Pointer to the results structure.
   * @return Void.
   */
  void forceImpl(const ForceInput &in, ForceOut *out) const override;

  /**
   * @brief Analytic Hessian-vector product @f$H v@f$.
   *
   * The pairs, minimum image, shift or switch of @c forceImpl, each pair
   * adding @f$K (v_i - v_j)@f$ to atom @c i and its negative to @c j, with
   * @f$K = \phi'' \hat d \hat d^T + (\phi' / r)(I - \hat d \hat d^T)@f$.
   * @a v and @a hv hold @c 3*in.nAtoms doubles. @c HessianVectorCache
   * (HessianVector.hpp) uses it in place of finite differences.
   */
  void hessianVectorImpl(const ForceInput &in, const double *v,
                         double *hv) const;

  /// forceImpl plus each pair energy and virial split half to each atom.
  void forcePerAtom(const ForceInput &in, ForceOut *out, const PerAtomOut &atoms) const;

  /// Staged.hpp: a pair potential has no densities.
  void densityStage(const HalfList &, double *) const {}

  /// Staged.hpp: the pair terms of forceImpl on a host's list.
  void forceStage(const HalfList &list, const double *densities, double *forces,
                  StageTally *tally) const;

  void setNeighbourMode(nlist::Mode mode) noexcept { mode_ = mode; }

  /// True when the last force call kept the Verlet list. Half the skin,
  /// the same predicate as `nlist::Builder::update`.
  [[nodiscard]] bool listReused() const noexcept { return neighbours_.reused(); }

  /// Pair indices copied out of the walk. A stream keeps none: the pair
  /// term accumulates during the cell walk.
  [[nodiscard]] std::size_t retainedPairs() const noexcept {
    return neighbours_.streaming() ? 0 : neighbours_.slots();
  }

  /// Skin added to the cutoff on a stored list. Stream uses none.
  [[nodiscard]] static double neighbourSkin(double cutoff) noexcept {
    return cutoff / 5.0;
  }

private:
  /// Bump when the kernel numerics change, so stale cache entries die.
  static constexpr uint64_t kKernelVersion = 2;

  /// Bring neighbours_ to the positions of @a in.
  void updateNeighbours(const ForceInput &in) const;

  void compute(const ForceInput &in, ForceOut *out, const PerAtomOut *atoms) const;

  /// Energy of one pair at distance r, and its force scale.
  inline void pairTerm(double r, double &energy, double &fscale) const;

  double De;           //!< Well depth parameter.
  double a;            //!< Range parameter.
  double re;           //!< Equilibrium pair distance.
  double cuttOffR;     //!< Distance beyond which the potential is truncated.
  double energyCutoff; //!< Potential energy value at the cutoff distance.
  MorseConfig m_config;
  QuinticSwitch m_switch{}; //!< In use when m_config.switch_width != 0.
  uint64_t m_paramsKey{0};
  nlist::Mode mode_ = nlist::Mode::Auto;
  mutable nlist::Neighbours neighbours_;
};

} // namespace rgpot
