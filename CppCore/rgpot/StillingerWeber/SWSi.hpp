#pragma once
// MIT License
// Copyright 2023--present rgpot developers

/**
 * @file SWSi.hpp
 * @author rgpot Developers
 * @date 2026-10-26
 * @brief Stillinger-Weber silicon, 1985 parameters.
 *
 * F. H. Stillinger and T. A. Weber, Phys. Rev. B 31, 5262 (1985),
 * doi:10.1103/PhysRevB.31.5262. Equations (2.7) and (2.9).
 * Epsilon is 3.4723e-12 erg, 2.16723920 eV. PotType::SWSi uses eOn's
 * rescaled epsilon, 2.16826 * 1.06767385 = 2.3149945 eV. The ratio of
 * the eOn value to this one is 1.068176741337. The factor is untraced.
 */

#include <cstddef>
#include <cstdint>

#include "rgpot/Potential.hpp"
#include "rgpot/nlist/HalfShell.hpp"
#include "rgpot/nlist/Neighbours.hpp"

namespace rgpot {

/**
 * @brief Stillinger-Weber silicon with the 1985 published parameters.
 * @ingroup rgpot_potentials
 *
 * One call evaluates a whole configuration. @c energyAndGradientBatch
 * evaluates many configurations that share a cell and a species list, so
 * a caller crosses into the library once per batch rather than once per
 * pair.
 */
class StillingerWeberSi : public Potential<StillingerWeberSi> {
public:
  /// Eq. (2.9) prints 3.4723e-12 erg per atom pair, 2.16723920 eV.
  /// 1 erg = 1e-7 J. The CODATA electron-volt is 1.602176634e-19 J.
  /// PotType::SWSi is 2.3149945 eV, larger by 1.068176741337.
  static constexpr double kEpsilon = 3.4723e-12 * 1.0e-7 / 1.602176634e-19;
  static constexpr double kSigma = 2.0951; //!< 0.20951 nm, Eq. (2.9).
  static constexpr double kA = 7.049556277;
  static constexpr double kB = 0.6022245584;
  static constexpr double kP = 4.0;
  static constexpr double kCut = 1.80; //!< Reduced cutoff a, Eq. (2.7).
  static constexpr double kLambda = 21.0;
  static constexpr double kGamma = 1.20;

  StillingerWeberSi();

  [[nodiscard]] PotCaps caps() const noexcept override {
    return {.reentrancy = Reentrancy::PerInstance, .stress = true};
  }

  [[nodiscard]] uint64_t paramsKey() const noexcept override {
    return m_paramsKey;
  }

  /// Forces, energy and the Cauchy stress sigma = (1/V) dE/dε in
  /// eV/Angstrom^3, the convention of stress.hpp. The virial sums each
  /// folded bond vector times the energy gradient with respect to it.
  void forceImpl(const ForceInput &in, ForceOut *out) const override;

  /// Auto, Stored and Stream all use the shared neighbour layer. One
  /// thread agrees with the old ordered walk to within 1e-12.
  void setNeighbourMode(nlist::Mode mode) noexcept { mode_ = mode; }

  /**
   * @brief Energy and dE/dx for one configuration.
   *
   * @param positions Row-major n_atoms by 3 coordinates, Angstrom.
   * @param atomic_numbers Length n_atoms. Unused: the 1985 set is silicon.
   * @param box Row-major 3 by 3 cell, Angstrom. Periodic in every direction.
   * @param n_atoms Atom count.
   * @param energy Receives the potential energy, eV.
   * @param gradient Receives dE/dx, length 3 * n_atoms, eV/Angstrom.
   *        The library force is the negation of this gradient.
   * @param neighbors Optional Verlet list. Null uses the internal list,
   *        with a skin of one fifth of the cutoff (none in Stream mode). A list is reused while each atom has moved by at
   *        most half its skin and the cell is unchanged.
   */
  void energyAndGradient(const double *positions, const int *atomic_numbers,
                         const double *box, std::size_t n_atoms, double *energy,
                         double *gradient,
                         nlist::NeighborList *neighbors = nullptr) const;

  /**
   * @brief Energy and dE/dx for a batch of configurations.
   *
   * @param n_config Number of configurations.
   * @param positions n_config blocks of n_atoms * 3 coordinates.
   * @param atomic_numbers Species shared by every configuration.
   * @param box Cell shared by every configuration.
   * @param n_atoms Atom count of each configuration.
   * @param energies Length n_config.
   * @param gradients Length n_config * n_atoms * 3.
   */
  void energyAndGradientBatch(std::size_t n_config, const double *positions,
                              const int *atomic_numbers, const double *box,
                              std::size_t n_atoms, double *energies,
                              double *gradients) const;

private:
  /// scale +1 writes forces, scale -1 writes dE/dx. A non-null strain
  /// gains dE/dε in eV, in the Voigt order of stress.hpp.
  void evaluate(const double *positions, const double *box, std::size_t n_atoms,
                double *energy, double *acc, double scale,
                nlist::NeighborList *neighbors, double *strain) const;

  uint64_t m_paramsKey{0};
  nlist::Mode mode_ = nlist::Mode::Auto;
  mutable nlist::Neighbours neighbours_;
};

} // namespace rgpot
