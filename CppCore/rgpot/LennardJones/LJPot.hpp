#pragma once
// MIT License
// Copyright 2023--present rgpot developers

/**
 * @brief Header file for the Lennard-Jones potential class.
 *
 * This file defines the @c LJPot class, which implements a standard  12-6
 * Lennard-Jones potential with a shifted cutoff for use in  atomic simulations.
 */

// clang-format off
#include <cmath>
#include <cstdint>
#include <utility>
#include <vector>
#include <stdexcept>
// clang-format on
#include "rgpot/Locality.hpp"
#include "rgpot/PairSwitch.hpp"
#include "rgpot/ParamHash.hpp"
#include "rgpot/PerAtom.hpp"
#include "rgpot/Potential.hpp"
#include "rgpot/Staged.hpp"
#include "rgpot/nlist/Neighbours.hpp"
#include "rgpot/types/AtomMatrix.hpp"

namespace rgpot {

/**
 * @brief Parameters for the shifted 12-6 Lennard-Jones potential.
 *
 * Plain aggregate with inline defaults: the parameter convention for
 * every classical potential (construct with designated initializers).
 */
struct LJConfig {
  double u0 = 1.0;      //!< Well depth (eV).
  double cutoff = 15.0; //!< Truncation distance (Angstrom).
  double psi = 1.0;     //!< Zero-crossing distance (Angstrom).
  /// Width (Angstrom) of the C^2 quintic switch (PairSwitch.hpp) that
  /// takes each pair term to zero over [cutoff - switch_width, cutoff].
  /// 0 keeps the shifted truncation, whose force jumps at the cutoff.
  double switch_width = 0.0;
};

/**
 * @class LJPot
 * @brief Implementation of a shifted 12-6 Lennard-Jones potential.
 * @ingroup rgpot_potentials
 */
class LJPot : public Potential<LJPot> {
public:
  LJPot() : LJPot(LJConfig{}) {}

  explicit LJPot(const LJConfig &c)
      : Potential(PotType::LJ), u0{c.u0}, cuttOffR{c.cutoff}, psi{c.psi},
        m_config{c} {
    // Shift so U(cuttOffR) = 0 (standard shifted 12-6 LJ).
    const double a = std::pow(psi / cuttOffR, 6.0);
    cuttOffU = 4.0 * u0 * a * (a - 1.0);
    if (c.switch_width != 0.0) {
      m_switch = QuinticSwitch::endingAt(c.cutoff, c.switch_width);
    }
    Fnv1a fp;
    fp.u64(kKernelVersion);
    fp.f64(c.u0);
    fp.f64(c.cutoff);
    fp.f64(c.psi);
    // Hashed only when set, so unswitched configs keep their cache keys.
    if (c.switch_width != 0.0) {
      fp.f64(c.switch_width);
    }
    m_paramsKey = fp.h;
  }

  ~LJPot() override;
  LJPot(const LJPot &) = default;
  LJPot &operator=(const LJPot &) = default;
  LJPot(LJPot &&) noexcept = default;
  LJPot &operator=(LJPot &&) noexcept = default;

  [[nodiscard]] const LJConfig &config() const noexcept { return m_config; }

  [[nodiscard]] PotCaps caps() const noexcept override {
    return {.reentrancy = Reentrancy::SharedInstance,
            .stress = true,
            .analyticHvp = true};
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
   * Two atoms on one point throw @c std::domain_error, as the energy does.
   * @a v and @a hv hold @c 3*in.nAtoms doubles. The pairs come from the
   * calling thread's kept neighbour list, as in @c forceImpl, so one
   * instance may serve any number of threads.
   */
  void hessianVectorImpl(const ForceInput &in, const double *v,
                         double *hv) const override;

  /// forceImpl plus each pair energy and virial split half to each atom.
  void forcePerAtom(const ForceInput &in, ForceOut *out, const PerAtomOut &atoms) const;

  /// Staged.hpp: a pair potential has no densities.
  void densityStage(const HalfList &, double *) const {}

  /// Staged.hpp: the pair terms of forceImpl on a host's list.
  void forceStage(const HalfList &list, const double *densities, double *forces,
                  StageTally *tally) const;

  void setNeighbourMode(nlist::Mode mode) noexcept { mode_ = mode; }

  /// True when the last force call on this thread kept the Verlet list.
  /// The test is half the skin, the same predicate as `nlist::Builder::update`.
  [[nodiscard]] bool listReused() const noexcept;

  /// Pair indices this potential kept on this thread. Zero in a streaming
  /// call: the pair term runs inside the cell walk and no half-list is
  /// copied out.
  [[nodiscard]] std::size_t retainedPairs() const noexcept;

private:
  /// Bump when the kernel numerics change, so stale cache entries die.
  static constexpr uint64_t kKernelVersion = 2;

  void compute(const ForceInput &in, ForceOut *out, const PerAtomOut *atoms) const;

  double u0;       //!< Well depth parameter.
  double cuttOffR; //!< Distance beyond which potential is truncated.
  double psi;      //!< Distance at which the inter-particle potential is zero.
  double cuttOffU; //!< Potential energy value at the cutoff distance.
  LJConfig m_config;
  QuinticSwitch m_switch{}; //!< In use when m_config.switch_width != 0.
  uint64_t m_paramsKey{0};
  nlist::Mode mode_ = nlist::Mode::Auto;
  mutable nlist::Neighbours neighbours_; //!< Unused. Kept so the potential object layout stays.
};

} // namespace rgpot
