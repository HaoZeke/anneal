#pragma once
// MIT License
// Copyright 2023--present rgpot developers

/**
 * @brief Header file for the free-boundary Lennard-Jones cluster potential.
 *
 * This file defines the @c LJClusterPot class, a 12-6 Lennard-Jones
 * potential evaluated without periodic boundary conditions. The kernel is
 * ported from eOn (https://github.com/TheochemUI/eOn,
 * client/potentials/LJCluster), BSD-3-Clause licensed, copyright the eOn
 * Development Team.
 */

// clang-format off
#include <cmath>
#include <cstdint>
// clang-format on
#include "rgpot/PairSwitch.hpp"
#include "rgpot/ParamHash.hpp"
#include "rgpot/PerAtom.hpp"
#include "rgpot/Potential.hpp"
#include "rgpot/nlist/Neighbours.hpp"
#include "rgpot/pot_caps.hpp"
#include "rgpot/types/AtomMatrix.hpp"

namespace rgpot {

/**
 * @brief Parameters for the free-boundary shifted 12-6 Lennard-Jones
 * potential.
 *
 * Plain aggregate with inline defaults (reduced units): construct with
 * designated initializers.
 */
struct LJClusterConfig {
  double u0 = 1.0;      //!< Well depth (eV).
  double cutoff = 15.0; //!< Truncation distance (Angstrom).
  double psi = 1.0;     //!< Zero-crossing distance (Angstrom).
  /// Width (Angstrom) of the C^2 quintic switch (PairSwitch.hpp) that
  /// takes each pair term to zero over [cutoff - switch_width, cutoff].
  /// 0 keeps the shifted truncation, whose force jumps at the cutoff.
  double switch_width = 0.0;
};

/**
 * @class LJClusterPot
 * @brief Shifted 12-6 Lennard-Jones for isolated clusters.
 *
 * Same pair kernel as @c LJPot with the minimum image convention switched
 * off, so an input cell never folds pair vectors.
 * @ingroup rgpot_potentials
 */
class LJClusterPot : public Potential<LJClusterPot> {
public:
  LJClusterPot() : LJClusterPot(LJClusterConfig{}) {}

  explicit LJClusterPot(const LJClusterConfig &c)
      : Potential(PotType::LJCluster),
        u0{c.u0},
        cuttOffR{c.cutoff},
        psi{c.psi},
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

  [[nodiscard]] const LJClusterConfig &config() const noexcept {
    return m_config;
  }

  [[nodiscard]] uint64_t paramsKey() const noexcept override {
    return m_paramsKey;
  }

  /// Free boundaries: the kernel ignores the cell and never applies the
  /// minimum image convention.
  [[nodiscard]] PotCaps caps() const noexcept override {
    return {.reentrancy = Reentrancy::PerInstance, .periodic = false};
  }

  /**
   * @brief Computes the forces and energy for a given configuration.
   * @param in Structure containing coordinates and cell info.
   * @param out Pointer to the results structure.
   * @return Void.
   */
  void forceImpl(const ForceInput &in, ForceOut *out) const override;

  /// forceImpl plus each pair energy and virial split half to each atom.
  /// Always walks the cells, so the energy can differ from forceImpl's
  /// direct all-pairs sum in the last bits.
  void forcePerAtom(const ForceInput &in, ForceOut *out, const PerAtomOut &atoms) const;

  void setNeighbourMode(nlist::Mode mode) noexcept { mode_ = mode; }

  [[nodiscard]] bool listReused() const noexcept { return neighbours_.reused(); }

  /// Zero when the call used the direct all-pairs kernel or a stream.
  [[nodiscard]] std::size_t retainedPairs() const noexcept { return retained_; }

private:
  /// Bump when the kernel numerics change, so stale cache entries die.
  static constexpr uint64_t kKernelVersion = 2;

  void compute(const ForceInput &in, ForceOut *out, const PerAtomOut *atoms) const;

  double u0;       //!< Well depth parameter.
  double cuttOffR; //!< Distance beyond which potential is truncated.
  double psi;      //!< Distance at which the inter-particle potential is zero.
  double cuttOffU; //!< Potential energy value at the cutoff distance.
  LJClusterConfig m_config;
  QuinticSwitch m_switch{}; //!< In use when m_config.switch_width != 0.
  uint64_t m_paramsKey{0};
  nlist::Mode mode_ = nlist::Mode::Auto;
  mutable nlist::Neighbours neighbours_;
  mutable std::size_t retained_ = 0;
};

} // namespace rgpot
