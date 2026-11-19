#pragma once
// MIT License
// Copyright 2023--present rgpot developers

/**
 * @brief Sutton-Chen potential.
 *
 * Sutton and Chen, Philosophical Magazine Letters 61, 139 (1990):
 *
 *   E = epsilon * [ 1/2 sum_i sum_{j != i} (a / r_ij)^n
 *                   - c * sum_i sqrt(rho_i) ]
 *   rho_i = sum_{j != i} (a / r_ij)^m
 *
 * Gold and platinum use n = 10, m = 8, c = 34.408. The cluster
 * references are in the reduced units epsilon = 1, a = 1 written on
 * the first line of the Doye and Wales point files
 * (doye.chem.ox.ac.uk/jon/structures/SC/10-8/). Those files are the
 * deposited Pt38 Oh and Pt75 D5h geometries. Doye and Wales attribute
 * n, m and c to the 1990 paper. Physical epsilon and a for both metals
 * are the presets in SuttonChenScales.hpp.
 *
 * optbench's globalopt Lennard-Jones problems truncate at cutoff 40
 * in the coordinate unit (problems/globalopt/lj75.toml). The same
 * truncation, with no energy shift, is the default. cutoff <= 0 keeps
 * every pair. On the deposited clusters every pair sits inside r = 3,
 * so the default and the full sum agree past the published digits.
 */

// clang-format off
#include <cstddef>
#include <cstdint>
#include <vector>
// clang-format on
#include "rgpot/KeptPotential.hpp"
#include "rgpot/Locality.hpp"
#include "rgpot/PairTerms.hpp"
#include "rgpot/ParamHash.hpp"
#include "rgpot/PerAtom.hpp"
#include "rgpot/Potential.hpp"
#include "rgpot/Staged.hpp"
#include "rgpot/nlist/Neighbours.hpp"
#include "rgpot/types/AtomMatrix.hpp"

#if defined(_WIN32)
#define RGPOT_SC_API __declspec(dllexport)
#else
#define RGPOT_SC_API __attribute__((visibility("default")))
#endif

extern "C" {

/// Plain parameter block for @c rgpot_sutton_chen_energy_forces.
/// Field order is part of the ABI.
struct rgpot_sc_config {
  double epsilon;
  double c;
  double a;
  int32_t n;
  int32_t m;
  double cutoff;
  int32_t periodic;
};

/// Energy and forces. Returns 0 on success, 1 on a bad argument.
/// The exported name is rgpot_sutton_chen_energy_forces. A trailing
/// `_force` is the Fortran bind(c) shape the symbol audit hides.
/// @a box may be null when @c periodic is 0. @a forces has room for
/// @c 3 * natoms doubles. Each thread keeps the potential of its last
/// call: one with the same parameters and atom count reuses its neighbour
/// update and pair terms, and any other builds a new one.
RGPOT_SC_API int
rgpot_sutton_chen_energy_forces(const rgpot_sc_config *cfg, int32_t natoms,
                                const double *positions, const double *box,
                                double *energy, double *forces);

} // extern "C"

namespace rgpot {

/**
 * @brief Parameters for the Sutton-Chen potential.
 *
 * Plain aggregate. Defaults are the reduced Pt/Au 10-8 potential.
 */
struct SuttonChenConfig {
  double epsilon = 1.0; //!< Energy unit. 1 in the cluster references.
  double c = 34.408;    //!< Dimensionless density coefficient (Pt, Au).
  double a = 1.0;       //!< Length scale. 1 in the cluster references.
  int n = 10;           //!< Repulsive exponent. 10 for Pt and Au.
  int m = 8;            //!< Density exponent. 8 for Pt and Au.
  /// Truncation distance in the coordinate unit. 0 keeps every pair.
  /// 40 matches the optbench globalopt Lennard-Jones cutoff.
  double cutoff = 40.0;
  /// Minimum image convention. Cluster references leave this false.
  bool periodic = false;
  /// Write the Cauchy stress sigma = (1/V) dE/dstrain of stress.hpp when the
  /// cell volume is positive.
  /// Off by default: has_stress and the tensor are left as the caller set
  /// them, and energy and forces are the same bits either way.
  bool stress = false;
};

/**
 * @class SuttonChenPot
 * @brief Sutton-Chen energy and analytic forces.
 * @ingroup rgpot_potentials
 */
class SuttonChenPot : public Potential<SuttonChenPot> {
public:
  SuttonChenPot() : SuttonChenPot(SuttonChenConfig{}) {}

  explicit SuttonChenPot(const SuttonChenConfig &c);

  [[nodiscard]] const SuttonChenConfig &config() const noexcept {
    return m_config;
  }

  [[nodiscard]] PotCaps caps() const noexcept override {
    return {.reentrancy = Reentrancy::PerInstance,
            .periodic = m_config.periodic,
            .stress = m_config.stress};
  }

  [[nodiscard]] uint64_t paramsKey() const noexcept override {
    return m_paramsKey;
  }

  /// The density and the pair term both stop at the cutoff. A cutoff <= 0
  /// keeps every pair, so there is no finite range. The staged evaluation
  /// exchanges rho.
  [[nodiscard]] Locality locality() const noexcept {
    return {.cutoff = m_config.cutoff > 0.0 ? m_config.cutoff : 0.0, .densities = 1};
  }

  void forceImpl(const ForceInput &in, ForceOut *out) const override;

  /// forceImpl plus per-atom terms: each atom's embedding
  /// -epsilon c sqrt(rho_i), and each pair energy and virial split half to
  /// each atom.
  void forcePerAtom(const ForceInput &in, ForceOut *out, const PerAtomOut &atoms) const;

  /// Staged.hpp: adds (a / r)^m to the density of each atom of a pair.
  void densityStage(const HalfList &list, double *densities) const;

  /// Staged.hpp: -epsilon c sqrt(rho_i) of each owned atom and the pair
  /// forces with the embedding weights of both atoms, tallied as
  /// forcePerAtom tallies them.
  void forceStage(const HalfList &list, const double *densities, double *forces,
                  StageTally *tally) const;

  /// Stored Verlet list or a streaming cell walk.
  void setNeighbourMode(nlist::Mode mode) noexcept { mode_ = mode; }

  /// True when the last force call kept the Verlet list.
  [[nodiscard]] bool listReused() const noexcept { return neighbours_.reused(); }

  /// Maximum pairs, over all threads, whose density-pass powers the
  /// two-pass force pass reads back (`PairTerms<3>::kDefaultLimit`,
  /// 1,398,101 pairs = 32 MiB at 24 bytes per pair, by default), and the
  /// upper-pair count bound (`total_up <= pairs`) for the three-phase
  /// stored OpenMP path. 0 skips the three-phase path and keeps no
  /// two-pass pair terms.
  void setPairTermLimit(std::size_t pairs) noexcept { term_limit_ = pairs; }

private:
  static constexpr uint64_t kKernelVersion = 1;
  using Terms = PairTerms<3>;

  void compute(const ForceInput &in, ForceOut *out, const PerAtomOut *atoms) const;

  SuttonChenConfig m_config;
  uint64_t m_paramsKey{0};
  nlist::Mode mode_ = nlist::Mode::Auto;
  std::size_t term_limit_ = Terms::kDefaultLimit;
  mutable nlist::Neighbours neighbours_;
  mutable std::vector<Terms> terms_;
  mutable std::vector<double> rho_;
  mutable std::vector<double> weight_;
};

template <>
RGPOT_SC_API KeptCounts Kept<SuttonChenPot>::counts() noexcept;

template <>
RGPOT_SC_API void Kept<SuttonChenPot>::release() noexcept;

} // namespace rgpot
