#pragma once
// MIT License
// Copyright 2023--present rgpot developers

/**
 * @brief Embedded-atom potentials read from DYNAMO tables.
 *
 * The three file layouts of the LAMMPS pair_eam page
 * (docs.lammps.org/pair_eam.html):
 *
 *   Funcfl          pair_style eam      one element: F(rho), Z(r), rho(r)
 *   Setfl           pair_style eam/alloy  F_a(rho), rho_a(r), r phi_ab(r)
 *   FinnisSinclair  pair_style eam/fs   F_a(rho), rho_ab(r), r phi_ab(r)
 *
 * E = sum_i F_a(rho_i) + sum_{i<j} phi_ab(r_ij), with a the element of
 * atom i, b that of j, and rho_i = sum_{j != i} rho_ab(r_ij), the density
 * a b neighbor puts at an a site. Setfl has rho_ab = rho_b. The
 * FinnisSinclair layout is the Finnis and Sinclair form, Phil. Mag. A 50,
 * 45 (1984).
 *
 * The tables go through the LAMMPS steps, so energies and forces follow
 * pair_style eam to rounding: funcfl arrays resampled onto lround((N-1) d
 * / d) points by four-point Lagrange, r phi = 27.2 * 0.529 * Z(r)^2, the
 * seven-coefficient cubic of PairEAM::interpolate, and F extended linearly
 * in the energy above rhomax. Pairs count when r < cutoff.
 *
 * Each atom is matched to an element by its atomic number against the
 * first field of the element line. LAMMPS ignores that field and maps by
 * name instead. Only one funcfl file is read, so funcfl is one element.
 */

// clang-format off
#include <cstdint>
#include <iosfwd>
#include <string>
#include <vector>
// clang-format on
#include "rgpot/Locality.hpp"
#include "rgpot/ParamHash.hpp"
#include "rgpot/PerAtom.hpp"
#include "rgpot/Potential.hpp"
#include "rgpot/Staged.hpp"
#include "rgpot/nlist/Neighbours.hpp"

namespace rgpot {

enum class EAMTableFormat { Funcfl, Setfl, FinnisSinclair };

/// Tabulated functions after the LAMMPS file2array step. Entry m of an
/// array sits at m * dr or m * drho.
struct EAMTable {
  int nrho = 0;
  double drho = 0.0;
  int nr = 0;
  double dr = 0.0;
  double cutoff = 0.0;
  double rhomax = 0.0;
  std::vector<int> z;                    //!< Atomic number per element.
  std::vector<std::vector<double>> frho; //!< F_a, one per element.
  /// rho_ab at index b * n + a, the density element b puts at an a site.
  std::vector<std::vector<double>> rhor;
  /// r phi_ab at index a (a + 1) / 2 + b for a >= b.
  std::vector<std::vector<double>> z2r;
};

/// Parse one table. Throws std::invalid_argument on a malformed file.
EAMTable read_eam_table(std::istream &in, EAMTableFormat format);

/**
 * @brief Parameters for EAMTablePot.
 *
 * Energies are in the table's units, eV and Angstrom for the LAMMPS metal
 * files.
 */
struct EAMTableConfig {
  std::string path; //!< Table file.
  EAMTableFormat format = EAMTableFormat::Setfl;
  /// Minimum image convention. The cutoff must sit under half of every
  /// face spacing of the cell.
  bool periodic = true;
  /// Write the Cauchy stress of stress.hpp when the cell volume is positive.
  bool stress = false;
  /// Verlet skin of the potential's own pair list, in the length unit of
  /// the table. forceImpl keeps the list while every atom stays within
  /// half of it (nlist::Neighbours). 0 rebuilds it whenever an atom moves.
  /// The skin changes when the list is rebuilt, not the result.
  double skin = 1.0;
};

/**
 * @class EAMTablePot
 * @brief Tabulated EAM or Finnis-Sinclair energy and forces.
 * @ingroup rgpot_potentials
 */
class EAMTablePot : public Potential<EAMTablePot> {
public:
  explicit EAMTablePot(const EAMTableConfig &c);
  EAMTablePot(const EAMTable &table, bool periodic, bool stress, double skin = 1.0);

  [[nodiscard]] PotCaps caps() const noexcept override {
    // The pair list is scratch rewritten by every force call.
    return {.reentrancy = Reentrancy::PerInstance,
            .periodic = m_periodic,
            .stress = m_stress};
  }

  [[nodiscard]] uint64_t paramsKey() const noexcept override {
    return m_paramsKey;
  }

  /// The density, the embedding argument and phi all stop at the table
  /// cutoff. The staged evaluation exchanges rho.
  [[nodiscard]] Locality locality() const noexcept {
    return {.cutoff = m_table.cutoff, .densities = 1};
  }

  void forceImpl(const ForceInput &in, ForceOut *out) const override;

  /// forceImpl plus per-atom terms: each atom's embedding F_a(rho_i), and
  /// each phi_ab and pair virial split half to each atom, as pair_style
  /// eam tallies eatom and vatom.
  void forcePerAtom(const ForceInput &in, ForceOut *out, const PerAtomOut &atoms) const;

  /// Staged.hpp: adds rho_ab(r) to the density of each atom of a pair.
  void densityStage(const HalfList &list, double *densities) const;

  /// Staged.hpp: F_a(rho_i) of each owned atom and the pair forces with
  /// F'(rho) of both atoms, tallied as forcePerAtom tallies them.
  void forceStage(const HalfList &list, const double *densities, double *forces,
                  StageTally *tally) const;

  /// Stream walks the cells on every call and keeps no list.
  void setNeighbourMode(nlist::Mode mode) noexcept { m_mode = mode; }

  /// True when the last forceImpl or forcePerAtom kept the pair list.
  [[nodiscard]] bool listReused() const noexcept { return m_neighbours.reused(); }

  /// Pair lists built by forceImpl and forcePerAtom so far.
  [[nodiscard]] std::uint64_t listBuilds() const noexcept { return m_neighbours.builds(); }

private:
  static constexpr uint64_t kKernelVersion = 1;

  void compute(const ForceInput &in, ForceOut *out, const PerAtomOut *atoms) const;

  EAMTable m_table;
  /// PairEAM::interpolate coefficients, seven per point, same layout as
  /// the m_table arrays.
  std::vector<std::vector<double>> m_frho, m_rhor, m_z2r;
  /// Element index of each atomic number, -1 for one the table lacks.
  std::vector<int> m_element;
  bool m_periodic;
  bool m_stress;
  double m_skin;
  uint64_t m_paramsKey{0};
  nlist::Mode m_mode = nlist::Mode::Auto;
  mutable nlist::Neighbours m_neighbours;
};

} // namespace rgpot
