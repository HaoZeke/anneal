#pragma once
// MIT License
// Copyright 2023--present rgpot developers

/**
 * @brief Two-stage evaluation of a local potential on a host's neighbour
 * list.
 *
 * A host that splits space over processes, such as LAMMPS, holds its own
 * atoms and copies of the atoms around them (periodic images, and atoms
 * that another process owns), and builds the neighbour list itself. A
 * potential with a finite range (Locality.hpp) that has the two members
 *
 *   void densityStage(const HalfList &, double *densities) const;
 *   void forceStage(const HalfList &, const double *densities,
 *                   double *forces, StageTally *tally) const;
 *
 * evaluates its host's atoms on that list with copies out to its cutoff,
 * not out to Locality::forceRange. The density stage adds the pair terms
 * of each atom's densities (the electron density of EAM and Sutton-Chen)
 * into an array the host has zeroed, Locality::densities consecutive
 * values per atom. The host then completes them: with HalfList::newton
 * it first adds the values written to copies into their owners, and in
 * every case it copies each owner's values to its copies. The force
 * stage takes the embedding of every atom from its densities and the pair
 * forces from the densities at both ends of each pair. This is the
 * pair_style eam scheme with one difference: the host communicates rho
 * where pair_style eam communicates F'(rho), and the force stage takes F'
 * of the copies too, so no third stage runs between the exchanges. A
 * pairwise potential has no densities; its density stage does nothing
 * and the host skips the exchange.
 *
 * The @c StagedPotential concept detects the two members, as
 * @c PerAtomPotential detects forcePerAtom, so they add no virtual
 * function to @c PotentialBase. The helpers in @c rgpot::staged hold the
 * loop over the list and the share of each pair that a host keeps.
 */

#include "rgpot/Locality.hpp"
#include "rgpot/PerAtom.hpp"

#include <cstddef>
#include <type_traits>

namespace rgpot {

/**
 * @brief A host's half neighbour list over its atoms and their copies.
 *
 * Atoms [0, nOwned) are the host's own and [nOwned, nAtoms) are copies.
 * Row k belongs to the owned atom i = rows[k], whose neighbours are
 * neighbours[i][0 .. count[i]): the layout of a LAMMPS NeighList (ilist,
 * numneigh, firstneigh), with @c count and @c neighbours indexed by atom.
 * @c indexMask clears any bits a host sets in an entry besides the index
 * (LAMMPS NEIGHMASK).
 *
 * A pair of owned atoms is listed once, in either row. A pair of an owned
 * atom and a copy is listed in the owned atom's row. With @c newton, only
 * one of the two hosts that hold such a pair lists it, and the stages
 * write the copy's share as well for the host to add to its owner
 * (LAMMPS newton_pair on). Without it both hosts list the pair and each
 * writes its own atom. Entries beyond the cutoff are skipped, so a list
 * built with a skin works. A copy sits at its image, so the separation of
 * a pair is the plain difference of its positions.
 */
struct HalfList {
  std::size_t nAtoms = 0;
  std::size_t nOwned = 0;
  const double *pos = nullptr; //!< 3 * nAtoms Cartesian coordinates.
  const int *atmnrs = nullptr; //!< Atomic number per atom.
  std::size_t nRows = 0;
  const int *rows = nullptr;
  const int *count = nullptr;
  const int *const *neighbours = nullptr;
  int indexMask = ~0;
  bool newton = false;
};

/**
 * @brief What forceStage adds up besides the forces.
 *
 * The host's share of the energy and of the virial W = -V sigma, in the
 * Voigt order of stress.hpp (xx yy zz yz xz xy): all of a pair that only
 * this host lists, half of a pair that both hosts list, and the site
 * energy of each owned atom. Per-atom arrays, when set, take the split of
 * PerAtom.hpp over nAtoms entries; a copy's entries are written only with
 * HalfList::newton, for the host to add to its owner.
 */
struct StageTally {
  double energy = 0.0;
  double virial[6] = {};
  PerAtomOut atoms{};
};

template <typename P>
concept StagedPotential =
    LocalPotential<P> &&
    requires(const P &pot, const HalfList &list, double *densities, const double *rows,
             double *forces, StageTally *tally) {
      pot.densityStage(list, densities);
      pot.forceStage(list, rows, forces, tally);
    };

namespace staged {

/// Calls fn(i, j, dx, dy, dz, r2, both) for every pair of @p list, with
/// d = r_i - r_j and @p both set when the stages write atom j as well as
/// the row atom i: j is owned, or the list is newton.
template <typename Fn> void forEachPair(const HalfList &list, Fn &&fn) {
  const double *x = list.pos;
  const auto owned = static_cast<long long>(list.nOwned);
  for (std::size_t k = 0; k < list.nRows; ++k) {
    const int i = list.rows[k];
    const double *xi = x + 3 * static_cast<std::size_t>(i);
    const double xi0 = xi[0];
    const double xi1 = xi[1];
    const double xi2 = xi[2];
    const int *row = list.neighbours[i];
    const int n = list.count[i];
    for (int q = 0; q < n; ++q) {
      const int j = row[q] & list.indexMask;
      const double *xj = x + 3 * static_cast<std::size_t>(j);
      const double dx = xi0 - xj[0];
      const double dy = xi1 - xj[1];
      const double dz = xi2 - xj[2];
      fn(i, j, dx, dy, dz, dx * dx + dy * dy + dz * dz, list.newton || j < owned);
    }
  }
}

/// Runs body(std::true_type{}) with a tally and body(std::false_type{})
/// without, so a stage compiles its loop once with the energy and virial
/// and once with the forces alone.
template <typename Body> void withTally(const StageTally *tally, Body &&body) {
  if (tally != nullptr) {
    body(std::true_type{});
  } else {
    body(std::false_type{});
  }
}

/// Adds a pair whose force on atom i is fscale * d, d = r_i - r_j: the
/// force on i, the opposite force on j when @p both, and with a tally the
/// host's share of the pair energy and virial (all of it when both, else
/// half) and half of each to atom i and, when both, to atom j.
template <bool Tally>
inline void addPair(std::bool_constant<Tally>, double *forces, StageTally *tally, int i, int j,
                    bool both, double energy, double fscale, double dx, double dy, double dz) {
  const double fx = fscale * dx;
  const double fy = fscale * dy;
  const double fz = fscale * dz;
  double *fi = forces + 3 * static_cast<std::size_t>(i);
  fi[0] += fx;
  fi[1] += fy;
  fi[2] += fz;
  if (both) {
    double *fj = forces + 3 * static_cast<std::size_t>(j);
    fj[0] -= fx;
    fj[1] -= fy;
    fj[2] -= fz;
  }
  if constexpr (Tally) {
    const double share = both ? 1.0 : 0.5;
    tally->energy += share * energy;
    const double c = share * fscale;
    double *w = tally->virial;
    w[0] += c * dx * dx;
    w[1] += c * dy * dy;
    w[2] += c * dz * dz;
    w[3] += c * dy * dz;
    w[4] += c * dx * dz;
    w[5] += c * dx * dy;
    const PerAtomOut &atoms = tally->atoms;
    if (atoms.energy != nullptr) {
      atoms.energy[i] += 0.5 * energy;
      if (both) {
        atoms.energy[j] += 0.5 * energy;
      }
    }
    tallyAtomVirial(atoms, static_cast<std::size_t>(i), 0.5, fscale, dx, dy, dz);
    if (both) {
      tallyAtomVirial(atoms, static_cast<std::size_t>(j), 0.5, fscale, dx, dy, dz);
    }
  }
}

/// Adds the site energy of the owned atom @p i.
inline void addSite(StageTally &tally, int i, double energy) {
  tally.energy += energy;
  if (tally.atoms.energy != nullptr) {
    tally.atoms.energy[i] += energy;
  }
}

} // namespace staged

} // namespace rgpot
