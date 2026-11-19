#pragma once
// MIT License
// Copyright 2023--present rgpot developers

/**
 * @brief Interaction range of a potential with a finite cutoff.
 *
 * A potential whose energy is a sum of site terms, E = sum_i E_i, where
 * each E_i depends only on the atoms within a distance @c cutoff of atom
 * i, has a non-virtual member
 *
 *   Locality locality() const noexcept;
 *
 * The @c LocalPotential concept detects it, the way @c PerAtomPotential
 * detects forcePerAtom, so a caller holding the concrete type can split
 * the work over regions of space without a new virtual on
 * @c PotentialBase or a new field in @c PotCaps.
 *
 * Then the force on atom k, -sum_i dE_i/dr_k, depends only on the atoms
 * within 2 * cutoff of k: the sites i within cutoff of k and their own
 * neighbours. When every term of E is a function of one pair distance
 * (@c pairwise), it depends only on the atoms within cutoff. That radius
 * is @c forceRange. A host can therefore evaluate the potential on its
 * own atoms plus every atom within @c forceRange of them, free of
 * periodic images, and keep the forces, the site energies and the
 * per-atom virials (PerAtom.hpp) of its own atoms: they equal those of
 * the whole system.
 *
 * A potential with the staged evaluation of Staged.hpp names how far the
 * copies reach. @c reach is 1 when an exchange of @c densities values
 * per atom carries what the neighbours of the neighbours contribute, so
 * the copies stop at @c cutoff. It is 2 when a three body term reads the
 * full neighbour list of each neighbour and the copies must reach
 * 2 * cutoff. @c full selects that list, the one pair_sw and pair_tersoff
 * build. @c ghostReach() is @c reach * cutoff. @c forceRange() stays the
 * distance that sets one force: @c cutoff when @c pairwise, otherwise
 * twice @c cutoff, whether or not a density exchange shortens the copies.
 *
 * A cutoff of 0 declares no finite range (a whole-system or long-range
 * potential, or a truncation switched off).
 */

#include <concepts>

namespace rgpot {

struct Locality {
  /// Range of a site energy, in the length unit of the potential.
  double cutoff = 0.0;
  /// Every energy term is a function of one pair distance.
  bool pairwise = false;
  /// Values per atom that the density stage of Staged.hpp sums over the
  /// neighbours and the force stage reads at both atoms of a pair. 0 for
  /// a potential without densities.
  int densities = 0;
  /// The staged neighbour list keeps every neighbour of a centre, as
  /// pair_sw and pair_tersoff request. False keeps the half list of a
  /// pair potential or a density potential.
  bool full = false;
  /// Copies the staged host must hold, in units of cutoff. 1 when the
  /// density exchange carries the second shell. 2 when a three body term
  /// needs each neighbour's full list, a reach of 2 * cutoff.
  int reach = 1;

  [[nodiscard]] constexpr bool finite() const noexcept { return cutoff > 0.0; }

  /// Distance within which the atoms set the force on one atom.
  [[nodiscard]] constexpr double forceRange() const noexcept {
    return pairwise ? cutoff : 2.0 * cutoff;
  }

  /// Distance of the copies a staged host holds. 0 when there is no
  /// finite range.
  [[nodiscard]] constexpr double ghostReach() const noexcept {
    return finite() ? static_cast<double>(reach) * cutoff : 0.0;
  }
};

template <typename P>
concept LocalPotential = requires(const P &pot) {
  { pot.locality() } noexcept -> std::same_as<Locality>;
};

} // namespace rgpot
