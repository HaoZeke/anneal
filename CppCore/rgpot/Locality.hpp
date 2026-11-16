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

  [[nodiscard]] constexpr bool finite() const noexcept { return cutoff > 0.0; }

  /// Distance within which the atoms set the force on one atom.
  [[nodiscard]] constexpr double forceRange() const noexcept {
    return pairwise ? cutoff : 2.0 * cutoff;
  }
};

template <typename P>
concept LocalPotential = requires(const P &pot) {
  { pot.locality() } noexcept -> std::same_as<Locality>;
};

} // namespace rgpot
