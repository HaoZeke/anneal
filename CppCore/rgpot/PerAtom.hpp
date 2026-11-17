#pragma once
// MIT License
// Copyright 2023--present rgpot developers

/**
 * @brief Per-atom energies and virials.
 *
 * A potential that can split its energy and virial over the atoms has a
 * non-virtual member
 *
 *   void forcePerAtom(const ForceInput &, ForceOut *, const PerAtomOut &) const;
 *
 * which fills @c ForceOut exactly as @c forceImpl does and also writes the
 * per-atom arrays. The @c PerAtomPotential concept detects it, so callers
 * that hold the concrete type (rgpot::spec::build, pair_style rgpot) can
 * offer per-atom output where it exists without a new virtual on
 * @c PotentialBase. Potentials without the member are unchanged.
 *
 * Pair terms go half to each atom. An embedding term F(rho_i) belongs to
 * atom i alone. The per-atom virial is W_i, energy units, with
 * sum_i W_i = -V sigma = -dE/dstrain: a pair whose force on atom i is
 * fscale * d, d = r_i - r_j, puts fscale * d (x) d / 2 on i and on j, the
 * split of LAMMPS ev_tally. Six entries per atom in the Voigt order of
 * stress.hpp: xx, yy, zz, yz, xz, xy.
 */

#include "rgpot/ForceStructs.hpp"

#include <algorithm>
#include <cstddef>

namespace rgpot {

/// Per-atom result arrays. A null pointer skips that quantity.
struct PerAtomOut {
  double *energy = nullptr; //!< nAtoms entries; they sum to ForceOut::energy.
  double *virial = nullptr; //!< 6 * nAtoms entries, xx yy zz yz xz xy per atom.
};

template <typename P>
concept PerAtomPotential = requires(const P &pot, const ForceInput &in, ForceOut *out,
                                    const PerAtomOut &atoms) {
  pot.forcePerAtom(in, out, atoms);
};

/// Zero the arrays of @p atoms for @p n atoms.
inline void clearPerAtom(const PerAtomOut &atoms, std::size_t n) {
  if (atoms.energy != nullptr) {
    std::fill(atoms.energy, atoms.energy + n, 0.0);
  }
  if (atoms.virial != nullptr) {
    std::fill(atoms.virial, atoms.virial + 6 * n, 0.0);
  }
}

/// Add @p share of the pair virial fscale * d (x) d to atom @p i.
inline void tallyAtomVirial(const PerAtomOut &atoms, std::size_t i, double share,
                            double fscale, double dx, double dy, double dz) {
  if (atoms.virial == nullptr) {
    return;
  }
  const double c = share * fscale;
  double *w = atoms.virial + 6 * i;
  w[0] += c * dx * dx;
  w[1] += c * dy * dy;
  w[2] += c * dz * dz;
  w[3] += c * dy * dz;
  w[4] += c * dx * dz;
  w[5] += c * dx * dy;
}

/// Half of a pair energy and of its virial to each of @p i and @p j.
inline void tallyPair(const PerAtomOut &atoms, std::size_t i, std::size_t j, double energy,
                      double fscale, double dx, double dy, double dz) {
  if (atoms.energy != nullptr) {
    atoms.energy[i] += 0.5 * energy;
    atoms.energy[j] += 0.5 * energy;
  }
  tallyAtomVirial(atoms, i, 0.5, fscale, dx, dy, dz);
  tallyAtomVirial(atoms, j, 0.5, fscale, dx, dy, dz);
}

} // namespace rgpot
