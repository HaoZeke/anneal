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
#include <vector>

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

/**
 * @brief Private per-atom arrays, one per slice, for a walk that writes
 * both atoms of a pair from any slice.
 *
 * Slice @c t tallies into @c slot(t); @c reduce adds the slots into the
 * caller's arrays in slice order. A null target allocates nothing and
 * hands out empty slots.
 */
class PerAtomTeam {
public:
  PerAtomTeam(const PerAtomOut *target, std::size_t n, int nthreads)
      : m_target{target}, m_n{n}, m_threads{nthreads < 1 ? 1 : nthreads} {
    if (m_target == nullptr) {
      return;
    }
    if (m_target->energy != nullptr) {
      m_energy.assign(static_cast<std::size_t>(m_threads) * n, 0.0);
    }
    if (m_target->virial != nullptr) {
      m_virial.assign(static_cast<std::size_t>(m_threads) * 6 * n, 0.0);
    }
  }

  [[nodiscard]] PerAtomOut slot(int t) {
    const auto k = static_cast<std::size_t>(t);
    return {m_energy.empty() ? nullptr : m_energy.data() + k * m_n,
            m_virial.empty() ? nullptr : m_virial.data() + k * 6 * m_n};
  }

  void reduce() const {
    if (m_target == nullptr) {
      return;
    }
    for (int t = 0; t < m_threads; ++t) {
      const auto k = static_cast<std::size_t>(t);
      if (!m_energy.empty()) {
        for (std::size_t a = 0; a < m_n; ++a) {
          m_target->energy[a] += m_energy[k * m_n + a];
        }
      }
      if (!m_virial.empty()) {
        for (std::size_t a = 0; a < 6 * m_n; ++a) {
          m_target->virial[a] += m_virial[k * 6 * m_n + a];
        }
      }
    }
  }

private:
  const PerAtomOut *m_target;
  std::size_t m_n;
  int m_threads;
  std::vector<double> m_energy;
  std::vector<double> m_virial;
};

} // namespace rgpot
