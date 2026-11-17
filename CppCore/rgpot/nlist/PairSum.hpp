#pragma once
// MIT License
// Copyright 2023--present rgpot developers

/**
 * @brief The pair sum of a two-body potential on the neighbour layer, with
 * the same bits on any number of threads.
 *
 * Below kTeamMinAtoms the half pairs are walked on the calling thread and
 * each term goes straight into the totals. From kTeamMinAtoms up, each atom
 * keeps the energy and strain of the pairs to its partners above it, and
 * the totals add those rows in atom order. One thread on a stored list or
 * an unbounded search walks the half pairs, which reach the force of an
 * atom through its partners in index order. A team walks the same rows by
 * atom range and each atom adds the force of all its partners in index
 * order itself; the vectors of forDirected make every term the negation of
 * the one-thread term, so the bits match. A stream reaches the partners of
 * an atom by cell, so it takes the walk by atom on any team.
 */

#include "rgpot/PerAtom.hpp"
#include "rgpot/Slices.hpp"
#include "rgpot/nlist/Neighbours.hpp"
#include "rgpot/stress.hpp"

#include <cstddef>
#include <cstdint>
#include <limits>
#include <vector>

// A walk calls the pair body once per pair, so the body and the term it
// calls go inline: out of line, the calls cost as much as the arithmetic.
#if defined(__GNUC__) || defined(__clang__)
#define RGPOT_PAIR_SUM_INLINE __attribute__((always_inline, flatten))
#else
#define RGPOT_PAIR_SUM_INLINE
#endif

namespace rgpot::nlist {

/// What a pair of atoms on one point does: nothing, or makes the energy NaN.
enum class Overlap : std::uint8_t { Skip, Poison };

namespace detail {

/// The half walk. With `Rows` a pair adds its energy to `energy[i]` and its
/// strain to `strain + 6 * i`; without, every pair adds to `energy[0]` and
/// `strain`. The body copies what it reads, so a walk keeps it in registers.
template <bool Rows, typename Term>
void halfPairs(const Neighbours &cells, std::size_t n, double cutoff2, Overlap overlap,
               const Term &term, double *force, double *energy, double *strain,
               const PerAtomOut *atoms) {
  cells.forHalf(0, static_cast<std::int64_t>(n),
                [=](std::int32_t i, std::int32_t j, double dx, double dy, double dz,
                    double r2) RGPOT_PAIR_SUM_INLINE {
                  if (!(i < j) || r2 > cutoff2) {
                    return;
                  }
                  const std::size_t a = Rows ? static_cast<std::size_t>(i) : 0;
                  if (!(r2 > 0.0)) {
                    if (overlap == Overlap::Poison) {
                      energy[a] = std::numeric_limits<double>::quiet_NaN();
                    }
                    return;
                  }
                  double pair_e = 0.0;
                  double fscale = 0.0;
                  term(i, j, r2, pair_e, fscale);
                  energy[a] += pair_e;
                  const double fx = fscale * dx;
                  const double fy = fscale * dy;
                  const double fz = fscale * dz;
                  force[3 * i] += fx;
                  force[3 * i + 1] += fy;
                  force[3 * i + 2] += fz;
                  force[3 * j] -= fx;
                  force[3 * j + 1] -= fy;
                  force[3 * j + 2] -= fz;
                  if (strain != nullptr) {
                    accumulatePairStrain(strain + 6 * a, fscale, dx, dy, dz);
                  }
                  if (atoms != nullptr) {
                    tallyPair(*atoms, static_cast<std::size_t>(i), static_cast<std::size_t>(j),
                              pair_e, fscale, dx, dy, dz);
                  }
                });
}

} // namespace detail

/**
 * Adds the pairs of `cells` within `cutoff2` over `n` atoms.
 *
 * `term(i, j, r2, energy, fscale)` gives the pair energy and the force on
 * i along d = r_i - r_j as fscale * d, with the same bits for (j, i). The
 * forces go into `force`, the energy into `energy`, the strain of
 * stress.hpp into `strain` when it is set, and half of each pair into the
 * arrays of `atoms` when it is set. A pair with r2 not above zero adds
 * nothing; with Overlap::Poison it makes the energy NaN.
 */
template <typename Term>
void sumPairs(const Neighbours &cells, std::size_t n, double cutoff2, Overlap overlap,
              const Term &term, double *force, double &energy, double *strain,
              const PerAtomOut *atoms) {
  if (n < kTeamMinAtoms) {
    detail::halfPairs<false>(cells, n, cutoff2, overlap, term, force, &energy, strain, atoms);
    return;
  }
  std::vector<double> rows(n, 0.0);
  std::vector<double> row_strain(strain != nullptr ? 6 * n : 0, 0.0);
  double *row_e = rows.data();
  double *row_s = strain != nullptr ? row_strain.data() : nullptr;
  const int slices = teamSize(n);
  if (slices == 1 && !(cells.streaming() && !cells.allPairs())) {
    detail::halfPairs<true>(cells, n, cutoff2, overlap, term, force, row_e, row_s, atoms);
  } else {
    cells.forRowTeam(slices, [=](int, std::int32_t i, std::int32_t j, double dx, double dy,
                                 double dz, double r2) RGPOT_PAIR_SUM_INLINE {
      if (r2 > cutoff2) {
        return;
      }
      const auto a = static_cast<std::size_t>(i);
      const bool upper = i < j;
      if (!(r2 > 0.0)) {
        if (upper && overlap == Overlap::Poison) {
          row_e[a] = std::numeric_limits<double>::quiet_NaN();
        }
        return;
      }
      double pair_e = 0.0;
      double fscale = 0.0;
      term(i, j, r2, pair_e, fscale);
      force[3 * a] += fscale * dx;
      force[3 * a + 1] += fscale * dy;
      force[3 * a + 2] += fscale * dz;
      if (upper) {
        row_e[a] += pair_e;
        if (row_s != nullptr) {
          accumulatePairStrain(row_s + 6 * a, fscale, dx, dy, dz);
        }
      }
      if (atoms != nullptr) {
        if (atoms->energy != nullptr) {
          atoms->energy[a] += 0.5 * pair_e;
        }
        tallyAtomVirial(*atoms, a, 0.5, fscale, dx, dy, dz);
      }
    });
  }
  for (std::size_t a = 0; a < n; ++a) {
    energy += rows[a];
  }
  if (strain != nullptr) {
    for (std::size_t a = 0; a < n; ++a) {
      for (std::size_t c = 0; c < 6; ++c) {
        strain[c] += row_strain[6 * a + c];
      }
    }
  }
}

} // namespace rgpot::nlist

#undef RGPOT_PAIR_SUM_INLINE
