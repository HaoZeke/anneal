// MIT License
// Copyright 2023--present rgpot developers

/**
 * @brief Implementation of the Lennard-Jones potential methods.
 *
 * This file contains the implementation of the force and energy
 * calculation for the Lennard-Jones potential, including periodic
 * boundary condition handling.
 */

// clang-format off
#include <array>
#include <cmath>
#include <limits>
#include <vector>
// clang-format on

#if defined(_OPENMP)
#include <omp.h>
#endif

#include "rgpot/LennardJones/LJPot.hpp"
#include "rgpot/nlist/NeighbourList.hpp"
#include "rgpot/simd/api.hpp"
#include "rgpot/simd/exact_fma.hpp"
#include "rgpot/stress.hpp"
#include "rgpot/types/AtomMatrix.hpp"
using rgpot::types::AtomMatrix;

namespace rgpot {
namespace {

inline void lj_pair(double r2, const simd::LjParams &p, double &energy, double &fscale) {
  const double inv_r2 = 1.0 / r2;
  const double sr2 = p.psi2 * inv_r2;
  const double sr4 = sr2 * sr2;
  const double a = sr4 * sr2;
  const double b = p.four_u0 * a;
  const double a_m1 = a - 1.0;
  const double v = b * a_m1;
  const double two_a = 2.0 * a;
  const double two_a_m1 = two_a - 1.0;
  const double six_b = 6.0 * b;
  const double f_un = (six_b * inv_r2) * two_a_m1;
  if (p.use_switch == 0) {
    energy = v - p.shift;
    fscale = f_un;
    return;
  }
  const double r = std::sqrt(r2);
  const double t_raw = (r - p.r_on) * p.inv_width;
  const double t = t_raw < 0.0 ? 0.0 : (t_raw > 1.0 ? 1.0 : t_raw);
  const double t2 = t * t;
  const double t3 = t2 * t;
  const double six_t = 6.0 * t;
  const double inner = 15.0 - six_t;
  const double poly = simd::fma_quick(t, inner, -10.0);
  const double s = simd::fma_quick(t3, poly, 1.0);
  const double omt = 1.0 - t;
  const double ds2 = (-30.0 * t2) * omt * omt;
  const double dsdr = ds2 * p.inv_width;
  energy = v * s;
  fscale = f_un * s - (v * dsdr) / r;
}

simd::LjParams lj_params(double u0, double psi, double shift, const LJConfig &c,
                         const QuinticSwitch &sw) {
  simd::LjParams params{};
  params.psi2 = psi * psi;
  params.four_u0 = 4.0 * u0;
  params.shift = shift;
  params.use_switch = 0;
  if (c.switch_width != 0.0) {
    params.use_switch = 1;
    params.r_on = sw.r_on;
    params.inv_width = sw.inv_width;
  }
  return params;
}

} // namespace

/**
 * @class LJPot
 * @details
 *
 * Pairwise interactions within the cutoff radius, minimum image convention
 * on an orthogonal or sheared box. Pairs come from the shared neighbour
 * layer: a cell list, a Verlet skin, and either a stored list or a
 * streaming walk. The pair term runs during that walk. A streaming call
 * does not copy a half-list. A stored call keeps candidates out to the
 * cutoff plus the skin and drops the rest from the current distances, so
 * a later call inside half the skin reuses the candidates. Several
 * threads split the atoms, keep a private force buffer, and add those
 * buffers in thread order. A nonzero switch width uses the quintic
 * switch. Cauchy stress is published from the kernel's strain accumulator.
 *
 * @warning The box is assumed to be orthogonal.
 *
 */
void LJPot::forceImpl(const ForceInput &in, ForceOut *out) const {
  compute(in, out, nullptr);
}

void LJPot::forcePerAtom(const ForceInput &in, ForceOut *out, const PerAtomOut &atoms) const {
  clearPerAtom(atoms, in.nAtoms);
  compute(in, out, &atoms);
}

void LJPot::compute(const ForceInput &in, ForceOut *out, const PerAtomOut *atoms) const {
  const long N = in.nAtoms;
  const double *R = in.pos;
  const double *box = in.box;
  double *F = out->F;
  double *U = &out->energy;
  double strain[6] = {};
  const double volume = cellVolume(box);
  *U = 0;
  for (long i = 0; i < N; i++) {
    F[3 * i] = 0;
    F[3 * i + 1] = 0;
    F[3 * i + 2] = 0;
  }
  if (N < 2 || cuttOffR <= 0.0) {
    publishCauchyStress(out, strain, volume);
    return;
  }

  const simd::LjParams params = lj_params(u0, psi, cuttOffU, m_config, m_switch);
  const auto n_atoms = static_cast<std::size_t>(N);
  // One angstrom of skin. A stored list keeps pairs out to cutoff + skin
  // and the kernel below drops those past the cutoff. Stream asks for no
  // skin: the bins are not reused unless the coordinates match.
  const double skin = mode_ == nlist::Mode::Stream ? 0.0 : 1.0;
  const std::array<bool, 3> periodic{{true, true, true}};
  nlist::updateCells(neighbours_, R, n_atoms, box, cuttOffR, skin, nlist::Image::Round,
                     mode_, periodic);
  const double cutoff2 = cuttOffR * cuttOffR;
  const int nt = nlist::threadsRequested();
  auto accept = [&](std::int32_t i, std::int32_t j, double dx, double dy, double dz,
                    double r2, double *force, double &energy, double *virial,
                    const PerAtomOut *share) {
    if (!(i < j) || r2 > cutoff2) {
      return;
    }
    // Two different atoms on the same point. Skipping the pair reports a
    // finite energy of zero and hides the singularity.
    if (!(r2 > 0.0)) {
      energy = std::numeric_limits<double>::quiet_NaN();
      return;
    }
    double pair_e = 0.0;
    double fscale = 0.0;
    lj_pair(r2, params, pair_e, fscale);
    energy += pair_e;
    const double fx = fscale * dx;
    const double fy = fscale * dy;
    const double fz = fscale * dz;
    force[3 * i] += fx;
    force[3 * i + 1] += fy;
    force[3 * i + 2] += fz;
    force[3 * j] -= fx;
    force[3 * j + 1] -= fy;
    force[3 * j + 2] -= fz;
    accumulatePairStrain(virial, fscale, dx, dy, dz);
    if (share != nullptr) {
      tallyPair(*share, static_cast<std::size_t>(i), static_cast<std::size_t>(j), pair_e, fscale,
                dx, dy, dz);
    }
  };
  if (nt <= 1) {
    nlist::visitHalf(neighbours_, 0, N,
                     [&](std::int32_t i, std::int32_t j, double dx, double dy, double dz,
                         double r2) { accept(i, j, dx, dy, dz, r2, F, *U, strain, atoms); });
  } else {
    std::vector<double> owned(static_cast<std::size_t>(nt) * n_atoms * 3, 0.0);
    std::vector<double> energy(static_cast<std::size_t>(nt), 0.0);
    std::vector<double> virial(static_cast<std::size_t>(nt) * 6, 0.0);
    PerAtomTeam team(atoms, n_atoms, nt);
    std::vector<PerAtomOut> slots(static_cast<std::size_t>(nt));
    for (int t = 0; t < nt; ++t) {
      slots[static_cast<std::size_t>(t)] = team.slot(t);
    }
    // One shared tile build. A visit from each thread would allocate the
    // cell scratch once per thread.
    neighbours_.forCellTeam(
        true, nt, [&](int t, std::int32_t i, std::int32_t j, double dx, double dy, double dz,
                      double r2) {
          double *local = owned.data() + static_cast<std::size_t>(t) * n_atoms * 3;
          double *st = virial.data() + static_cast<std::size_t>(t) * 6;
          accept(i, j, dx, dy, dz, r2, local, energy[static_cast<std::size_t>(t)], st,
                 atoms != nullptr ? &slots[static_cast<std::size_t>(t)] : nullptr);
        });
    team.reduce();
    for (int t = 0; t < nt; ++t) {
      *U += energy[static_cast<std::size_t>(t)];
      const double *local = owned.data() + static_cast<std::size_t>(t) * n_atoms * 3;
      for (std::size_t k = 0; k < n_atoms * 3; ++k) {
        F[k] += local[k];
      }
      for (int c = 0; c < 6; ++c) {
        strain[c] += virial[static_cast<std::size_t>(t) * 6 + static_cast<std::size_t>(c)];
      }
    }
  }
  publishCauchyStress(out, strain, volume);
}

void LJPot::forceStage(const HalfList &list, const double *, double *forces,
                       StageTally *tally) const {
  if (cuttOffR <= 0.0) {
    return;
  }
  const simd::LjParams params = lj_params(u0, psi, cuttOffU, m_config, m_switch);
  const double cutoff2 = cuttOffR * cuttOffR;
  staged::withTally(tally, [&](auto tallied) {
    staged::forEachPair(
        list, [&](int i, int j, double dx, double dy, double dz, double r2, bool both) {
          if (r2 > cutoff2) {
            return;
          }
          if (!(r2 > 0.0)) {
            if constexpr (decltype(tallied)::value) {
              tally->energy = std::numeric_limits<double>::quiet_NaN();
            }
            return;
          }
          double pair_e = 0.0;
          double fscale = 0.0;
          lj_pair(r2, params, pair_e, fscale);
          staged::addPair(tallied, forces, tally, i, j, both, pair_e, fscale, dx, dy, dz);
        });
  });
}

} // namespace rgpot
