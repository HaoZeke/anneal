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
#include <algorithm>
#include <array>
#include <cmath>
#include <limits>
#include <vector>
// clang-format on

#include "rgpot/LennardJones/LJPot.hpp"
#include "rgpot/nlist/NeighbourList.hpp"
#include "rgpot/nlist/PairSum.hpp"
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
 * a later call inside half the skin reuses the candidates. nlist::sumPairs
 * adds the pairs, with the same bits on any number of threads. A nonzero
 * switch width uses the quintic switch. Cauchy stress is published from
 * the kernel's strain accumulator.
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
  updateNeighbours(in);
  const double cutoff2 = cuttOffR * cuttOffR;
  // Two different atoms on the same point make the energy NaN. Skipping
  // the pair would report a finite energy of zero and hide the singularity.
  nlist::sumPairs(
      neighbours_, n_atoms, cutoff2, nlist::Overlap::Poison,
      [&params](std::int32_t, std::int32_t, double r2, double &pair_e, double &fscale) {
        lj_pair(r2, params, pair_e, fscale);
      },
      F, *U, strain, atoms);
  publishCauchyStress(out, strain, volume);
}

void LJPot::updateNeighbours(const ForceInput &in) const {
  // One angstrom of skin. A stored list keeps pairs out to cutoff + skin
  // and the kernels drop those past the cutoff. Stream asks for no skin:
  // the bins are not reused unless the coordinates match.
  const double skin = mode_ == nlist::Mode::Stream ? 0.0 : 1.0;
  const std::array<bool, 3> periodic{{true, true, true}};
  nlist::updateCells(neighbours_, in.pos, static_cast<std::size_t>(in.nAtoms), in.box,
                     cuttOffR, skin, nlist::Image::Round, mode_, periodic);
}

/**
 * @details
 *
 * With @f$a = (\psi/r)^6@f$ the bare pair term @f$V = 4 u_0 a (a - 1)@f$
 * has @f$V' = (4 u_0 / r)(6 a - 12 a^2)@f$ and
 * @f$V'' = (4 u_0 / r^2)(156 a^2 - 42 a)@f$. The shift is a constant and
 * drops out. The switch multiplies the bare term by @f$S@f$, so
 * @f$\phi' = V' S + V S'@f$ and @f$\phi'' = V'' S + 2 V' S' + V S''@f$.
 */
void LJPot::hessianVectorImpl(const ForceInput &in, const double *v, double *hv) const {
  const auto N = static_cast<long>(in.nAtoms);
  std::fill_n(hv, 3 * in.nAtoms, 0.0);
  if (N < 2 || cuttOffR <= 0.0) {
    return;
  }
  updateNeighbours(in);
  const double cutoff2 = cuttOffR * cuttOffR;
  const double psi2 = psi * psi;
  const double fourU0 = 4.0 * u0;
  const bool switched = m_config.switch_width != 0.0;
  nlist::visitHalf(neighbours_, 0, N,
                   [&](std::int32_t i, std::int32_t j, double dx, double dy, double dz,
                       double r2) {
    if (!(i < j) || r2 > cutoff2) {
      return;
    }
    if (!(r2 > 0.0)) {
      throw std::domain_error(
          "LJPot Hessian-vector product is undefined for coincident atoms");
    }
    const double r = std::sqrt(r2);
    const double inv_r = 1.0 / r;
    const double sr2 = psi2 / r2;
    const double a = sr2 * sr2 * sr2;
    double d1 = fourU0 * inv_r * (6.0 * a - 12.0 * a * a);
    double d2 = fourU0 * inv_r * inv_r * (156.0 * a * a - 42.0 * a);
    if (switched) {
      const double bare = fourU0 * a * (a - 1.0);
      const auto s = m_switch(r);
      const double s2 = m_switch.d2sdr2(r);
      d2 = d2 * s.s + 2.0 * d1 * s.dsdr + bare * s2;
      d1 = d1 * s.s + bare * s.dsdr;
    }
    const double ux = dx * inv_r;
    const double uy = dy * inv_r;
    const double uz = dz * inv_r;
    const double *vi = v + 3 * static_cast<std::size_t>(i);
    const double *vj = v + 3 * static_cast<std::size_t>(j);
    const double wx = vi[0] - vj[0];
    const double wy = vi[1] - vj[1];
    const double wz = vi[2] - vj[2];
    const double radial = d1 * inv_r;
    const double along = (d2 - radial) * (ux * wx + uy * wy + uz * wz);
    const double kx = along * ux + radial * wx;
    const double ky = along * uy + radial * wy;
    const double kz = along * uz + radial * wz;
    double *hi = hv + 3 * static_cast<std::size_t>(i);
    double *hj = hv + 3 * static_cast<std::size_t>(j);
    hi[0] += kx;
    hi[1] += ky;
    hi[2] += kz;
    hj[0] -= kx;
    hj[1] -= ky;
    hj[2] -= kz;
  });
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
