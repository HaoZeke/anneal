// MIT License
// Copyright 2023--present rgpot developers

/**
 * @brief Implementation of the free-boundary Lennard-Jones cluster
 * potential.
 *
 * Ported from eOn (https://github.com/TheochemUI/eOn,
 * client/potentials/LJCluster), BSD-3-Clause licensed, copyright the eOn
 * Development Team.
 */

// clang-format off
#include <algorithm>
#include <array>
#include <cmath>
#include <vector>
// clang-format on

#include "rgpot/LennardJones/LJClusterKernel.hpp"
#include "rgpot/LennardJones/LJClusterPot.hpp"
#include "rgpot/nlist/NeighbourList.hpp"
#include "rgpot/simd/api.hpp"
#include "rgpot/simd/exact_fma.hpp"

namespace rgpot {

/**
 * @class LJClusterPot
 * @details
 *
 * Every pair within the cutoff contributes, with vectors taken straight
 * from the coordinates: periodicity is off, so the walk never folds a
 * separation into the cell. A cluster whose bounding box fits inside
 * the cutoff, and whose call is not a stream, walks ``i < j`` in SoA
 * form. Every other call runs the pair term on the shared cell walk.
 * A stream copies no pair index. A nonzero switch width skips the
 * direct walk, which has no switch. A degenerate input cell is
 * replaced by a 1e6 Angstrom cube.
 */
void LJClusterPot::forceImpl(const ForceInput &in, ForceOut *out) const {
  compute(in, out, nullptr);
}

void LJClusterPot::forcePerAtom(const ForceInput &in, ForceOut *out,
                                const PerAtomOut &atoms) const {
  clearPerAtom(atoms, in.nAtoms);
  compute(in, out, &atoms);
}

void LJClusterPot::compute(const ForceInput &in, ForceOut *out,
                           const PerAtomOut *atoms) const {
  const auto N = static_cast<long>(in.nAtoms);
  const double *R = in.pos;
  double *F = out->F;
  double *U = &out->energy;
  *U = 0.0;
  for (long i = 0; i < N; i++) {
    F[3 * i] = 0.0;
    F[3 * i + 1] = 0.0;
    F[3 * i + 2] = 0.0;
  }
  if (N < 2) {
    retained_ = 0;
    return;
  }

  const double cutoff = (cuttOffR > 0.0) ? cuttOffR : 1.0e6;

  static constexpr double free_box[9] = {1e6, 0.0, 0.0, 0.0, 1e6,
                                         0.0, 0.0, 0.0, 1e6};
  const bool box_ok = in.box != nullptr && in.box[0] > 0.0 && in.box[4] > 0.0 &&
                      in.box[8] > 0.0;
  const double *box_use = box_ok ? in.box : free_box;

  double minx = R[0];
  double maxx = minx;
  double miny = R[1];
  double maxy = miny;
  double minz = R[2];
  double maxz = minz;
  for (long i = 1; i < N; ++i) {
    const double x = R[3 * i];
    const double y = R[3 * i + 1];
    const double z = R[3 * i + 2];
    minx = std::min(minx, x);
    maxx = std::max(maxx, x);
    miny = std::min(miny, y);
    maxy = std::max(maxy, y);
    minz = std::min(minz, z);
    maxz = std::max(maxz, z);
  }
  const double ex = maxx - minx;
  const double ey = maxy - miny;
  const double ez = maxz - minz;
  const bool switched = m_config.switch_width != 0.0;
  // Every pair is at most the box diagonal. When that diagonal sits on
  // the cutoff, the direct walk and the cutoff test agree on the set.
  // The direct walk is the shifted pair term, so a quintic switch stays
  // on the cell walk. A stream never takes the direct walk: that walk
  // is an all-pairs kernel, and a stream has to stay on the shared cells.
  if (!switched && mode_ != nlist::Mode::Stream && atoms == nullptr &&
      ((ex * ex + ey * ey) + ez * ez <= cutoff * cutoff)) {
    double energyAcc = 0.0;
    lj_cluster_direct(R, static_cast<std::size_t>(N), cutoff, u0, psi, cuttOffU, F,
                      &energyAcc);
    *U = energyAcc;
    retained_ = 0;
    return;
  }

  simd::LjParams params{};
  params.psi2 = psi * psi;
  params.four_u0 = 4.0 * u0;
  params.shift = cuttOffU;
  params.use_switch = 0;
  if (switched) {
    params.use_switch = 1;
    params.r_on = m_switch.r_on;
    params.inv_width = m_switch.inv_width;
  }
  const double skin = mode_ == nlist::Mode::Stream ? 0.0 : cutoff / 5.0;
  const std::array<bool, 3> open{{false, false, false}};
  nlist::updateCells(neighbours_, R, static_cast<std::size_t>(N), box_use, cutoff, skin,
                     nlist::Image::Round, mode_, open);
  retained_ = neighbours_.streaming() ? 0 : neighbours_.slots();
  const double cutoff2 = cutoff * cutoff;
  auto accept = [&](std::int32_t i, std::int32_t j, double dx, double dy, double dz,
                    double r2, double *force, double &energy, const PerAtomOut *share) {
    if (!(i < j) || !(r2 > 0.0) || r2 > cutoff2) {
      return;
    }
    const double inv_r2 = 1.0 / r2;
    const double sr2 = params.psi2 * inv_r2;
    const double a = (sr2 * sr2) * sr2;
    const double b = params.four_u0 * a;
    double pair_e = 0.0;
    double fscale = 0.0;
    if (params.use_switch == 0) {
      pair_e = b * (a - 1.0) - params.shift;
      fscale = ((6.0 * b) * inv_r2) * ((2.0 * a) - 1.0);
    } else {
      const double r = std::sqrt(r2);
      const double t_raw = (r - params.r_on) * params.inv_width;
      const double t = t_raw < 0.0 ? 0.0 : (t_raw > 1.0 ? 1.0 : t_raw);
      const double t2 = t * t;
      const double t3 = t2 * t;
      const double poly = simd::fma_quick(t, 15.0 - 6.0 * t, -10.0);
      const double s = simd::fma_quick(t3, poly, 1.0);
      const double dsdr = ((-30.0 * t2) * (1.0 - t) * (1.0 - t)) * params.inv_width;
      const double v = b * (a - 1.0);
      const double f_un = ((6.0 * b) * inv_r2) * ((2.0 * a) - 1.0);
      pair_e = v * s;
      fscale = f_un * s - (v * dsdr) / r;
    }
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
    if (share != nullptr) {
      tallyPair(*share, static_cast<std::size_t>(i), static_cast<std::size_t>(j), pair_e, fscale,
                dx, dy, dz);
    }
  };
  const int nt = nlist::threadsRequested();
  if (nt <= 1) {
    nlist::visitHalf(neighbours_, 0, N,
                     [&](std::int32_t i, std::int32_t j, double dx, double dy, double dz,
                         double r2) { accept(i, j, dx, dy, dz, r2, F, *U, atoms); });
  } else {
    std::vector<double> owned(static_cast<std::size_t>(nt) * static_cast<std::size_t>(N) * 3,
                              0.0);
    std::vector<double> partial(static_cast<std::size_t>(nt), 0.0);
    PerAtomTeam team(atoms, static_cast<std::size_t>(N), nt);
    std::vector<PerAtomOut> slots(static_cast<std::size_t>(nt));
    for (int t = 0; t < nt; ++t) {
      slots[static_cast<std::size_t>(t)] = team.slot(t);
    }
    nlist::forEachSlice(nt, [&](int t) {
      const auto range = nlist::partition(N, t, nt);
      double *local =
          owned.data() + static_cast<std::size_t>(t) * static_cast<std::size_t>(N) * 3;
      double local_e = 0.0;
      nlist::visitDirected(
          neighbours_, range.begin, range.end,
          [&](std::int32_t i, std::int32_t j, double dx, double dy, double dz, double r2) {
            accept(i, j, dx, dy, dz, r2, local, local_e,
                   atoms != nullptr ? &slots[static_cast<std::size_t>(t)] : nullptr);
          });
      partial[static_cast<std::size_t>(t)] = local_e;
    });
    team.reduce();
    for (int t = 0; t < nt; ++t) {
      *U += partial[static_cast<std::size_t>(t)];
      const double *local =
          owned.data() + static_cast<std::size_t>(t) * static_cast<std::size_t>(N) * 3;
      for (long k = 0; k < 3 * N; ++k) {
        F[k] += local[k];
      }
    }
  }
}

} // namespace rgpot
