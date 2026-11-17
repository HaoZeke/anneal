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
#include "rgpot/LennardJones/LJPair.hpp"
#include "rgpot/PairEval.hpp"
#include "rgpot/nlist/NeighbourList.hpp"
#include "rgpot/nlist/PairSum.hpp"

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

  const double *box_use = freeBoxOrCell(in.box);

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

  const simd::LjParams params =
      makeLjParams(u0, psi, cuttOffU, m_config.switch_width, m_switch);
  const double skin = mode_ == nlist::Mode::Stream ? 0.0 : cutoff / 5.0;
  const std::array<bool, 3> open{{false, false, false}};
  nlist::updateCells(neighbours_, R, static_cast<std::size_t>(N), box_use, cutoff, skin,
                     nlist::Image::Round, mode_, open);
  retained_ = neighbours_.streaming() ? 0 : neighbours_.slots();
  const double cutoff2 = cutoff * cutoff;
  auto term = [&params](std::int32_t, std::int32_t, double r2, double &pair_e, double &fscale) {
    evalLjPair(r2, params, pair_e, fscale);
  };
  nlist::sumPairs(neighbours_, static_cast<std::size_t>(N), cutoff2, nlist::Overlap::Skip, term,
                  F, *U, nullptr, atoms);
}

} // namespace rgpot
