// MIT License
// Copyright 2023--present rgpot developers

/**
 * @file SWSi.cc
 * @author rgpot Developers
 * @date 2026-10-26
 * @brief Stillinger-Weber silicon kernel, 1985 parameters.
 */

#include "rgpot/StillingerWeber/SWSi.hpp"

#include "rgpot/CentreTerms.hpp"
#include "rgpot/ParamHash.hpp"
#include "rgpot/nlist/NeighbourList.hpp"
#include "rgpot/stress.hpp"

#include <algorithm>
#include <cmath>
#include <cstring>
#include <stdexcept>
#include <vector>

namespace rgpot {
namespace {

struct Bond {
  int j;
  double x, y, z, r, gexp, dgdr;
};

struct PairTerm {
  bool live;
  double f2;
  double dedr;
  double gexp;
  double dgdr;
};

PairTerm pairTerm(double r) {
  const double u = r / StillingerWeberSi::kSigma;
  const double gap = u - StillingerWeberSi::kCut;
  // The factor exp(1/(u-a)) underflows to zero before u reaches a.
  if (!(gap < 0.0) || gap > -1.0 / 700.0) {
    return {};
  }
  const double s = 1.0 / gap;
  const double inv_u = 1.0 / u;
  const double u2 = inv_u * inv_u;
  const double u_m4 = u2 * u2;
  const double g = StillingerWeberSi::kB * u_m4 - 1.0;
  const double es = std::exp(s);
  const double f2 = StillingerWeberSi::kA * g * es;
  const double dgdu =
      -StillingerWeberSi::kP * StillingerWeberSi::kB * u_m4 * inv_u;
  const double dfdu = StillingerWeberSi::kA * es * (dgdu - g * s * s);
  const double dedr =
      StillingerWeberSi::kEpsilon * dfdu / StillingerWeberSi::kSigma;
  const double gexp = std::exp(StillingerWeberSi::kGamma * s);
  const double dgdr =
      gexp * (-StillingerWeberSi::kGamma * s * s) / StillingerWeberSi::kSigma;
  return {true, f2, dedr, gexp, dgdr};
}

} // namespace

StillingerWeberSi::StillingerWeberSi() : Potential(PotType::SWSi1985) {
  Fnv1a fp;
  fp.u64(1);
  fp.f64(kEpsilon);
  fp.f64(kSigma);
  fp.f64(kA);
  fp.f64(kB);
  fp.f64(kP);
  fp.f64(kCut);
  fp.f64(kLambda);
  fp.f64(kGamma);
  m_paramsKey = fp.h;
}

void StillingerWeberSi::forceImpl(const ForceInput &in, ForceOut *out) const {
  if (out == nullptr) {
    throw std::invalid_argument("StillingerWeberSi: null ForceOut");
  }
  out->variance = 0.0;
  out->has_stress = 0;
  for (double &s : out->stress) {
    s = 0.0;
  }
  if (in.nAtoms > 0 &&
      (in.pos == nullptr || in.box == nullptr || out->F == nullptr)) {
    throw std::invalid_argument("StillingerWeberSi: null force buffer");
  }
  double strain[6] = {};
  evaluate(in.pos, in.box, in.nAtoms, &out->energy, out->F, 1.0, nullptr,
           strain);
  publishCauchyStress(out, strain, cellVolume(in.box));
}

void StillingerWeberSi::energyAndGradient(
    const double *positions, const int *atomic_numbers, const double *box,
    std::size_t n_atoms, double *energy, double *gradient,
    nlist::NeighborList *neighbors) const {
  (void)atomic_numbers;
  if (energy == nullptr ||
      (n_atoms > 0 &&
       (positions == nullptr || box == nullptr || gradient == nullptr))) {
    throw std::invalid_argument("StillingerWeberSi: null energy buffer");
  }
  // Gradient is dE/dx, the negation of the force written by forceImpl.
  evaluate(positions, box, n_atoms, energy, gradient, -1.0, neighbors, nullptr);
}

void StillingerWeberSi::energyAndGradientBatch(
    std::size_t n_config, const double *positions, const int *atomic_numbers,
    const double *box, std::size_t n_atoms, double *energies,
    double *gradients) const {
  if (n_config == 0) {
    return;
  }
  if (energies == nullptr ||
      (n_atoms > 0 &&
       (positions == nullptr || box == nullptr || gradients == nullptr))) {
    throw std::invalid_argument("StillingerWeberSi: null batch buffer");
  }
  const std::size_t stride = n_atoms * 3;
  for (std::size_t c = 0; c < n_config; ++c) {
    energyAndGradient(positions + c * stride, atomic_numbers, box, n_atoms,
                      energies + c, gradients + c * stride);
  }
}

void StillingerWeberSi::evaluate(const double *positions, const double *box,
                                 std::size_t n_atoms, double *energy,
                                 double *acc, double scale,
                                 nlist::NeighborList *neighbors,
                                 double *strain) const {
  *energy = 0.0;
  const std::size_t n3 = n_atoms * 3;
  for (std::size_t i = 0; i < n3; ++i) {
    acc[i] = 0.0;
  }
  if (n_atoms < 2) {
    return;
  }

  const double cut = kCut * kSigma;
  const double cut2 = cut * cut;
  const int n = static_cast<int>(n_atoms);

  const double skin = (neighbors != nullptr && neighbors->skin > 0.0)
                          ? neighbors->skin
                          : (mode_ == nlist::Mode::Stream ? 0.0 : cut / 5.0);
  const auto before = neighbours_.builds();
  nlist::updateCells(neighbours_, positions, n_atoms, box, cut, skin, nlist::Image::Round,
                     mode_, {true, true, true});
  if (neighbors != nullptr && neighbors->skin > 0.0 &&
      (neighbours_.builds() != before || !neighbors->built)) {
    neighbors->ref.assign(positions, positions + 3 * n_atoms);
    std::memcpy(neighbors->box, box, 9 * sizeof(double));
    neighbors->n = n_atoms;
    neighbors->cutoff = cut;
    neighbors->skin_used = neighbors->skin;
    neighbors->built = true;
  }

  // One atom of bonds. A stream reads the bins directly, so the scratch
  // is that atom and not a copy of every bond. A stored call reads the
  // index row, which is already paid for.
  auto load = [&](int i, std::vector<Bond> &nb) {
    nb.clear();
    auto take = [&](std::int32_t, std::int32_t j, double dx, double dy, double dz, double r2) {
      if (!std::isfinite(dx) || !std::isfinite(dy) || !std::isfinite(dz)) {
        throw std::invalid_argument("StillingerWeberSi: non-finite position");
      }
      if (!(r2 > 0.0) || r2 >= cut2) {
        return;
      }
      const double r = std::sqrt(r2);
      const PairTerm t = pairTerm(r);
      if (!t.live) {
        return;
      }
      nb.push_back(Bond{static_cast<int>(j), -dx, -dy, -dz, r, t.gexp, t.dgdr});
    };
    if (neighbours_.streaming()) {
      neighbours_.forAtom(static_cast<std::int32_t>(i), take);
    } else {
      nlist::visitDirected(neighbours_, i, static_cast<std::int64_t>(i) + 1, take);
    }
    std::sort(nb.begin(), nb.end(), [](const Bond &a, const Bond &b) { return a.j < b.j; });
  };
  // The terms of centre i, into a CentreDirect or a CentreTerms.
  auto centre = [&](int i, const std::vector<Bond> &nb, auto &sink) {
    const int m = static_cast<int>(nb.size());
    for (int a = 0; a < m; ++a) {
      const Bond &bj = nb[static_cast<std::size_t>(a)];
      if (bj.j <= i) {
        continue;
      }
      const PairTerm t = pairTerm(bj.r);
      if (!t.live) {
        continue;
      }
      sink.energy(kEpsilon * t.f2);
      const double inv_r = 1.0 / bj.r;
      const double sx = scale * t.dedr * bj.x * inv_r;
      const double sy = scale * t.dedr * bj.y * inv_r;
      const double sz = scale * t.dedr * bj.z * inv_r;
      sink.pair(static_cast<std::size_t>(a), bj.j, sx, sy, sz);
      if (sink.strained()) {
        // dE/d(bj) is (dedr / r) bj. bj is the negative of the d in
        // stress.hpp when the walk stores r_i - r_j. The term is
        // quadratic in that vector, so the sign does not change it.
        sink.pairStrain(-t.dedr * inv_r, bj.x, bj.y, bj.z);
      }
    }
    for (int a = 0; a < m; ++a) {
      const Bond &bj = nb[static_cast<std::size_t>(a)];
      for (int b = a + 1; b < m; ++b) {
        const Bond &bk = nb[static_cast<std::size_t>(b)];
        const double inv_r1 = 1.0 / bj.r;
        const double inv_r2 = 1.0 / bk.r;
        const double inv = inv_r1 * inv_r2;
        double cth = (bj.x * bk.x + bj.y * bk.y + bj.z * bk.z) * inv;
        if (cth > 1.0) {
          cth = 1.0;
        } else if (cth < -1.0) {
          cth = -1.0;
        }
        const double ang = cth + (1.0 / 3.0);
        const double ang2 = ang * ang;
        sink.energy(kEpsilon * kLambda * bj.gexp * bk.gexp * ang2);
        const double pref = kEpsilon * kLambda;
        const double d_edcth = pref * bj.gexp * bk.gexp * 2.0 * ang;
        const double d_edr1 = pref * bj.dgdr * bk.gexp * ang2;
        const double d_edr2 = pref * bj.gexp * bk.dgdr * ang2;
        const double g1x =
            d_edr1 * bj.x * inv_r1 +
            d_edcth * (bk.x * inv - cth * bj.x * inv_r1 * inv_r1);
        const double g1y =
            d_edr1 * bj.y * inv_r1 +
            d_edcth * (bk.y * inv - cth * bj.y * inv_r1 * inv_r1);
        const double g1z =
            d_edr1 * bj.z * inv_r1 +
            d_edcth * (bk.z * inv - cth * bj.z * inv_r1 * inv_r1);
        const double g2x =
            d_edr2 * bk.x * inv_r2 +
            d_edcth * (bj.x * inv - cth * bk.x * inv_r2 * inv_r2);
        const double g2y =
            d_edr2 * bk.y * inv_r2 +
            d_edcth * (bj.y * inv - cth * bk.y * inv_r2 * inv_r2);
        const double g2z =
            d_edr2 * bk.z * inv_r2 +
            d_edcth * (bj.z * inv - cth * bk.z * inv_r2 * inv_r2);
        // g is dE/d(neighbor). Force on that atom is -g.
        sink.triplet(static_cast<std::size_t>(a), bj.j, static_cast<std::size_t>(b), bk.j, scale,
                     g1x, g1y, g1z, g2x, g2y, g2z);
        if (sink.strained()) {
          sink.bondStrain(bj.x, bj.y, bj.z, g1x, g1y, g1z);
          sink.bondStrain(bk.x, bk.y, bk.z, g2x, g2y, g2z);
        }
      }
    }
  };

  if (n_atoms < kTeamMinAtoms) {
    std::vector<Bond> nb;
    for (int i = 0; i < n; ++i) {
      load(i, nb);
      CentreDirect sink(i, acc, energy, strain);
      centre(i, nb, sink);
    }
    return;
  }
  sumCentres<std::vector<Bond>>(
      n_atoms,
      [&](std::int32_t i, CentreTerms &terms, std::vector<Bond> &nb) {
        load(i, nb);
        terms.reset(i, nb);
        centre(i, nb, terms);
      },
      acc, *energy, strain);
}

} // namespace rgpot
