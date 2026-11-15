// MIT License
// Copyright 2023--present rgpot developers

/**
 * @brief Implementation of the ZBL potential methods.
 *
 * Ported from eOn (https://github.com/TheochemUI/eOn,
 * client/potentials/ZBL), BSD-3-Clause licensed, copyright the eOn
 * Development Team, which adapts the GPL-licensed LAMMPS ``pair_zbl``
 * kernel. Constants follow LAMMPS ``metal`` units: eV, Angstrom.
 */

// clang-format off
#include <algorithm>
#include <array>
#include <cmath>
#include <iterator>
#include <stdexcept>
#include <vector>
// clang-format on

#include "rgpot/ZBL/ZBLPot.hpp"
#include "rgpot/nlist/NeighbourList.hpp"
#include "rgpot/stress.hpp"

#if defined(_OPENMP)
#include <omp.h>
#endif

namespace rgpot {

namespace {

// Universal ZBL screening function coefficients (pair_zbl_const.h).
constexpr double kC1 = 0.02817;
constexpr double kC2 = 0.28022;
constexpr double kC3 = 0.50986;
constexpr double kC4 = 0.18175;

// Screening exponents as carried by the LAMMPS source, which differ in the
// last digits from the published values.
constexpr double kD1 = 3.19980;
constexpr double kD2 = 0.94229;
constexpr double kD3 = 0.40290;
constexpr double kD4 = 0.20162;

constexpr double kPZBL = 0.23;
/// Screening length prefactor (Angstrom).
constexpr double kA0 = 0.46850;
/// Coulomb conversion in LAMMPS 'metal' units (eV Angstrom).
constexpr double kQQR2E = 14.399645;

/// Unswitched screened Coulomb energy.
double e_zbl(const ZblPairCoeffs &p, double r) {
  const double r_inv = 1.0 / r;
  const double sum = kC4 * std::exp(-p.d1 * r) + kC3 * std::exp(-p.d2 * r) +
                     kC2 * std::exp(-p.d3 * r) + kC1 * std::exp(-p.d4 * r);
  return p.zze * sum * r_inv;
}

/// First derivative of the unswitched energy with respect to r.
double dzbldr(const ZblPairCoeffs &p, double r) {
  const double r_inv = 1.0 / r;
  const double e1 = std::exp(-p.d1 * r);
  const double e2 = std::exp(-p.d2 * r);
  const double e3 = std::exp(-p.d3 * r);
  const double e4 = std::exp(-p.d4 * r);

  const double sum = kC4 * e1 + kC3 * e2 + kC2 * e3 + kC1 * e4;
  const double sum_p =
      -kC4 * p.d1 * e1 - kC3 * p.d2 * e2 - kC2 * p.d3 * e3 - kC1 * p.d4 * e4;

  return p.zze * (sum_p - sum * r_inv) * r_inv;
}

/// Second derivative of the unswitched energy with respect to r.
double d2zbldr2(const ZblPairCoeffs &p, double r) {
  const double r_inv = 1.0 / r;
  const double e1 = std::exp(-p.d1 * r);
  const double e2 = std::exp(-p.d2 * r);
  const double e3 = std::exp(-p.d3 * r);
  const double e4 = std::exp(-p.d4 * r);

  const double sum = kC4 * e1 + kC3 * e2 + kC2 * e3 + kC1 * e4;
  const double sum_p =
      kC4 * e1 * p.d1 + kC3 * e2 * p.d2 + kC2 * e3 * p.d3 + kC1 * e4 * p.d4;
  const double sum_pp = kC4 * e1 * p.d1 * p.d1 + kC3 * e2 * p.d2 * p.d2 +
                        kC2 * e3 * p.d3 * p.d3 + kC1 * e4 * p.d4 * p.d4;

  return p.zze * (sum_pp + 2.0 * sum_p * r_inv + 2.0 * sum * r_inv * r_inv) *
         r_inv;
}

} // namespace

ZBLPot::ZBLPot(const ZBLConfig &c)
    : Potential(PotType::ZBL), cut_inner{c.cut_inner}, cut_global{c.cut_global},
      cut_inner_sq{c.cut_inner * c.cut_inner}, m_config{c} {
  if (!(cut_inner > 0.0) || !(cut_inner < cut_global)) {
    throw std::invalid_argument(
        "ZBLPot: invalid cutoffs, require 0.0 < cut_inner < cut_global.");
  }
  Fnv1a fp;
  fp.u64(kKernelVersion);
  fp.f64(c.cut_inner);
  fp.f64(c.cut_global);
  m_paramsKey = fp.h;
}

std::shared_ptr<const ZblTables>
ZBLPot::buildTables(std::vector<int> uniqueZ) const {
  auto tables = std::make_shared<ZblTables>();
  tables->z = std::move(uniqueZ);
  const std::size_t n = tables->z.size();
  tables->pairs.assign(n * n, ZblPairCoeffs{});

  const double tc = cut_global - cut_inner;
  for (std::size_t i = 0; i < n; ++i) {
    for (std::size_t j = i; j < n; ++j) {
      const double zi = static_cast<double>(tables->z[i]);
      const double zj = static_cast<double>(tables->z[j]);
      const double a_inv = (std::pow(zi, kPZBL) + std::pow(zj, kPZBL)) / kA0;

      ZblPairCoeffs p;
      p.d1 = kD1 * a_inv;
      p.d2 = kD2 * a_inv;
      p.d3 = kD3 * a_inv;
      p.d4 = kD4 * a_inv;
      p.zze = zi * zj * kQQR2E;

      // Switching polynomial: energy and force both vanish at cut_global.
      const double fc = e_zbl(p, cut_global);
      const double fcp = dzbldr(p, cut_global);
      const double fcpp = d2zbldr2(p, cut_global);
      const double swa = (-3.0 * fcp + tc * fcpp) / (tc * tc);
      const double swb = (2.0 * fcp - tc * fcpp) / (tc * tc * tc);
      const double swc = -fc + (tc / 2.0) * fcp - (tc * tc / 12.0) * fcpp;

      p.sw1 = swa;
      p.sw2 = swb;
      p.sw3 = swa / 3.0;
      p.sw4 = swb / 4.0;
      p.sw5 = swc;

      tables->pairs[i * n + j] = p;
      tables->pairs[j * n + i] = p;
    }
  }
  return tables;
}

std::shared_ptr<const ZblTables> ZBLPot::tablesFor(const int *atomicNrs,
                                                   long N) const {
  std::vector<int> uniqueZ(atomicNrs, atomicNrs + N);
  std::sort(uniqueZ.begin(), uniqueZ.end());
  uniqueZ.erase(std::unique(uniqueZ.begin(), uniqueZ.end()), uniqueZ.end());

  {
    std::lock_guard<std::mutex> lock(m_tablesMtx);
    if (m_tables && m_tables->z == uniqueZ) {
      return m_tables;
    }
  }

  auto fresh = buildTables(std::move(uniqueZ));
  std::lock_guard<std::mutex> lock(m_tablesMtx);
  m_tables = fresh;
  return fresh;
}

/**
 * @class ZBLPot
 * @details
 *
 * Every pair within ``cut_global`` contributes. Periodicity is off, so
 * the walk does not fold a separation into the cell. The pair term runs
 * during the shared cell walk. A stream copies no pair index. A stored
 * list keeps candidates out to the cutoff plus one fifth of the cutoff.
 */
void ZBLPot::forceImpl(const ForceInput &in, ForceOut *out) const {
  compute(in, out, nullptr);
}

void ZBLPot::forcePerAtom(const ForceInput &in, ForceOut *out, const PerAtomOut &atoms) const {
  clearPerAtom(atoms, in.nAtoms);
  compute(in, out, &atoms);
}

void ZBLPot::compute(const ForceInput &in, ForceOut *out, const PerAtomOut *atoms) const {
  const auto N = static_cast<long>(in.nAtoms);
  const double *R = in.pos;
  double *F = out->F;
  double *U = &out->energy;
  double strain[6] = {};
  const double volume = cellVolume(in.box);
  publishCauchyStress(out, strain, volume);
  *U = 0.0;
  std::fill(F, F + 3 * N, 0.0);
  if (N < 2) {
    return;
  }

  const auto tables = tablesFor(in.atmnrs, N);
  const std::size_t nTypes = tables->z.size();
  const ZblPairCoeffs *coeffs = tables->pairs.data();

  // Per-atom type index, so the pair loop indexes flat tables instead of
  // searching per pair.
  std::vector<int> typeIdx(static_cast<std::size_t>(N));
  for (long k = 0; k < N; ++k) {
    const auto it =
        std::lower_bound(tables->z.begin(), tables->z.end(), in.atmnrs[k]);
    typeIdx[static_cast<std::size_t>(k)] =
        static_cast<int>(std::distance(tables->z.begin(), it));
  }

  static constexpr double free_box[9] = {1e6, 0.0, 0.0, 0.0, 1e6,
                                         0.0, 0.0, 0.0, 1e6};
  const bool box_ok = in.box != nullptr && in.box[0] > 0.0 && in.box[4] > 0.0 &&
                      in.box[8] > 0.0;
  const double *box_use = box_ok ? in.box : free_box;
  // Auto stores below two million atoms. An explicit stream keeps the
  // pruned stencil in the shared cache.
  nlist::Mode mode = mode_;
  if (mode == nlist::Mode::Auto && N < 2000000) {
    mode = nlist::Mode::Stored;
  }
  const double skin = mode == nlist::Mode::Stream ? 0.0 : neighbourSkin(cut_global);
  const std::array<bool, 3> open{{false, false, false}};
  nlist::updateCells(neighbours_, R, static_cast<std::size_t>(N), box_use, cut_global, skin,
                     nlist::Image::Round, mode, open);
  const double rInner = cut_inner;
  const double rInnerSq = cut_inner_sq;
  const int *types = typeIdx.data();
  const double cutoff2 = cut_global * cut_global;
  const int nt = nlist::threadsRequested();
  auto accept = [&](std::int32_t i, std::int32_t j, double dx, double dy, double dz,
                    double r2, double *force, double &energy, double *virial,
                    const PerAtomOut *share) {
    if (!(i < j) || !(r2 > 0.0) || r2 > cutoff2) {
      return;
    }
    const auto ti = static_cast<std::size_t>(types[i]);
    const auto tj = static_cast<std::size_t>(types[j]);
    const ZblPairCoeffs &coeff = coeffs[ti * nTypes + tj];
    const double r = std::sqrt(r2);
    double energy_pair = e_zbl(coeff, r) + coeff.sw5;
    double dEdr = dzbldr(coeff, r);
    if (r2 > rInnerSq) {
      const double t = r - rInner;
      energy_pair += t * t * t * (coeff.sw3 + coeff.sw4 * t);
      dEdr += t * t * (coeff.sw1 + coeff.sw2 * t);
    }
    const double fscale = -dEdr / r;
    energy += energy_pair;
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
      tallyPair(*share, static_cast<std::size_t>(i), static_cast<std::size_t>(j), energy_pair,
                fscale, dx, dy, dz);
    }
  };
  double energy = 0.0;
  if (nt <= 1) {
    nlist::visitHalf(neighbours_, 0, N,
                     [&](std::int32_t i, std::int32_t j, double dx, double dy, double dz,
                         double r2) { accept(i, j, dx, dy, dz, r2, F, energy, strain, atoms); });
  } else {
    std::vector<double> owned(static_cast<std::size_t>(nt) * static_cast<std::size_t>(N) * 3,
                              0.0);
    std::vector<double> partial(static_cast<std::size_t>(nt), 0.0);
    std::vector<double> virial(static_cast<std::size_t>(nt) * 6, 0.0);
    PerAtomTeam team(atoms, static_cast<std::size_t>(N), nt);
    std::vector<PerAtomOut> slots(static_cast<std::size_t>(nt));
    for (int t = 0; t < nt; ++t) {
      slots[static_cast<std::size_t>(t)] = team.slot(t);
    }
    neighbours_.forCellTeam(
        true, nt, [&](int t, std::int32_t i, std::int32_t j, double dx, double dy, double dz,
                      double r2) {
          double *local =
              owned.data() + static_cast<std::size_t>(t) * static_cast<std::size_t>(N) * 3;
          double *st = virial.data() + static_cast<std::size_t>(t) * 6;
          accept(i, j, dx, dy, dz, r2, local, partial[static_cast<std::size_t>(t)], st,
                 atoms != nullptr ? &slots[static_cast<std::size_t>(t)] : nullptr);
        });
    team.reduce();
    for (int t = 0; t < nt; ++t) {
      energy += partial[static_cast<std::size_t>(t)];
      const double *local =
          owned.data() + static_cast<std::size_t>(t) * static_cast<std::size_t>(N) * 3;
      for (long k = 0; k < 3 * N; ++k) {
        F[k] += local[k];
      }
      for (int c = 0; c < 6; ++c) {
        strain[c] += virial[static_cast<std::size_t>(t) * 6 + static_cast<std::size_t>(c)];
      }
    }
  }
  *U = energy;
  publishCauchyStress(out, strain, volume);
}

} // namespace rgpot
