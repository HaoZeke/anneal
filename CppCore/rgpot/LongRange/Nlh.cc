// MIT License
// Copyright 2023--present rgpot developers

#include "rgpot/LongRange/Nlh.hpp"

#include "rgpot/ParamHash.hpp"
#include "rgpot/stress.hpp"
#include "rgpot/units.hpp"

#include <algorithm>
#include <cmath>
#include <stdexcept>

namespace rgpot::lr {
namespace {

constexpr double kKe = units::HARTREE_TO_EV * units::BOHR_TO_ANGSTROM;

#include "rgpot/LongRange/NlhTable.inc"

double sigma(double x) {
  return x > 0.0 ? std::exp(-1.0 / x) : 0.0;
}

double sigma_prime(double x) {
  return x > 0.0 ? std::exp(-1.0 / x) / (x * x) : 0.0;
}

void switch_w(double r, double off, double *w, double *dw) {
  if (!(r < off)) {
    *w = 0.0;
    *dw = 0.0;
    return;
  }
  const double c = r / off;
  const double s1 = sigma(1.0 - c);
  const double s2 = sigma(c);
  const double den = s1 + s2;
  *w = s1 / den;
  const double ds1 = -sigma_prime(1.0 - c);
  const double ds2 = sigma_prime(c);
  const double dfdc = (ds1 * den - s1 * (ds1 + ds2)) / (den * den);
  *dw = dfdc / off;
}

} // namespace

NlhPot::NlhPot(NlhConfig cfg)
    : Potential(PotType::LrNlh), cfg_(cfg),
      lists_(std::make_shared<DualList>(cfg.cutoff, cfg.cutoff, cfg.skin)) {
  if (!(cfg_.cutoff > cfg_.switch_off) || !(cfg_.switch_off > 0.0)) {
    throw std::invalid_argument("NLH cutoff must sit past the switch");
  }
  Fnv1a fp;
  fp.u64(1);
  fp.f64(cfg_.cutoff);
  fp.f64(cfg_.switch_off);
  fp.f64(cfg_.skin);
  params_ = fp.h;
}

void NlhPot::forceImpl(const ForceInput &in, ForceOut *out) const {
  const auto n = in.nAtoms;
  std::fill(out->F, out->F + 3 * n, 0.0);
  out->energy = 0.0;
  out->variance = 0.0;
  out->has_stress = 0;
  if (n < 2) {
    return;
  }
  if (in.atmnrs == nullptr) {
    throw std::invalid_argument("NLH needs atomic numbers");
  }
  lists_->update(in.pos, n, in.box, false);
  double strain[6] = {};
  for (const Pair &pr : lists_->short_pairs()) {
    if (!(pr.r > 0.0) || pr.r >= cfg_.cutoff) {
      continue;
    }
    int zi = in.atmnrs[pr.i];
    int zj = in.atmnrs[pr.j];
    if (zi < 1 || zj < 1 || zi > kNlhZMax || zj > kNlhZMax) {
      throw std::invalid_argument("NLH atomic number is outside H..U");
    }
    if (zi > zj) {
      std::swap(zi, zj);
    }
    const int base = ((zi - 1) * kNlhZMax + (zj - 1)) * 3;
    const double zprod = static_cast<double>(in.atmnrs[pr.i]) * static_cast<double>(in.atmnrs[pr.j]);
    double y = 0.0;
    double yp = 0.0;
    for (int k = 0; k < 3; ++k) {
      const double a = kNlhA[base + k];
      const double b = kNlhB[base + k];
      const double e = a * std::exp(-b * pr.r);
      y += e;
      yp += -b * e;
    }
    double w = 0.0;
    double dw = 0.0;
    switch_w(pr.r, cfg_.switch_off, &w, &dw);
    const double x = kKe * zprod / pr.r;
    const double xp = -kKe * zprod / (pr.r * pr.r);
    const double e = w * x * y;
    const double dedr = dw * x * y + w * xp * y + w * x * yp;
    out->energy += e;
    const double invr = 1.0 / pr.r;
    const double fx = dedr * pr.x * invr;
    out->F[3 * pr.i] += fx;
    out->F[3 * pr.i + 1] += dedr * pr.y * invr;
    out->F[3 * pr.i + 2] += dedr * pr.z * invr;
    out->F[3 * pr.j] -= fx;
    out->F[3 * pr.j + 1] -= dedr * pr.y * invr;
    out->F[3 * pr.j + 2] -= dedr * pr.z * invr;
    accumulatePairStrain(strain, -dedr * invr, -pr.x, -pr.y, -pr.z);
  }
  const double volume = cellVolume(in.box);
  if (volume > 0.0) {
    publishCauchyStress(out, strain, volume);
  }
}

} // namespace rgpot::lr
