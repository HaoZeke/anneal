// MIT License
// Copyright 2023--present rgpot developers

#include "rgpot/LongRange/Dispersion.hpp"
#include "rgpot/LongRange/Electro.hpp"
#include "rgpot/LongRange/Lattice.hpp"
#include "rgpot/LongRange/Nlh.hpp"
#include "rgpot/LongRange/SumPot.hpp"
#include "rgpot/units.hpp"

#include <algorithm>
#include <cmath>
#include <iomanip>
#include <iostream>
#include <limits>
#include <numeric>
#include <random>
#include <string>
#include <vector>

#ifdef _OPENMP
#include <omp.h>
#endif

using rgpot::ForceInput;
using rgpot::ForceOut;
using rgpot::lr::CellGeom;
using rgpot::lr::CoulombConfig;
using rgpot::lr::CoulombMethod;
using rgpot::lr::CoulombPot;
using rgpot::lr::DispersionConfig;
using rgpot::lr::DispersionKind;
using rgpot::lr::DispersionPot;
using rgpot::lr::DualList;
using rgpot::lr::FixedCharges;
using rgpot::lr::FixedRatios;
using rgpot::lr::NlhPot;
using rgpot::lr::SumPot;
using rgpot::lr::cell_from_box;
using rgpot::lr::so3lr_erf_pair;

namespace {

int g_fails = 0;

void check(bool ok, const char *what) {
  if (!ok) {
    std::cerr << "FAIL " << what << '\n';
    ++g_fails;
  }
}

void check_close(double got, double ref, double tol, const char *what) {
  const double err = std::abs(got - ref);
  const double scale = std::max(1.0, std::abs(ref));
  if (!(err <= tol * scale) || !std::isfinite(got)) {
    std::cerr << "FAIL " << what << " got " << got << " ref " << ref << " err " << err << '\n';
    ++g_fails;
  }
}

struct Eval {
  double energy = 0;
  std::vector<double> f;
  double stress[9]{};
  bool has_stress = false;
};

template <class Pot>
Eval run(Pot &pot, const std::vector<double> &pos, const std::vector<int> &z, const double *box) {
  Eval e;
  e.f.assign(pos.size(), 0.0);
  ForceOut out{};
  out.F = e.f.data();
  ForceInput in{pos.size() / 3, pos.data(), z.data(), box};
  pot.forceImpl(in, &out);
  e.energy = out.energy;
  e.has_stress = out.has_stress != 0;
  std::copy(out.stress, out.stress + 9, e.stress);
  return e;
}

double fd_force(auto &&energy, std::vector<double> pos, std::size_t i, double step) {
  const double x = pos[i];
  pos[i] = x + step;
  const double ep = energy(pos);
  pos[i] = x - step;
  const double em = energy(pos);
  return -(ep - em) / (2.0 * step);
}

void test_lists() {
  DualList lists(3.0, 6.0, 0.4);
  std::mt19937_64 rng(1);
  std::uniform_real_distribution<double> u(0.2, 9.8);
  std::vector<double> pos;
  for (int i = 0; i < 40; ++i) {
    pos.push_back(u(rng));
    pos.push_back(u(rng));
    pos.push_back(u(rng));
  }
  const double box[9] = {10, 0.4, 0.1, 0.2, 11, 0.3, 0.0, -0.2, 9};
  lists.update(pos.data(), 40, box, true);
  check(lists.short_rebuilt() && lists.long_rebuilt(), "first list build rebuilds");
  check(lists.short_pairs().size() < lists.long_pairs().size(), "long list is the larger cutoff");
  auto brute = [&](double cut) {
    std::size_t n = 0;
    const CellGeom g = cell_from_box(box);
    for (int i = 0; i < 40; ++i) {
      for (int j = i; j < 40; ++j) {
        for (int a = -2; a <= 2; ++a) {
          for (int b = -2; b <= 2; ++b) {
            for (int c = -2; c <= 2; ++c) {
              if (i == j && a == 0 && b == 0 && c == 0) {
                continue;
              }
              double dx = pos[3 * j] - pos[3 * i];
              double dy = pos[3 * j + 1] - pos[3 * i + 1];
              double dz = pos[3 * j + 2] - pos[3 * i + 2];
              dx += a * g.H[0][0] + b * g.H[1][0] + c * g.H[2][0];
              dy += a * g.H[0][1] + b * g.H[1][1] + c * g.H[2][1];
              dz += a * g.H[0][2] + b * g.H[1][2] + c * g.H[2][2];
              const double r2 = dx * dx + dy * dy + dz * dz;
              if (r2 < cut * cut && r2 > 0.0) {
                const bool keep = i < j || (i == j && (a > 0 || (a == 0 && (b > 0 || (b == 0 && c > 0)))));
                if (keep || i != j) {
                  if (i < j || (i == j && a >= 0 && (a > 0 || b > 0 || (b == 0 && c > 0)))) {
                    ++n;
                  }
                }
              }
            }
          }
        }
      }
    }
    return n;
  };
  // Compare the long list to a minimum-image brute force at the same cutoff.
  std::size_t brute_n = 0;
  for (int i = 0; i < 40; ++i) {
    for (int j = i + 1; j < 40; ++j) {
      double dx = pos[3 * j] - pos[3 * i];
      double dy = pos[3 * j + 1] - pos[3 * i + 1];
      double dz = pos[3 * j + 2] - pos[3 * i + 2];
      const CellGeom g = cell_from_box(box);
      double s[3];
      double r[3] = {dx, dy, dz};
      g.fractional(r, s);
      for (int c = 0; c < 3; ++c) {
        s[c] -= std::round(s[c]);
      }
      double w[3] = {};
      for (int c = 0; c < 3; ++c) {
        w[c] = s[0] * g.H[0][c] + s[1] * g.H[1][c] + s[2] * g.H[2][c];
      }
      if (w[0] * w[0] + w[1] * w[1] + w[2] * w[2] < 36.0) {
        ++brute_n;
      }
    }
  }
  (void)brute;
  check(lists.long_pairs().size() >= brute_n, "long list covers the minimum image");
  for (double &x : pos) {
    x += 0.01;
  }
  lists.update(pos.data(), 40, box, true);
  check(!lists.long_rebuilt(), "a small move stays inside the skin");
  pos[0] += 2.0;
  lists.update(pos.data(), 40, box, true);
  check(lists.long_rebuilt(), "a large move rebuilds the skin");
}

void test_madelung() {
  const double ke = rgpot::units::HARTREE_TO_EV * rgpot::units::BOHR_TO_ANGSTROM;
  const double a = 2.0;
  const double box[9] = {a, 0, 0, 0, a, 0, 0, 0, a};
  std::vector<double> pos;
  std::vector<int> z(8, 11);
  std::vector<double> q;
  const double na[4][3] = {{0, 0, 0}, {0.5, 0.5, 0}, {0.5, 0, 0.5}, {0, 0.5, 0.5}};
  const double cl[4][3] = {{0.5, 0, 0}, {0, 0.5, 0}, {0, 0, 0.5}, {0.5, 0.5, 0.5}};
  for (const auto &p : na) {
    pos.insert(pos.end(), {p[0] * a, p[1] * a, p[2] * a});
    q.push_back(1.0);
  }
  for (const auto &p : cl) {
    pos.insert(pos.end(), {p[0] * a, p[1] * a, p[2] * a});
    q.push_back(-1.0);
    z[q.size() - 1] = 17;
  }
  CoulombConfig cfg;
  cfg.method = CoulombMethod::Ewald;
  cfg.cutoff = 8.0;
  cfg.alpha = 2.5;
  cfg.kmax = 12;
  cfg.short_cutoff = 3.0;
  cfg.skin = 0.2;
  cfg.periodic = true;
  CoulombPot pot(cfg, std::make_shared<FixedCharges>(q, 0.0));
  const Eval e = run(pot, pos, z, box);
  std::cout << "CASE nacl " << e.energy << '\n';
  const double M = 1.7475645946331821;
  const double expect = -4.0 * M * ke / 1.0;
  check_close(e.energy, expect, 2e-6, "NaCl Madelung");

  std::vector<double> cs = {0, 0, 0, 0.5 * a, 0.5 * a, 0.5 * a};
  std::vector<int> z2 = {55, 17};
  CoulombPot cscl(cfg, std::make_shared<FixedCharges>(std::vector<double>{1.0, -1.0}, 0.0));
  const Eval e2 = run(cscl, cs, z2, box);
  const double Mcs = 1.762674773070988;
  const double rnn = a * std::sqrt(3.0) / 2.0;
  check_close(e2.energy, -Mcs * ke / rnn, 2e-5, "CsCl Madelung");
}

void test_forces_and_stress() {
  const double box[9] = {6, 0.3, 0.1, -0.2, 5.5, 0.2, 0.1, -0.4, 6.2};
  std::vector<double> pos = {0.4, 0.5, 0.6, 2.2, 1.1, 0.8, 1.0, 3.3, 2.4, 4.2, 2.2, 3.5};
  std::vector<int> z = {1, 1, 8, 8};
  std::vector<double> q = {0.4, -0.2, 0.5, -0.7};
  CoulombConfig cfg;
  cfg.method = CoulombMethod::Ewald;
  cfg.cutoff = 5.0;
  cfg.alpha = 0.45;
  cfg.kmax = 8;
  cfg.short_cutoff = 2.5;
  cfg.periodic = true;
  auto pot = std::make_shared<CoulombPot>(cfg, std::make_shared<FixedCharges>(q, 0.0));
  const Eval e0 = run(*pot, pos, z, box);
  for (std::size_t i = 0; i < pos.size(); ++i) {
    const double f = fd_force(
        [&](const std::vector<double> &p) { return run(*pot, p, z, box).energy; }, pos, i, 1e-5);
    check_close(e0.f[i], f, 2e-5, "Ewald force");
  }
  check(e0.has_stress, "Ewald stress is set");
  const double eps = 1e-6;
  auto strain_xx = [&](double eps_xx) {
    std::vector<double> moved = pos;
    double box2[9];
    std::copy(box, box + 9, box2);
    for (int row = 0; row < 3; ++row) {
      box2[3 * row] *= (1.0 + eps_xx);
    }
    for (std::size_t i = 0; i < moved.size(); i += 3) {
      moved[i] *= (1.0 + eps_xx);
    }
    return run(*pot, moved, z, box2).energy;
  };
  const double dE = (strain_xx(eps) - strain_xx(-eps)) / (2.0 * eps);
  const double volume = std::abs(cell_from_box(box).volume);
  check_close(e0.stress[0], dE / volume, 5e-4, "Ewald xx stress");

  cfg.method = CoulombMethod::SPME;
  cfg.grid[0] = cfg.grid[1] = cfg.grid[2] = 128;
  cfg.spline_order = 6;
  CoulombPot spme(cfg, std::make_shared<FixedCharges>(q, 0.0));
  const Eval es = run(spme, pos, z, box);
  std::cout << "CASE ewald_cluster " << e0.energy << '\n';
  std::cout << "CASE spme_cluster " << es.energy << '\n';
  check_close(es.energy, e0.energy, 2e-3, "SPME matches Ewald");
  for (std::size_t i = 0; i < pos.size(); ++i) {
    const double f = fd_force(
        [&](const std::vector<double> &p) { return run(spme, p, z, box).energy; }, pos, i, 1e-5);
    check_close(es.f[i], f, 5e-4, "SPME force");
  }
}

void test_dsf() {
  std::vector<double> pos = {0, 0, 0, 1.2, 0.2, -0.1, 2.5, 1.0, 0.4};
  std::vector<int> z = {1, 1, 8};
  std::vector<double> q = {0.3, -0.4, 0.1};
  CoulombConfig cfg;
  cfg.method = CoulombMethod::DSF;
  cfg.periodic = false;
  cfg.cutoff = 8.0;
  cfg.alpha = 0.25;
  cfg.short_cutoff = 3.0;
  CoulombPot pot(cfg, std::make_shared<FixedCharges>(q, 0.0));
  const Eval e0 = run(pot, pos, z, nullptr);
  for (std::size_t i = 0; i < pos.size(); ++i) {
    const double f = fd_force(
        [&](const std::vector<double> &p) { return run(pot, p, z, nullptr).energy; }, pos, i, 1e-6);
    check_close(e0.f[i], f, 2e-6, "DSF force");
  }
  double ene = 0, dedr = 0;
  so3lr_erf_pair(0.2, -0.3, 1.4, 2.0, 6.0, 9.0, 14.399645351950548, &ene, &dedr);
  check(std::isfinite(ene) && dedr != 0.0, "SO3LR erf pair is finite");
  const double step = 1e-6;
  double ep = 0, em = 0, dummy = 0;
  so3lr_erf_pair(0.2, -0.3, 1.4 + step, 2.0, 6.0, 9.0, 14.399645351950548, &ep, &dummy);
  so3lr_erf_pair(0.2, -0.3, 1.4 - step, 2.0, 6.0, 9.0, 14.399645351950548, &em, &dummy);
  check_close(dedr, (ep - em) / (2.0 * step), 1e-6, "SO3LR erf derivative");
}

void test_qdo_nlh_mbd() {
  std::vector<double> pos = {0, 0, 0, 1.4, 0.1, 0.0, 0.2, 1.6, 0.3};
  std::vector<int> z = {6, 1, 8};
  DispersionConfig cfg;
  cfg.kind = DispersionKind::QDO;
  cfg.cutoff = 12;
  cfg.short_cutoff = 3;
  cfg.periodic = false;
  cfg.qdo_damped = true;
  DispersionPot qdo(cfg, std::make_shared<FixedRatios>(std::vector<double>{1.0, 0.8, 1.1},
                                                           std::vector<double>{}));
  const Eval e0 = run(qdo, pos, z, nullptr);
  std::cout << "CASE qdo_cho " << e0.energy << '\n';
  check(e0.energy < 0.0, "QDO dispersion is attractive");
  for (std::size_t i = 0; i < pos.size(); ++i) {
    const double f = fd_force(
        [&](const std::vector<double> &p) { return run(qdo, p, z, nullptr).energy; }, pos, i, 1e-6);
    check_close(e0.f[i], f, 2e-5, "QDO force");
  }
  DispersionPot bare(cfg, std::make_shared<FixedRatios>(std::vector<double>{1, 1, 1},
                                                            std::vector<double>{1, 1, 1}));
  check(std::abs(run(bare, pos, z, nullptr).energy - e0.energy) > 1e-6, "C6 ratios change the energy");

  NlhPot nlh;
  const Eval en = run(nlh, pos, z, nullptr);
  check(en.energy > 0.0, "NLH is repulsive");
  for (std::size_t i = 0; i < 3; ++i) {
    const double f = fd_force(
        [&](const std::vector<double> &p) { return run(nlh, p, z, nullptr).energy; }, pos, i, 1e-6);
    check_close(en.f[i], f, 2e-5, "NLH force");
  }

  DispersionConfig mcfg;
  mcfg.kind = DispersionKind::MBD;
  mcfg.periodic = false;
  mcfg.cutoff = 20;
  mcfg.short_cutoff = 4;
  mcfg.mbd_beta = 0.83;
  DispersionPot mbd(mcfg);
  const Eval em = run(mbd, pos, z, nullptr);
  std::cout << "CASE mbd_cho " << em.energy << '\n';
  check(std::isfinite(em.energy), "MBD energy is finite");
  for (std::size_t i = 0; i < 3; ++i) {
    const double f = fd_force(
        [&](const std::vector<double> &p) { return run(mbd, p, z, nullptr).energy; }, pos, i, 2e-4);
    check_close(em.f[i], f, 5e-3, "MBD force");
  }
}

class Bias final : public rgpot::Potential<Bias> {
public:
  Bias() : rgpot::Potential<Bias>(rgpot::PotType::UNKNOWN) {}
  void forceImpl(const ForceInput &in, ForceOut *out) const override {
    out->energy = 0.5;
    for (std::size_t i = 0; i < in.nAtoms; ++i) {
      out->F[3 * i] += 0.2;
    }
    out->has_stress = 0;
    out->variance = 0.1;
  }
};

void test_library_dispersion() {
#if defined(RGPOT_HAS_DFTD3)
  {
  std::vector<double> pos = {0, 0, 0, 0.96, 0, 0, -0.24, 0.93, 0};
  std::vector<int> z = {8, 1, 1};
  rgpot::lr::DispersionConfig cfg;
  cfg.kind = rgpot::lr::DispersionKind::D3BJ;
  cfg.functional = "pbe";
  cfg.atm = true;
  cfg.periodic = false;
  cfg.cutoff = 20;
  cfg.short_cutoff = 4;
  rgpot::lr::DispersionPot d3(cfg);
  const Eval e = run(d3, pos, z, nullptr);
  std::cout << "CASE d3_water " << e.energy;
  for (double f : e.f) {
    std::cout << ' ' << f;
  }
  std::cout << '\n';
  check(e.energy < 0.0, "D3 energy is negative");
  for (std::size_t i = 0; i < 3; ++i) {
    const double f = fd_force(
        [&](const std::vector<double> &p) { return run(d3, p, z, nullptr).energy; }, pos, i, 1e-6);
    check_close(e.f[i], f, 2e-5, "D3 force");
  }
  }
#else
  std::cout << "CASE d3_water skipped\n";
#endif
#if defined(RGPOT_HAS_DFTD4)
  {
  std::vector<double> pos = {0, 0, 0, 0.96, 0, 0, -0.24, 0.93, 0};
  std::vector<int> z = {8, 1, 1};
  rgpot::lr::DispersionConfig cfg;
  cfg.kind = rgpot::lr::DispersionKind::D4;
  cfg.functional = "pbe";
  cfg.atm = true;
  cfg.charge = 0.0;
  cfg.periodic = false;
  cfg.cutoff = 20;
  cfg.short_cutoff = 4;
  rgpot::lr::DispersionPot d4(cfg);
  const Eval e = run(d4, pos, z, nullptr);
  std::cout << "CASE d4_water " << e.energy;
  for (double f : e.f) {
    std::cout << ' ' << f;
  }
  std::cout << '\n';
  check(e.energy < 0.0, "D4 energy is negative");
  }
#else
  std::cout << "CASE d4_water skipped\n";
#endif
}

void test_sum_and_charge() {
  std::vector<double> pos = {0, 0, 0, 2, 0, 0};
  std::vector<int> z = {1, 1};
  CoulombConfig cfg;
  cfg.method = CoulombMethod::DSF;
  cfg.periodic = false;
  cfg.cutoff = 6;
  cfg.alpha = 0.2;
  cfg.short_cutoff = 3;
  auto coul = std::make_unique<CoulombPot>(cfg, std::make_shared<FixedCharges>(std::vector<double>{0.5, -0.5}, 0.0));
  const Eval alone = run(*coul, pos, z, nullptr);
  SumPot sum;
  sum.add(std::move(coul));
  sum.add(std::make_unique<Bias>());
  const Eval both = run(sum, pos, z, nullptr);
  check_close(both.energy, alone.energy + 0.5, 1e-12, "sum energy");
  check_close(both.f[0], alone.f[0] + 0.2, 1e-12, "sum force");
  check_close(both.f[1], alone.f[1], 1e-12, "sum leaves the other component");
  bool threw = false;
  try {
    CoulombPot bad(cfg, std::make_shared<FixedCharges>(std::vector<double>{0.5, -0.2}, 0.0));
    run(bad, pos, z, nullptr);
  } catch (const std::invalid_argument &) {
    threw = true;
  }
  check(threw, "total charge is checked");
}

void test_nve() {
  const double a = 5.64;
  const double box[9] = {a, 0, 0, 0, a, 0, 0, 0, a};
  std::vector<double> pos;
  std::vector<double> q;
  std::vector<int> z;
  const double basis[4][3] = {{0, 0, 0}, {0.5, 0.5, 0}, {0.5, 0, 0.5}, {0, 0.5, 0.5}};
  for (const auto &p : basis) {
    pos.insert(pos.end(), {p[0] * a, p[1] * a, p[2] * a});
    q.push_back(1);
    z.push_back(11);
  }
  for (const auto &p : basis) {
    pos.insert(pos.end(), {(p[0] + 0.5) * a, p[1] * a, p[2] * a});
    q.push_back(-1);
    z.push_back(17);
  }
  // The second loop is not CsCl/NaCl. Use the NaCl decoration instead.
  pos.clear();
  q.clear();
  z.clear();
  const double na[4][3] = {{0, 0, 0}, {0.5, 0.5, 0}, {0.5, 0, 0.5}, {0, 0.5, 0.5}};
  const double cl[4][3] = {{0.5, 0, 0}, {0, 0.5, 0}, {0, 0, 0.5}, {0.5, 0.5, 0.5}};
  for (const auto &p : na) {
    pos.insert(pos.end(), {p[0] * a, p[1] * a, p[2] * a});
    q.push_back(1);
    z.push_back(23);
  }
  for (const auto &p : cl) {
    pos.insert(pos.end(), {p[0] * a, p[1] * a, p[2] * a});
    q.push_back(-1);
    z.push_back(17);
  }
  CoulombConfig cfg;
  cfg.cutoff = 8;
  cfg.alpha = 0.35;
  cfg.kmax = 6;
  cfg.short_cutoff = 3;
  cfg.skin = 0.5;
  CoulombPot pot(cfg, std::make_shared<FixedCharges>(q, 0.0));
  std::vector<double> vel(pos.size(), 0.0);
  vel[0] = 0.002;
  vel[4] = -0.001;
  const double mass = 22.99;
  const double conv = 0.009648533212;
  const double dt = 1.0;
  auto energy = [&]() {
    const Eval e = run(pot, pos, z, box);
    double ke = 0.0;
    for (std::size_t i = 0; i < vel.size(); ++i) {
      ke += 0.5 * mass * vel[i] * vel[i] / conv;
    }
    return e.energy + ke;
  };
  // velocity unit: Angstrom/fs. ke = 0.5 m v^2 / conv, with conv = a per (F/m).
  double e1 = 0;
  for (int step = 0; step < 40; ++step) {
    const Eval e = run(pot, pos, z, box);
    if (step == 0) {
      double kin = 0;
      for (double v : vel) {
        kin += 0.5 * mass * v * v / conv;
      }
      e1 = e.energy + kin;
    }
    for (std::size_t i = 0; i < pos.size(); ++i) {
      vel[i] += 0.5 * dt * e.f[i] / mass * conv;
      pos[i] += dt * vel[i];
    }
    const Eval e2 = run(pot, pos, z, box);
    for (std::size_t i = 0; i < pos.size(); ++i) {
      vel[i] += 0.5 * dt * e2.f[i] / mass * conv;
    }
  }
  double kin = 0;
  for (double v : vel) {
    kin += 0.5 * mass * v * v / conv;
  }
  const Eval ef = run(pot, pos, z, box);
  check_close(ef.energy + kin, e1, 2e-4, "NVE energy");
  (void)energy;
}

struct Hook final : rgpot::lr::ChargeResponse {
  void accumulate(const ForceInput &in, const double *dE_dq, double *force) const override {
    for (std::size_t i = 0; i < in.nAtoms; ++i) {
      force[3 * i] += 0.01 * dE_dq[i];
    }
  }
};

struct Ratios final : rgpot::lr::RatioResponse {
  void jacobians(const ForceInput &, const double **dh, const double **dc6) const override {
    static double hder[9] = {0.05, 0, 0, 0, 0, 0, 0, 0, 0};
    static double cder[9] = {0, 0.02, 0, 0, 0, 0, 0, 0, 0};
    *dh = hder;
    *dc6 = cder;
  }
};

void fill_charges(const ForceInput &, double *q, double *total, void *) {
  q[0] = 0.25;
  q[1] = -0.25;
  *total = 0.0;
}

void test_edges() {
  bool threw = false;
  try {
    DualList bad(0.0, 1.0, 0.1);
  } catch (const std::invalid_argument &) {
    threw = true;
  }
  check(threw, "non-positive cutoff is rejected");
  threw = false;
  try {
    DualList bad(5.0, 1.0, 0.1);
  } catch (const std::invalid_argument &) {
    threw = true;
  }
  check(threw, "short cutoff cannot exceed the long cutoff");
  DualList empty(1.0, 2.0, 0.1);
  empty.update(nullptr, 1, nullptr, false);
  check(empty.long_pairs().empty(), "one atom has no pairs");
  threw = false;
  try {
    std::vector<double> pos = {0, 0, 0, 1, 0, 0};
    empty.update(nullptr, 2, nullptr, false);
  } catch (const std::invalid_argument &) {
    threw = true;
  }
  check(threw, "null positions are rejected");

  CoulombConfig cfg;
  cfg.method = CoulombMethod::DSF;
  cfg.periodic = false;
  cfg.cutoff = 5;
  cfg.alpha = 0.2;
  cfg.short_cutoff = 2;
  std::vector<double> pos = {0, 0, 0, 1.1, 0.2, 0};
  std::vector<int> z = {1, 1};
  threw = false;
  try {
    CoulombPot pot(cfg, nullptr);
  } catch (const std::invalid_argument &) {
    threw = true;
  }
  check(threw, "a charge source is required");
  threw = false;
  try {
    rgpot::lr::FnCharges(nullptr, nullptr);
  } catch (const std::invalid_argument &) {
    threw = true;
  }
  check(threw, "a null charge callback is rejected");
  CoulombPot hooked(cfg, std::make_shared<rgpot::lr::FnCharges>(&fill_charges, nullptr),
                    std::make_shared<Hook>());
  const Eval eh = run(hooked, pos, z, nullptr);
  check(std::isfinite(eh.energy), "backend charges evaluate");

  threw = false;
  try {
    CoulombPot pot(cfg, std::make_shared<FixedCharges>(std::vector<double>{1.0}, 1.0));
    run(pot, pos, z, nullptr);
  } catch (const std::invalid_argument &) {
    threw = true;
  }
  check(threw, "charge count is checked");

  double ene = 0, dedr = 0;
  so3lr_erf_pair(1, -1, 12, 2, 4, 8, 1, &ene, &dedr);
  check(ene == 0.0 && dedr == 0.0, "erf pair is zero past the cutoff");
  so3lr_erf_pair(1, -1, 6, 2, 4, 8, 1, &ene, &dedr);
  check(std::isfinite(ene), "erf pair is finite inside the switch");

  cfg.method = CoulombMethod::SPME;
  cfg.periodic = true;
  cfg.spline_order = 2;
  const double box[9] = {8, 0, 0, 0, 8, 0, 0, 0, 8};
  threw = false;
  try {
    CoulombPot pot(cfg, std::make_shared<FixedCharges>(std::vector<double>{0.2, -0.2}, 0.0));
    run(pot, pos, z, box);
  } catch (const std::invalid_argument &) {
    threw = true;
  }
  check(threw, "SPME order below 3 is rejected");
  const double badbox[9] = {1, 0, 0, 2, 0, 0, 3, 0, 0};
  cfg.spline_order = 4;
  cfg.method = CoulombMethod::Ewald;
  cfg.kmax = 2;
  threw = false;
  try {
    CoulombPot pot(cfg, std::make_shared<FixedCharges>(std::vector<double>{0.2, -0.2}, 0.0));
    run(pot, pos, z, badbox);
  } catch (const std::exception &) {
    threw = true;
  }
  check(threw, "a singular cell is rejected");

  auto qdo = rgpot::lr::make_dispersion(rgpot::lr::DispersionKind::QDO);
  auto mbd = rgpot::lr::make_dispersion(rgpot::lr::DispersionKind::MBD);
  auto d3 = rgpot::lr::make_dispersion(rgpot::lr::DispersionKind::D3BJ);
  auto d4 = rgpot::lr::make_dispersion(rgpot::lr::DispersionKind::D4);
  check(std::string(qdo->name()) == "QDO" && qdo->kind() == rgpot::lr::DispersionKind::QDO, "QDO name");
  check(std::string(mbd->name()) == "MBD@rsSCS", "MBD name");
  check(std::string(d3->name()) == "D3BJ" && std::string(d4->name()) == "D4", "library names");

  DispersionConfig dc;
  dc.kind = DispersionKind::QDO;
  dc.qdo_legacy = true;
  dc.qdo_damped = false;
  dc.switch_width = 1.0;
  dc.cutoff = 6;
  dc.short_cutoff = 3;
  dc.periodic = false;
  std::vector<double> tri = {0, 0, 0, 1.5, 0.2, 0.1, 0.3, 1.8, 0.4};
  std::vector<int> z3 = {1, 6, 8};
  DispersionPot legacy(dc, std::make_shared<FixedRatios>(std::vector<double>{1, 1, 1}, std::vector<double>{1, 1, 1}),
                       std::make_shared<Ratios>());
  const Eval el = run(legacy, tri, z3, nullptr);
  check(std::isfinite(el.energy), "legacy undamped QDO with a ratio response");
  const double cell[9] = {12, 0, 0, 0, 12, 0, 0, 0, 12};
  dc.periodic = true;
  DispersionPot periodic(dc);
  const Eval ep = run(periodic, tri, z3, cell);
  check(ep.has_stress, "periodic QDO publishes stress");

  NlhPot nlh;
  const Eval en = run(nlh, tri, z3, cell);
  check(en.has_stress, "NLH publishes stress in a cell");
  threw = false;
  try {
    NlhPot bad(rgpot::lr::NlhConfig{1.0, 0.2, 1.5});
  } catch (const std::invalid_argument &) {
    threw = true;
  }
  check(threw, "NLH cutoff must clear the switch");

  SumPot sum;
  sum.add(std::make_unique<CoulombPot>(cfg, std::make_shared<FixedCharges>(std::vector<double>{0.2, -0.2, 0.0}, 0.0)));
  sum.add(std::make_unique<Bias>());
  DispersionConfig scfg;
  scfg.kind = DispersionKind::QDO;
  scfg.cutoff = 6;
  scfg.short_cutoff = 2;
  sum.add(std::make_unique<DispersionPot>(scfg));
  sum.add(std::make_unique<NlhPot>());
  auto shared = std::make_shared<DualList>(2.0, 5.0, 0.4);
  sum.share_neighbours(shared);
  const Eval es = run(sum, tri, z3, box);
  check(es.has_stress, "the sum keeps child stress");
  threw = false;
  try {
    sum.add(nullptr);
  } catch (const std::invalid_argument &) {
    threw = true;
  }
  check(threw, "a null summand is rejected");

  DispersionConfig d3cfg;
  d3cfg.kind = DispersionKind::D3BJ;
  d3cfg.functional = "pbe";
  d3cfg.cutoff = 12;
  d3cfg.short_cutoff = 3;
  DispersionPot d3pot(d3cfg);
  check(mbd->kind() == rgpot::lr::DispersionKind::MBD, "MBD kind");
  check(d3->kind() == rgpot::lr::DispersionKind::D3BJ && d4->kind() == rgpot::lr::DispersionKind::D4,
        "library kinds");
  threw = false;
  try {
    DispersionConfig badz;
    badz.kind = DispersionKind::MBD;
    badz.cutoff = 8;
    badz.short_cutoff = 3;
    DispersionPot pot(badz);
    std::vector<int> zz = {0, 1, 1};
    run(pot, tri, zz, nullptr);
  } catch (const std::exception &) {
    threw = true;
  }
  check(threw, "MBD rejects an unknown element");
  threw = false;
  try {
    DispersionConfig badh;
    badh.kind = DispersionKind::MBD;
    badh.cutoff = 8;
    badh.short_cutoff = 3;
    DispersionPot pot(badh, std::make_shared<FixedRatios>(std::vector<double>{0, 1, 1}, std::vector<double>{}));
    run(pot, tri, z3, nullptr);
  } catch (const std::exception &) {
    threw = true;
  }
  check(threw, "MBD rejects a zero ratio");
  threw = false;
  try {
    DispersionConfig qcfg;
    qcfg.kind = DispersionKind::QDO;
    qcfg.cutoff = 8;
    qcfg.short_cutoff = 3;
    DispersionPot pot(qcfg);
    std::vector<int> zz = {200, 1, 1};
    run(pot, tri, zz, nullptr);
  } catch (const std::invalid_argument &) {
    threw = true;
  }
  check(threw, "QDO rejects an unknown element");
  threw = false;
  try {
    DispersionPot pot(dc, std::make_shared<FixedRatios>(std::vector<double>{1.0}, std::vector<double>{}));
    run(pot, tri, z3, nullptr);
  } catch (const std::invalid_argument &) {
    threw = true;
  }
  check(threw, "ratio length is checked");
  threw = false;
  try {
    DispersionPot pot(dc);
    ForceOut out{};
    std::vector<double> f(6, 0);
    out.F = f.data();
    ForceInput in{2, pos.data(), nullptr, nullptr};
    pot.forceImpl(in, &out);
  } catch (const std::invalid_argument &) {
    threw = true;
  }
  check(threw, "dispersion requires atomic numbers");
  threw = false;
  try {
    DispersionConfig badm = d3cfg;
    badm.functional = "not-a-functional";
    DispersionPot pot(badm);
    run(pot, pos, z, nullptr);
  } catch (const std::runtime_error &) {
    threw = true;
  }
  check(threw, "an unknown D3 functional is rejected");
  threw = false;
  try {
    NlhPot pot;
    ForceOut out{};
    std::vector<double> f(6, 0);
    out.F = f.data();
    ForceInput in{2, pos.data(), nullptr, nullptr};
    pot.forceImpl(in, &out);
  } catch (const std::invalid_argument &) {
    threw = true;
  }
  check(threw, "NLH requires atomic numbers");
  threw = false;
  try {
    NlhPot pot;
    std::vector<int> zz = {99, 1};
    run(pot, pos, zz, nullptr);
  } catch (const std::invalid_argument &) {
    threw = true;
  }
  check(threw, "NLH rejects Z beyond uranium");

  std::vector<double> dedq(2);
  rgpot::lr::ElecOut er;
  std::vector<double> ff(6, 0);
  rgpot::lr::CoulombConfig ec = cfg;
  ec.method = CoulombMethod::Ewald;
  ec.kmax = 3;
  ec.cutoff = 6;
  ec.alpha = 0.4;
  std::vector<rgpot::lr::Pair> nopairs;
  rgpot::lr::evaluate_electrostatics(CoulombMethod::Ewald, ec, pos.data(),
                                    std::vector<double>{0.3, -0.3}.data(), 2, 0.0, box, nopairs, ff.data(),
                                    dedq.data(), &er);
  check(std::isfinite(dedq[0]), "Ewald charge derivative");
  ec.method = CoulombMethod::SPME;
  ec.grid[0] = ec.grid[1] = ec.grid[2] = 16;
  ec.spline_order = 4;
  rgpot::lr::evaluate_electrostatics(CoulombMethod::SPME, ec, pos.data(),
                                    std::vector<double>{0.3, -0.3}.data(), 2, 0.0, box, nopairs, ff.data(),
                                    dedq.data(), &er);
  check(std::isfinite(er.energy), "SPME charge derivative");
  threw = false;
  try {
    rgpot::lr::evaluate_electrostatics(CoulombMethod::Ewald, ec, pos.data(), nullptr, 2, 0, box, nopairs,
                                      ff.data(), nullptr, &er);
  } catch (const std::invalid_argument &) {
    threw = true;
  }
  check(threw, "null charges are rejected");
  threw = false;
  try {
    std::vector<double> nq = {std::nan(""), 0};
    rgpot::lr::evaluate_electrostatics(CoulombMethod::DSF, ec, pos.data(), nq.data(), 2, 0, nullptr, nopairs,
                                      ff.data(), nullptr, &er);
  } catch (const std::invalid_argument &) {
    threw = true;
  }
  check(threw, "a non-finite charge is rejected");
  threw = false;
  try {
    ec.alpha = -1;
    rgpot::lr::evaluate_electrostatics(CoulombMethod::DSF, ec, pos.data(),
                                      std::vector<double>{0.1, -0.1}.data(), 2, 0, nullptr, nopairs, ff.data(),
                                      nullptr, &er);
  } catch (const std::invalid_argument &) {
    threw = true;
  }
  check(threw, "a negative alpha is rejected");
  threw = false;
  try {
    std::vector<double> badq = {0.2, -0.2};
    ec.alpha = 0.3;
    ec.method = CoulombMethod::Ewald;
    rgpot::lr::evaluate_electrostatics(CoulombMethod::Ewald, ec, pos.data(), badq.data(), 2,
                                      std::numeric_limits<double>::infinity(), box, nopairs, ff.data(),
                                      nullptr, &er);
  } catch (const std::invalid_argument &) {
    threw = true;
  }
  check(threw, "a non-finite total charge is rejected");
  threw = false;
  try {
    ec.kmax = 0;
    rgpot::lr::evaluate_electrostatics(CoulombMethod::Ewald, ec, pos.data(),
                                      std::vector<double>{0.2, -0.2}.data(), 2, 0, box, nopairs,
                                      ff.data(), nullptr, &er);
  } catch (const std::invalid_argument &) {
    threw = true;
  }
  check(threw, "Ewald kmax of zero is rejected");
  threw = false;
  try {
    std::vector<double> badpos = {0, 0, std::numeric_limits<double>::quiet_NaN(), 1, 0, 0};
    ec.kmax = 2;
    rgpot::lr::evaluate_electrostatics(CoulombMethod::DSF, ec, badpos.data(),
                                      std::vector<double>{0.2, -0.2}.data(), 2, 0, nullptr, nopairs,
                                      ff.data(), nullptr, nullptr);
  } catch (const std::invalid_argument &) {
    threw = true;
  }
  check(threw, "a non-finite position is rejected");
  threw = false;
  try {
    DispersionConfig badc6;
    badc6.kind = DispersionKind::QDO;
    badc6.cutoff = 6;
    badc6.short_cutoff = 2;
    DispersionPot pot(badc6, std::make_shared<FixedRatios>(std::vector<double>{1, 1},
                                                          std::vector<double>{1}));
    run(pot, pos, z, nullptr);
  } catch (const std::invalid_argument &) {
    threw = true;
  }
  check(threw, "a short C6 ratio vector is rejected");
  threw = false;
  try {
    DispersionConfig bad4;
    bad4.kind = DispersionKind::D4;
    bad4.functional = "not-a-functional";
    bad4.cutoff = 8;
    bad4.short_cutoff = 2;
    DispersionPot pot(bad4);
    run(pot, pos, z, nullptr);
  } catch (const std::runtime_error &) {
    threw = true;
  }
  check(threw, "an unknown D4 functional is rejected");
  {
    DispersionConfig close_cfg;
    close_cfg.kind = DispersionKind::MBD;
    close_cfg.cutoff = 8;
    close_cfg.short_cutoff = 2;
    close_cfg.mbd_beta = 0.1;
    DispersionPot pot(close_cfg);
    std::vector<double> piled = {0, 0, 0, 0.05, 0, 0, 0.08, 0.02, 0};
    try {
      run(pot, piled, z3, nullptr);
    } catch (const std::exception &) {
    }
  }

  run(d3pot, pos, z, nullptr);
  run(d3pot, pos, z, nullptr);
  DispersionConfig d4cfg = d3cfg;
  d4cfg.kind = DispersionKind::D4;
  d4cfg.charge = 0;
  DispersionPot d4pot(d4cfg);
  run(d4pot, pos, z, nullptr);
  run(d4pot, pos, z, nullptr);
}

} // namespace

int main() {
  std::cout << std::setprecision(17);
#ifdef _OPENMP
  omp_set_num_threads(1);
#endif
  try {
    test_lists();
    test_madelung();
    test_forces_and_stress();
    test_dsf();
    test_qdo_nlh_mbd();
    test_library_dispersion();
    test_sum_and_charge();
    test_edges();
    test_nve();
  } catch (const std::exception &ex) {
    std::cerr << "FAIL exception " << ex.what() << '\n';
    return 1;
  }
  if (g_fails != 0) {
    std::cerr << g_fails << " checks failed\n";
    return 1;
  }
  std::cout << "lr_phys_test ok\n";
  return 0;
}
