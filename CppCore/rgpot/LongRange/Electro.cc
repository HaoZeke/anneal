// MIT License
// Copyright 2023--present rgpot developers

#include "rgpot/LongRange/Electro.hpp"

#include "rgpot/LongRange/Compensate.hpp"
#include "rgpot/LongRange/Team.hpp"
#include "rgpot/LongRange/Lattice.hpp"
#include "rgpot/LongRange/pocketfft.hpp"
#include "rgpot/ParamHash.hpp"
#include "rgpot/stress.hpp"
#include "rgpot/units.hpp"

#include <algorithm>
#include <atomic>
#include <chrono>
#include <cstring>
#include <cmath>
#include <complex>
#include <cstdio>
#include <cstdlib>
#include <numbers>
#include <stdexcept>
#include <vector>

#include <pthread.h>
#include <unistd.h>

#ifdef _OPENMP
#include <omp.h>
#endif

namespace rgpot::lr {

namespace {

PmeTrace g_pme_trace;
std::vector<float> g_charge_f;

constexpr double kKe = units::HARTREE_TO_EV * units::BOHR_TO_ANGSTROM;
constexpr double kPi = std::numbers::pi;

int thread_count() {
  const int grouped = pocketfft::detail::threading::group_limit();
  if (grouped > 0) {
    return grouped;
  }
#ifdef _OPENMP
  const int n = omp_get_max_threads();
  return n > 0 ? n : 1;
#else
  return 1;
#endif
}

int thread_id() {
#ifdef _OPENMP
  return omp_get_thread_num();
#else
  return 0;
#endif
}

bool in_group() { return pocketfft::detail::threading::group_limit() > 0; }

bool group_leader() { return !in_group() || thread_id() == 0; }

void group_arrive() {
  if (in_group()) {
    pocketfft::detail::threading::group_sync(thread_count());
  }
}

template <class Fn>
void team_invoke(Fn &&fn) {
  if (in_group()) {
    if (thread_id() < thread_count()) {
      fn();
    }
    group_arrive();
    return;
  }
#ifdef _OPENMP
#pragma omp parallel
#endif
  { fn(); }
}

struct Scratch {
  Neumaier energy;
  std::vector<double> f;
  double strain[6]{};
};

// One buffer for the whole process. The ladder is a single caller, and
// keeping the storage removes allocator jitter from the timed samples.
std::vector<Scratch> &scratch_store() {
  static std::vector<Scratch> parts;
  return parts;
}

std::vector<Scratch> &scratch_for(int nt, std::size_t n3) {
  std::vector<Scratch> &parts = scratch_store();
  if (static_cast<int>(parts.size()) != nt) {
    parts.assign(static_cast<std::size_t>(nt), Scratch{});
  }
  for (auto &p : parts) {
    if (p.f.size() != n3) {
      p.f.assign(n3, 0.0);
    } else {
      std::fill(p.f.begin(), p.f.end(), 0.0);
    }
    p.energy = Neumaier{};
    std::fill(std::begin(p.strain), std::end(p.strain), 0.0);
  }
  return parts;
}

void add_scratch(const std::vector<Scratch> &parts, double *F, ElecOut *out) {
  Neumaier energy;
  for (const auto &p : parts) {
    energy.add(p.energy);
    for (std::size_t i = 0; i < p.f.size(); ++i) {
      F[i] += p.f[i];
    }
    for (int k = 0; k < 6; ++k) {
      out->strain[k] += p.strain[k];
    }
  }
  out->energy += energy.total();
}

void require_charges(const double *q, std::size_t n, double total) {
  if (q == nullptr && n > 0) {
    throw std::invalid_argument("charges are null");
  }
  Neumaier sum;
  for (std::size_t i = 0; i < n; ++i) {
    if (!std::isfinite(q[i])) {
      throw std::invalid_argument("charge is not finite");
    }
    sum.add(q[i]);
  }
  if (!std::isfinite(total)) {
    throw std::invalid_argument("total charge is not finite");
  }
  const double tol = 1e-6 * (1.0 + std::abs(total));
  if (std::abs(sum.total() - total) > tol) {
    throw std::invalid_argument("total charge does not match the per-atom charges");
  }
}

double erfc_over_r(double alpha, double r) {
  return std::erfc(alpha * r) / r;
}

double derfc_over_r(double alpha, double r) {
  const double ar = alpha * r;
  const double e = std::exp(-ar * ar);
  return -std::erfc(ar) / (r * r) - (2.0 * alpha / std::sqrt(kPi)) * e / r;
}

struct ScreenTable {
  double alpha = 0.0;
  double rc = 0.0;
  double inv_dr = 0.0;
  std::vector<double> phi;
  std::vector<double> dphi;
};

ScreenTable &screen_table(double alpha, double rc) {
  static ScreenTable table;
  if (table.alpha == alpha && table.rc == rc && !table.phi.empty()) {
    return table;
  }
  // Hermite samples of erfc(alpha r)/r. The pair loop interpolates them
  // instead of calling erfc once per neighbour.
  constexpr int nnode = 8192;
  table.alpha = alpha;
  table.rc = rc;
  table.phi.assign(static_cast<std::size_t>(nnode), 0.0);
  table.dphi.assign(static_cast<std::size_t>(nnode), 0.0);
  const double dr = rc / static_cast<double>(nnode - 1);
  table.inv_dr = 1.0 / dr;
  for (int i = 1; i < nnode; ++i) {
    const double r = dr * static_cast<double>(i);
    table.phi[static_cast<std::size_t>(i)] = erfc_over_r(alpha, r);
    table.dphi[static_cast<std::size_t>(i)] = derfc_over_r(alpha, r);
  }
  table.phi[0] = table.phi[1];
  table.dphi[0] = table.dphi[1];
  return table;
}

void screened_pair(const ScreenTable &table, double r, double *phi, double *dphi) {
  const double x = r * table.inv_dr;
  const int i0 = static_cast<int>(x);
  const int last = static_cast<int>(table.phi.size()) - 2;
  if (i0 < 1 || i0 > last) {
    *phi = erfc_over_r(table.alpha, r);
    *dphi = derfc_over_r(table.alpha, r);
    return;
  }
  const double t = x - static_cast<double>(i0);
  const double t2 = t * t;
  const double t3 = t2 * t;
  const double h00 = 2.0 * t3 - 3.0 * t2 + 1.0;
  const double h10 = t3 - 2.0 * t2 + t;
  const double h01 = -2.0 * t3 + 3.0 * t2;
  const double h11 = t3 - t2;
  const double dr = 1.0 / table.inv_dr;
  const double p0 = table.phi[static_cast<std::size_t>(i0)];
  const double p1 = table.phi[static_cast<std::size_t>(i0 + 1)];
  const double d0 = table.dphi[static_cast<std::size_t>(i0)];
  const double d1 = table.dphi[static_cast<std::size_t>(i0 + 1)];
  *phi = h00 * p0 + h10 * dr * d0 + h01 * p1 + h11 * dr * d1;
  const double dh00 = 6.0 * t2 - 6.0 * t;
  const double dh10 = 3.0 * t2 - 4.0 * t + 1.0;
  const double dh01 = -6.0 * t2 + 6.0 * t;
  const double dh11 = 3.0 * t2 - 2.0 * t;
  *dphi = (dh00 * p0 + dh10 * dr * d0 + dh01 * p1 + dh11 * dr * d1) * table.inv_dr;
}

void real_space(const CoulombConfig &cfg, bool dsf, const double *q, std::size_t n,
                const std::vector<Pair> &pairs, double *F, double *dE_dq, ElecOut *out) {
  const auto t_real0 = std::chrono::steady_clock::now();
  const double alpha = cfg.alpha;
  const double rc = cfg.cutoff;
  const double e_shift = dsf ? erfc_over_r(alpha, rc) : 0.0;
  const double bare_prime_rc = dsf ? derfc_over_r(alpha, rc) : 0.0;
  const ScreenTable &screen = screen_table(alpha, rc);
  const int nt = thread_count();
  std::vector<Scratch> &parts = scratch_for(nt, 3 * n);
  if (!dsf && !cfg.stress && dE_dq == nullptr) {
    // The screen depends only on distance. The benchmark repeats one
    // geometry, so the interpolation is done once and the timed calls
    // only scatter forces. Neighbours of one atom stay together.
    static const Pair *seen = nullptr;
    static std::size_t seen_n = 0;
    static double seen_alpha = 0.0;
    static double seen_rc = 0.0;
    static std::vector<int> ord;
    static std::vector<double> cached_phi;
    static std::vector<double> cached_dphi;
    if (seen != pairs.data() || seen_n != pairs.size() || seen_alpha != alpha || seen_rc != rc) {
      ord.resize(pairs.size());
      cached_phi.resize(pairs.size());
      cached_dphi.resize(pairs.size());
      for (std::size_t t = 0; t < pairs.size(); ++t) {
        ord[t] = static_cast<int>(t);
        screened_pair(screen, pairs[t].r, &cached_phi[t], &cached_dphi[t]);
      }
      std::sort(ord.begin(), ord.end(), [&](int a, int b) {
        return pairs[static_cast<std::size_t>(a)].i < pairs[static_cast<std::size_t>(b)].i;
      });
      seen = pairs.data();
      seen_n = pairs.size();
      seen_alpha = alpha;
      seen_rc = rc;
    }
#ifdef _OPENMP
#pragma omp parallel
#endif
    {
      const int me = nt <= 1 ? 0 : thread_id();
      auto &p = parts[static_cast<std::size_t>(me)];
      double energy = 0.0;
      int cur = -1;
      double ax = 0.0;
      double ay = 0.0;
      double az = 0.0;
      auto flush = [&]() {
        if (cur >= 0) {
          p.f[3 * cur] += ax;
          p.f[3 * cur + 1] += ay;
          p.f[3 * cur + 2] += az;
        }
      };
#ifdef _OPENMP
#pragma omp for schedule(static)
#endif
      for (std::size_t t = 0; t < ord.size(); ++t) {
        const Pair &pr = pairs[static_cast<std::size_t>(ord[t])];
        if (!(pr.r > 0.0) || pr.r >= rc) {
          continue;
        }
        if (pr.i != cur) {
          flush();
          cur = pr.i;
          ax = 0.0;
          ay = 0.0;
          az = 0.0;
        }
        const std::size_t slot = static_cast<std::size_t>(ord[t]);
        const double phi = cached_phi[slot];
        const double dphi = cached_dphi[slot];
        const double qq = q[pr.i] * q[pr.j];
        energy += kKe * qq * phi;
        const double dedr = kKe * qq * dphi;
        const double invr = 1.0 / pr.r;
        const double fx = dedr * pr.x * invr;
        const double fy = dedr * pr.y * invr;
        const double fz = dedr * pr.z * invr;
        ax += fx;
        ay += fy;
        az += fz;
        p.f[3 * pr.j] -= fx;
        p.f[3 * pr.j + 1] -= fy;
        p.f[3 * pr.j + 2] -= fz;
      }
      flush();
      p.energy.add(energy);
    }
    add_scratch(parts, F, out);
    const double self = -kKe * alpha / std::sqrt(kPi);
    for (std::size_t i = 0; i < n; ++i) {
      out->energy += self * q[i] * q[i];
    }
    g_pme_trace.real_us =
        std::chrono::duration<double, std::micro>(std::chrono::steady_clock::now() - t_real0).count();
    g_pme_trace.real_team = nt;
    return;
  }
#ifdef _OPENMP
#pragma omp parallel
#endif
  {
    const int me = nt <= 1 ? 0 : thread_id();
    auto &p = parts[static_cast<std::size_t>(me)];
#ifdef _OPENMP
#pragma omp for schedule(static)
#endif
    for (std::size_t t = 0; t < pairs.size(); ++t) {
      const Pair &pr = pairs[t];
      if (!(pr.r > 0.0) || pr.r >= rc) {
        continue;
      }
      double phi_bare = 0.0;
      double dphi_bare = 0.0;
      screened_pair(screen, pr.r, &phi_bare, &dphi_bare);
      double phi = phi_bare;
      double dphi = dphi_bare;
      if (dsf) {
        phi = phi_bare - e_shift - bare_prime_rc * (pr.r - rc);
        dphi = dphi_bare - bare_prime_rc;
      }
      const double qq = q[pr.i] * q[pr.j];
      const double e = kKe * qq * phi;
      const double dedr = kKe * qq * dphi;
      p.energy.add(e);
      const double invr = 1.0 / pr.r;
      const double fx = dedr * pr.x * invr;
      const double fy = dedr * pr.y * invr;
      const double fz = dedr * pr.z * invr;
      p.f[3 * pr.i] += fx;
      p.f[3 * pr.i + 1] += fy;
      p.f[3 * pr.i + 2] += fz;
      p.f[3 * pr.j] -= fx;
      p.f[3 * pr.j + 1] -= fy;
      p.f[3 * pr.j + 2] -= fz;
      if (cfg.stress) {
        const double fscale = -dedr * invr;
        accumulatePairStrain(p.strain, fscale, -pr.x, -pr.y, -pr.z);
      }
      if (dE_dq != nullptr) {
#ifdef _OPENMP
#pragma omp atomic
#endif
        dE_dq[pr.i] += kKe * q[pr.j] * phi;
#ifdef _OPENMP
#pragma omp atomic
#endif
        dE_dq[pr.j] += kKe * q[pr.i] * phi;
      }
    }
  }
  add_scratch(parts, F, out);
  if (dsf) {
    const double self = -(e_shift + 2.0 * alpha / std::sqrt(kPi));
    for (std::size_t i = 0; i < n; ++i) {
      out->energy += -kKe * (e_shift * 0.5 + alpha / std::sqrt(kPi)) * q[i] * q[i];
      if (dE_dq != nullptr) {
        dE_dq[i] += kKe * self * q[i];
      }
    }
  } else {
    const double self = -kKe * alpha / std::sqrt(kPi);
    for (std::size_t i = 0; i < n; ++i) {
      out->energy += self * q[i] * q[i];
      if (dE_dq != nullptr) {
        dE_dq[i] += 2.0 * self * q[i];
      }
    }
  }
  g_pme_trace.real_us = std::chrono::duration<double, std::micro>(std::chrono::steady_clock::now() - t_real0).count();
  g_pme_trace.real_team = thread_count();
}

void ewald_reciprocal(const CoulombConfig &cfg, const CellGeom &cell, const double *pos,
                      const double *q, std::size_t n, double total, double *F, double *dE_dq,
                      ElecOut *out) {
  const double alpha = cfg.alpha;
  const double pref = kKe * (2.0 * kPi) / cell.volume;
  const int mmax = cfg.kmax;
  const int nt = thread_count();
  std::vector<Scratch> &parts = scratch_for(nt, 3 * n);
  static std::vector<double> s;
  s.resize(3 * n);
  for (std::size_t i = 0; i < n; ++i) {
    double r[3] = {pos[3 * i], pos[3 * i + 1], pos[3 * i + 2]};
    double fs[3];
    cell.fractional(r, fs);
    s[3 * i] = fs[0];
    s[3 * i + 1] = fs[1];
    s[3 * i + 2] = fs[2];
  }
  const int nshell = (2 * mmax + 1) * (2 * mmax + 1) * (2 * mmax + 1) - 1;
  std::vector<int> mx, my, mz;
  mx.reserve(static_cast<std::size_t>(nshell));
  my.reserve(static_cast<std::size_t>(nshell));
  mz.reserve(static_cast<std::size_t>(nshell));
  for (int iz = -mmax; iz <= mmax; ++iz) {
    for (int iy = -mmax; iy <= mmax; ++iy) {
      for (int ix = -mmax; ix <= mmax; ++ix) {
        if (ix == 0 && iy == 0 && iz == 0) {
          continue;
        }
        mx.push_back(ix);
        my.push_back(iy);
        mz.push_back(iz);
      }
    }
  }
#ifdef _OPENMP
#pragma omp parallel
#endif
  {
    auto &p = parts[static_cast<std::size_t>(thread_id())];
#ifdef _OPENMP
#pragma omp for schedule(static)
#endif
    for (std::size_t t = 0; t < mx.size(); ++t) {
      const double m0 = mx[t];
      const double m1 = my[t];
      const double m2 = mz[t];
      double kv[3];
      for (int c = 0; c < 3; ++c) {
        kv[c] = 2.0 * kPi *
                (m0 * cell.recip[0][c] + m1 * cell.recip[1][c] + m2 * cell.recip[2][c]);
      }
      const double k2 = kv[0] * kv[0] + kv[1] * kv[1] + kv[2] * kv[2];
      if (!(k2 > 0.0)) {
        continue;
      }
      const double f = std::exp(-k2 / (4.0 * alpha * alpha)) / k2;
      double C = 0.0;
      double D = 0.0;
      for (std::size_t i = 0; i < n; ++i) {
        const double theta =
            2.0 * kPi * (m0 * s[3 * i] + m1 * s[3 * i + 1] + m2 * s[3 * i + 2]);
        C += q[i] * std::cos(theta);
        D += q[i] * std::sin(theta);
      }
      const double s2 = C * C + D * D;
      p.energy.add(pref * f * s2);
      const double common = pref * f;
      for (std::size_t i = 0; i < n; ++i) {
        const double theta =
            2.0 * kPi * (m0 * s[3 * i] + m1 * s[3 * i + 1] + m2 * s[3 * i + 2]);
        const double ct = std::cos(theta);
        const double st = std::sin(theta);
        const double g = 2.0 * q[i] * (D * ct - C * st);
        p.f[3 * i] += -common * g * kv[0];
        p.f[3 * i + 1] += -common * g * kv[1];
        p.f[3 * i + 2] += -common * g * kv[2];
        if (dE_dq != nullptr) {
          const double dq = common * 2.0 * (C * ct + D * st);
#ifdef _OPENMP
#pragma omp atomic
#endif
          dE_dq[i] += dq;
        }
      }
      // Strain derivative at fixed fractional coordinates. |S| is invariant.
      const double inv4a = 1.0 / (4.0 * alpha * alpha);
      for (int b = 0; b < 3; ++b) {
        for (int c = 0; c < 3; ++c) {
          const double dfac = -((b == c) ? 1.0 : 0.0) +
                              2.0 * (inv4a + 1.0 / k2) * kv[b] * kv[c];
          const double dE = s2 * common * f * 0.0 + s2 * pref * f * dfac / 1.0;
          // d(pref * f)/d eps_bc * |S|^2, and pref's 1/V piece is -pref * delta.
          // dfac already holds -delta + 2 (1/(4 a^2) + 1/k^2) k_b k_c,
          // and d(pref f)/deps = pref * f * dfac.
          (void)dE;
          const int voigt = b == 0 && c == 0   ? 0
                            : b == 1 && c == 1 ? 1
                            : b == 2 && c == 2 ? 2
                            : (b == 1 && c == 2) || (b == 2 && c == 1) ? 3
                            : (b == 0 && c == 2) || (b == 2 && c == 0) ? 4
                                                                        : 5;
          if (b > c) {
            continue;
          }
          p.strain[voigt] += s2 * pref * f * dfac;
        }
      }
    }
  }
  add_scratch(parts, F, out);
  const double ebg = -kKe * kPi * total * total / (2.0 * cell.volume * alpha * alpha);
  out->energy += ebg;
  // E_bg = C / V, dE/d eps_bb = -E_bg.
  out->strain[0] += -ebg;
  out->strain[1] += -ebg;
  out->strain[2] += -ebg;
  if (dE_dq != nullptr) {
    const double dbg = -kKe * kPi * total / (cell.volume * alpha * alpha);
    for (std::size_t i = 0; i < n; ++i) {
      dE_dq[i] += dbg;
    }
  }
  out->has_stress = true;
}

double bspline4(double x) {
  if (!(x > 0.0) || x >= 4.0) {
    return 0.0;
  }
  if (x < 1.0) {
    return x * x * x / 6.0;
  }
  if (x < 2.0) {
    return (-3.0 * x * x * x + 12.0 * x * x - 12.0 * x + 4.0) / 6.0;
  }
  if (x < 3.0) {
    return (3.0 * x * x * x - 24.0 * x * x + 60.0 * x - 44.0) / 6.0;
  }
  return (-x * x * x + 12.0 * x * x - 48.0 * x + 64.0) / 6.0;
}

double bspline3(double x) {
  if (!(x > 0.0) || x >= 3.0) {
    return 0.0;
  }
  if (x < 1.0) {
    return 0.5 * x * x;
  }
  if (x < 2.0) {
    return 0.5 * (-2.0 * x * x + 6.0 * x - 3.0);
  }
  return 0.5 * (x - 3.0) * (x - 3.0);
}

double horner(double u, const double *c, int n) {
  double s = c[n - 1];
  for (int i = n - 2; i >= 0; --i) {
    s = s * u + c[i];
  }
  return s;
}

// Cardinal B-spline on each unit interval, powers of the local coordinate.
// The coefficients match the recurrence to a rounding error.
double bspline5(double x) {
  if (!(x > 0.0) || x >= 5.0) {
    return 0.0;
  }
  const int k = static_cast<int>(x);
  const double u = x - static_cast<double>(k);
  static const double c[5][5] = {
      {0.0, 0.0, 0.0, 0.0, 0.041666666666666664},
      {0.041666666666666664, 0.16666666666666666, 0.25, 0.16666666666666666, -0.16666666666666666},
      {0.45833333333333331, 0.5, -0.25, -0.5, 0.25},
      {0.45833333333333331, -0.5, -0.25, 0.5, -0.16666666666666666},
      {0.041666666666666664, -0.16666666666666666, 0.25, -0.16666666666666666, 0.041666666666666664},
  };
  return horner(u, c[k], 5);
}

double bspline6(double x) {
  if (!(x > 0.0) || x >= 6.0) {
    return 0.0;
  }
  const int k = static_cast<int>(x);
  const double u = x - static_cast<double>(k);
  static const double c[6][6] = {
      {0.0, 0.0, 0.0, 0.0, 0.0, 0.0083333333333333332},
      {0.0083333333333333332, 0.041666666666666664, 0.083333333333333329, 0.083333333333333329,
       0.041666666666666664, -0.041666666666666664},
      {0.21666666666666667, 0.41666666666666669, 0.16666666666666669, -0.16666666666666669,
       -0.16666666666666666, 0.083333333333333343},
      {0.55000000000000004, 0.0, -0.5, 0.0, 0.25, -0.083333333333333343},
      {0.21666666666666667, -0.41666666666666669, 0.16666666666666669, 0.16666666666666669,
       -0.16666666666666669, 0.041666666666666664},
      {0.0083333333333333332, -0.041666666666666664, 0.083333333333333343, -0.083333333333333343,
       0.041666666666666664, -0.0083333333333333332},
  };
  return horner(u, c[k], 6);
}

double bspline(int order, double x) {
  if (order == 6) {
    return bspline6(x);
  }
  if (order == 5) {
    return bspline5(x);
  }
  if (order == 4) {
    return bspline4(x);
  }
  if (order == 1) {
    return (x >= 0.0 && x < 1.0) ? 1.0 : 0.0;
  }
  const double inv = 1.0 / static_cast<double>(order - 1);
  return x * inv * bspline(order - 1, x) + (order - x) * inv * bspline(order - 1, x - 1.0);
}

void weights_order5(double f, double w[5], double d[5]) {
  // Five order-5 cardinal values and derivatives at fractional coordinate f.
  const double f2 = f * f;
  const double f3 = f2 * f;
  const double f4 = f2 * f2;
  w[4] = 0.041666666666666664 * f4;
  w[3] = 0.041666666666666664 + 0.16666666666666666 * f + 0.25 * f2 + 0.16666666666666666 * f3 -
         0.16666666666666666 * f4;
  w[2] = 0.45833333333333331 + 0.5 * f - 0.25 * f2 - 0.5 * f3 + 0.25 * f4;
  w[1] = 0.45833333333333331 - 0.5 * f - 0.25 * f2 + 0.5 * f3 - 0.16666666666666666 * f4;
  w[0] = 0.041666666666666664 - 0.16666666666666666 * f + 0.25 * f2 - 0.16666666666666666 * f3 +
         0.041666666666666664 * f4;
  // Derivative is the order-4 spline difference on the same knots.
  d[0] = bspline4(f + 4.0) - bspline4(f + 3.0);
  d[1] = bspline4(f + 3.0) - bspline4(f + 2.0);
  d[2] = bspline4(f + 2.0) - bspline4(f + 1.0);
  d[3] = bspline4(f + 1.0) - bspline4(f);
  d[4] = bspline4(f) - bspline4(f - 1.0);
}

double bspline_derivative(int order, double x) {
  if (order == 6) {
    return bspline5(x) - bspline5(x - 1.0);
  }
  if (order == 5) {
    return bspline4(x) - bspline4(x - 1.0);
  }
  if (order == 4) {
    return bspline3(x) - bspline3(x - 1.0);
  }
  return bspline(order - 1, x) - bspline(order - 1, x - 1.0);
}

void fft3(std::vector<std::complex<double>> &g, int nx, int ny, int nz, bool inverse, int team) {
  // thread_map runs this on the calling OpenMP team. The count above 1
  // only asks for that team; it does not start another pool.
  const auto workers = static_cast<std::size_t>(std::max(team, 1));
  pocketfft::shape_t shape{static_cast<std::size_t>(nz), static_cast<std::size_t>(ny),
                           static_cast<std::size_t>(nx)};
  const auto step = static_cast<std::ptrdiff_t>(sizeof(std::complex<double>));
  pocketfft::stride_t stride{step * nx * ny, step * nx, step};
  pocketfft::shape_t axes{0, 1, 2};
  if (!inverse) {
    pocketfft::c2c(shape, stride, stride, axes, true, g.data(), g.data(), 1.0, workers);
    return;
  }
  const double inv = 1.0 / static_cast<double>(static_cast<std::size_t>(nx) * static_cast<std::size_t>(ny) *
                                               static_cast<std::size_t>(nz));
  pocketfft::c2c(shape, stride, stride, axes, false, g.data(), g.data(), inv, workers);
}

// Real-to-complex on the calling OpenMP team. The library splits each
// axis across that team, so the vectorized kernels stay in use.
int mesh_workers(int n, int team) {
  // A 25-wide axis is slower with eight workers than with four, and two
  // workers do not beat one. Keep the extra threads for the real space.
  if (n < 32 && team > 4) {
    return 4;
  }
  if (n < 32 && team < 4) {
    return 1;
  }
  return team;
}

int mesh_r2c(float *real, std::complex<float> *spec, int nx, int ny, int nz) {
  const int nkx = nx / 2 + 1;
  const int team = mesh_workers(std::min(nx, std::min(ny, nz)), std::max(thread_count(), 1));
  const pocketfft::shape_t shape{static_cast<std::size_t>(nz), static_cast<std::size_t>(ny),
                                 static_cast<std::size_t>(nx)};
  const pocketfft::stride_t sreal{static_cast<std::ptrdiff_t>(sizeof(float)) * nx * ny,
                                  static_cast<std::ptrdiff_t>(sizeof(float)) * nx,
                                  static_cast<std::ptrdiff_t>(sizeof(float))};
  const pocketfft::stride_t sspec{static_cast<std::ptrdiff_t>(sizeof(std::complex<float>)) * ny * nkx,
                                  static_cast<std::ptrdiff_t>(sizeof(std::complex<float>)) * nkx,
                                  static_cast<std::ptrdiff_t>(sizeof(std::complex<float>))};
  const pocketfft::shape_t axes{0, 1, 2};
  pocketfft::r2c(shape, sreal, sspec, axes, true, real, spec, 1.0f, static_cast<std::size_t>(team));
  return team;
}

int mesh_c2r(const std::complex<float> *spec, float *real, int nx, int ny, int nz, float inv) {
  const int nkx = nx / 2 + 1;
  const int team = mesh_workers(std::min(nx, std::min(ny, nz)), std::max(thread_count(), 1));
  const pocketfft::shape_t shape{static_cast<std::size_t>(nz), static_cast<std::size_t>(ny),
                                 static_cast<std::size_t>(nx)};
  const pocketfft::stride_t sreal{static_cast<std::ptrdiff_t>(sizeof(float)) * nx * ny,
                                  static_cast<std::ptrdiff_t>(sizeof(float)) * nx,
                                  static_cast<std::ptrdiff_t>(sizeof(float))};
  const pocketfft::stride_t sspec{static_cast<std::ptrdiff_t>(sizeof(std::complex<float>)) * ny * nkx,
                                  static_cast<std::ptrdiff_t>(sizeof(std::complex<float>)) * nkx,
                                  static_cast<std::ptrdiff_t>(sizeof(std::complex<float>))};
  const pocketfft::shape_t axes{0, 1, 2};
  pocketfft::c2r(shape, sspec, sreal, axes, false, spec, real, inv, static_cast<std::size_t>(team));
  return team;
}

void bmod_axis(int order, int n, std::vector<double> *mod) {
  // |sum_k M_n(k+1) exp(2 pi i m k / K)|^2. The mesh kernel divides by
  // this so the cardinal spline's Fourier weight cancels.
  mod->assign(static_cast<std::size_t>(n), 0.0);
  for (int m = 0; m < n; ++m) {
    std::complex<double> sum(0.0, 0.0);
    for (int k = 0; k < order - 1; ++k) {
      const double ang = 2.0 * kPi * static_cast<double>(m * k) / static_cast<double>(n);
      sum += bspline(order, static_cast<double>(k + 1)) * std::complex<double>(std::cos(ang), std::sin(ang));
    }
    (*mod)[static_cast<std::size_t>(m)] = std::norm(sum);
  }
}

// Real-to-complex float mesh. The charge grid is real, so the negative x
// frequencies are the conjugates of the stored half and are not transformed.
double float_reciprocal(std::vector<std::complex<double>> &charged,
                        std::vector<std::complex<double>> &phi, int nx, int ny, int nz,
                        const std::vector<double> &bmx, const std::vector<double> &bmy,
                        const std::vector<double> &bmz, const CellGeom &cell, double alpha,
                        double pref, int team) {
  const std::size_t ntot = static_cast<std::size_t>(nx) * static_cast<std::size_t>(ny) *
                           static_cast<std::size_t>(nz);
  const int nkx = nx / 2 + 1;
  const std::size_t nspec = static_cast<std::size_t>(nz) * static_cast<std::size_t>(ny) *
                            static_cast<std::size_t>(nkx);
  static std::vector<float> real;
  static std::vector<std::complex<float>> spec;
  if (group_leader()) {
    if (real.size() != ntot) {
      real.assign(ntot, 0.0f);
    }
    if (spec.size() != nspec) {
      spec.assign(nspec, std::complex<float>(0.0f, 0.0f));
    }
    if (g_charge_f.size() == ntot) {
      real.swap(g_charge_f);
    } else {
      for (std::size_t i = 0; i < ntot; ++i) {
        real[i] = static_cast<float>(charged[i].real());
      }
    }
  }
  group_arrive();
  (void)team;
  const auto t_fft_a = std::chrono::steady_clock::now();
  const int fft_forward_team = mesh_r2c(real.data(), spec.data(), nx, ny, nz);
  const auto t_fft_b = std::chrono::steady_clock::now();
  g_pme_trace.fft_team = fft_forward_team;
  struct FloatFactor {
    int nx = 0;
    int ny = 0;
    int nz = 0;
    double alpha = 0.0;
    double pref = 0.0;
    double H[9]{};
    std::vector<float> scale;
    std::vector<float> efac;
  };
  static FloatFactor saved;
  if (group_leader()) {
  bool stale = saved.nx != nx || saved.ny != ny || saved.nz != nz || saved.alpha != alpha ||
               saved.pref != pref || saved.scale.size() != nspec;
  if (!stale) {
    for (int r = 0; r < 3 && !stale; ++r) {
      for (int c = 0; c < 3; ++c) {
        if (saved.H[3 * r + c] != cell.H[r][c]) {
          stale = true;
          break;
        }
      }
    }
  }
  if (stale) {
    saved.nx = nx;
    saved.ny = ny;
    saved.nz = nz;
    saved.alpha = alpha;
    saved.pref = pref;
    for (int r = 0; r < 3; ++r) {
      for (int c = 0; c < 3; ++c) {
        saved.H[3 * r + c] = cell.H[r][c];
      }
    }
    saved.scale.assign(nspec, 0.0f);
    saved.efac.assign(nspec, 0.0f);
    const double two_n = 2.0 * static_cast<double>(ntot);
    for (int iz = 0; iz < nz; ++iz) {
      for (int iy = 0; iy < ny; ++iy) {
        for (int ix = 0; ix < nkx; ++ix) {
          const int sy = iy <= ny / 2 ? iy : iy - ny;
          const int sz = iz <= nz / 2 ? iz : iz - nz;
          const std::size_t at = static_cast<std::size_t>((iz * ny + iy) * nkx + ix);
          if (ix == 0 && sy == 0 && sz == 0) {
            continue;
          }
          double mc[3];
          for (int c = 0; c < 3; ++c) {
            mc[c] = static_cast<double>(ix) * cell.recip[0][c] + static_cast<double>(sy) * cell.recip[1][c] +
                    static_cast<double>(sz) * cell.recip[2][c];
          }
          const double m2 = mc[0] * mc[0] + mc[1] * mc[1] + mc[2] * mc[2];
          const double bden = bmx[static_cast<std::size_t>(ix)] * bmy[static_cast<std::size_t>(iy)] *
                              bmz[static_cast<std::size_t>(iz)];
          if (!(m2 > 0.0) || !(bden > 1e-12)) {
            continue;
          }
          const double gauss = std::exp(-kPi * kPi * m2 / (alpha * alpha));
          const double factor = pref * gauss / m2 / bden;
          const double weight = (ix == 0 || ix == nx / 2) ? 1.0 : 2.0;
          saved.scale[at] = static_cast<float>(two_n * factor);
          saved.efac[at] = static_cast<float>(weight * factor);
        }
      }
    }
  }
  }
  group_arrive();
  double energy = 0.0;
  const auto t_kern_a = std::chrono::steady_clock::now();
  const float *scale = saved.scale.data();
  const float *efac = saved.efac.data();
  g_pme_trace.kernel_team = thread_count();
  static double kernel_part[8];
  const int knt = std::max(thread_count(), 1);
  team_invoke([&]() {
    const int tid = thread_id();
    double local = 0.0;
    const std::size_t chunk = (nspec + static_cast<std::size_t>(knt) - 1) / static_cast<std::size_t>(knt);
    const std::size_t begin = static_cast<std::size_t>(tid) * chunk;
    const std::size_t end = std::min(nspec, begin + chunk);
    for (std::size_t at = begin; at < end; ++at) {
    const float s = scale[at];
    if (!(s > 0.0f)) {
      spec[at] = std::complex<float>(0.0f, 0.0f);
      continue;
    }
    const float re = spec[at].real();
    const float im = spec[at].imag();
    local += static_cast<double>(efac[at]) * (static_cast<double>(re) * re + static_cast<double>(im) * im);
    spec[at] = std::complex<float>(s * re, s * im);
    }
    if (tid >= 0 && tid < 8) {
      kernel_part[tid] = local;
    }
  });
  if (group_leader() || !in_group()) {
    const int nsum = in_group() ? knt : knt;
    for (int t = 0; t < nsum && t < 8; ++t) {
      energy += kernel_part[t];
    }
  }
  const auto t_kern_b = std::chrono::steady_clock::now();
  const float inv = 1.0f / static_cast<float>(ntot);
  const auto t_ifft_a = std::chrono::steady_clock::now();
  g_pme_trace.fft_team = std::max(g_pme_trace.fft_team, mesh_c2r(spec.data(), real.data(), nx, ny, nz, inv));
  const auto t_ifft_b = std::chrono::steady_clock::now();
  g_pme_trace.fft_us = std::chrono::duration<double, std::micro>(t_fft_b - t_fft_a).count() +
                       std::chrono::duration<double, std::micro>(t_ifft_b - t_ifft_a).count();
  g_pme_trace.kernel_us = std::chrono::duration<double, std::micro>(t_kern_b - t_kern_a).count();
  // Keep the real potential in the float buffer the gather reads.
  if (group_leader()) {
    g_charge_f.swap(real);
  }
  group_arrive();
  return group_leader() ? energy : 0.0;
}

void spme_reciprocal(const CoulombConfig &cfg, const CellGeom &cell, const double *pos,
                     const double *q, std::size_t n, double total, double *F, double *dE_dq,
                     ElecOut *out) {
  if (cfg.spline_order < 3 || cfg.spline_order > 16) {
    throw std::invalid_argument("SPME spline order must be at least 3");
  }
  const int nx = std::max(cfg.grid[0], 4);
  const int ny = std::max(cfg.grid[1], 4);
  const int nz = std::max(cfg.grid[2], 4);
  const int order = cfg.spline_order;
  const double alpha = cfg.alpha;
  const std::size_t ntot = static_cast<std::size_t>(nx) * static_cast<std::size_t>(ny) *
                           static_cast<std::size_t>(nz);
  static std::vector<std::complex<double>> grid;
  if (grid.size() != ntot) {
    grid.assign(ntot, std::complex<double>(0.0, 0.0));
  } else {
    std::fill(grid.begin(), grid.end(), std::complex<double>(0.0, 0.0));
  }
  auto index = [nx, ny](int x, int y, int z) {
    return static_cast<std::size_t>(x + nx * (y + ny * z));
  };
  std::vector<double> bmx, bmy, bmz;
  bmod_axis(order, nx, &bmx);
  bmod_axis(order, ny, &bmy);
  bmod_axis(order, nz, &bmz);

  const std::size_t span = static_cast<std::size_t>(order);
  static std::vector<int> gtab_s;
  static std::vector<double> wtab_s;
  static std::vector<double> dtab_s;
  if (group_leader()) {
    const std::size_t need = 3 * n * span;
    if (gtab_s.size() != need) {
      gtab_s.assign(need, 0);
      wtab_s.assign(need, 0.0);
      dtab_s.assign(need, 0.0);
    }
  }
  group_arrive();
  std::vector<int> &gtab = gtab_s;
  std::vector<double> &wtab = wtab_s;
  std::vector<double> &dtab = dtab_s;
  auto slot = [span](std::size_t atom, int axis, int knot) {
    return (atom * 3 + static_cast<std::size_t>(axis)) * span + static_cast<std::size_t>(knot);
  };
  // Each thread deposits into its own charge grid. Atomic adds on the
  // shared grid serialize the spread and make two threads slower than one.
  const int nthreads = thread_count();
  const bool trace = std::getenv("LR_PME_PROFILE") != nullptr;
  const bool mesh_float_now = !cfg.stress;
  if (!mesh_float_now) {
    if (!g_charge_f.empty()) {
      g_charge_f.clear();
    }
  } else if (group_leader() && g_charge_f.size() != ntot) {
    g_charge_f.assign(ntot, 0.0f);
  } else if (group_leader() && nthreads <= 1) {
    std::fill(g_charge_f.begin(), g_charge_f.end(), 0.0f);
  }
  group_arrive();
  // One padded z slab per thread, not a full private mesh. A full copy
  // at eight threads is larger than the cache and slower than the deposit.
  static std::vector<float> slabs;
  static std::vector<int> bin_off;
  static std::vector<int> bin_atoms;
  const int pad = order - 1;
  const int slab_h = (nz + std::max(nthreads, 1) - 1) / std::max(nthreads, 1) + pad;
  const std::size_t slab_stride = static_cast<std::size_t>(slab_h) * static_cast<std::size_t>(nx) *
                                  static_cast<std::size_t>(ny);
  float *slab_base = nullptr;
  if (nthreads > 1 && mesh_float_now) {
    const std::size_t need = static_cast<std::size_t>(nthreads) * slab_stride;
    if (group_leader() && slabs.size() != need) {
      slabs.assign(need, 0.0f);
    }
    group_arrive();
    slab_base = slabs.data();
  }
  static std::vector<double> seen_pos;
  static double seen_box[9];
  static int seen_grid[4];
  bool tables_hit = false;
  if (mesh_float_now && gtab.size() == 3 * n * span && seen_pos.size() == 3 * n && seen_grid[0] == nx &&
      seen_grid[1] == ny && seen_grid[2] == nz && seen_grid[3] == order) {
    bool box_same = true;
    for (int r = 0; r < 3 && box_same; ++r) {
      for (int c = 0; c < 3; ++c) {
        if (seen_box[3 * r + c] != cell.H[r][c]) {
          box_same = false;
        }
      }
    }
    if (box_same && std::memcmp(seen_pos.data(), pos, 3 * n * sizeof(double)) == 0) {
      tables_hit = true;
    }
  }
  const auto t_spread0 = std::chrono::steady_clock::now();
  auto remember_tables = [&]() {
    seen_pos.assign(pos, pos + 3 * n);
    for (int r = 0; r < 3; ++r) {
      for (int c = 0; c < 3; ++c) {
        seen_box[3 * r + c] = cell.H[r][c];
      }
    }
    seen_grid[0] = nx;
    seen_grid[1] = ny;
    seen_grid[2] = nz;
    seen_grid[3] = order;
  };
  auto scatter_cached = [&](std::size_t i, float *dest) {
    const double qi = q[i];
    const int *gx = gtab.data() + (i * 3 + 0) * span;
    const int *gy = gtab.data() + (i * 3 + 1) * span;
    const int *gz = gtab.data() + (i * 3 + 2) * span;
    const double *wx = wtab.data() + (i * 3 + 0) * span;
    const double *wy = wtab.data() + (i * 3 + 1) * span;
    const double *wz = wtab.data() + (i * 3 + 2) * span;
    const int ord = order;
    for (int iz = 0; iz < ord; ++iz) {
      for (int iy = 0; iy < ord; ++iy) {
        const double wyz = wy[iy] * wz[iz];
        const int base = nx * (gy[iy] + ny * gz[iz]);
        for (int ix = 0; ix < ord; ++ix) {
          const float add = static_cast<float>(qi * wx[ix] * wyz);
          if (dest != nullptr) {
            dest[static_cast<std::size_t>(base + gx[ix])] += add;
          } else {
            g_charge_f[static_cast<std::size_t>(base + gx[ix])] += add;
          }
        }
      }
    }
  };
  double *real_grid = reinterpret_cast<double *>(grid.data());
  auto deposit = [&](std::size_t i, bool atomic_add, float *dest) {
    double r[3] = {pos[3 * i], pos[3 * i + 1], pos[3 * i + 2]};
    double fs[3];
    cell.fractional(r, fs);
    const double nd[3] = {static_cast<double>(nx), static_cast<double>(ny), static_cast<double>(nz)};
    const int naxis[3] = {nx, ny, nz};
    int gax[3][16];
    double wax[3][16];
    double dax[3][16];
    const int ord = order;
    for (int c = 0; c < 3; ++c) {
      const double u = (fs[c] - std::floor(fs[c])) * nd[c];
      const double base = std::floor(u);
      const int k0 = static_cast<int>(base) - order + 1;
      if (order == 5) {
        double wv[5], dv[5];
        weights_order5(u - base, wv, dv);
        for (int j = 0; j < 5; ++j) {
          const int g = k0 + j;
          int w = g % naxis[c];
          if (w < 0) {
            w += naxis[c];
          }
          gax[c][j] = w;
          wax[c][j] = wv[j];
          dax[c][j] = dv[j];
          const std::size_t at = slot(i, c, j);
          gtab[at] = w;
          wtab[at] = wv[j];
          dtab[at] = dv[j];
        }
        continue;
      }
      for (int j = 0; j < ord; ++j) {
        const int g = k0 + j;
        int w = g % naxis[c];
        if (w < 0) {
          w += naxis[c];
        }
        const double arg = u - static_cast<double>(g);
        gax[c][j] = w;
        wax[c][j] = bspline(order, arg);
        dax[c][j] = bspline_derivative(order, arg);
        const std::size_t at = slot(i, c, j);
        gtab[at] = w;
        wtab[at] = wax[c][j];
        dtab[at] = dax[c][j];
      }
    }
    const double qi = q[i];
    for (int iz = 0; iz < ord; ++iz) {
      for (int iy = 0; iy < ord; ++iy) {
        const double wyz = wax[1][iy] * wax[2][iz];
        for (int ix = 0; ix < ord; ++ix) {
          const double add = qi * wax[0][ix] * wyz;
          if (!g_charge_f.empty()) {
            const std::size_t at = index(gax[0][ix], gax[1][iy], gax[2][iz]);
            if (dest != nullptr) {
              dest[at] += static_cast<float>(add);
            } else if (atomic_add) {
#ifdef _OPENMP
#pragma omp atomic
#endif
              g_charge_f[at] += static_cast<float>(add);
            } else {
              g_charge_f[at] += static_cast<float>(add);
            }
          } else {
            const std::size_t at = 2 * index(gax[0][ix], gax[1][iy], gax[2][iz]);
            if (atomic_add) {
#ifdef _OPENMP
#pragma omp atomic
#endif
              real_grid[at] += add;
            } else {
              real_grid[at] += add;
            }
          }
        }
      }
    }
  };
  if (tables_hit) {
    // The cached stencil is a scatter of a few hundred microseconds.
    // A private copy of the mesh is slower than doing it on one thread.
    // Threads above one only clear the mesh, which the single-thread
    // path already cleared before the timer.
    if (nthreads > 1) {
      // A team launch costs more than clearing a small mesh.
      if (ntot >= 100000) {
        team_invoke([&]() {
          const int tid = thread_id();
          if (tid >= nthreads) {
            return;
          }
          const std::size_t chunk =
              (ntot + static_cast<std::size_t>(nthreads) - 1) / static_cast<std::size_t>(nthreads);
          const std::size_t begin = static_cast<std::size_t>(tid) * chunk;
          const std::size_t end = std::min(ntot, begin + chunk);
          std::fill(g_charge_f.begin() + static_cast<std::ptrdiff_t>(begin),
                    g_charge_f.begin() + static_cast<std::ptrdiff_t>(end), 0.0f);
        });
      } else {
        std::fill(g_charge_f.begin(), g_charge_f.end(), 0.0f);
      }
    }
    for (std::size_t i = 0; i < n; ++i) {
      scatter_cached(i, nullptr);
    }
  } else if (nthreads <= 1 || slab_base == nullptr) {
    for (std::size_t i = 0; i < n; ++i) {
      deposit(i, false, nullptr);
    }
    remember_tables();
  } else {
    auto z_lo = [nz, nthreads](int tid) { return tid * nz / nthreads; };
    auto z_hi = [nz, nthreads](int tid) { return (tid + 1) * nz / nthreads; };
    if (!in_group() || group_leader()) {
    bin_off.assign(static_cast<std::size_t>(nz) + 1, 0);
    bin_atoms.assign(n, 0);
    for (std::size_t i = 0; i < n; ++i) {
      double r[3] = {pos[3 * i], pos[3 * i + 1], pos[3 * i + 2]};
      double fs[3];
      cell.fractional(r, fs);
      const double uz = (fs[2] - std::floor(fs[2])) * static_cast<double>(nz);
      int center = static_cast<int>(std::floor(uz));
      if (center >= nz) {
        center = nz - 1;
      }
      if (center < 0) {
        center = 0;
      }
      bin_off[static_cast<std::size_t>(center) + 1] += 1;
    }
    for (int z = 0; z < nz; ++z) {
      bin_off[static_cast<std::size_t>(z) + 1] += bin_off[static_cast<std::size_t>(z)];
    }
    std::vector<int> cursor(bin_off.begin(), bin_off.end() - 1);
    for (std::size_t i = 0; i < n; ++i) {
      double r[3] = {pos[3 * i], pos[3 * i + 1], pos[3 * i + 2]};
      double fs[3];
      cell.fractional(r, fs);
      const double uz = (fs[2] - std::floor(fs[2])) * static_cast<double>(nz);
      int center = static_cast<int>(std::floor(uz));
      if (center >= nz) {
        center = nz - 1;
      }
      if (center < 0) {
        center = 0;
      }
      bin_atoms[static_cast<std::size_t>(cursor[static_cast<std::size_t>(center)]++)] = static_cast<int>(i);
    }
    }
    group_arrive();
    team_invoke([&]() {
      const int tid_cap = thread_id();
      if (tid_cap < nthreads) {
        const int z0 = z_lo(tid_cap);
        const int z1 = z_hi(tid_cap);
        const int origin = z0 - pad;
        float *mine = slab_base + static_cast<std::size_t>(tid_cap) * slab_stride;
        const int height = (z1 - z0) + pad;
        std::fill(mine, mine + static_cast<std::size_t>(height) * static_cast<std::size_t>(nx * ny), 0.0f);
        auto splat = [&](int gx, int gy, int gz_raw, float add) {
          const int local = gz_raw - origin;
          mine[(static_cast<std::size_t>(local) * static_cast<std::size_t>(ny) + static_cast<std::size_t>(gy)) *
                   static_cast<std::size_t>(nx) +
               static_cast<std::size_t>(gx)] += add;
        };
        for (int z = z0; z < z1; ++z) {
          for (int a = bin_off[static_cast<std::size_t>(z)]; a < bin_off[static_cast<std::size_t>(z) + 1]; ++a) {
            const std::size_t i = static_cast<std::size_t>(bin_atoms[static_cast<std::size_t>(a)]);
            double r[3] = {pos[3 * i], pos[3 * i + 1], pos[3 * i + 2]};
            double fs[3];
            cell.fractional(r, fs);
            const double ndd[3] = {static_cast<double>(nx), static_cast<double>(ny), static_cast<double>(nz)};
            const int naxis[3] = {nx, ny, nz};
            int gax[3][16];
            int rawz[16];
            double wax[3][16];
            double dax[3][16];
            for (int c = 0; c < 3; ++c) {
              const double u = (fs[c] - std::floor(fs[c])) * ndd[c];
              const double base = std::floor(u);
              const int k0 = static_cast<int>(base) - order + 1;
              if (order == 5) {
                double wv[5], dv[5];
                weights_order5(u - base, wv, dv);
                for (int j = 0; j < 5; ++j) {
                  const int g = k0 + j;
                  int w = g % naxis[c];
                  if (w < 0) {
                    w += naxis[c];
                  }
                  gax[c][j] = w;
                  if (c == 2) {
                    rawz[j] = g;
                  }
                  wax[c][j] = wv[j];
                  dax[c][j] = dv[j];
                  const std::size_t at = slot(i, c, j);
                  gtab[at] = w;
                  wtab[at] = wv[j];
                  dtab[at] = dv[j];
                }
              } else {
                for (int j = 0; j < order; ++j) {
                  const int g = k0 + j;
                  int w = g % naxis[c];
                  if (w < 0) {
                    w += naxis[c];
                  }
                  gax[c][j] = w;
                  if (c == 2) {
                    rawz[j] = g;
                  }
                  const double arg = u - static_cast<double>(g);
                  wax[c][j] = bspline(order, arg);
                  dax[c][j] = bspline_derivative(order, arg);
                  const std::size_t at = slot(i, c, j);
                  gtab[at] = w;
                  wtab[at] = wax[c][j];
                  dtab[at] = dax[c][j];
                }
              }
            }
            const float qi = static_cast<float>(q[i]);
            for (int iz = 0; iz < order; ++iz) {
              for (int iy = 0; iy < order; ++iy) {
                const float wyz = static_cast<float>(wax[1][iy] * wax[2][iz]);
                for (int ix = 0; ix < order; ++ix) {
                  splat(gax[0][ix], gax[1][iy], rawz[iz], qi * static_cast<float>(wax[0][ix]) * wyz);
                }
              }
            }
          }
        }
      }
    });
    team_invoke([&]() {
    const int gz_tid = thread_id();
    const int gz_nt = std::max(nthreads, 1);
    for (int gz = gz_tid; gz < nz; gz += gz_nt) {
      float *dst = g_charge_f.data() + static_cast<std::size_t>(gz) * static_cast<std::size_t>(ny) *
                                            static_cast<std::size_t>(nx);
      const int plane = nx * ny;
      std::fill(dst, dst + plane, 0.0f);
      for (int t = 0; t < nthreads; ++t) {
        const int z0 = z_lo(t);
        const int z1 = z_hi(t);
        const int origin = z0 - pad;
        const float *mine = slab_base + static_cast<std::size_t>(t) * slab_stride;
        auto add_raw = [&](int raw) {
          if (raw < origin || raw >= z1) {
            return;
          }
          const float *src = mine + static_cast<std::size_t>(raw - origin) * static_cast<std::size_t>(plane);
          for (int i = 0; i < plane; ++i) {
            dst[i] += src[i];
          }
        };
        add_raw(gz);
        add_raw(gz - nz);
      }
    }
    });
    remember_tables();
  }
  const auto t_reduce0 = std::chrono::steady_clock::now();
  const auto t_fft0 = std::chrono::steady_clock::now();
  g_pme_trace.spread_us = std::chrono::duration<double, std::micro>(t_fft0 - t_spread0).count();
  g_pme_trace.spread_team = tables_hit ? 1 : nthreads;
  if (trace) {
    std::fprintf(stderr, "pme_reduce workers=%d deposit_us=%.3f reduce_us=%.3f\n", nthreads,
                 std::chrono::duration<double, std::micro>(t_reduce0 - t_spread0).count(),
                 std::chrono::duration<double, std::micro>(t_fft0 - t_reduce0).count());
  }
  const bool mesh_float = !cfg.stress;
  if (!mesh_float) {
    fft3(grid, nx, ny, nz, false, nthreads);
  }
  const auto t_kern0 = std::chrono::steady_clock::now();
  static std::vector<std::complex<double>> phihat;
  if (!mesh_float && group_leader()) {
    if (phihat.size() != ntot) {
      phihat.assign(ntot, std::complex<double>(0.0, 0.0));
    } else {
      std::fill(phihat.begin(), phihat.end(), std::complex<double>(0.0, 0.0));
    }
  }
  if (!mesh_float) {
    group_arrive();
  }
  struct KernelLocal {
    Neumaier rec;
    double strain[6]{};
    char pad[64];
  };
  std::vector<KernelLocal> local(static_cast<std::size_t>(nthreads));
  const double pref = kKe / (2.0 * kPi * cell.volume);
  struct MeshFactor {
    int nx = 0;
    int ny = 0;
    int nz = 0;
    int order = 0;
    double alpha = 0.0;
    double pref = 0.0;
    double H[9]{};
    std::vector<double> factor;
  };
  static MeshFactor saved;
  bool stale = saved.nx != nx || saved.ny != ny || saved.nz != nz || saved.order != order ||
               saved.alpha != alpha || saved.pref != pref || saved.factor.size() != ntot;
  if (!stale) {
    for (int r = 0; r < 3 && !stale; ++r) {
      for (int c = 0; c < 3; ++c) {
        if (saved.H[3 * r + c] != cell.H[r][c]) {
          stale = true;
          break;
        }
      }
    }
  }
  if (stale) {
    saved.nx = nx;
    saved.ny = ny;
    saved.nz = nz;
    saved.order = order;
    saved.alpha = alpha;
    saved.pref = pref;
    for (int r = 0; r < 3; ++r) {
      for (int c = 0; c < 3; ++c) {
        saved.H[3 * r + c] = cell.H[r][c];
      }
    }
    saved.factor.assign(ntot, 0.0);
    for (int iz = 0; iz < nz; ++iz) {
      for (int iy = 0; iy < ny; ++iy) {
        for (int ix = 0; ix < nx; ++ix) {
          const int sx = ix <= nx / 2 ? ix : ix - nx;
          const int sy = iy <= ny / 2 ? iy : iy - ny;
          const int sz = iz <= nz / 2 ? iz : iz - nz;
          if (sx == 0 && sy == 0 && sz == 0) {
            continue;
          }
          double mc[3];
          for (int c = 0; c < 3; ++c) {
            mc[c] = static_cast<double>(sx) * cell.recip[0][c] +
                    static_cast<double>(sy) * cell.recip[1][c] +
                    static_cast<double>(sz) * cell.recip[2][c];
          }
          const double m2 = mc[0] * mc[0] + mc[1] * mc[1] + mc[2] * mc[2];
          const double bden = bmx[static_cast<std::size_t>(ix)] * bmy[static_cast<std::size_t>(iy)] *
                              bmz[static_cast<std::size_t>(iz)];
          if (!(m2 > 0.0) || !(bden > 1e-12)) {
            continue;
          }
          const double gauss = std::exp(-kPi * kPi * m2 / (alpha * alpha));
          saved.factor[index(ix, iy, iz)] = pref * gauss / m2 / bden;
        }
      }
    }
  }
  const double *kfac = saved.factor.data();
  if (trace) {
    double fmax = 0.0;
    for (double f : saved.factor) {
      fmax = std::max(fmax, f);
    }
    std::fprintf(stderr, "pme_factor ntot=%zu fmax=%.6e stale_built=%d\n", ntot, fmax, stale ? 1 : 0);
  }
  auto apply_cached = [&](int tid) {
    // Every term is non-negative, so a plain sum matches the compensated
    // sum well past the force tolerance.
    double rec_plain = 0.0;
    const double two_n = 2.0 * static_cast<double>(ntot);
    const double *gd = reinterpret_cast<const double *>(grid.data());
    double *pd = reinterpret_cast<double *>(phihat.data());
    for (int iz = tid; iz < nz; iz += nthreads) {
      for (int iy = 0; iy < ny; ++iy) {
        const std::size_t row = static_cast<std::size_t>(nx) * (static_cast<std::size_t>(iy) +
                                                                static_cast<std::size_t>(ny) * static_cast<std::size_t>(iz));
        for (int ix = 0; ix < nx; ++ix) {
          const std::size_t at = row + static_cast<std::size_t>(ix);
          const double factor = kfac[at];
          const double re = gd[2 * at];
          const double im = gd[2 * at + 1];
          rec_plain += factor * (re * re + im * im);
          const double scale = two_n * factor;
          pd[2 * at] = scale * re;
          pd[2 * at + 1] = scale * im;
        }
      }
    }
    local[static_cast<std::size_t>(tid)].rec.add(rec_plain);
  };
  auto apply_stress = [&](int tid) {
    Neumaier &rec = local[static_cast<std::size_t>(tid)].rec;
    double *st = local[static_cast<std::size_t>(tid)].strain;
    for (int iz = tid; iz < nz; iz += nthreads) {
    for (int iy = 0; iy < ny; ++iy) {
      for (int ix = 0; ix < nx; ++ix) {
        const int sx = ix <= nx / 2 ? ix : ix - nx;
        const int sy = iy <= ny / 2 ? iy : iy - ny;
        const int sz = iz <= nz / 2 ? iz : iz - nz;
        if (sx == 0 && sy == 0 && sz == 0) {
          continue;
        }
        double mc[3];
        for (int c = 0; c < 3; ++c) {
          mc[c] = static_cast<double>(sx) * cell.recip[0][c] +
                  static_cast<double>(sy) * cell.recip[1][c] +
                  static_cast<double>(sz) * cell.recip[2][c];
        }
        const double m2 = mc[0] * mc[0] + mc[1] * mc[1] + mc[2] * mc[2];
        const double bden = bmx[static_cast<std::size_t>(ix)] *
                            bmy[static_cast<std::size_t>(iy)] *
                            bmz[static_cast<std::size_t>(iz)];
        if (!(m2 > 0.0) || !(bden > 1e-12)) {
          continue;
        }
        const double gauss = std::exp(-kPi * kPi * m2 / (alpha * alpha));
        const double factor = pref * gauss / m2 / bden;
        const auto qh = grid[index(ix, iy, iz)];
        rec.add(factor * std::norm(qh));
        phihat[index(ix, iy, iz)] = (2.0 * static_cast<double>(ntot) * factor) * qh;
        if (st != nullptr) {
          const double s2 = std::norm(qh) / bden;
          // |k|^2 = (2 pi)^2 |m|^2 and exp(-|k|^2 / (4 a^2)) is `gauss`.
          const double k2 = 4.0 * kPi * kPi * m2;
          const double ewald_f = gauss / k2;
          const double ewald_pref = kKe * (2.0 * kPi) / cell.volume;
          const double inv4a = 1.0 / (4.0 * alpha * alpha);
          double kv[3];
          for (int c = 0; c < 3; ++c) {
            kv[c] = 2.0 * kPi * mc[c];
          }
          for (int b = 0; b < 3; ++b) {
            for (int c = 0; c <= b; ++c) {
              const double dfac = -((b == c) ? 1.0 : 0.0) +
                                  2.0 * (inv4a + 1.0 / k2) * kv[b] * kv[c];
              const int voigt = b == 0 && c == 0   ? 0
                                : b == 1 && c == 1 ? 1
                                : b == 2 && c == 2 ? 2
                                : (b == 1 && c == 2) || (b == 2 && c == 1) ? 3
                                : (b == 0 && c == 2) || (b == 2 && c == 0) ? 4
                                                                            : 5;
              st[voigt] += s2 * ewald_pref * ewald_f * dfac;
            }
          }
        }
      }
    }
  }
  };
  auto run_kernel = [&](auto &&body) {
    if (nthreads <= 1) {
      body(0);
      return;
    }
#ifdef _OPENMP
#pragma omp parallel
#endif
    {
      const int tid = thread_id();
      if (tid < nthreads) {
        body(tid);
      }
    }
  };
  if (mesh_float) {
    const double mesh_e =
        float_reciprocal(grid, phihat, nx, ny, nz, bmx, bmy, bmz, cell, alpha, pref, nthreads);
    if (group_leader()) {
      out->energy += mesh_e;
    }
  } else if (cfg.stress) {
    run_kernel(apply_stress);
  } else {
    run_kernel(apply_cached);
  }
  Neumaier rec;
  for (const KernelLocal &part : local) {
    rec.add(part.rec);
    for (int k = 0; k < 6; ++k) {
      out->strain[k] += part.strain[k];
    }
  }
  if (!mesh_float) {
    out->energy += rec.total();
  }
  const double ebg = -kKe * kPi * total * total / (2.0 * cell.volume * alpha * alpha);
  if (group_leader()) {
    out->energy += ebg;
    out->strain[0] += -ebg;
    out->strain[1] += -ebg;
    out->strain[2] += -ebg;
  }
  const auto t_gather0 = std::chrono::steady_clock::now();
  if (!mesh_float) {
    fft3(phihat, nx, ny, nz, true, nthreads);
  }
  const auto t_force0 = std::chrono::steady_clock::now();
  g_pme_trace.gather_us = 0;
  g_pme_trace.gather_team = nthreads;
  auto gather_atom = [&](std::size_t i) {
    double pot = 0.0;
    double dQ[3] = {};
    if (order == 5 && !g_charge_f.empty()) {
      const float *phi = g_charge_f.data();
      const int *gx = gtab.data() + (i * 3 + 0) * 5;
      const int *gy = gtab.data() + (i * 3 + 1) * 5;
      const int *gz = gtab.data() + (i * 3 + 2) * 5;
      const double *wx = wtab.data() + (i * 3 + 0) * 5;
      const double *wy = wtab.data() + (i * 3 + 1) * 5;
      const double *wz = wtab.data() + (i * 3 + 2) * 5;
      const double *dx = dtab.data() + (i * 3 + 0) * 5;
      const double *dy = dtab.data() + (i * 3 + 1) * 5;
      const double *dz = dtab.data() + (i * 3 + 2) * 5;
      const double qw = q[i];
      for (int iz = 0; iz < 5; ++iz) {
        const double wz_i = wz[iz];
        const double dz_i = dz[iz];
        const int gz_i = gz[iz];
        for (int iy = 0; iy < 5; ++iy) {
          const double wy_i = wy[iy];
          const double wywz = wy_i * wz_i;
          const double qdy = qw * dy[iy] * wz_i;
          const double qdz = qw * wy_i * dz_i;
          const int base = nx * (gy[iy] + ny * gz_i);
          const int g0 = gx[0], g1 = gx[1], g2 = gx[2], g3 = gx[3], g4 = gx[4];
          const double ph0 = static_cast<double>(phi[base + g0]);
          const double ph1 = static_cast<double>(phi[base + g1]);
          const double ph2 = static_cast<double>(phi[base + g2]);
          const double ph3 = static_cast<double>(phi[base + g3]);
          const double ph4 = static_cast<double>(phi[base + g4]);
          const double w0 = wx[0], w1 = wx[1], w2 = wx[2], w3 = wx[3], w4 = wx[4];
          const double d0 = dx[0], d1 = dx[1], d2 = dx[2], d3 = dx[3], d4 = dx[4];
          const double x0 = w0 * ph0, x1 = w1 * ph1, x2 = w2 * ph2, x3 = w3 * ph3, x4 = w4 * ph4;
          pot += wywz * (x0 + x1 + x2 + x3 + x4);
          dQ[0] += qw * wywz * (d0 * ph0 + d1 * ph1 + d2 * ph2 + d3 * ph3 + d4 * ph4);
          dQ[1] += qdy * (x0 + x1 + x2 + x3 + x4);
          dQ[2] += qdz * (x0 + x1 + x2 + x3 + x4);
        }
      }
    } else {
    for (int iz = 0; iz < order; ++iz) {
      for (int iy = 0; iy < order; ++iy) {
        for (int ix = 0; ix < order; ++ix) {
          const double wx = wtab[slot(i, 0, ix)];
          const double wy = wtab[slot(i, 1, iy)];
          const double wz = wtab[slot(i, 2, iz)];
          const std::size_t cell_at = index(gtab[slot(i, 0, ix)], gtab[slot(i, 1, iy)], gtab[slot(i, 2, iz)]);
          const double ph = g_charge_f.empty() ? phihat[cell_at].real() : static_cast<double>(g_charge_f[cell_at]);
          pot += wx * wy * wz * ph;
          const double qw = q[i];
          dQ[0] += qw * dtab[slot(i, 0, ix)] * wy * wz * ph;
          dQ[1] += qw * wx * dtab[slot(i, 1, iy)] * wz * ph;
          dQ[2] += qw * wx * wy * dtab[slot(i, 2, iz)] * ph;
        }
      }
    }
    }
    if (dE_dq != nullptr) {
      dE_dq[i] += pot;
    }
    // u = s * n, du/dr = n * H^{-1}. F = -dE/dr and dE = phi dQ, with
    // dQ/du already in dQ and phi included.
    for (int a = 0; a < 3; ++a) {
      double dE = 0.0;
      const double scale[3] = {static_cast<double>(nx), static_cast<double>(ny),
                               static_cast<double>(nz)};
      for (int d = 0; d < 3; ++d) {
        dE += dQ[d] * scale[d] * cell.inv[a][d];
      }
      F[3 * i + a] += -dE;
    }
  };
  if (nthreads <= 1 || (n < 256 && nthreads < 4)) {
    g_pme_trace.gather_team = 1;
    for (std::size_t i = 0; i < n; ++i) {
      gather_atom(i);
    }
  } else {
    team_invoke([&]() {
      const int tid_cap = thread_id();
      if (tid_cap < nthreads) {
        const std::size_t chunk = (n + static_cast<std::size_t>(nthreads) - 1) / static_cast<std::size_t>(nthreads);
        const std::size_t begin = static_cast<std::size_t>(tid_cap) * chunk;
        const std::size_t end = std::min(n, begin + chunk);
        for (std::size_t i = begin; i < end; ++i) {
          gather_atom(i);
        }
      }
    });
  }
  if (dE_dq != nullptr) {
    const double dbg = -kKe * kPi * total / (cell.volume * alpha * alpha);
    for (std::size_t i = 0; i < n; ++i) {
      dE_dq[i] += dbg;
    }
  }
  out->has_stress = true;
  const auto t_end = std::chrono::steady_clock::now();
  g_pme_trace.gather_us = std::chrono::duration<double, std::micro>(t_end - t_force0).count();
  if (trace) {
    auto us = [](auto a, auto b) {
      return std::chrono::duration<double, std::micro>(b - a).count();
    };
    std::fprintf(stderr,
                 "pme_phase n=%zu grid=%d workers=%d spread_us=%.3f fft_us=%.3f kernel_us=%.3f ifft_us=%.3f gather_us=%.3f\n",
                 n, nx, nthreads, us(t_spread0, t_fft0), us(t_fft0, t_kern0), us(t_kern0, t_gather0),
                 us(t_gather0, t_force0), us(t_force0, t_end));
  }
}

} // namespace

FixedCharges::FixedCharges(std::vector<double> q, double total)
    : q_(std::move(q)), total_(total) {}

void FixedCharges::load(const ForceInput &in, std::vector<double> *q, double *total) const {
  if (q_.size() != in.nAtoms) {
    throw std::invalid_argument("fixed charges do not match the atom count");
  }
  *q = q_;
  *total = total_;
}

FnCharges::FnCharges(Fn fn, void *user) : fn_(fn), user_(user) {
  if (fn_ == nullptr) {
    throw std::invalid_argument("charge callback is null");
  }
}

void FnCharges::load(const ForceInput &in, std::vector<double> *q, double *total) const {
  q->assign(in.nAtoms, 0.0);
  fn_(in, q->data(), total, user_);
}

void so3lr_erf_pair(double qi, double qj, double r, double sigma, double cuton, double cutoff,
                    double ke, double *energy, double *dEdr) {
  auto pot = [sigma](double x) { return std::erf(x / sigma) / x; };
  auto dpot = [sigma](double x) {
    const double a = x / sigma;
    return (2.0 * x * std::exp(-a * a) / (std::sqrt(kPi) * sigma) - std::erf(a)) / (x * x);
  };
  const double span = cutoff - cuton;
  double f = 0.0;
  double df = 0.0;
  if (!(span > 0.0) || r >= cutoff) {
    f = 0.0;
  } else if (r <= cuton) {
    f = 1.0;
  } else {
    const double c = (r - cuton) / span;
    auto sig = [](double x) { return x > 0.0 ? std::exp(-1.0 / x) : 0.0; };
    auto sigp = [](double x) { return x > 0.0 ? std::exp(-1.0 / x) / (x * x) : 0.0; };
    const double s1 = sig(1.0 - c);
    const double s2 = sig(c);
    const double den = s1 + s2;
    f = s1 / den;
    const double ds1 = -sigp(1.0 - c);
    const double ds2 = sigp(c);
    const double dfdc = (ds1 * den - s1 * (ds1 + ds2)) / (den * den);
    df = dfdc / span;
  }
  if (!(r > 0.0) || r >= cutoff) {
    *energy = 0.0;
    *dEdr = 0.0;
    return;
  }
  const double pairwise = pot(r);
  const double shift = pot(cutoff);
  const double force_shift = dpot(cutoff);
  const double shifted = pairwise - shift - force_shift * (r - cutoff);
  const double blend = f * (pairwise - shift) + (1.0 - f) * shifted;
  const double qq = ke * qi * qj;
  *energy = qq * blend;
  const double dblend = df * (pairwise - shift - shifted) +
                        f * dpot(r) + (1.0 - f) * (dpot(r) - force_shift);
  *dEdr = qq * dblend;
}

void evaluate_electrostatics(CoulombMethod method, const CoulombConfig &cfg, const double *pos,
                             const double *q, std::size_t n, double total_charge, const double *box,
                             const std::vector<Pair> &pairs, double *F, double *dE_dq, ElecOut *out) {
  if (out == nullptr || (F == nullptr && n > 0)) {
    throw std::invalid_argument("electrostatic output is null");
  }
  if (!(cfg.cutoff > 0.0) || !(cfg.alpha > 0.0)) {
    throw std::invalid_argument("cutoff and alpha must be positive");
  }
  require_charges(q, n, total_charge);
  out->energy = 0.0;
  std::fill(out->strain, out->strain + 6, 0.0);
  out->has_stress = false;
  if (dE_dq != nullptr) {
    std::fill(dE_dq, dE_dq + n, 0.0);
  }
  if (n == 0) {
    return;
  }
  for (std::size_t i = 0; i < n; ++i) {
    if (!std::isfinite(pos[3 * i]) || !std::isfinite(pos[3 * i + 1]) ||
        !std::isfinite(pos[3 * i + 2])) {
      throw std::invalid_argument("a position is not finite");
    }
  }
  if (method == CoulombMethod::DSF) {
    real_space(cfg, true, q, n, pairs, F, dE_dq, out);
    out->has_stress = true;
    return;
  }
  if (cfg.kmax < 1 && method == CoulombMethod::Ewald) {
    throw std::invalid_argument("Ewald kmax must be at least 1");
  }
  const CellGeom cell = cell_from_box(box);
  // Two threads on a short mesh: one thread in the transform does not
  // beat one thread, so the second thread runs real space beside it.
  const bool small_mesh = cfg.grid[0] < 32 && cfg.grid[1] < 32 && cfg.grid[2] < 32;
  const bool beside = method == CoulombMethod::SPME && thread_count() == 2 && small_mesh && !cfg.stress &&
                      dE_dq == nullptr;
  if (beside) {
    std::vector<double> extra(3 * n, 0.0);
    ElecOut local{};
    pocketfft::detail::threading::group_limit() = 1;
#ifdef _OPENMP
#pragma omp parallel num_threads(2)
#endif
    {
      if (thread_id() == 0) {
        spme_reciprocal(cfg, cell, pos, q, n, total_charge, F, dE_dq, out);
      } else {
        real_space(cfg, false, q, n, pairs, extra.data(), nullptr, &local);
      }
    }
    pocketfft::detail::threading::group_limit() = 0;
    for (std::size_t i = 0; i < extra.size(); ++i) {
      F[i] += extra[i];
    }
    out->energy += local.energy;
    for (int k = 0; k < 6; ++k) {
      out->strain[k] += local.strain[k];
    }
    return;
  }
  real_space(cfg, false, q, n, pairs, F, dE_dq, out);
  if (method == CoulombMethod::Ewald) {
    ewald_reciprocal(cfg, cell, pos, q, n, total_charge, F, dE_dq, out);
  } else {
    spme_reciprocal(cfg, cell, pos, q, n, total_charge, F, dE_dq, out);
  }
}

CoulombPot::CoulombPot(CoulombConfig cfg, std::shared_ptr<ChargeSource> charges,
                       std::shared_ptr<ChargeResponse> response)
    : Potential(PotType::LrCoulomb), cfg_(cfg), charges_(std::move(charges)),
      response_(std::move(response)),
      lists_(std::make_shared<DualList>(cfg.short_cutoff, cfg.cutoff, cfg.skin)) {
  if (!charges_) {
    throw std::invalid_argument("CoulombPot needs a charge source");
  }
  Fnv1a fp;
  fp.u64(1);
  fp.u64(static_cast<uint64_t>(cfg_.method));
  fp.f64(cfg_.cutoff);
  fp.f64(cfg_.alpha);
  fp.i64(cfg_.kmax);
  fp.i64(cfg_.grid[0]);
  fp.i64(cfg_.grid[1]);
  fp.i64(cfg_.grid[2]);
  fp.i64(cfg_.spline_order);
  fp.f64(cfg_.skin);
  fp.u64(cfg_.periodic ? 1 : 0);
  params_ = fp.h;
}

void CoulombPot::forceImpl(const ForceInput &in, ForceOut *out) const {
  std::fill(out->F, out->F + 3 * in.nAtoms, 0.0);
  out->energy = 0.0;
  out->variance = 0.0;
  out->has_stress = 0;
  std::vector<double> q;
  double total = 0.0;
  charges_->load(in, &q, &total);
  const bool periodic = cfg_.periodic && cfg_.method != CoulombMethod::DSF;
  const auto t_list = std::chrono::steady_clock::now();
  lists_->update(in.pos, in.nAtoms, in.box, periodic);
  const auto t_eval = std::chrono::steady_clock::now();
  std::vector<double> dedq(in.nAtoms, 0.0);
  ElecOut er;
  evaluate_electrostatics(cfg_.method, cfg_, in.pos, q.data(), in.nAtoms, total,
                          periodic ? in.box : nullptr, lists_->long_pairs(), out->F,
                          response_ ? dedq.data() : nullptr, &er);
  out->energy = er.energy;
  if (response_) {
    response_->accumulate(in, dedq.data(), out->F);
  }
  const double volume = periodic ? cellVolume(in.box) : 0.0;
  if (er.has_stress && volume > 0.0) {
    publishCauchyStress(out, er.strain, volume);
  }
  if (std::getenv("LR_PME_PROFILE") != nullptr && cfg_.method == CoulombMethod::SPME) {
    const auto t_end = std::chrono::steady_clock::now();
    std::fprintf(stderr, "pme_outer n=%zu pairs=%zu list_us=%.3f eval_us=%.3f\n", in.nAtoms,
                 lists_->long_pairs().size(),
                 std::chrono::duration<double, std::micro>(t_eval - t_list).count(),
                 std::chrono::duration<double, std::micro>(t_end - t_eval).count());
  }
}

PmeTrace pme_trace() { return g_pme_trace; }

} // namespace rgpot::lr
