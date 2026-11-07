// MIT License
// Copyright 2023--present rgpot developers

#include "rgpot/LongRange/FastD3.hpp"

#include "rgpot/LongRange/Pairs.hpp"
#include "rgpot/units.hpp"

#include <algorithm>
#include <span>
#include <cmath>
#include <stdexcept>
#include <string>
#include <vector>

#ifdef _OPENMP
#include <omp.h>
#endif

namespace rgpot::lr {
namespace {

#include "rgpot/LongRange/D3Tables.inc"

constexpr double kBohr = units::BOHR_TO_ANGSTROM;
constexpr double kMctcBohr = 0.52917721067;
constexpr double kHartree = units::HARTREE_TO_EV;
constexpr int kRef = 7;

double r4r2_of(int z) {
  const double raw = kD3R4OverR2[z - 1];
  return std::sqrt(0.5 * raw * std::sqrt(static_cast<double>(z)));
}

double rcov_of(int z) {
  return (4.0 / 3.0) * kD3CovAng[z - 1] / kMctcBohr;
}

double vdw_of(int za, int zb) {
  int a = za;
  int b = zb;
  if (a < b) {
    std::swap(a, b);
  }
  const int ic = b + a * (a - 1) / 2;
  return kD3VdwAng[ic - 1] / kMctcBohr;
}

double ref_c6(int iref, int jref, int za, int zb) {
  int ir = iref;
  int jr = jref;
  int a = za;
  int b = zb;
  if (a <= b) {
    std::swap(ir, jr);
    std::swap(a, b);
  }
  const int ic = b + a * (a - 1) / 2;
  return kD3C6[ir + kRef * jr + kRef * kRef * (ic - 1)];
}

struct Weight {
  double gw[kRef]{};
  double dg[kRef]{};
  int nref = 0;
};

Weight weight_of(int z, double cn) {
  Weight w;
  w.nref = kD3Nref[z - 1];
  const double *ref = kD3RefCn + static_cast<std::size_t>(z - 1) * kRef;
  double dmin = 1e300;
  for (int a = 0; a < w.nref; ++a) {
    const double d = ref[a] - cn;
    dmin = std::min(dmin, d * d);
  }
  double norm = 0.0;
  for (int a = 0; a < w.nref; ++a) {
    const double d = ref[a] - cn;
    w.gw[a] = std::exp(-4.0 * (d * d - dmin));
    norm += w.gw[a];
  }
  double mean = 0.0;
  for (int a = 0; a < w.nref; ++a) {
    w.gw[a] /= norm;
    const double slope = 2.0 * 4.0 * (ref[a] - cn);
    mean += w.gw[a] * slope;
    w.dg[a] = slope;
  }
  for (int a = 0; a < w.nref; ++a) {
    w.dg[a] = w.gw[a] * (w.dg[a] - mean);
  }
  return w;
}

void c6_of(const Weight &wi, const Weight &wj, int zi, int zj, double *c6, double *dci,
           double *dcj) {
  double c = 0.0;
  double di = 0.0;
  double dj = 0.0;
  for (int a = 0; a < wi.nref; ++a) {
    for (int b = 0; b < wj.nref; ++b) {
      const double ref = ref_c6(a, b, zi, zj);
      c += wi.gw[a] * wj.gw[b] * ref;
      di += wi.dg[a] * wj.gw[b] * ref;
      dj += wi.gw[a] * wj.dg[b] * ref;
    }
  }
  *c6 = c;
  *dci = di;
  *dcj = dj;
}

struct Nb {
  int j = 0;
  double x = 0.0;
  double y = 0.0;
  double z = 0.0;
  double r = 0.0;
};

void smooth_cutoff(double r, double cutoff, double width, double *sw, double *dsw) {
  // Callers reject distances at or beyond the cutoff before this point.
  (void)r;
  (void)cutoff;
  (void)width;
  *sw = 1.0;
  *dsw = 0.0;
}

double triple_scale(int i, int j, int k) {
  if (i == j) {
    return i == k ? 1.0 / 6.0 : 0.5;
  }
  if (i != k && j != k) {
    return 1.0;
  }
  return 0.5;
}

int thread_count() {
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

} // namespace

bool d3bj_param(const char *functional, D3BjParam *out) {
  if (functional == nullptr) {
    return false;
  }
  const std::string name(functional);
  if (name == "pbe") {
    *out = D3BjParam{};
    return true;
  }
  if (name == "blyp") {
    *out = D3BjParam{1.0, 2.6996, 1.0, 0.4298, 4.2359, 14.0};
    return true;
  }
  if (name == "b3lyp") {
    *out = D3BjParam{1.0, 1.9889, 1.0, 0.3981, 4.4211, 14.0};
    return true;
  }
  return false;
}

void d3bj_evaluate(const double *pos, const int *z, std::size_t n, const double *box, bool periodic,
                   const std::vector<Pair> &pairs, const D3BjParam &param, double cut2, double cut3,
                   double cut_cn, double *energy, double *F, double strain[6]) {
  (void)pos;
  (void)box;
  (void)periodic;
  const double to_b = 1.0 / kBohr;
  const double reach = std::max(cut2, std::max(cut3, cut_cn));
  const int nt_build = thread_count();
  std::vector<int> deg(static_cast<std::size_t>(nt_build) * n, 0);
#ifdef _OPENMP
#pragma omp parallel
#endif
  {
    const int tid = thread_id();
    int *mine = deg.data() + static_cast<std::size_t>(tid) * n;
    const std::size_t begin = pairs.size() * static_cast<std::size_t>(tid) / static_cast<std::size_t>(nt_build);
    const std::size_t end = pairs.size() * static_cast<std::size_t>(tid + 1) / static_cast<std::size_t>(nt_build);
    for (std::size_t t = begin; t < end; ++t) {
      const Pair &pr = pairs[t];
      if (!(pr.r < reach)) {
        continue;
      }
      mine[pr.i] += 1;
      if (pr.i != pr.j) {
        mine[pr.j] += 1;
      }
    }
  }
  std::vector<int> off(n + 1, 0);
  std::vector<int> base(static_cast<std::size_t>(nt_build) * n, 0);
  for (std::size_t i = 0; i < n; ++i) {
    int cursor = off[i];
    for (int t = 0; t < nt_build; ++t) {
      base[static_cast<std::size_t>(t) * n + i] = cursor;
      cursor += deg[static_cast<std::size_t>(t) * n + i];
    }
    off[i + 1] = cursor;
  }
  std::vector<Nb> nb(static_cast<std::size_t>(off[n]));
#ifdef _OPENMP
#pragma omp parallel
#endif
  {
    const int tid = thread_id();
    int *slot = base.data() + static_cast<std::size_t>(tid) * n;
    const std::size_t begin = pairs.size() * static_cast<std::size_t>(tid) / static_cast<std::size_t>(nt_build);
    const std::size_t end = pairs.size() * static_cast<std::size_t>(tid + 1) / static_cast<std::size_t>(nt_build);
    for (std::size_t t = begin; t < end; ++t) {
      const Pair &pr = pairs[t];
      if (!(pr.r < reach)) {
        continue;
      }
      Nb a;
      a.j = pr.j;
      a.x = pr.x * to_b;
      a.y = pr.y * to_b;
      a.z = pr.z * to_b;
      a.r = pr.r * to_b;
      nb[static_cast<std::size_t>(slot[pr.i]++)] = a;
      if (pr.i != pr.j) {
        Nb b = a;
        b.j = pr.i;
        b.x = -a.x;
        b.y = -a.y;
        b.z = -a.z;
        nb[static_cast<std::size_t>(slot[pr.j]++)] = b;
      }
    }
  }
  const auto neigh = [&](std::size_t i) {
    return std::span<const Nb>(nb.data() + off[i], static_cast<std::size_t>(off[i + 1] - off[i]));
  };
  const double cut2_b = cut2 * to_b;
  const double cut3_b = cut3 * to_b;
  const double cutn_b = cut_cn * to_b;

  std::vector<double> cn(n, 0.0);
  for (std::size_t i = 0; i < n; ++i) {
    for (const Nb &a : neigh(i)) {
      if (a.j > static_cast<int>(i) || a.r >= cutn_b) {
        continue;
      }
      const double rc = rcov_of(z[i]) + rcov_of(z[a.j]);
      const double count = 1.0 / (1.0 + std::exp(-16.0 * (rc / a.r - 1.0)));
      cn[i] += count;
      if (a.j != static_cast<int>(i)) {
        cn[static_cast<std::size_t>(a.j)] += count;
      }
    }
  }
  std::vector<Weight> w(n);
  std::vector<double> r4(n);
  for (std::size_t i = 0; i < n; ++i) {
    if (z[i] < 1 || z[i] > 103) {
      throw std::invalid_argument("D3 atomic number is outside the reference table");
    }
    w[i] = weight_of(z[i], cn[i]);
    r4[i] = r4r2_of(z[i]);
  }

  const int nt = thread_count();
  std::vector<double> ene_th(static_cast<std::size_t>(nt), 0.0);
  std::vector<std::vector<double>> dedcn_th(static_cast<std::size_t>(nt), std::vector<double>(n, 0.0));
  std::vector<std::vector<double>> f_th(static_cast<std::size_t>(nt), std::vector<double>(3 * n, 0.0));
  std::vector<std::vector<double>> sig_th(static_cast<std::size_t>(nt), std::vector<double>(9, 0.0));

#ifdef _OPENMP
#pragma omp parallel
#endif
  {
    const int tid =
#ifdef _OPENMP
        omp_get_thread_num();
#else
        0;
#endif
    auto &dedcn = dedcn_th[static_cast<std::size_t>(tid)];
    auto &floc = f_th[static_cast<std::size_t>(tid)];
    auto &sig = sig_th[static_cast<std::size_t>(tid)];
    double ene = 0.0;
#ifdef _OPENMP
#pragma omp for schedule(static)
#endif
    for (std::size_t i = 0; i < n; ++i) {
      for (const Nb &a : neigh(i)) {
        if (a.j > static_cast<int>(i) || a.r >= cut2_b) {
          continue;
        }
        double c6 = 0.0;
        double dci = 0.0;
        double dcj = 0.0;
        c6_of(w[i], w[static_cast<std::size_t>(a.j)], z[i], z[a.j], &c6, &dci, &dcj);
        const double rrij = 3.0 * r4[i] * r4[static_cast<std::size_t>(a.j)];
        const double r0 = param.a1 * std::sqrt(rrij) + param.a2;
        const double r0_2 = r0 * r0;
        const double r0_6 = r0_2 * r0_2 * r0_2;
        const double r0_8 = r0_6 * r0_2;
        const double r2 = a.r * a.r;
        const double r4p = r2 * r2;
        const double t6 = 1.0 / (r4p * r2 + r0_6);
        const double t8 = 1.0 / (r4p * r4p + r0_8);
        double sw = 1.0;
        double dsw = 0.0;
        smooth_cutoff(a.r, cut2_b, 0.0, &sw, &dsw);
        const double d6 = -6.0 * r4p * t6 * t6;
        const double d8 = -8.0 * r4p * r2 * t8 * t8;
        const double edisp0 = param.s6 * t6 + param.s8 * rrij * t8;
        const double gdisp0 = param.s6 * d6 + param.s8 * rrij * d8;
        const double edisp = sw * edisp0;
        const double gdisp = sw * gdisp0 + dsw * edisp0 / a.r;
        const double dE = -c6 * edisp * 0.5;
        // vec in the Fortran kernel is r_i - r_j. Our Nb vector is r_j - r_i.
        const double gx = c6 * gdisp * a.x;
        const double gy = c6 * gdisp * a.y;
        const double gz = c6 * gdisp * a.z;
        ene += (a.j == static_cast<int>(i) ? dE : 2.0 * dE);
        dedcn[i] -= dci * edisp;
        floc[3 * i] += gx;
        floc[3 * i + 1] += gy;
        floc[3 * i + 2] += gz;
        if (a.j != static_cast<int>(i)) {
          dedcn[static_cast<std::size_t>(a.j)] -= dcj * edisp;
          floc[3 * a.j] -= gx;
          floc[3 * a.j + 1] -= gy;
          floc[3 * a.j + 2] -= gz;
          sig[0] += gx * (-a.x);
          sig[4] += gy * (-a.y);
          sig[8] += gz * (-a.z);
          sig[1] += gy * (-a.x);
          sig[3] += gx * (-a.y);
          sig[2] += gz * (-a.x);
          sig[6] += gx * (-a.z);
          sig[5] += gz * (-a.y);
          sig[7] += gy * (-a.z);
        }
      }
    }
#ifdef _OPENMP
#pragma omp for schedule(dynamic, 32)
#endif
    for (std::size_t i = 0; i < n; ++i) {
      for (const Nb &a : neigh(i)) {
        if (a.j > static_cast<int>(i) || a.r >= cut3_b) {
          continue;
        }
        double c6ij = 0.0;
        double dij_i = 0.0;
        double dij_j = 0.0;
        c6_of(w[i], w[static_cast<std::size_t>(a.j)], z[i], z[a.j], &c6ij, &dij_i, &dij_j);
        if (!(std::abs(c6ij) > 0.0)) {
          continue;
        }
        const double r0ij = (4.0 / 3.0) * vdw_of(z[a.j], z[i]);
        double swij = 1.0;
        double dswij = 0.0;
        smooth_cutoff(a.r, cut3_b, 0.0, &swij, &dswij);
        for (const Nb &b : neigh(i)) {
          if (b.j > a.j || b.r >= cut3_b) {
            continue;
          }
          double c6ik = 0.0;
          double dik_i = 0.0;
          double dik_k = 0.0;
          c6_of(w[i], w[static_cast<std::size_t>(b.j)], z[i], z[b.j], &c6ik, &dik_i, &dik_k);
          double c6jk = 0.0;
          double djk_j = 0.0;
          double djk_k = 0.0;
          c6_of(w[static_cast<std::size_t>(a.j)], w[static_cast<std::size_t>(b.j)], z[a.j], z[b.j],
                &c6jk, &djk_j, &djk_k);
          if (!(std::abs(c6ik) > 0.0) || !(std::abs(c6jk) > 0.0)) {
            continue;
          }
          const double vjkx = b.x - a.x;
          const double vjky = b.y - a.y;
          const double vjkz = b.z - a.z;
          const double r2jk = vjkx * vjkx + vjky * vjky + vjkz * vjkz;
          const double rjk = std::sqrt(r2jk);
          if (!(rjk > 0.0) || rjk >= cut3_b) {
            continue;
          }
          double swik = 1.0;
          double dswik = 0.0;
          double swjk = 1.0;
          double dswjk = 0.0;
          smooth_cutoff(b.r, cut3_b, 0.0, &swik, &dswik);
          smooth_cutoff(rjk, cut3_b, 0.0, &swjk, &dswjk);
          const double r2ij = a.r * a.r;
          const double r2ik = b.r * b.r;
          const double r2 = r2ij * r2ik * r2jk;
          const double r1 = std::sqrt(r2);
          const double r3 = r2 * r1;
          const double r5 = r3 * r2;
          const double r0ik = (4.0 / 3.0) * vdw_of(z[b.j], z[i]);
          const double r0jk = (4.0 / 3.0) * vdw_of(z[b.j], z[a.j]);
          const double r0 = r0ij * r0ik * r0jk;
          // s-dftd3 evaluates the three-body damping at alp + 2.
          const double alp_atm = param.alp + 2.0;
          const double alp3 = alp_atm / 3.0;
          const double fdmp = 1.0 / (1.0 + 6.0 * std::pow(r0 / r1, alp3));
          const double ang = 0.375 * (r2ij + r2jk - r2ik) * (r2ij - r2jk + r2ik) *
                                 (-r2ij + r2jk + r2ik) / r5 +
                             1.0 / r3;
          const double c9 = -param.s9 * std::sqrt(std::abs(c6ij * c6ik * c6jk));
          const double triple = triple_scale(static_cast<int>(i), a.j, b.j);
          const double sw = swij * swik * swjk;
          const double dE0 = ang * fdmp * c9;
          const double dE = dE0 * triple * sw;
          const double r0r = std::pow(r0 / r1, alp3);
          const double dfdmp = -2.0 * alp_atm * r0r * fdmp * fdmp;
          auto dang_ij = [&]() {
            return -0.375 *
                   (r2ij * r2ij * r2ij + r2ij * r2ij * (r2jk + r2ik) +
                    r2ij * (3.0 * r2jk * r2jk + 2.0 * r2jk * r2ik + 3.0 * r2ik * r2ik) -
                    5.0 * (r2jk - r2ik) * (r2jk - r2ik) * (r2jk + r2ik)) /
                   r5;
          };
          const double dij = dang_ij();
          const double dik =
              -0.375 *
              (r2ik * r2ik * r2ik + r2ik * r2ik * (r2jk + r2ij) +
               r2ik * (3.0 * r2jk * r2jk + 2.0 * r2jk * r2ij + 3.0 * r2ij * r2ij) -
               5.0 * (r2jk - r2ij) * (r2jk - r2ij) * (r2jk + r2ij)) /
              r5;
          const double djk =
              -0.375 *
              (r2jk * r2jk * r2jk + r2jk * r2jk * (r2ik + r2ij) +
               r2jk * (3.0 * r2ik * r2ik + 2.0 * r2ik * r2ij + 3.0 * r2ij * r2ij) -
               5.0 * (r2ik - r2ij) * (r2ik - r2ij) * (r2ik + r2ij)) /
              r5;
          // Fortran vij is r_j - r_i, which is our Nb vector.
          const double scale_ij = sw * c9 * (-dij * fdmp + ang * dfdmp) / r2ij;
          const double gijx = scale_ij * a.x - dE0 * dswij / a.r * swik * swjk * a.x;
          const double gijy = scale_ij * a.y - dE0 * dswij / a.r * swik * swjk * a.y;
          const double gijz = scale_ij * a.z - dE0 * dswij / a.r * swik * swjk * a.z;
          const double scale_ik = sw * c9 * (-dik * fdmp + ang * dfdmp) / r2ik;
          const double gikx = scale_ik * b.x - dE0 * dswik / b.r * swij * swjk * b.x;
          const double giky = scale_ik * b.y - dE0 * dswik / b.r * swij * swjk * b.y;
          const double gikz = scale_ik * b.z - dE0 * dswik / b.r * swij * swjk * b.z;
          const double scale_jk = sw * c9 * (-djk * fdmp + ang * dfdmp) / r2jk;
          const double gjkx = scale_jk * (b.x - a.x) - dE0 * dswjk / rjk * swij * swik * (b.x - a.x);
          const double gjky = scale_jk * (b.y - a.y) - dE0 * dswjk / rjk * swij * swik * (b.y - a.y);
          const double gjkz = scale_jk * (b.z - a.z) - dE0 * dswjk / rjk * swij * swik * (b.z - a.z);
          ene -= dE;
          floc[3 * i] -= (gijx + gikx) * triple;
          floc[3 * i + 1] -= (gijy + giky) * triple;
          floc[3 * i + 2] -= (gijz + gikz) * triple;
          floc[3 * a.j] += (gijx - gjkx) * triple;
          floc[3 * a.j + 1] += (gijy - gjky) * triple;
          floc[3 * a.j + 2] += (gijz - gjkz) * triple;
          floc[3 * b.j] += (gikx + gjkx) * triple;
          floc[3 * b.j + 1] += (giky + gjky) * triple;
          floc[3 * b.j + 2] += (gikz + gjkz) * triple;
          dedcn[i] -= dE * 0.5 * (dij_i / c6ij + dik_i / c6ik);
          dedcn[static_cast<std::size_t>(a.j)] -= dE * 0.5 * (dij_j / c6ij + djk_j / c6jk);
          dedcn[static_cast<std::size_t>(b.j)] -= dE * 0.5 * (dik_k / c6ik + djk_k / c6jk);
        }
      }
    }
    ene_th[static_cast<std::size_t>(tid)] = ene;
  }

  double ene = 0.0;
  std::vector<double> dedcn(n, 0.0);
  const bool apply_cn_force = true;
  std::vector<double> grad(3 * n, 0.0);
  for (int t = 0; t < nt; ++t) {
    ene += ene_th[static_cast<std::size_t>(t)];
    for (std::size_t i = 0; i < n; ++i) {
      dedcn[i] += dedcn_th[static_cast<std::size_t>(t)][i];
    }
    for (std::size_t i = 0; i < grad.size(); ++i) {
      grad[i] += f_th[static_cast<std::size_t>(t)][i];
    }
  }
  for (std::size_t i = 0; i < n && apply_cn_force; ++i) {
    for (const Nb &a : neigh(i)) {
      if (a.j > static_cast<int>(i) || a.r >= cutn_b) {
        continue;
      }
      const double rc = rcov_of(z[i]) + rcov_of(z[a.j]);
      const double expterm = std::exp(-16.0 * (rc / a.r - 1.0));
      const double cf = 1.0 / (1.0 + expterm);
      const double dcf = -cf * (1.0 - cf) * 16.0 * rc / (a.r * a.r);
      const double cx = dcf * a.x / a.r;
      const double cy = dcf * a.y / a.r;
      const double cz = dcf * a.z / a.r;
      const double couple = dedcn[i] + dedcn[static_cast<std::size_t>(a.j)];
      grad[3 * i] -= cx * couple;
      grad[3 * i + 1] -= cy * couple;
      grad[3 * i + 2] -= cz * couple;
      if (a.j != static_cast<int>(i)) {
        grad[3 * a.j] += cx * couple;
        grad[3 * a.j + 1] += cy * couple;
        grad[3 * a.j + 2] += cz * couple;
      }
    }
  }
  std::vector<double> sig(9, 0.0);
  for (int t = 0; t < nt; ++t) {
    for (int c = 0; c < 9; ++c) {
      sig[static_cast<std::size_t>(c)] += sig_th[static_cast<std::size_t>(t)][static_cast<std::size_t>(c)];
    }
  }
  *energy = ene * kHartree;
  if (F != nullptr) {
    for (std::size_t i = 0; i < grad.size(); ++i) {
      F[i] += grad[i] * units::NEG_GRAD_TO_FORCE;
    }
  }
  strain[0] = sig[0] * kHartree;
  strain[1] = sig[4] * kHartree;
  strain[2] = sig[8] * kHartree;
  strain[3] = sig[5] * kHartree;
  strain[4] = sig[2] * kHartree;
  strain[5] = sig[1] * kHartree;
}

} // namespace rgpot::lr
