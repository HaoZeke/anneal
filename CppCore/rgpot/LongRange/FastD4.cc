// MIT License
// Copyright 2023--present rgpot developers

#include "rgpot/LongRange/FastD4.hpp"
#ifndef POCKETFFT_CACHE_SIZE
#define POCKETFFT_CACHE_SIZE 16
#endif
#include "rgpot/LongRange/pocketfft.hpp"

#include "rgpot/LongRange/Lattice.hpp"
#include "rgpot/LongRange/Team.hpp"
#include "rgpot/units.hpp"

#include <algorithm>
#include <cmath>
#include <complex>
#include <cstdlib>
#include <numbers>
#include <span>
#include <stdexcept>
#include <string>
#include <vector>

#ifdef _OPENMP
#include <omp.h>
#endif

namespace rgpot::lr {
namespace {

// When a worker is building one column of the dense charge matrix it
// must not start another OpenMP team inside the transform.
thread_local int g_fft_team = 0;

constexpr double kPi = std::numbers::pi;
constexpr double kBohr = units::BOHR_TO_ANGSTROM;
constexpr double kHartree = units::HARTREE_TO_EV;
constexpr double kSqrtPi = 1.7724538509055160273;
constexpr double kSqrt2Pi = 0.79788456080286535588; // sqrt(2/pi)
constexpr double kGa = 3.0;
constexpr double kGc = 2.0;
constexpr double kWf = 6.0;
constexpr double kCnK = 7.5;
constexpr double kK4 = 4.10451;
constexpr double kK5 = 19.08857;
constexpr double kK6 = 2.0 * 11.28174 * 11.28174;
constexpr double kEeqCut = 25.0; // Bohr
constexpr double kCnMax = 8.0;
constexpr double kReg = 1.0e-14;

#include "rgpot/LongRange/D4Tables.inc"

struct Nb {
  int j = 0;
  double x = 0.0;
  double y = 0.0;
  double z = 0.0;
  double r = 0.0;
};

struct Gw {
  double w[7]{};
  double dn[7]{};
  double dq[7]{};
};

int elem(int z) {
  if (z < 1 || z > 103) {
    throw std::invalid_argument("D4 atomic number is outside the EEQ table");
  }
  return z - 1;
}

double zeta_of(double a, double c, double qref, double qmod) {
  if (qmod < 0.0) {
    return std::exp(a);
  }
  return std::exp(a * (1.0 - std::exp(c * (1.0 - qref / qmod))));
}

double dzeta_of(double a, double c, double qref, double qmod) {
  if (qmod < 0.0) {
    return 0.0;
  }
  const double z = zeta_of(a, c, qref, qmod);
  return -a * c * std::exp(c * (1.0 - qref / qmod)) * z * qref / (qmod * qmod);
}

double ref_c6(int za, int zb, int ia, int ib) {
  const double *aa = kD4Alpha + (static_cast<std::size_t>(za) * 7 + static_cast<std::size_t>(ia)) * 23;
  const double *bb = kD4Alpha + (static_cast<std::size_t>(zb) * 7 + static_cast<std::size_t>(ib)) * 23;
  double s = 0.0;
  for (int f = 0; f < 23; ++f) {
    s += aa[f] * bb[f] * kTrapzW[f];
  }
  return 3.0 / kPi * s;
}

void weights_of(int z, double cn, double q, Gw *g) {
  const int iz = elem(z);
  const int nref = kD4Nref[iz];
  const double zi = kD4Zeff[iz];
  const double gi = kD4Hard[iz] * kGc;
  double norm = 0.0;
  double dnorm = 0.0;
  for (int ir = 0; ir < nref; ++ir) {
    const double cnref = kD4RefCn[iz * 7 + ir];
    const int ng = kD4Ngw[iz * 7 + ir];
    for (int ig = 1; ig <= ng; ++ig) {
      const double wf = static_cast<double>(ig) * kWf;
      const double gw = std::exp(-wf * (cn - cnref) * (cn - cnref));
      norm += gw;
      dnorm += 2.0 * wf * (cnref - cn) * gw;
    }
  }
  const double tiny = std::sqrt(1e-300);
  if (std::abs(norm) > tiny) {
    norm = 1.0 / norm;
  } else {
    norm = 0.0;
  }
  double maxcn = 0.0;
  for (int ir = 0; ir < nref; ++ir) {
    maxcn = std::max(maxcn, kD4RefCn[iz * 7 + ir]);
  }
  for (int ir = 0; ir < 7; ++ir) {
    g->w[ir] = g->dn[ir] = g->dq[ir] = 0.0;
  }
  for (int ir = 0; ir < nref; ++ir) {
    const double cnref = kD4RefCn[iz * 7 + ir];
    const int ng = kD4Ngw[iz * 7 + ir];
    double expw = 0.0;
    double expd = 0.0;
    for (int ig = 1; ig <= ng; ++ig) {
      const double wf = static_cast<double>(ig) * kWf;
      const double gw = std::exp(-wf * (cn - cnref) * (cn - cnref));
      expw += gw;
      expd += 2.0 * wf * (cnref - cn) * gw;
    }
    double gwk = expw * norm;
    if (!std::isfinite(gwk) || norm == 0.0) {
      gwk = std::abs(maxcn - cnref) < 1e-12 ? 1.0 : 0.0;
    }
    double dg = norm * (expd - expw * dnorm * norm);
    if (!std::isfinite(dg) || norm == 0.0) {
      dg = 0.0;
    }
    const double qref = kD4RefQ[iz * 7 + ir] + zi;
    const double zscale = zeta_of(kGa, gi, qref, q + zi);
    g->w[ir] = gwk * zscale;
    g->dn[ir] = dg * zscale;
    g->dq[ir] = gwk * dzeta_of(kGa, gi, qref, q + zi);
  }
}

void c6_of(const Gw &a, const Gw &b, int za, int zb, double *c6, double *dni, double *dnj, double *dqi,
           double *dqj) {
  const int ia = elem(za);
  const int ib = elem(zb);
  double s = 0.0;
  double sni = 0.0;
  double snj = 0.0;
  double sqi = 0.0;
  double sqj = 0.0;
  for (int i = 0; i < kD4Nref[ia]; ++i) {
    for (int j = 0; j < kD4Nref[ib]; ++j) {
      const double r = ref_c6(ia, ib, i, j);
      s += a.w[i] * b.w[j] * r;
      sni += a.dn[i] * b.w[j] * r;
      snj += a.w[i] * b.dn[j] * r;
      sqi += a.dq[i] * b.w[j] * r;
      sqj += a.w[i] * b.dq[j] * r;
    }
  }
  *c6 = s;
  *dni = sni;
  *dnj = snj;
  *dqi = sqi;
  *dqj = sqj;
}

double en_factor(int za, int zb) {
  const double d = std::abs(kD4En[elem(za)] - kD4En[elem(zb)]);
  return kK4 * std::exp(-(d + kK5) * (d + kK5) / kK6);
}

double erf_count(double r, double rc) {
  const double arg = -kCnK * (r - rc) / rc;
  return 0.5 * (1.0 + std::erf(arg));
}

double erf_dcount(double r, double rc) {
  const double ex = kCnK * (r - rc) / rc;
  return -(kCnK * std::exp(-ex * ex)) / (kSqrtPi * rc);
}

double log_cut(double cn) {
  return std::log(1.0 + std::exp(kCnMax)) - std::log(1.0 + std::exp(kCnMax - cn));
}

double dlog_cut(double cn) { return std::exp(kCnMax) / (std::exp(kCnMax) + std::exp(cn)); }

struct Shell {
  std::vector<int> off;
  std::vector<Nb> nb;
  std::span<const Nb> of(std::size_t i) const {
    return std::span<const Nb>(nb.data() + off[i], static_cast<std::size_t>(off[i + 1] - off[i]));
  }
};

Shell shell_of(const std::vector<Pair> &pairs, std::size_t n) {
  Shell s;
  s.off.assign(n + 1, 0);
  const double to_b = 1.0 / kBohr;
  for (const Pair &pr : pairs) {
    s.off[static_cast<std::size_t>(pr.i) + 1] += 1;
    if (pr.i != pr.j) {
      s.off[static_cast<std::size_t>(pr.j) + 1] += 1;
    }
  }
  for (std::size_t i = 0; i < n; ++i) {
    s.off[i + 1] += s.off[i];
  }
  s.nb.resize(static_cast<std::size_t>(s.off[n]));
  std::vector<int> cur(s.off.begin(), s.off.end() - 1);
  for (const Pair &pr : pairs) {
    Nb a;
    a.j = pr.j;
    a.x = pr.x * to_b;
    a.y = pr.y * to_b;
    a.z = pr.z * to_b;
    a.r = pr.r * to_b;
    s.nb[static_cast<std::size_t>(cur[static_cast<std::size_t>(pr.i)]++)] = a;
    if (pr.i != pr.j) {
      Nb b = a;
      b.j = pr.i;
      b.x = -a.x;
      b.y = -a.y;
      b.z = -a.z;
      s.nb[static_cast<std::size_t>(cur[static_cast<std::size_t>(pr.j)]++)] = b;
    }
  }
  return s;
}

void accumulate_cn(const Shell &sh, const int *z, std::size_t n, double cut_b, bool electroneg,
                   std::vector<double> *cn) {
  cn->assign(n, 0.0);
  for (std::size_t i = 0; i < n; ++i) {
    for (const Nb &a : sh.of(i)) {
      if (a.j > static_cast<int>(i) || a.r >= cut_b || !(a.r > 1e-12)) {
        continue;
      }
      const double rc = kD4Rcov[elem(z[i])] + kD4Rcov[elem(z[a.j])];
      double count = erf_count(a.r, rc);
      if (electroneg) {
        count *= en_factor(z[i], z[a.j]);
      }
      (*cn)[i] += count;
      if (a.j != static_cast<int>(i)) {
        (*cn)[static_cast<std::size_t>(a.j)] += count;
      }
    }
  }
}

void mat_inv(const double *box, double inv[9], double *vol);
double bspline4(double x);

void fft3_lines(std::vector<std::complex<double>> &g, int ng, bool inverse) {
  int team = 1;
#ifdef _OPENMP
  team = std::max(1, omp_get_max_threads());
#endif
  // Column assembly of the dense charge system runs one serial transform
  // per thread. The outer parallel loop is the team.
  if (g_fft_team > 0) {
    team = g_fft_team;
  }
  const int cap = axis_workers(ng, team);
  const double ntot = static_cast<double>(ng) * static_cast<double>(ng) * static_cast<double>(ng);
  const double scale = inverse ? 1.0 / ntot : 1.0;
  const auto step = static_cast<std::ptrdiff_t>(sizeof(std::complex<double>));
  auto axis = [&](int which, int tid, double fct) {
    const int along = which == 0 ? ng : ng;
    const int mine0 = tid * along / cap;
    const int mine1 = (tid + 1) * along / cap;
    if (mine1 <= mine0) {
      return;
    }
    std::complex<double> *base = g.data();
    pocketfft::shape_t shape;
    pocketfft::stride_t stride{step * ng * ng, step * ng, step};
    if (which == 0) {
      base += static_cast<std::size_t>(mine0) * static_cast<std::size_t>(ng);
      shape = {static_cast<std::size_t>(ng), static_cast<std::size_t>(mine1 - mine0), static_cast<std::size_t>(ng)};
    } else {
      base += static_cast<std::size_t>(mine0) * static_cast<std::size_t>(ng) * static_cast<std::size_t>(ng);
      shape = {static_cast<std::size_t>(mine1 - mine0), static_cast<std::size_t>(ng), static_cast<std::size_t>(ng)};
    }
    pocketfft::shape_t axes{static_cast<std::size_t>(which)};
    pocketfft::c2c(shape, stride, stride, axes, !inverse, base, base, fct, 1);
  };
  if (cap <= 1) {
    axis(2, 0, 1.0);
    axis(1, 0, 1.0);
    axis(0, 0, scale);
    return;
  }
#ifdef _OPENMP
#pragma omp parallel num_threads(cap)
#endif
  {
    const int tid =
#ifdef _OPENMP
        omp_get_thread_num();
#else
        0;
#endif
    axis(2, tid, 1.0);
#ifdef _OPENMP
#pragma omp barrier
#endif
    axis(1, tid, 1.0);
#ifdef _OPENMP
#pragma omp barrier
#endif
    axis(0, tid, scale);
  }
}

double bspline_order(int order, double x) {
  if (order == 1) {
    return (x >= 0.0 && x < 1.0) ? 1.0 : 0.0;
  }
  const double inv = 1.0 / static_cast<double>(order - 1);
  return x * inv * bspline_order(order - 1, x) + (order - x) * inv * bspline_order(order - 1, x - 1.0);
}

void bmod_axis(int n, std::vector<double> *mod) {
  constexpr int order = 4;
  mod->assign(static_cast<std::size_t>(n), 0.0);
  for (int m = 0; m < n; ++m) {
    std::complex<double> sum(0.0, 0.0);
    for (int k = 0; k < order - 1; ++k) {
      const double ang = 2.0 * kPi * static_cast<double>(m * k) / static_cast<double>(n);
      sum += bspline_order(order, static_cast<double>(k + 1)) * std::complex<double>(std::cos(ang), std::sin(ang));
    }
    const double angb = 2.0 * kPi * static_cast<double>((order - 1) * m) / static_cast<double>(n);
    const std::complex<double> num(std::cos(angb), std::sin(angb));
    const std::complex<double> b = num / sum;
    (*mod)[static_cast<std::size_t>(m)] = std::norm(b);
  }
}

double bspline4(double x) {
  // Order-4 cardinal B-spline on x in [0, 4).
  if (x <= 0.0 || x >= 4.0) {
    return 0.0;
  }
  if (x < 1.0) {
    return x * x * x / 6.0;
  }
  if (x < 2.0) {
    const double t = x - 1.0;
    return (1.0 + 3.0 * t + 3.0 * t * t - 3.0 * t * t * t) / 6.0;
  }
  if (x < 3.0) {
    const double t = x - 2.0;
    return (4.0 - 6.0 * t * t + 3.0 * t * t * t) / 6.0;
  }
  const double t = 4.0 - x;
  return t * t * t / 6.0;
}

struct Operator {
  const double *pos = nullptr; // Bohr
  const double *box = nullptr; // Angstrom
  const int *z = nullptr;
  const Shell *real = nullptr;
  std::vector<double> diag;
  std::vector<double> recip_kernel;
  double alpha = 0.2;
  bool periodic = false;
  std::size_t n = 0;
  static constexpr int kGrid = 32;
  mutable std::vector<std::complex<double>> grid;
  mutable std::vector<double> acc_buf;
  mutable std::vector<int> sp_at;
  mutable std::vector<double> sp_w;
  mutable std::vector<double> ker;
  mutable bool ker_ready = false;

  void prepare_ker() const {
    if (ker_ready) {
      return;
    }
    ker.assign(real->nb.size(), 0.0);
    for (std::size_t i = 0; i < n; ++i) {
      const double radi = kD4Rad[elem(z[i])];
      for (std::size_t k = static_cast<std::size_t>(real->off[i]);
           k < static_cast<std::size_t>(real->off[i + 1]); ++k) {
        const Nb &a = real->nb[k];
        if (a.j > static_cast<int>(i) || !(a.r > 1e-12)) {
          continue;
        }
        const double radj = kD4Rad[elem(z[a.j])];
        const double gam = 1.0 / std::sqrt(radi * radi + radj * radj);
        double value = std::erf(gam * a.r) / a.r;
        if (periodic) {
          value -= std::erf(alpha * a.r) / a.r;
        }
        ker[k] = value;
      }
    }
    ker_ready = true;
  }

  void prepare_spread() const {
    if (sp_at.size() == n * 64) {
      return;
    }
    sp_at.assign(n * 64, 0);
    sp_w.assign(n * 64, 0.0);
    double inv[9];
    double vol = 0.0;
    mat_inv(box, inv, &vol);
    constexpr int ng = kGrid;
    auto index = [](int x, int y, int z) {
      return x + ng * (y + ng * z);
    };
    for (std::size_t i = 0; i < n; ++i) {
      const double rx = pos[3 * i] * kBohr;
      const double ry = pos[3 * i + 1] * kBohr;
      const double rz = pos[3 * i + 2] * kBohr;
      double fx = inv[0] * rx + inv[1] * ry + inv[2] * rz;
      double fy = inv[3] * rx + inv[4] * ry + inv[5] * rz;
      double fz = inv[6] * rx + inv[7] * ry + inv[8] * rz;
      fx -= std::floor(fx);
      fy -= std::floor(fy);
      fz -= std::floor(fz);
      const double ux = fx * ng;
      const double uy = fy * ng;
      const double uz = fz * ng;
      const int kx = static_cast<int>(std::floor(ux)) - 3;
      const int ky = static_cast<int>(std::floor(uy)) - 3;
      const int kz = static_cast<int>(std::floor(uz)) - 3;
      std::size_t slot = i * 64;
      for (int iz = 0; iz < 4; ++iz) {
        int wz = (kz + iz) % ng;
        if (wz < 0) {
          wz += ng;
        }
        const double tz = bspline4(uz - static_cast<double>(kz + iz));
        for (int iy = 0; iy < 4; ++iy) {
          int wy = (ky + iy) % ng;
          if (wy < 0) {
            wy += ng;
          }
          const double ty = bspline4(uy - static_cast<double>(ky + iy));
          for (int ix = 0; ix < 4; ++ix) {
            int wx = (kx + ix) % ng;
            if (wx < 0) {
              wx += ng;
            }
            sp_at[slot] = index(wx, wy, wz);
            sp_w[slot] = bspline4(ux - static_cast<double>(kx + ix)) * ty * tz;
            ++slot;
          }
        }
      }
    }
  }

  void prepare_kernel() {
    recip_kernel.assign(static_cast<std::size_t>(kGrid) * kGrid * kGrid, 0.0);
    if (!periodic || box == nullptr) {
      return;
    }
    double inv[9];
    double vol = 0.0;
    mat_inv(box, inv, &vol);
    const double vol_b = vol / (kBohr * kBohr * kBohr);
    const double fac = 4.0 * kPi / vol_b;
    std::vector<double> bmx, bmy, bmz;
    bmod_axis(kGrid, &bmx);
    bmod_axis(kGrid, &bmy);
    bmod_axis(kGrid, &bmz);
    for (int iz = 0; iz < kGrid; ++iz) {
      for (int iy = 0; iy < kGrid; ++iy) {
        for (int ix = 0; ix < kGrid; ++ix) {
          const int sx = ix <= kGrid / 2 ? ix : ix - kGrid;
          const int sy = iy <= kGrid / 2 ? iy : iy - kGrid;
          const int sz = iz <= kGrid / 2 ? iz : iz - kGrid;
          if (sx == 0 && sy == 0 && sz == 0) {
            continue;
          }
          const double gx = (inv[0] * sx + inv[3] * sy + inv[6] * sz) * kBohr * 2.0 * kPi;
          const double gy = (inv[1] * sx + inv[4] * sy + inv[7] * sz) * kBohr * 2.0 * kPi;
          const double gz = (inv[2] * sx + inv[5] * sy + inv[8] * sz) * kBohr * 2.0 * kPi;
          const double g2 = gx * gx + gy * gy + gz * gz;
          const double bden = bmx[static_cast<std::size_t>(ix)] * bmy[static_cast<std::size_t>(iy)] *
                              bmz[static_cast<std::size_t>(iz)];
          recip_kernel[static_cast<std::size_t>(ix + kGrid * (iy + kGrid * iz))] =
              fac * std::exp(-0.25 * g2 / (alpha * alpha)) / g2 / std::max(bden, 1e-30);
        }
      }
    }
  }

  std::vector<std::complex<double>> &mesh_buf() const {
    // One buffer per worker so column assembly can run side by side.
    thread_local std::vector<std::complex<double>> local;
    return local;
  }

  void add_reciprocal(const double *q, double *y) const {
    if (!periodic || recip_kernel.empty()) {
      return;
    }
    constexpr int ng = kGrid;
    const std::size_t ntot = static_cast<std::size_t>(ng) * ng * ng;
    auto &mesh = mesh_buf();
    if (mesh.size() != ntot) {
      mesh.assign(ntot, std::complex<double>(0.0, 0.0));
    } else {
      std::fill(mesh.begin(), mesh.end(), std::complex<double>(0.0, 0.0));
    }
    prepare_spread();
    for (std::size_t i = 0; i < n; ++i) {
      const double qi = q[i];
      const std::size_t base = i * 64;
      for (int k = 0; k < 64; ++k) {
        mesh[static_cast<std::size_t>(sp_at[base + static_cast<std::size_t>(k)])] +=
            qi * sp_w[base + static_cast<std::size_t>(k)];
      }
    }
    fft3_lines(mesh, ng, false);
    for (std::size_t m = 0; m < ntot; ++m) {
      mesh[m] *= recip_kernel[m];
    }
    fft3_lines(mesh, ng, true);
    const double nscale = static_cast<double>(ntot);
    for (std::size_t i = 0; i < n; ++i) {
      const std::size_t base = i * 64;
      double acc = 0.0;
      for (int k = 0; k < 64; ++k) {
        acc += mesh[static_cast<std::size_t>(sp_at[base + static_cast<std::size_t>(k)])].real() *
               sp_w[base + static_cast<std::size_t>(k)];
      }
      // The inverse FFT already divides by ng^3. The forward transform of a
      // cosine series needs that factor back so A matches sum_G w cos.
      y[i] += acc * nscale;
    }
  }

  // One thread, no nested team. Column assembly calls this.
  void apply_one(const double *q, double *y) const {
    prepare_ker();
    for (std::size_t i = 0; i < n; ++i) {
      y[i] = diag[i] * q[i];
    }
    for (std::size_t i = 0; i < n; ++i) {
      const std::size_t begin = static_cast<std::size_t>(real->off[i]);
      const std::size_t end = static_cast<std::size_t>(real->off[i + 1]);
      for (std::size_t k = begin; k < end; ++k) {
        const Nb &a = real->nb[k];
        if (a.j > static_cast<int>(i) || !(a.r > 1e-12)) {
          continue;
        }
        const double coeff = ker[k];
        y[i] += coeff * q[a.j];
        if (a.j != static_cast<int>(i)) {
          y[a.j] += coeff * q[i];
        }
      }
    }
    add_reciprocal(q, y);
  }

  void apply(const double *q, double *y) const {
    std::fill(y, y + n, 0.0);
#ifdef _OPENMP
#pragma omp parallel for schedule(static)
#endif
    for (std::size_t i = 0; i < n; ++i) {
      y[i] = diag[i] * q[i];
    }
    // Pair kernel. Writes to both ends, so each thread keeps a private vector.
    const int nt =
#ifdef _OPENMP
        std::max(1, omp_get_max_threads());
#else
        1;
#endif
    if (acc_buf.size() != static_cast<std::size_t>(nt) * n) {
      acc_buf.assign(static_cast<std::size_t>(nt) * n, 0.0);
    } else {
      std::fill(acc_buf.begin(), acc_buf.end(), 0.0);
    }
    double *acc = acc_buf.data();
    prepare_ker();
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
      double *loc = acc + static_cast<std::size_t>(tid) * n;
#ifdef _OPENMP
#pragma omp for schedule(static)
#endif
      for (std::size_t i = 0; i < n; ++i) {
        const std::size_t begin = static_cast<std::size_t>(real->off[i]);
        const std::size_t end = static_cast<std::size_t>(real->off[i + 1]);
        for (std::size_t k = begin; k < end; ++k) {
          const Nb &a = real->nb[k];
          const double coeff = ker[k];
          if (a.j > static_cast<int>(i) || !(a.r > 1e-12)) {
            continue;
          }
          loc[i] += coeff * q[a.j];
          if (a.j != static_cast<int>(i)) {
            loc[a.j] += coeff * q[i];
          }
        }
      }
    }
    for (int t = 0; t < nt; ++t) {
      const double *loc = acc + static_cast<std::size_t>(t) * n;
      for (std::size_t i = 0; i < n; ++i) {
        y[i] += loc[i];
      }
    }
    add_reciprocal(q, y);
  }

  // Dense LU of A. N matvecs, then one factorization reused for every
  // right hand side. At N=128 that is cheaper than four conjugate gradients.
  mutable std::vector<double> fact;
  mutable std::vector<int> ipiv;
  mutable bool factored = false;

  void factorize() const {
    if (factored) {
      return;
    }
    prepare_ker();
    if (periodic) {
      prepare_spread();
    }
    fact.assign(n * n, 0.0);
    const int cols = static_cast<int>(n);
#ifdef _OPENMP
#pragma omp parallel for schedule(static)
#endif
    for (int j = 0; j < cols; ++j) {
      g_fft_team = 1;
      std::vector<double> basis(n, 0.0), column(n, 0.0);
      basis[static_cast<std::size_t>(j)] = 1.0;
      apply_one(basis.data(), column.data());
      for (std::size_t i = 0; i < n; ++i) {
        fact[i + n * static_cast<std::size_t>(j)] = column[i];
      }
      g_fft_team = 0;
    }
    ipiv.assign(n, 0);
    for (int k = 0; k < cols; ++k) {
      int piv = k;
      double best = std::abs(fact[static_cast<std::size_t>(k) + n * static_cast<std::size_t>(k)]);
      for (int i = k + 1; i < cols; ++i) {
        const double v = std::abs(fact[static_cast<std::size_t>(i) + n * static_cast<std::size_t>(k)]);
        if (v > best) {
          best = v;
          piv = i;
        }
      }
      ipiv[static_cast<std::size_t>(k)] = piv;
      if (!(best > 0.0)) {
        throw std::runtime_error("EEQ matrix is singular");
      }
      if (piv != k) {
        for (int j = 0; j < cols; ++j) {
          std::swap(fact[static_cast<std::size_t>(k) + n * static_cast<std::size_t>(j)],
                    fact[static_cast<std::size_t>(piv) + n * static_cast<std::size_t>(j)]);
        }
      }
      const double diag = fact[static_cast<std::size_t>(k) + n * static_cast<std::size_t>(k)];
      for (int i = k + 1; i < cols; ++i) {
        const double f = fact[static_cast<std::size_t>(i) + n * static_cast<std::size_t>(k)] / diag;
        fact[static_cast<std::size_t>(i) + n * static_cast<std::size_t>(k)] = f;
        for (int j = k + 1; j < cols; ++j) {
          fact[static_cast<std::size_t>(i) + n * static_cast<std::size_t>(j)] -=
              f * fact[static_cast<std::size_t>(k) + n * static_cast<std::size_t>(j)];
        }
      }
    }
    factored = true;
  }

  void solve_linear(const double *b, double *x) const {
    factorize();
    std::vector<double> y(b, b + n);
    const int cols = static_cast<int>(n);
    for (int k = 0; k < cols; ++k) {
      const int piv = ipiv[static_cast<std::size_t>(k)];
      if (piv != k) {
        std::swap(y[static_cast<std::size_t>(k)], y[static_cast<std::size_t>(piv)]);
      }
    }
    for (int i = 0; i < cols; ++i) {
      double s = y[static_cast<std::size_t>(i)];
      for (int k = 0; k < i; ++k) {
        s -= fact[static_cast<std::size_t>(i) + n * static_cast<std::size_t>(k)] * y[static_cast<std::size_t>(k)];
      }
      y[static_cast<std::size_t>(i)] = s;
    }
    for (int i = cols - 1; i >= 0; --i) {
      double s = y[static_cast<std::size_t>(i)];
      for (int k = i + 1; k < cols; ++k) {
        s -= fact[static_cast<std::size_t>(i) + n * static_cast<std::size_t>(k)] * y[static_cast<std::size_t>(k)];
      }
      y[static_cast<std::size_t>(i)] =
          s / fact[static_cast<std::size_t>(i) + n * static_cast<std::size_t>(i)];
    }
    std::copy(y.begin(), y.end(), x);
  }
};

double dot(const double *a, const double *b, std::size_t n) {
  // Serial so the conjugate-gradient solution does not depend on the thread count.
  double s = 0.0;
  for (std::size_t i = 0; i < n; ++i) {
    s += a[i] * b[i];
  }
  return s;
}

void cg(const Operator &op, const double *b, double *x, int maxit, double tol) {
  const std::size_t n = op.n;
  std::vector<double> r(n), p(n), ap(n);
  op.apply(x, ap.data());
  for (std::size_t i = 0; i < n; ++i) {
    r[i] = b[i] - ap[i];
    p[i] = r[i];
  }
  double rr = dot(r.data(), r.data(), n);
  const double bnorm = std::sqrt(dot(b, b, n)) + 1e-30;
  for (int it = 0; it < maxit && std::sqrt(rr) > tol * bnorm; ++it) {
    op.apply(p.data(), ap.data());
    const double pap = dot(p.data(), ap.data(), n);
    if (!(std::abs(pap) > 1e-30)) {
      break;
    }
    const double alpha = rr / pap;
    for (std::size_t i = 0; i < n; ++i) {
      x[i] += alpha * p[i];
      r[i] -= alpha * ap[i];
    }
    const double rr2 = dot(r.data(), r.data(), n);
    const double beta = rr2 / rr;
    for (std::size_t i = 0; i < n; ++i) {
      p[i] = r[i] + beta * p[i];
    }
    rr = rr2;
  }
}

void solve_charges(const Operator &op, const double *xvec, double qtot, std::vector<double> *q) {
  const std::size_t n = op.n;
  std::vector<double> ones(n, 1.0), q0(n, 0.0), q1(n, 0.0);
  for (std::size_t i = 0; i < n; ++i) {
    q0[i] = xvec[i] / op.diag[i];
    q1[i] = 1.0 / op.diag[i];
  }
  // N^3 is a fraction of one matvec once N is this small. Above it the
  // conjugate gradient, which never forms A, is the faster solve.
  // The crossover is measured; LR_EEQ_DIRECT overrides it for that measurement.
  const std::size_t direct_n = []() {
    if (const char *env = std::getenv("LR_EEQ_DIRECT")) {
      return static_cast<std::size_t>(std::atoi(env));
    }
    return static_cast<std::size_t>(160);
  }();
  if (n <= direct_n) {
    op.solve_linear(xvec, q0.data());
    op.solve_linear(ones.data(), q1.data());
  } else {
    const int maxit = 80;
    cg(op, xvec, q0.data(), maxit, 1e-10);
    cg(op, ones.data(), q1.data(), maxit, 1e-10);
  }
  double s0 = 0.0;
  double s1 = 0.0;
  for (std::size_t i = 0; i < n; ++i) {
    s0 += q0[i];
    s1 += q1[i];
  }
  const double beta = (qtot - s0) / s1;
  q->resize(n);
  for (std::size_t i = 0; i < n; ++i) {
    (*q)[i] = q0[i] + beta * q1[i];
  }
}

double ewald_alpha(const CellGeom &cell) {
  const double a0 = std::sqrt(cell.H[0][0] * cell.H[0][0] + cell.H[0][1] * cell.H[0][1] +
                              cell.H[0][2] * cell.H[0][2]);
  const double a1 = std::sqrt(cell.H[1][0] * cell.H[1][0] + cell.H[1][1] * cell.H[1][1] +
                              cell.H[1][2] * cell.H[1][2]);
  const double a2 = std::sqrt(cell.H[2][0] * cell.H[2][0] + cell.H[2][1] * cell.H[2][1] +
                              cell.H[2][2] * cell.H[2][2]);
  const double dlen = std::min(a0, std::min(a1, a2)) / kBohr;
  const double b0 = std::sqrt(cell.recip[0][0] * cell.recip[0][0] + cell.recip[0][1] * cell.recip[0][1] +
                              cell.recip[0][2] * cell.recip[0][2]);
  const double b1 = std::sqrt(cell.recip[1][0] * cell.recip[1][0] + cell.recip[1][1] * cell.recip[1][1] +
                              cell.recip[1][2] * cell.recip[1][2]);
  const double b2 = std::sqrt(cell.recip[2][0] * cell.recip[2][0] + cell.recip[2][1] * cell.recip[2][1] +
                              cell.recip[2][2] * cell.recip[2][2]);
  const double rlen = 2.0 * kPi * std::min(b0, std::min(b1, b2)) * kBohr;
  const double vol = cell.volume / (kBohr * kBohr * kBohr);
  const double tol = 1.4901161193847656e-8;
  auto rec = [&](double alpha, double g) {
    return 4.0 * kPi * std::exp(-0.25 * g * g / (alpha * alpha)) / (vol * g * g);
  };
  auto dir = [&](double alpha, double r) { return std::erfc(alpha * r) / r; };
  auto diff = [&](double alpha) {
    return (rec(alpha, 4.0 * rlen) - rec(alpha, 5.0 * rlen)) -
           (dir(alpha, 2.0 * dlen) - dir(alpha, 3.0 * dlen));
  };
  double alpha = 1.0e-8;
  double gap = diff(alpha);
  int stat = 0;
  while (gap < -tol && alpha < 1e300) {
    alpha *= 2.0;
    gap = diff(alpha);
  }
  if (!(alpha < 1e300)) {
    stat = 1;
  } else if (alpha == 1.0e-8) {
    stat = 2;
  }
  double alpl = 0.5 * alpha;
  if (stat == 0) {
    while (gap < tol && alpha < 1e300) {
      alpha *= 2.0;
      gap = diff(alpha);
    }
    if (!(alpha < 1e300)) {
      stat = 3;
    }
  }
  if (stat == 0) {
    double alpr = alpha;
    alpha = 0.5 * (alpl + alpr);
    gap = diff(alpha);
    for (int it = 0; it <= 30 && std::abs(gap) > tol; ++it) {
      if (gap < 0.0) {
        alpl = alpha;
      } else {
        alpr = alpha;
      }
      alpha = 0.5 * (alpl + alpr);
      gap = diff(alpha);
      if (it == 30) {
        stat = 4;
      }
    }
  }
  if (stat != 0) {
    alpha = 0.25;
  }
  return alpha;
}

double ewald_pair(const double *ri, const double *rj, double gam, double alpha, const CellGeom &cell) {
  const double to_b = 1.0 / kBohr;
  double acc = 0.0;
  const double eps = 1.4901161193847656e-8;
  const double vol = cell.volume / (kBohr * kBohr * kBohr);
  double base[3] = {(rj[0] - ri[0]) * to_b, (rj[1] - ri[1]) * to_b, (rj[2] - ri[2]) * to_b};
  for (int ia = -2; ia <= 2; ++ia) {
    for (int ib = -2; ib <= 2; ++ib) {
      for (int ic = -2; ic <= 2; ++ic) {
        double v[3];
        for (int c = 0; c < 3; ++c) {
          v[c] = base[c] + (ia * cell.H[0][c] + ib * cell.H[1][c] + ic * cell.H[2][c]) * to_b;
        }
        const double r = std::sqrt(v[0] * v[0] + v[1] * v[1] + v[2] * v[2]);
        if (r >= eps) {
          acc += std::erf(gam * r) / r - std::erf(alpha * r) / r;
        }
        if (ia == 0 && ib == 0 && ic == 0) {
          continue;
        }
        const double gx =
            2.0 * kPi * (ia * cell.recip[0][0] + ib * cell.recip[1][0] + ic * cell.recip[2][0]) * kBohr;
        const double gy =
            2.0 * kPi * (ia * cell.recip[0][1] + ib * cell.recip[1][1] + ic * cell.recip[2][1]) * kBohr;
        const double gz =
            2.0 * kPi * (ia * cell.recip[0][2] + ib * cell.recip[1][2] + ic * cell.recip[2][2]) * kBohr;
        const double g2 = gx * gx + gy * gy + gz * gz;
        if (g2 < eps) {
          continue;
        }
        const double w = 4.0 * kPi * std::exp(-0.25 * g2 / (alpha * alpha)) / (vol * g2);
        acc += w * std::cos(gx * base[0] + gy * base[1] + gz * base[2]);
      }
    }
  }
  return acc;
}

void dense_solve_3d(const double *pos, const int *z, std::size_t n, const double *box, const double *xvec,
                    double qtot, std::vector<double> *q) {
  const CellGeom cell = cell_from_box(box);
  const double alpha = ewald_alpha(cell);
  const int m = static_cast<int>(n) + 1;
  std::vector<double> amat(static_cast<std::size_t>(m * m), 0.0);
  std::vector<double> rhs(static_cast<std::size_t>(m), 0.0);
  auto A = [&](int i, int j) -> double & { return amat[static_cast<std::size_t>(i * m + j)]; };
  for (std::size_t i = 0; i < n; ++i) {
    rhs[i] = xvec[i];
    const double radi = kD4Rad[elem(z[i])];
    for (std::size_t j = i; j < n; ++j) {
      const double radj = kD4Rad[elem(z[j])];
      const double gam = 1.0 / std::sqrt(radi * radi + radj * radj);
      double tmp = ewald_pair(pos + 3 * i, pos + 3 * j, gam, alpha, cell);
      if (i == j) {
        tmp += kD4Eta[elem(z[i])] + kSqrt2Pi / radi - 2.0 * alpha / kSqrtPi;
      }
      A(static_cast<int>(i), static_cast<int>(j)) = tmp;
      A(static_cast<int>(j), static_cast<int>(i)) = tmp;
    }
    A(static_cast<int>(i), m - 1) = 1.0;
    A(m - 1, static_cast<int>(i)) = 1.0;
  }
  rhs[n] = qtot;
  for (int k = 0; k < m; ++k) {
    int piv = k;
    double best = std::abs(A(k, k));
    for (int i = k + 1; i < m; ++i) {
      if (std::abs(A(i, k)) > best) {
        best = std::abs(A(i, k));
        piv = i;
      }
    }
    if (!(best > 0.0)) {
      throw std::runtime_error("EEQ matrix is singular");
    }
    if (piv != k) {
      for (int j = k; j < m; ++j) {
        std::swap(A(k, j), A(piv, j));
      }
      std::swap(rhs[static_cast<std::size_t>(k)], rhs[static_cast<std::size_t>(piv)]);
    }
    const double diag = A(k, k);
    for (int i = k + 1; i < m; ++i) {
      const double f = A(i, k) / diag;
      A(i, k) = f;
      for (int j = k + 1; j < m; ++j) {
        A(i, j) -= f * A(k, j);
      }
      rhs[static_cast<std::size_t>(i)] -= f * rhs[static_cast<std::size_t>(k)];
    }
  }
  for (int i = m - 1; i >= 0; --i) {
    double s = rhs[static_cast<std::size_t>(i)];
    for (int j = i + 1; j < m; ++j) {
      s -= A(i, j) * rhs[static_cast<std::size_t>(j)];
    }
    rhs[static_cast<std::size_t>(i)] = s / A(i, i);
  }
  q->assign(rhs.begin(), rhs.begin() + static_cast<std::ptrdiff_t>(n));
}

void dense_solve(const double *pos, const int *z, std::size_t n, const double *xvec, double qtot,
                 std::vector<double> *q) {
  const int m = static_cast<int>(n) + 1;
  std::vector<double> a(static_cast<std::size_t>(m * m), 0.0);
  std::vector<double> rhs(static_cast<std::size_t>(m), 0.0);
  auto A = [&](int i, int j) -> double & { return a[static_cast<std::size_t>(i * m + j)]; };
  for (std::size_t i = 0; i < n; ++i) {
    const double radi = kD4Rad[elem(z[i])];
    A(static_cast<int>(i), static_cast<int>(i)) = kD4Eta[elem(z[i])] + kSqrt2Pi / radi;
    rhs[i] = xvec[i];
    for (std::size_t j = i + 1; j < n; ++j) {
      const double dx = pos[3 * j] - pos[3 * i];
      const double dy = pos[3 * j + 1] - pos[3 * i + 1];
      const double dz = pos[3 * j + 2] - pos[3 * i + 2];
      const double r = std::sqrt(dx * dx + dy * dy + dz * dz);
      const double radj = kD4Rad[elem(z[j])];
      const double gam = 1.0 / (radi * radi + radj * radj);
      const double tmp = std::erf(std::sqrt(r * r * gam)) / r;
      A(static_cast<int>(i), static_cast<int>(j)) = tmp;
      A(static_cast<int>(j), static_cast<int>(i)) = tmp;
    }
    A(static_cast<int>(i), m - 1) = 1.0;
    A(m - 1, static_cast<int>(i)) = 1.0;
  }
  rhs[n] = qtot;
  for (int k = 0; k < m; ++k) {
    int piv = k;
    double best = std::abs(A(k, k));
    for (int i = k + 1; i < m; ++i) {
      const double v = std::abs(A(i, k));
      if (v > best) {
        best = v;
        piv = i;
      }
    }
    if (!(best > 0.0)) {
      throw std::runtime_error("EEQ matrix is singular");
    }
    if (piv != k) {
      for (int j = k; j < m; ++j) {
        std::swap(A(k, j), A(piv, j));
      }
      std::swap(rhs[static_cast<std::size_t>(k)], rhs[static_cast<std::size_t>(piv)]);
    }
    const double diag = A(k, k);
    for (int i = k + 1; i < m; ++i) {
      const double f = A(i, k) / diag;
      A(i, k) = f;
      for (int j = k + 1; j < m; ++j) {
        A(i, j) -= f * A(k, j);
      }
      rhs[static_cast<std::size_t>(i)] -= f * rhs[static_cast<std::size_t>(k)];
    }
  }
  for (int i = m - 1; i >= 0; --i) {
    double s = rhs[static_cast<std::size_t>(i)];
    for (int j = i + 1; j < m; ++j) {
      s -= A(i, j) * rhs[static_cast<std::size_t>(j)];
    }
    rhs[static_cast<std::size_t>(i)] = s / A(i, i);
  }
  q->assign(rhs.begin(), rhs.begin() + static_cast<std::ptrdiff_t>(n));
}

double triple_weight(int i, int j, int k) {
  if (i == j) {
    return i == k ? 1.0 / 6.0 : 0.5;
  }
  if (i != k && j != k) {
    return 1.0;
  }
  return 0.5;
}

void mat_inv(const double *box, double inv[9], double *vol) {
  // Lattice vectors are (box[0..2]), (box[3..5]), (box[6..8]).
  const double a0 = box[0], a1 = box[1], a2 = box[2];
  const double b0 = box[3], b1 = box[4], b2 = box[5];
  const double c0 = box[6], c1 = box[7], c2 = box[8];
  const double v = a0 * (b1 * c2 - b2 * c1) - a1 * (b0 * c2 - b2 * c0) + a2 * (b0 * c1 - b1 * c0);
  *vol = std::abs(v);
  const double s = 1.0 / v;
  // Inverse of the matrix whose columns are the lattice vectors.
  inv[0] = s * (b1 * c2 - b2 * c1);
  inv[1] = s * (b2 * c0 - b0 * c2);
  inv[2] = s * (b0 * c1 - b1 * c0);
  inv[3] = s * (c1 * a2 - c2 * a1);
  inv[4] = s * (c2 * a0 - c0 * a2);
  inv[5] = s * (c0 * a1 - c1 * a0);
  inv[6] = s * (a1 * b2 - a2 * b1);
  inv[7] = s * (a2 * b0 - a0 * b2);
  inv[8] = s * (a0 * b1 - a1 * b0);
}

void cn_force(const Shell &sh, const int *z, std::size_t n, double cut_b, bool electroneg, bool do_log,
              const std::vector<double> &cn_raw, const std::vector<double> &dedcn, double *grad) {
  for (std::size_t i = 0; i < n; ++i) {
    const double chain = do_log ? dlog_cut(cn_raw[i]) : 1.0;
    for (const Nb &a : sh.of(i)) {
      if (a.j > static_cast<int>(i) || a.r >= cut_b || !(a.r > 1e-12)) {
        continue;
      }
      const double rc = kD4Rcov[elem(z[i])] + kD4Rcov[elem(z[a.j])];
      const double den = electroneg ? en_factor(z[i], z[a.j]) : 1.0;
      const double dcf = den * erf_dcount(a.r, rc);
      const double couple = dedcn[i] * (do_log ? dlog_cut(cn_raw[i]) : 1.0) +
                            dedcn[static_cast<std::size_t>(a.j)] * (do_log ? dlog_cut(cn_raw[a.j]) : 1.0);
      (void)chain;
      // Nb points from i to j, while the CN derivative uses r_i - r_j.
      const double cx = dcf * (-a.x) / a.r;
      const double cy = dcf * (-a.y) / a.r;
      const double cz = dcf * (-a.z) / a.r;
      grad[3 * i] += cx * couple;
      grad[3 * i + 1] += cy * couple;
      grad[3 * i + 2] += cz * couple;
      if (a.j != static_cast<int>(i)) {
        grad[3 * a.j] -= cx * couple;
        grad[3 * a.j + 1] -= cy * couple;
        grad[3 * a.j + 2] -= cz * couple;
      }
    }
  }
}

} // namespace

bool d4bj_param(const char *functional, D4BjParam *out) {
  if (functional == nullptr) {
    return false;
  }
  if (std::string(functional) == "pbe") {
    *out = D4BjParam{};
    return true;
  }
  return false;
}

void d4_evaluate(const double *pos, const int *z, std::size_t n, const double *box, bool periodic,
                 double charge, const D4BjParam &param, double cut2, double cut3, double cut_cn,
                 double *energy, double *F, double strain[6]) {
  const double to_b = 1.0 / kBohr;
  const double reach = std::max(cut2, std::max(cut3, cut_cn));
  double ewald_cut = reach;
  if (periodic && box != nullptr) {
    const double la = std::sqrt(box[0] * box[0] + box[1] * box[1] + box[2] * box[2]);
    const double lb = std::sqrt(box[3] * box[3] + box[4] * box[4] + box[5] * box[5]);
    const double lc = std::sqrt(box[6] * box[6] + box[7] * box[7] + box[8] * box[8]);
    ewald_cut = std::min(0.45 * std::min(la, std::min(lb, lc)), 18.0);
    ewald_cut = std::max(ewald_cut, reach);
  } else {
    ewald_cut = std::max(reach, 40.0);
  }
  std::vector<Pair> built;
  const std::vector<Pair> *use = nullptr;
  DualList lists(std::min(3.0, ewald_cut * 0.5), ewald_cut, 0.0);
  if (!periodic || box == nullptr) {
    // Molecules use every pair. The list cutoff is past the molecule.
    lists.update(pos, n, nullptr, false);
  } else {
    lists.update(pos, n, box, true);
  }
  use = &lists.long_pairs();
  (void)built;
  const Shell sh = shell_of(*use, n);
  std::vector<double> cn_disp;
  std::vector<double> cn_eeq_raw(n, 0.0);
  accumulate_cn(sh, z, n, cut_cn * to_b, true, &cn_disp);
  accumulate_cn(sh, z, n, std::min(kEeqCut, ewald_cut * to_b), false, &cn_eeq_raw);
  std::vector<double> cn_eeq(n);
  for (std::size_t i = 0; i < n; ++i) {
    cn_eeq[i] = log_cut(cn_eeq_raw[i]);
  }
  std::vector<double> xvec(n);
  for (std::size_t i = 0; i < n; ++i) {
    const double c = cn_eeq[i];
    xvec[i] = -kD4Chi[elem(z[i])] + kD4Kcn[elem(z[i])] * c / std::sqrt(c + kReg);
  }
  std::vector<double> q;
  Operator op;
  op.n = n;
  op.z = z;
  op.real = &sh;
  op.periodic = periodic && box != nullptr;
  std::vector<double> pos_b(3 * n);
  for (std::size_t i = 0; i < pos_b.size(); ++i) {
    pos_b[i] = pos[i] * to_b;
  }
  op.pos = pos_b.data();
  op.box = box;
  op.diag.assign(n, 0.0);
  op.alpha = 3.2 / (ewald_cut * to_b);
  for (std::size_t i = 0; i < n; ++i) {
    op.diag[i] = kD4Eta[elem(z[i])] + kSqrt2Pi / kD4Rad[elem(z[i])];
    if (op.periodic) {
      op.diag[i] -= 2.0 * op.alpha / kSqrtPi;
    }
  }
  if (op.periodic && n <= 64) {
    // Same real and reciprocal lattice sums dftd4 uses for the EEQ matrix.
    dense_solve_3d(pos, z, n, box, xvec.data(), charge, &q);
  } else if (op.periodic) {
    op.prepare_kernel();
    solve_charges(op, xvec.data(), charge, &q);
  } else if (n <= 96) {
    dense_solve(pos_b.data(), z, n, xvec.data(), charge, &q);
  } else {
    solve_charges(op, xvec.data(), charge, &q);
  }

  std::vector<Gw> gw(n), gw0(n);
  std::vector<double> r4(n);
  for (std::size_t i = 0; i < n; ++i) {
    weights_of(z[i], cn_disp[i], q[i], &gw[i]);
    weights_of(z[i], cn_disp[i], 0.0, &gw0[i]);
    r4[i] = kD4R4[elem(z[i])];
  }

  const double cut2_b = cut2 * to_b;
  const double cut3_b = cut3 * to_b;
  double ene = 0.0;
  std::vector<double> grad(3 * n, 0.0);
  std::vector<double> dedcn(n, 0.0);
  std::vector<double> dedq(n, 0.0);
  int pair_nt = 1;
#ifdef _OPENMP
  pair_nt = std::max(1, omp_get_max_threads());
#endif
  std::vector<std::vector<double>> pair_g(static_cast<std::size_t>(pair_nt), std::vector<double>(3 * n, 0.0));
  std::vector<std::vector<double>> pair_c(static_cast<std::size_t>(pair_nt), std::vector<double>(n, 0.0));
  std::vector<std::vector<double>> pair_q(static_cast<std::size_t>(pair_nt), std::vector<double>(n, 0.0));
  std::vector<double> pair_e(static_cast<std::size_t>(pair_nt), 0.0);
#ifdef _OPENMP
#pragma omp parallel for schedule(static)
#endif
  for (std::size_t i = 0; i < n; ++i) {
    const int tid =
#ifdef _OPENMP
        omp_get_thread_num();
#else
        0;
#endif
    double *const G = pair_g[static_cast<std::size_t>(tid)].data();
    double *const C = pair_c[static_cast<std::size_t>(tid)].data();
    double *const Q = pair_q[static_cast<std::size_t>(tid)].data();
    double &eadd = pair_e[static_cast<std::size_t>(tid)];
      for (const Nb &a : sh.of(i)) {
        if (a.j > static_cast<int>(i) || a.r >= cut2_b) {
          continue;
        }
        double c6 = 0, dni = 0, dnj = 0, dqi = 0, dqj = 0;
        c6_of(gw[i], gw[static_cast<std::size_t>(a.j)], z[i], z[a.j], &c6, &dni, &dnj, &dqi, &dqj);
        const double rrij = 3.0 * r4[i] * r4[static_cast<std::size_t>(a.j)];
        const double r0 = param.a1 * std::sqrt(rrij) + param.a2;
        const double r0_2 = r0 * r0;
        const double r0_6 = r0_2 * r0_2 * r0_2;
        const double r0_8 = r0_6 * r0_2;
        const double r2 = a.r * a.r;
        const double r4p = r2 * r2;
        const double t6 = 1.0 / (r4p * r2 + r0_6);
        const double t8 = 1.0 / (r4p * r4p + r0_8);
        const double d6 = -6.0 * r4p * t6 * t6;
        const double d8 = -8.0 * r4p * r2 * t8 * t8;
        const double edisp = param.s6 * t6 + param.s8 * rrij * t8;
        const double gdisp = param.s6 * d6 + param.s8 * rrij * d8;
        const double pair = -c6 * edisp;
        eadd += (a.j == static_cast<int>(i) ? 0.5 * pair : pair);
        // dftd4 accumulates the charge and CN derivatives with the full
        // kernel on a self image, while the energy of that image stays halved.
        C[i] += dni * (-edisp);
        Q[i] += dqi * (-edisp);
        if (a.j != static_cast<int>(i)) {
          C[static_cast<std::size_t>(a.j)] += dnj * (-edisp);
          Q[static_cast<std::size_t>(a.j)] += dqj * (-edisp);
        }
        const double gx = c6 * gdisp * a.x;
        const double gy = c6 * gdisp * a.y;
        const double gz = c6 * gdisp * a.z;
        G[3 * i] += gx;
        G[3 * i + 1] += gy;
        G[3 * i + 2] += gz;
        if (a.j != static_cast<int>(i)) {
          G[3 * a.j] -= gx;
          G[3 * a.j + 1] -= gy;
          G[3 * a.j + 2] -= gz;
        }
      }
  }
  for (int t = 0; t < pair_nt; ++t) {
    ene += pair_e[static_cast<std::size_t>(t)];
    const auto &lg = pair_g[static_cast<std::size_t>(t)];
    const auto &lc = pair_c[static_cast<std::size_t>(t)];
    const auto &lq = pair_q[static_cast<std::size_t>(t)];
    for (std::size_t k = 0; k < grad.size(); ++k) grad[k] += lg[k];
    for (std::size_t k = 0; k < n; ++k) { dedcn[k] += lc[k]; dedq[k] += lq[k]; }
  }
  if (param.s9 != 0.0) {
    int atm_nt = 1;
#ifdef _OPENMP
    atm_nt = std::max(1, omp_get_max_threads());
#endif
    std::vector<std::vector<double>> atm_g(static_cast<std::size_t>(atm_nt), std::vector<double>(3 * n, 0.0));
    std::vector<std::vector<double>> atm_c(static_cast<std::size_t>(atm_nt), std::vector<double>(n, 0.0));
    std::vector<double> atm_e(static_cast<std::size_t>(atm_nt), 0.0);
#ifdef _OPENMP
#pragma omp parallel for schedule(static)
#endif
    for (std::size_t i = 0; i < n; ++i) {
      const int tid =
#ifdef _OPENMP
          omp_get_thread_num();
#else
          0;
#endif
      double *const G = atm_g[static_cast<std::size_t>(tid)].data();
      double *const C = atm_c[static_cast<std::size_t>(tid)].data();
      double &eadd = atm_e[static_cast<std::size_t>(tid)];
        for (const Nb &a : sh.of(i)) {
          if (a.j > static_cast<int>(i) || a.r >= cut3_b) {
            continue;
          }
          double c6ij = 0, dni = 0, dnj = 0, dqi = 0, dqj = 0;
          c6_of(gw0[i], gw0[static_cast<std::size_t>(a.j)], z[i], z[a.j], &c6ij, &dni, &dnj, &dqi, &dqj);
          if (!(std::abs(c6ij) > 0.0)) {
            continue;
          }
          const double r0ij =
              param.a1 * std::sqrt(3.0 * r4[i] * r4[static_cast<std::size_t>(a.j)]) + param.a2;
          for (const Nb &b : sh.of(i)) {
            if (b.j > a.j || b.r >= cut3_b) {
              continue;
            }
            double c6ik = 0, dik_i = 0, dik_k = 0, u1 = 0, u2 = 0;
            c6_of(gw0[i], gw0[static_cast<std::size_t>(b.j)], z[i], z[b.j], &c6ik, &dik_i, &dik_k, &u1, &u2);
            double c6jk = 0, djk_j = 0, djk_k = 0;
            c6_of(gw0[static_cast<std::size_t>(a.j)], gw0[static_cast<std::size_t>(b.j)], z[a.j], z[b.j],
                  &c6jk, &djk_j, &djk_k, &u1, &u2);
            if (!(std::abs(c6ik) > 0.0) || !(std::abs(c6jk) > 0.0)) {
              continue;
            }
            const double vjkx = b.x - a.x;
            const double vjky = b.y - a.y;
            const double vjkz = b.z - a.z;
            const double rjk = std::sqrt(vjkx * vjkx + vjky * vjky + vjkz * vjkz);
            if (!(rjk > 0.0) || rjk >= cut3_b) {
              continue;
            }
            const double r2ij = a.r * a.r;
            const double r2ik = b.r * b.r;
            const double r2jk = rjk * rjk;
            const double r2 = r2ij * r2ik * r2jk;
            const double r1 = std::sqrt(r2);
            const double r3 = r2 * r1;
            const double r5 = r3 * r2;
            const double r0ik =
                param.a1 * std::sqrt(3.0 * r4[i] * r4[static_cast<std::size_t>(b.j)]) + param.a2;
            const double r0jk = param.a1 * std::sqrt(3.0 * r4[static_cast<std::size_t>(a.j)] *
                                                     r4[static_cast<std::size_t>(b.j)]) +
                                param.a2;
            const double r0 = r0ij * r0ik * r0jk;
            const double alp3 = param.alp / 3.0;
            const double fdmp = 1.0 / (1.0 + 6.0 * std::pow(r0 / r1, alp3));
            const double ang = 0.375 * (r2ij + r2jk - r2ik) * (r2ij - r2jk + r2ik) *
                                   (-r2ij + r2jk + r2ik) / r5 +
                               1.0 / r3;
            const double c9 = -param.s9 * std::sqrt(std::abs(c6ij * c6ik * c6jk));
            const double triple = triple_weight(static_cast<int>(i), a.j, b.j);
            const double dE = ang * fdmp * c9 * triple;
            eadd -= dE;
            C[i] -= dE * 0.5 * (dni / c6ij + dik_i / c6ik);
            C[static_cast<std::size_t>(a.j)] -= dE * 0.5 * (dnj / c6ij + djk_j / c6jk);
            C[static_cast<std::size_t>(b.j)] -= dE * 0.5 * (dik_k / c6ik + djk_k / c6jk);
            const double r0r = std::pow(r0 / r1, alp3);
            const double dfdmp = -2.0 * param.alp * r0r * fdmp * fdmp;
            const double dij =
                -0.375 *
                (r2ij * r2ij * r2ij + r2ij * r2ij * (r2jk + r2ik) +
                 r2ij * (3.0 * r2jk * r2jk + 2.0 * r2jk * r2ik + 3.0 * r2ik * r2ik) -
                 5.0 * (r2jk - r2ik) * (r2jk - r2ik) * (r2jk + r2ik)) /
                r5;
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
            const double scale_ij = c9 * (-dij * fdmp + ang * dfdmp) / r2ij;
            const double scale_ik = c9 * (-dik * fdmp + ang * dfdmp) / r2ik;
            const double scale_jk = c9 * (-djk * fdmp + ang * dfdmp) / r2jk;
            const double gijx = scale_ij * a.x;
            const double gijy = scale_ij * a.y;
            const double gijz = scale_ij * a.z;
            const double gikx = scale_ik * b.x;
            const double giky = scale_ik * b.y;
            const double gikz = scale_ik * b.z;
            const double gjkx = scale_jk * (b.x - a.x);
            const double gjky = scale_jk * (b.y - a.y);
            const double gjkz = scale_jk * (b.z - a.z);
            G[3 * i] -= (gijx + gikx) * triple;
            G[3 * i + 1] -= (gijy + giky) * triple;
            G[3 * i + 2] -= (gijz + gikz) * triple;
            G[3 * a.j] += (gijx - gjkx) * triple;
            G[3 * a.j + 1] += (gijy - gjky) * triple;
            G[3 * a.j + 2] += (gijz - gjkz) * triple;
            G[3 * b.j] += (gikx + gjkx) * triple;
            G[3 * b.j + 1] += (giky + gjky) * triple;
            G[3 * b.j + 2] += (gikz + gjkz) * triple;
          }
        }
    }
    for (int t = 0; t < atm_nt; ++t) {
      ene += atm_e[static_cast<std::size_t>(t)];
      const auto &lg = atm_g[static_cast<std::size_t>(t)];
      const auto &lc = atm_c[static_cast<std::size_t>(t)];
      for (std::size_t k = 0; k < grad.size(); ++k) grad[k] += lg[k];
      for (std::size_t k = 0; k < n; ++k) dedcn[k] += lc[k];
    }
  }
  cn_force(sh, z, n, cut_cn * to_b, true, false, cn_disp, dedcn, grad.data());

  // Charge response: A lambda = dEdq with a vanishing charge sum, then
  // contract the same real and reciprocal kernels used to build A.
  if (op.periodic && n <= 64) {
    const CellGeom cell = cell_from_box(box);
    const double alpha = ewald_alpha(cell);
    std::vector<double> y;
    // The saddle solve enforces a vanishing charge sum, which is the
    // adjoint of the same Ewald matrix used for the charges.
    dense_solve_3d(pos, z, n, box, dedq.data(), 0.0, &y);
    const double to_b = 1.0 / kBohr;
    const double eps = 1.4901161193847656e-8;
    const double vol = cell.volume / (kBohr * kBohr * kBohr);
    auto dradial = [](double c, double r) {
      const double arg = c * r;
      return 2.0 * c * std::exp(-arg * arg) / (kSqrtPi * r) - std::erf(arg) / (r * r);
    };
    for (std::size_t i = 0; i < n; ++i) {
      for (std::size_t j = i; j < n; ++j) {
        const double gam =
            1.0 / std::sqrt(kD4Rad[elem(z[i])] * kD4Rad[elem(z[i])] + kD4Rad[elem(z[j])] * kD4Rad[elem(z[j])]);
        const double couple = (i == j) ? y[i] * q[i] : y[i] * q[j] + y[j] * q[i];
        if (!(std::abs(couple) > 0.0)) {
          continue;
        }
        double base[3] = {(pos[3 * j] - pos[3 * i]) * to_b, (pos[3 * j + 1] - pos[3 * i + 1]) * to_b,
                          (pos[3 * j + 2] - pos[3 * i + 2]) * to_b};
        double gx = 0, gy = 0, gz = 0;
        for (int ia = -2; ia <= 2; ++ia) {
          for (int ib = -2; ib <= 2; ++ib) {
            for (int ic = -2; ic <= 2; ++ic) {
              double v[3];
              for (int c = 0; c < 3; ++c) {
                v[c] = base[c] + (ia * cell.H[0][c] + ib * cell.H[1][c] + ic * cell.H[2][c]) * to_b;
              }
              const double r = std::sqrt(v[0] * v[0] + v[1] * v[1] + v[2] * v[2]);
              if (r >= eps) {
                const double d = (dradial(gam, r) - dradial(alpha, r)) / r;
                gx += d * v[0];
                gy += d * v[1];
                gz += d * v[2];
              }
              if (ia == 0 && ib == 0 && ic == 0) {
                continue;
              }
              const double gxv =
                  2.0 * kPi * (ia * cell.recip[0][0] + ib * cell.recip[1][0] + ic * cell.recip[2][0]) * kBohr;
              const double gyv =
                  2.0 * kPi * (ia * cell.recip[0][1] + ib * cell.recip[1][1] + ic * cell.recip[2][1]) * kBohr;
              const double gzv =
                  2.0 * kPi * (ia * cell.recip[0][2] + ib * cell.recip[1][2] + ic * cell.recip[2][2]) * kBohr;
              const double g2 = gxv * gxv + gyv * gyv + gzv * gzv;
              if (g2 < eps) {
                continue;
              }
              const double w = 4.0 * kPi * std::exp(-0.25 * g2 / (alpha * alpha)) / (vol * g2);
              const double s = -w * std::sin(gxv * base[0] + gyv * base[1] + gzv * base[2]);
              gx += s * gxv;
              gy += s * gyv;
              gz += s * gzv;
            }
          }
        }
        // Same direction as the mesh response: atom i receives dA/d(rj-ri).
        grad[3 * i] += couple * gx;
        grad[3 * i + 1] += couple * gy;
        grad[3 * i + 2] += couple * gz;
        if (i != j) {
          grad[3 * j] -= couple * gx;
          grad[3 * j + 1] -= couple * gy;
          grad[3 * j + 2] -= couple * gz;
        }
      }
    }
    std::vector<double> eeq_w(n);
    for (std::size_t i = 0; i < n; ++i) {
      const double c = cn_eeq[i];
      const double dx = kD4Kcn[elem(z[i])] * (0.5 * c + kReg) / std::pow(c + kReg, 1.5);
      eeq_w[i] = y[i] * dx;
    }
    cn_force(sh, z, n, std::min(kEeqCut, ewald_cut * to_b), false, true, cn_eeq_raw, eeq_w, grad.data());
  } else if (op.periodic || n > 96) {
    std::vector<double> y(n, 0.0), ones(n, 1.0), zsol(n, 0.0);
    for (std::size_t i = 0; i < n; ++i) {
      y[i] = dedq[i] / op.diag[i];
      zsol[i] = 1.0 / op.diag[i];
    }
    const std::size_t direct_n = []() {
      if (const char *env = std::getenv("LR_EEQ_DIRECT")) {
        return static_cast<std::size_t>(std::atoi(env));
      }
      return static_cast<std::size_t>(160);
    }();
    // Systems at or under 64 periodic atoms keep the lattice factorization
    // used for the charges. This response is the conjugate gradient one.
    const bool lattice = op.periodic && op.n <= 64;
    if (op.n <= direct_n && !lattice) {
      op.solve_linear(dedq.data(), y.data());
      op.solve_linear(ones.data(), zsol.data());
    } else {
      cg(op, dedq.data(), y.data(), 80, 1e-10);
      cg(op, ones.data(), zsol.data(), 80, 1e-10);
    }
    double sy = 0.0;
    double sz = 0.0;
    for (std::size_t i = 0; i < n; ++i) {
      sy += y[i];
      sz += zsol[i];
    }
    const double beta = -sy / sz;
    for (std::size_t i = 0; i < n; ++i) {
      y[i] += beta * zsol[i];
    }
    for (std::size_t i = 0; i < n; ++i) {
      const double c = cn_eeq[i];
      const double dx = kD4Kcn[elem(z[i])] * (0.5 * c + kReg) / std::pow(c + kReg, 1.5);
      dedcn[i] = y[i] * dx;
    }
    // Reuse dedcn as the EEQ-CN weight. cn_eeq_raw is the pre-log value.
    std::vector<double> eeq_w(n);
    for (std::size_t i = 0; i < n; ++i) {
      eeq_w[i] = dedcn[i];
    }
    cn_force(sh, z, n, std::min(kEeqCut, ewald_cut * to_b), false, true, cn_eeq_raw, eeq_w, grad.data());
    for (std::size_t i = 0; i < n; ++i) {
      const double radi = kD4Rad[elem(z[i])];
      for (const Nb &a : sh.of(i)) {
        if (a.j > static_cast<int>(i) || !(a.r > 1e-12)) {
          continue;
        }
        const double radj = kD4Rad[elem(z[a.j])];
        const double gam = 1.0 / std::sqrt(radi * radi + radj * radj);
        auto dtmp = [](double gamv, double r) {
          const double r2 = r * r;
          const double arg = gamv * gamv * r2;
          return 2.0 * gamv * std::exp(-arg) / (kSqrtPi * r2) - std::erf(std::sqrt(arg)) / (r2 * r);
        };
        double d = dtmp(gam, a.r);
        if (op.periodic) {
          d -= dtmp(op.alpha, a.r);
        }
        const double couple = y[i] * q[a.j] + y[static_cast<std::size_t>(a.j)] * q[i];
        const double gx = couple * d * a.x;
        const double gy = couple * d * a.y;
        const double gz = couple * d * a.z;
        grad[3 * i] += gx;
        grad[3 * i + 1] += gy;
        grad[3 * i + 2] += gz;
        if (a.j != static_cast<int>(i)) {
          grad[3 * a.j] -= gx;
          grad[3 * a.j + 1] -= gy;
          grad[3 * a.j + 2] -= gz;
        }
      }
    }
  }

  *energy = ene * kHartree;
  if (F != nullptr) {
    for (std::size_t i = 0; i < grad.size(); ++i) {
      F[i] += grad[i] * units::NEG_GRAD_TO_FORCE;
    }
  }
  std::fill(strain, strain + 6, 0.0);
}

} // namespace rgpot::lr
