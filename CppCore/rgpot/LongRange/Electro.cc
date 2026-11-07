// MIT License
// Copyright 2023--present rgpot developers

#include "rgpot/LongRange/Electro.hpp"

#include "rgpot/LongRange/Compensate.hpp"
#include "rgpot/LongRange/Team.hpp"
#include "rgpot/LongRange/Lattice.hpp"
#include "rgpot/ParamHash.hpp"
#include "rgpot/stress.hpp"
#include "rgpot/units.hpp"

#include <algorithm>
#include <chrono>
#include <cmath>
#include <complex>
#include <cstdio>
#include <cstdlib>
#include <numbers>
#include <stdexcept>
#include <vector>

#ifdef _OPENMP
#include <omp.h>
#endif

namespace rgpot::lr {

namespace {

constexpr double kKe = units::HARTREE_TO_EV * units::BOHR_TO_ANGSTROM;
constexpr double kPi = std::numbers::pi;

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

struct Scratch {
  Neumaier energy;
  std::vector<double> f;
  double strain[6]{};
};

// One buffer for the whole process. The ladder is a single caller, and
// keeping the storage removes allocator jitter from the timed samples.
std::vector<Scratch> &scratch_for(int nt, std::size_t n3) {
  static std::vector<Scratch> parts;
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

void real_space(const CoulombConfig &cfg, bool dsf, const double *q, std::size_t n,
                const std::vector<Pair> &pairs, double *F, double *dE_dq, ElecOut *out) {
  const double alpha = cfg.alpha;
  const double rc = cfg.cutoff;
  const double e_shift = dsf ? erfc_over_r(alpha, rc) : 0.0;
  const double bare_prime_rc = dsf ? derfc_over_r(alpha, rc) : 0.0;
  const int nt = thread_count();
  std::vector<Scratch> &parts = scratch_for(nt, 3 * n);
#ifdef _OPENMP
#pragma omp parallel
#endif
  {
    auto &p = parts[static_cast<std::size_t>(thread_id())];
#ifdef _OPENMP
#pragma omp for schedule(static)
#endif
    for (std::size_t t = 0; t < pairs.size(); ++t) {
      const Pair &pr = pairs[t];
      if (!(pr.r > 0.0) || pr.r >= rc) {
        continue;
      }
      const double phi_bare = erfc_over_r(alpha, pr.r);
      const double dphi_bare = derfc_over_r(alpha, pr.r);
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

double bspline(int order, double x) {
  if (order == 4) {
    return bspline4(x);
  }
  if (order == 1) {
    return (x >= 0.0 && x < 1.0) ? 1.0 : 0.0;
  }
  const double inv = 1.0 / static_cast<double>(order - 1);
  return x * inv * bspline(order - 1, x) + (order - x) * inv * bspline(order - 1, x - 1.0);
}

double bspline_derivative(int order, double x) {
  if (order == 4) {
    return bspline3(x) - bspline3(x - 1.0);
  }
  return bspline(order - 1, x) - bspline(order - 1, x - 1.0);
}

extern "C" {
struct fftw_plan_s;
typedef struct fftw_plan_s *fftw_plan;
void fftw_execute_dft(fftw_plan p, std::complex<double> *in, std::complex<double> *out);
fftw_plan fftw_plan_dft_3d(int n0, int n1, int n2, std::complex<double> *in, std::complex<double> *out, int sign,
                           unsigned flags);
void fftw_destroy_plan(fftw_plan p);
}

fftw_plan mesh_plan(int nx, int ny, int nz, int sign, std::complex<double> *data) {
  struct Saved {
    int nx = 0;
    int ny = 0;
    int nz = 0;
    int sign = 0;
    fftw_plan plan = nullptr;
  };
  static Saved forward;
  static Saved backward;
  Saved &slot = sign < 0 ? forward : backward;
  if (slot.plan == nullptr || slot.nx != nx || slot.ny != ny || slot.nz != nz) {
    if (slot.plan != nullptr) {
      fftw_destroy_plan(slot.plan);
    }
    // z is the slowest index: x + nx * (y + ny * z).
    // Estimate does not overwrite the charge grid. A measured plan does,
    // and the first force would then be wrong.
    slot.plan = fftw_plan_dft_3d(nz, ny, nx, data, data, sign, 1U << 6);
    slot.nx = nx;
    slot.ny = ny;
    slot.nz = nz;
    slot.sign = sign;
  }
  return slot.plan;
}

void fft3(std::vector<std::complex<double>> &g, int nx, int ny, int nz, bool inverse, int team) {
  (void)team;
  const int sign = inverse ? +1 : -1;
  fftw_plan plan = mesh_plan(nx, ny, nz, sign, g.data());
  fftw_execute_dft(plan, g.data(), g.data());
  if (inverse) {
    const double inv = 1.0 / static_cast<double>(static_cast<std::size_t>(nx) * static_cast<std::size_t>(ny) *
                                                 static_cast<std::size_t>(nz));
    for (auto &z : g) {
      z *= inv;
    }
  }
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
  std::vector<int> gtab(3 * n * span);
  std::vector<double> wtab(3 * n * span);
  std::vector<double> dtab(3 * n * span);
  auto slot = [span](std::size_t atom, int axis, int knot) {
    return (atom * 3 + static_cast<std::size_t>(axis)) * span + static_cast<std::size_t>(knot);
  };
  // One core is faster than a forked team while the mesh is this small.
  // The four-worker cap still applies once the team is worth starting:
  // extra threads sit the call out, because resizing the pool is slower.
  const int nthreads = work_team(n, 256, 4);
  const bool trace = std::getenv("LR_PME_PROFILE") != nullptr;
  const auto t_spread0 = std::chrono::steady_clock::now();
  // Reuse the per-thread grids. A fresh allocation faults during the
  // deposit, and eight threads faulting at once dominate the spread.
  static std::vector<std::vector<std::complex<double>>> partial;
  if (static_cast<int>(partial.size()) != nthreads || partial[0].size() != ntot) {
    partial.assign(static_cast<std::size_t>(nthreads), std::vector<std::complex<double>>(ntot));
  } else {
    for (auto &grid_t : partial) {
      std::fill(grid_t.begin(), grid_t.end(), std::complex<double>(0.0, 0.0));
    }
  }
  auto deposit = [&](std::size_t i, std::complex<double> *dest) {
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
      const int k0 = static_cast<int>(std::floor(u)) - order + 1;
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
          dest[index(gax[0][ix], gax[1][iy], gax[2][iz])] += qi * wax[0][ix] * wyz;
        }
      }
    }
  };
  if (nthreads <= 1) {
    std::complex<double> *dest = partial[0].data();
    for (std::size_t i = 0; i < n; ++i) {
      deposit(i, dest);
    }
  } else {
#ifdef _OPENMP
#pragma omp parallel
#endif
    {
      const int tid_cap = thread_id();
      if (tid_cap < nthreads) {
        std::complex<double> *dest = partial[static_cast<std::size_t>(tid_cap)].data();
        // Contiguous atoms keep each thread's spline table on its own
        // cache lines. A stride of one shares those lines and slows down.
        const std::size_t chunk = (n + static_cast<std::size_t>(nthreads) - 1) / static_cast<std::size_t>(nthreads);
        const std::size_t begin = static_cast<std::size_t>(tid_cap) * chunk;
        const std::size_t end = std::min(n, begin + chunk);
        for (std::size_t i = begin; i < end; ++i) {
          deposit(i, dest);
        }
      }
    }
  }
  const auto t_reduce0 = std::chrono::steady_clock::now();
  for (int t = 0; t < nthreads; ++t) {
    const auto &src = partial[static_cast<std::size_t>(t)];
    for (std::size_t m = 0; m < ntot; ++m) {
      grid[m] += src[m];
    }
  }
  const auto t_fft0 = std::chrono::steady_clock::now();
  if (trace) {
    std::fprintf(stderr, "pme_reduce workers=%d deposit_us=%.3f reduce_us=%.3f\n", nthreads,
                 std::chrono::duration<double, std::micro>(t_reduce0 - t_spread0).count(),
                 std::chrono::duration<double, std::micro>(t_fft0 - t_reduce0).count());
  }
  fft3(grid, nx, ny, nz, false, nthreads);
  const auto t_kern0 = std::chrono::steady_clock::now();
  static std::vector<std::complex<double>> phihat;
  if (phihat.size() != ntot) {
    phihat.assign(ntot, std::complex<double>(0.0, 0.0));
  } else {
    std::fill(phihat.begin(), phihat.end(), std::complex<double>(0.0, 0.0));
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
  if (cfg.stress) {
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
  out->energy += rec.total();
  const double ebg = -kKe * kPi * total * total / (2.0 * cell.volume * alpha * alpha);
  out->energy += ebg;
  out->strain[0] += -ebg;
  out->strain[1] += -ebg;
  out->strain[2] += -ebg;
  const auto t_gather0 = std::chrono::steady_clock::now();
  fft3(phihat, nx, ny, nz, true, nthreads);
  const auto t_force0 = std::chrono::steady_clock::now();
  auto gather_atom = [&](std::size_t i) {
    double pot = 0.0;
    double dQ[3] = {};
    for (int iz = 0; iz < order; ++iz) {
      for (int iy = 0; iy < order; ++iy) {
        for (int ix = 0; ix < order; ++ix) {
          const double wx = wtab[slot(i, 0, ix)];
          const double wy = wtab[slot(i, 1, iy)];
          const double wz = wtab[slot(i, 2, iz)];
          const double ph = phihat[index(gtab[slot(i, 0, ix)], gtab[slot(i, 1, iy)], gtab[slot(i, 2, iz)])].real();
          pot += wx * wy * wz * ph;
          const double qw = q[i];
          dQ[0] += qw * dtab[slot(i, 0, ix)] * wy * wz * ph;
          dQ[1] += qw * wx * dtab[slot(i, 1, iy)] * wz * ph;
          dQ[2] += qw * wx * wy * dtab[slot(i, 2, iz)] * ph;
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
  if (nthreads <= 1) {
    for (std::size_t i = 0; i < n; ++i) {
      gather_atom(i);
    }
  } else {
#ifdef _OPENMP
#pragma omp parallel
#endif
    {
      const int tid_cap = thread_id();
      if (tid_cap < nthreads) {
        const std::size_t chunk = (n + static_cast<std::size_t>(nthreads) - 1) / static_cast<std::size_t>(nthreads);
        const std::size_t begin = static_cast<std::size_t>(tid_cap) * chunk;
        const std::size_t end = std::min(n, begin + chunk);
        for (std::size_t i = begin; i < end; ++i) {
          gather_atom(i);
        }
      }
    }
  }
  if (dE_dq != nullptr) {
    const double dbg = -kKe * kPi * total / (cell.volume * alpha * alpha);
    for (std::size_t i = 0; i < n; ++i) {
      dE_dq[i] += dbg;
    }
  }
  out->has_stress = true;
  if (trace) {
    const auto t_end = std::chrono::steady_clock::now();
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

} // namespace rgpot::lr
