// MIT License
// Copyright 2023--present rgpot developers

#include "rgpot/LongRange/Dispersion.hpp"

#include "rgpot/LongRange/FastD3.hpp"
#include "rgpot/LongRange/FastD4.hpp"
#include "rgpot/LongRange/Compensate.hpp"
#include "rgpot/LongRange/Lattice.hpp"
#include "rgpot/ParamHash.hpp"
#include "rgpot/stress.hpp"
#include "rgpot/units.hpp"

#include <algorithm>
#include <array>
#include <cmath>
#include <complex>
#include <numbers>
#include <stdexcept>
#include <vector>

#ifdef _OPENMP
#include <omp.h>
#endif

#if defined(RGPOT_HAS_DFTD3)
#include <dftd3.h>
#endif
#if defined(RGPOT_HAS_DFTD4)
#include <dftd4.h>
#endif

namespace rgpot::lr {
namespace {

constexpr double kPi = std::numbers::pi;
constexpr double kBohr = units::BOHR_TO_ANGSTROM;
constexpr double kHartree = units::HARTREE_TO_EV;

#include "rgpot/LongRange/QdoTable.inc"
#include "rgpot/LongRange/TsTable.inc"

const double kMbdFreq[] = {0,
                           99.337689021500196,
                           18.530637634124265,
                           7.3054811925383083,
                           3.7544165810186088,
                           2.1970213664137797,
                           1.3806926931529042,
                           0.90224227310125316,
                           0.59999999999999998,
                           0.39900591086536169,
                           0.26073868702666625,
                           0.16385821526516678,
                           0.095887068531518307,
                           0.049278068139809568,
                           0.019427286157550135,
                           0.0036240021641945307};
const double kMbdWeight[] = {0,
                             255.95773870445413,
                             21.460547728644038,
                             5.5809087182011936,
                             2.2053264790411649,
                             1.0839821838000383,
                             0.60861360457306679,
                             0.37317221033628684,
                             0.24309389031067308,
                             0.16503097021054849,
                             0.11493441875220682,
                             0.080845543303314407,
                             0.056323608919503651,
                             0.037645213621037406,
                             0.022498966849061337,
                             0.0093377589930294866};
constexpr int kMbdNfreq = 16;

struct Mat {
  int n = 0;
  std::vector<double> a;
  explicit Mat(int n_) : n(n_), a(static_cast<std::size_t>(n_ * n_), 0.0) {}
  double &operator()(int i, int j) { return a[static_cast<std::size_t>(i * n + j)]; }
  double operator()(int i, int j) const { return a[static_cast<std::size_t>(i * n + j)]; }
};

bool invert(Mat a, Mat *inv) {
  const int n = a.n;
  *inv = Mat(n);
  for (int i = 0; i < n; ++i) {
    (*inv)(i, i) = 1.0;
  }
  for (int col = 0; col < n; ++col) {
    int piv = col;
    double best = std::abs(a(col, col));
    for (int r = col + 1; r < n; ++r) {
      const double v = std::abs(a(r, col));
      if (v > best) {
        best = v;
        piv = r;
      }
    }
    if (!(best > 1e-14)) {
      return false;
    }
    if (piv != col) {
      for (int c = 0; c < n; ++c) {
        std::swap(a(col, c), a(piv, c));
        std::swap((*inv)(col, c), (*inv)(piv, c));
      }
    }
    const double diag = a(col, col);
    for (int c = 0; c < n; ++c) {
      a(col, c) /= diag;
      (*inv)(col, c) /= diag;
    }
    for (int r = 0; r < n; ++r) {
      if (r == col) {
        continue;
      }
      const double f = a(r, col);
      for (int c = 0; c < n; ++c) {
        a(r, c) -= f * a(col, c);
        (*inv)(r, c) -= f * (*inv)(col, c);
      }
    }
  }
  return true;
}

void jacobi(Mat a, std::vector<double> *eval) {
  const int n = a.n;
  eval->assign(static_cast<std::size_t>(n), 0.0);
  for (int sweep = 0; sweep < 100; ++sweep) {
    double off = 0.0;
    for (int p = 0; p < n; ++p) {
      for (int q = p + 1; q < n; ++q) {
        off += std::abs(a(p, q));
      }
    }
    if (off < 1e-14 * n) {
      break;
    }
    for (int p = 0; p < n; ++p) {
      for (int q = p + 1; q < n; ++q) {
        const double apq = a(p, q);
        if (std::abs(apq) < 1e-18) {
          continue;
        }
        const double app = a(p, p);
        const double aqq = a(q, q);
        const double tau = (aqq - app) / (2.0 * apq);
        const double t = std::copysign(1.0 / (std::abs(tau) + std::sqrt(1.0 + tau * tau)), tau);
        const double c = 1.0 / std::sqrt(1.0 + t * t);
        const double s = t * c;
        a(p, p) = app - t * apq;
        a(q, q) = aqq + t * apq;
        a(p, q) = 0.0;
        a(q, p) = 0.0;
        for (int r = 0; r < n; ++r) {
          if (r == p || r == q) {
            continue;
          }
          const double arp = a(r, p);
          const double arq = a(r, q);
          a(r, p) = c * arp - s * arq;
          a(p, r) = a(r, p);
          a(r, q) = s * arp + c * arq;
          a(q, r) = a(r, q);
        }
      }
    }
  }
  for (int i = 0; i < n; ++i) {
    (*eval)[static_cast<std::size_t>(i)] = a(i, i);
  }
}

struct TB {
  double t[3][3];
};

TB t_bare(double x, double y, double z) {
  const double r2 = x * x + y * y + z * z;
  const double r = std::sqrt(r2);
  const double r5 = r2 * r2 * r;
  TB t{};
  const double rvec[3] = {x, y, z};
  for (int a = 0; a < 3; ++a) {
    for (int b = 0; b < 3; ++b) {
      const double delta = a == b ? r2 : 0.0;
      t.t[a][b] = (-3.0 * rvec[a] * rvec[b] + delta) / r5;
    }
  }
  return t;
}

TB t_erf(double x, double y, double z, double sigma) {
  const TB bare = t_bare(x, y, z);
  const double r = std::sqrt(x * x + y * y + z * z);
  const double r5 = r * r * r * r * r;
  const double zeta = r / sigma;
  const double theta = 2.0 * zeta / std::sqrt(kPi) * std::exp(-zeta * zeta);
  const double erf_theta = std::erf(zeta) - theta;
  const double extra = 2.0 * zeta * zeta * theta;
  const double rvec[3] = {x, y, z};
  TB t{};
  for (int a = 0; a < 3; ++a) {
    for (int b = 0; b < 3; ++b) {
      t.t[a][b] = erf_theta * bare.t[a][b] + extra * rvec[a] * rvec[b] / r5;
    }
  }
  return t;
}

double fermi(double r, double s_vdw, double a) {
  return 1.0 / (1.0 + std::exp(-a * (r / s_vdw - 1.0)));
}

void add_block(Mat *m, int i, int j, double scale, const TB &t) {
  for (int a = 0; a < 3; ++a) {
    for (int b = 0; b < 3; ++b) {
      (*m)(3 * i + a, 3 * j + b) += scale * t.t[a][b];
    }
  }
}

struct Scaled {
  std::vector<double> alpha;
  std::vector<double> c6;
  std::vector<double> rvdw;
};

Scaled scale_ts(const int *z, std::size_t n, const std::vector<double> &h,
                const std::vector<double> *c6ratio) {
  Scaled s;
  s.alpha.resize(n);
  s.c6.resize(n);
  s.rvdw.resize(n);
  for (std::size_t i = 0; i < n; ++i) {
    const int zi = z[i];
    if (zi < 1 || zi > kTsZMax) {
      throw std::invalid_argument("MBD atomic number is outside the TS table");
    }
    const double hi = h[i];
    const double ratio = c6ratio != nullptr ? (*c6ratio)[i] : hi * hi;
    s.alpha[i] = kTsAlpha[zi - 1] * hi;
    s.c6[i] = kTsC6[zi - 1] * ratio;
    s.rvdw[i] = kTsRvdw[zi - 1] * std::cbrt(hi);
    if (!(s.alpha[i] > 0.0) || !(s.c6[i] > 0.0) || !(s.rvdw[i] > 0.0)) {
      throw std::invalid_argument("MBD parameters must be positive");
    }
  }
  return s;
}

void accumulate_T(Mat *m, const std::vector<Pair> &pairs, const std::vector<double> &alpha,
                  const std::vector<double> &rvdw, double beta, double a, bool gg) {
  const int n = static_cast<int>(alpha.size());
  for (const Pair &pr : pairs) {
    const double rb = pr.r / kBohr;
    if (!(rb > 1e-8)) {
      continue;
    }
    const double s_vdw = beta * (rvdw[static_cast<std::size_t>(pr.i)] + rvdw[static_cast<std::size_t>(pr.j)]);
    const double f = fermi(rb, s_vdw, a);
    const double x = pr.x / kBohr;
    const double y = pr.y / kBohr;
    const double z = pr.z / kBohr;
    TB t{};
    if (gg) {
      const double ai = alpha[static_cast<std::size_t>(pr.i)];
      const double aj = alpha[static_cast<std::size_t>(pr.j)];
      const double si = std::cbrt(std::sqrt(2.0 / kPi) * ai / 3.0);
      const double sj = std::cbrt(std::sqrt(2.0 / kPi) * aj / 3.0);
      const double sigma = std::sqrt(si * si + sj * sj);
      t = t_erf(x, y, z, sigma);
      for (int u = 0; u < 3; ++u) {
        for (int v = 0; v < 3; ++v) {
          t.t[u][v] *= (1.0 - f);
        }
      }
    } else {
      t = t_bare(x, y, z);
      for (int u = 0; u < 3; ++u) {
        for (int v = 0; v < 3; ++v) {
          t.t[u][v] *= f;
        }
      }
    }
    const double mirror = pr.i == pr.j ? 2.0 : 1.0;
    add_block(m, pr.i, pr.j, mirror, t);
    if (pr.i != pr.j) {
      add_block(m, pr.j, pr.i, 1.0, t);
    }
    (void)n;
  }
}

#include "rgpot/LongRange/MbdEwald.inc"

double mbd_energy_hartree(const double *pos, std::size_t n, const int *z, const std::vector<Pair> &pairs,
                          const std::vector<double> &h, const std::vector<double> *c6ratio,
                          const DispersionConfig &cfg, const double *box) {
  if (cfg.periodic) {
    return mbd_periodic_hartree(pos, n, z, box, h, c6ratio, cfg);
  }
  std::vector<Pair> gas;
  const std::vector<Pair> *use = &pairs;
  if (!cfg.periodic) {
    for (std::size_t i = 0; i < n; ++i) {
      for (std::size_t j = i + 1; j < n; ++j) {
        Pair p;
        p.i = static_cast<int>(i);
        p.j = static_cast<int>(j);
        p.x = pos[3 * j] - pos[3 * i];
        p.y = pos[3 * j + 1] - pos[3 * i + 1];
        p.z = pos[3 * j + 2] - pos[3 * i + 2];
        p.r = std::sqrt(p.x * p.x + p.y * p.y + p.z * p.z);
        gas.push_back(p);
      }
    }
    use = &gas;
  }
  const std::vector<Pair> &pp = *use;
  const Scaled base = scale_ts(z, n, h, c6ratio);
  std::vector<double> alpha_w(static_cast<std::size_t>(kMbdNfreq) * n);
  for (int f = 0; f < kMbdNfreq; ++f) {
    for (std::size_t i = 0; i < n; ++i) {
      const double omega = (4.0 / 3.0) * base.c6[i] / (base.alpha[i] * base.alpha[i]);
      const double u = kMbdFreq[f];
      const double ad = base.alpha[i] / (1.0 + (u / omega) * (u / omega));
      alpha_w[static_cast<std::size_t>(f) * n + i] = ad;
    }
    Mat tgg(static_cast<int>(3 * n));
    std::vector<double> ad(n);
    for (std::size_t i = 0; i < n; ++i) {
      ad[i] = alpha_w[static_cast<std::size_t>(f) * n + i];
      for (int a = 0; a < 3; ++a) {
        tgg(3 * static_cast<int>(i) + a, 3 * static_cast<int>(i) + a) = 1.0 / ad[i];
      }
    }
    accumulate_T(&tgg, pp, ad, base.rvdw, cfg.mbd_beta, cfg.mbd_a, true);
    Mat inv(0);
    if (!invert(tgg, &inv)) {
      throw std::runtime_error("MBD screening matrix is singular");
    }
    for (std::size_t i = 0; i < n; ++i) {
      double tr = 0.0;
      for (int a = 0; a < 3; ++a) {
        for (std::size_t j = 0; j < n; ++j) {
          tr += inv(3 * static_cast<int>(i) + a, 3 * static_cast<int>(j) + a);
        }
      }
      alpha_w[static_cast<std::size_t>(f) * n + i] = tr / 3.0;
    }
  }
  std::vector<double> alpha(n), c6(n), rvdw(n), omega(n);
  for (std::size_t i = 0; i < n; ++i) {
    alpha[i] = alpha_w[i];
    double acc = 0.0;
    for (int f = 0; f < kMbdNfreq; ++f) {
      const double aw = alpha_w[static_cast<std::size_t>(f) * n + i];
      acc += kMbdWeight[f] * aw * aw;
    }
    c6[i] = 3.0 / kPi * acc;
    rvdw[i] = base.rvdw[i] * std::cbrt(alpha[i] / base.alpha[i]);
    omega[i] = (4.0 / 3.0) * c6[i] / (alpha[i] * alpha[i]);
  }
  Mat ham(static_cast<int>(3 * n));
  for (std::size_t i = 0; i < n; ++i) {
    for (int a = 0; a < 3; ++a) {
      ham(3 * static_cast<int>(i) + a, 3 * static_cast<int>(i) + a) = omega[i] * omega[i];
    }
  }
  // off-diagonal blocks are pre_i pre_j T, with pre = omega sqrt(alpha).
  // accumulate_T writes T. Scale per block afterwards by walking pairs again.
  Mat bare(static_cast<int>(3 * n));
  accumulate_T(&bare, pp, alpha, rvdw, cfg.mbd_beta, cfg.mbd_a, false);
  for (int i = 0; i < static_cast<int>(n); ++i) {
    const double pre_i = omega[static_cast<std::size_t>(i)] * std::sqrt(alpha[static_cast<std::size_t>(i)]);
    for (int j = 0; j < static_cast<int>(n); ++j) {
      const double pre = pre_i * omega[static_cast<std::size_t>(j)] *
                         std::sqrt(alpha[static_cast<std::size_t>(j)]);
      for (int a = 0; a < 3; ++a) {
        for (int b = 0; b < 3; ++b) {
          ham(3 * i + a, 3 * j + b) += pre * bare(3 * i + a, 3 * j + b);
        }
      }
    }
  }
  std::vector<double> eval;
  jacobi(ham, &eval);
  double ene = 0.0;
  double wsum = 0.0;
  for (std::size_t i = 0; i < n; ++i) {
    wsum += omega[i];
  }
  for (double lam : eval) {
    if (lam < -1e-8) {
      throw std::runtime_error("MBD eigenvalue is negative");
    }
    ene += std::sqrt(std::max(lam, 0.0));
  }
  return 0.5 * ene - 1.5 * wsum;
}

class MbdModel final : public DispersionModel {
public:
  [[nodiscard]] const char *name() const override { return "MBD@rsSCS"; }
  [[nodiscard]] DispersionKind kind() const override { return DispersionKind::MBD; }
  void evaluate(const ForceInput &in, const std::vector<Pair> &pairs, const std::vector<double> &h,
                const std::vector<double> *c6, const DispersionConfig &cfg, double *energy, double *F,
                double strain[6], int *has_stress) const override {
    const double eh = mbd_energy_hartree(in.pos, in.nAtoms, in.atmnrs, pairs, h, c6, cfg, in.box);
    *energy = eh * kHartree;
    *has_stress = 0;
    std::fill(strain, strain + 6, 0.0);
    if (F == nullptr) {
      return;
    }
    constexpr double step = 1e-5;
    for (std::size_t i = 0; i < in.nAtoms; ++i) {
      for (int c = 0; c < 3; ++c) {
        std::vector<double> pos(in.pos, in.pos + 3 * in.nAtoms);
        pos[3 * i + static_cast<std::size_t>(c)] += step;
        const double ep =
            mbd_energy_hartree(pos.data(), in.nAtoms, in.atmnrs, pairs, h, c6, cfg, in.box);
        pos[3 * i + static_cast<std::size_t>(c)] -= 2.0 * step;
        const double em =
            mbd_energy_hartree(pos.data(), in.nAtoms, in.atmnrs, pairs, h, c6, cfg, in.box);
        const double dedr = (ep - em) / (2.0 * step) * kHartree;
        F[3 * i + static_cast<std::size_t>(c)] += -dedr;
      }
    }
  }

};

double gamma_of(double alpha) {
  const double fine = 7.2973525693e-3;
  const double radius = std::pow(fine, -4.0 / 21.0) * std::pow(alpha, 1.0 / 7.0);
  const double sigma = -0.00078893 * radius * radius * radius + 0.04125273 * radius * radius +
                       0.24428889 * radius - 0.00433008;
  return 0.5 / (sigma * sigma);
}

class QdoModel final : public DispersionModel {
public:
  [[nodiscard]] const char *name() const override { return "QDO"; }
  [[nodiscard]] DispersionKind kind() const override { return DispersionKind::QDO; }
  void evaluate(const ForceInput &in, const std::vector<Pair> &pairs, const std::vector<double> &h,
                const std::vector<double> *c6ratio, const DispersionConfig &cfg, double *energy,
                double *F, double strain[6], int *has_stress) const override {
    const bool legacy = cfg.qdo_legacy;
    const int zmax = legacy ? kQdoLegacyZMax : kQdoZMax;
    const double *alpha_tab = legacy ? kQdoAlphaLegacy : kQdoAlpha;
    const double *c6_tab = legacy ? kQdoC6Legacy : kQdoC6;
    std::vector<double> alpha(in.nAtoms), c6(in.nAtoms);
    for (std::size_t i = 0; i < in.nAtoms; ++i) {
      const int z = in.atmnrs[i];
      if (z < 1 || z > zmax) {
        throw std::invalid_argument("QDO atomic number is outside the free-atom table");
      }
      const double hi = h[i];
      const double ratio = c6ratio != nullptr ? (*c6ratio)[i] : hi * hi;
      alpha[i] = alpha_tab[z - 1] * hi;
      c6[i] = c6_tab[z - 1] * ratio;
    }
    int nt = 1;
#ifdef _OPENMP
    nt = omp_get_max_threads();
#endif
    std::vector<Neumaier> enes(static_cast<std::size_t>(nt));
    std::vector<std::vector<double>> forces(static_cast<std::size_t>(nt));
    std::vector<std::array<double, 6>> strains(static_cast<std::size_t>(nt));
    if (F != nullptr) {
      for (auto &buf : forces) {
        buf.assign(3 * in.nAtoms, 0.0);
      }
    }
#ifdef _OPENMP
#pragma omp parallel
#endif
    {
#ifdef _OPENMP
      const int tid = omp_get_thread_num();
#else
      const int tid = 0;
#endif
      auto &ene = enes[static_cast<std::size_t>(tid)];
      auto &floc = forces[static_cast<std::size_t>(tid)];
      auto &sloc = strains[static_cast<std::size_t>(tid)];
#ifdef _OPENMP
#pragma omp for schedule(static)
#endif
      for (std::size_t ti = 0; ti < pairs.size(); ++ti) {
      const Pair &pr = pairs[ti];
      const double R = pr.r / kBohr;
      if (!(R > 1e-8)) {
        continue;
      }
      const double ai = alpha[static_cast<std::size_t>(pr.i)];
      const double aj = alpha[static_cast<std::size_t>(pr.j)];
      const double ci = c6[static_cast<std::size_t>(pr.i)];
      const double cj = c6[static_cast<std::size_t>(pr.j)];
      const double aij = 0.5 * (ai + aj);
      const double denom = ai * ai * cj + aj * aj * ci;
      const double c6ij = 2.0 * ci * cj * ai * aj / denom;
      const double gamma = gamma_of(aij);
      const double c8 = 5.0 / gamma * c6ij;
      const double c10 = 245.0 / 8.0 / (gamma * gamma) * c6ij;
      const double p = cfg.qdo_scale * 2.0 * 2.54 * std::pow(aij, 1.0 / 7.0);
      auto u = [](double Rpow, double ppow) { return Rpow + ppow; };
      const double r6 = std::pow(R, 6.0);
      const double r8 = std::pow(R, 8.0);
      const double r10 = std::pow(R, 10.0);
      const double p6 = std::pow(p, 6.0);
      const double p8 = std::pow(p, 8.0);
      const double p10 = std::pow(p, 10.0);
      double vh = 0.0;
      double dv = 0.0;
      if (cfg.qdo_damped) {
        const double u6 = u(r6, p6);
        const double u8 = u(r8, p8);
        const double u10 = u(r10, p10);
        vh = -c6ij / u6 - c8 / u8 - c10 / u10;
        dv = c6ij * 6.0 * std::pow(R, 5.0) / (u6 * u6) + c8 * 8.0 * std::pow(R, 7.0) / (u8 * u8) +
             c10 * 10.0 * std::pow(R, 9.0) / (u10 * u10);
      } else {
        vh = -c6ij / r6 - c8 / r8 - c10 / r10;
        dv = c6ij * 6.0 / std::pow(R, 7.0) + c8 * 8.0 / std::pow(R, 9.0) + c10 * 10.0 / std::pow(R, 11.0);
      }
      double w = 1.0;
      double dw = 0.0;
      if (cfg.switch_width > 0.0) {
        const double on = cfg.cutoff - cfg.switch_width;
        if (pr.r >= cfg.cutoff) {
          continue;
        }
        if (pr.r > on) {
          const double c = (pr.r - on) / cfg.switch_width;
          auto sig = [](double x) { return x > 0.0 ? std::exp(-1.0 / x) : 0.0; };
          auto sigp = [](double x) { return x > 0.0 ? std::exp(-1.0 / x) / (x * x) : 0.0; };
          const double s1 = sig(1.0 - c);
          const double s2 = sig(c);
          const double den = s1 + s2;
          w = s1 / den;
          const double ds1 = -sigp(1.0 - c);
          const double ds2 = sigp(c);
          dw = ((ds1 * den - s1 * (ds1 + ds2)) / (den * den)) / cfg.switch_width;
        }
      }
      const double e_ev = w * vh * kHartree;
      const double dedr_A = (dw * vh * kHartree) + w * (dv * kHartree / kBohr);
      ene.add(e_ev);
      if (F != nullptr) {
        const double invr = 1.0 / pr.r;
        const double fx = dedr_A * pr.x * invr;
        floc[3 * pr.i] += fx;
        floc[3 * pr.i + 1] += dedr_A * pr.y * invr;
        floc[3 * pr.i + 2] += dedr_A * pr.z * invr;
        floc[3 * pr.j] -= fx;
        floc[3 * pr.j + 1] -= dedr_A * pr.y * invr;
        floc[3 * pr.j + 2] -= dedr_A * pr.z * invr;
        accumulatePairStrain(sloc.data(), -dedr_A * invr, -pr.x, -pr.y, -pr.z);
      }
      }
    }
    Neumaier ene;
    for (const auto &part : enes) {
      ene.add(part);
    }
    std::fill(strain, strain + 6, 0.0);
    for (const auto &sloc : strains) {
      for (int k = 0; k < 6; ++k) {
        strain[k] += sloc[k];
      }
    }
    if (F != nullptr) {
      for (const auto &floc : forces) {
        for (std::size_t i = 0; i < floc.size(); ++i) {
          F[i] += floc[i];
        }
      }
    }
    *energy = ene.total();
    *has_stress = 1;
  }
};

#if defined(RGPOT_HAS_DFTD3)
class D3Model final : public DispersionModel {
public:
  [[nodiscard]] const char *name() const override { return "D3BJ"; }
  [[nodiscard]] DispersionKind kind() const override { return DispersionKind::D3BJ; }
  void evaluate(const ForceInput &in, const std::vector<Pair> &pairs, const std::vector<double> &,
                const std::vector<double> *, const DispersionConfig &cfg, double *energy, double *F,
                double strain[6], int *has_stress) const override {
    D3BjParam bj;
    if (!d3bj_param(cfg.functional.c_str(), &bj)) {
      throw std::runtime_error("s-dftd3 rational damping failed");
    }
    if (!cfg.atm) {
      bj.s9 = 0.0;
    }
    const double bohr = units::BOHR_TO_ANGSTROM;
    const double cut2 = (cfg.lib_disp2 > 0.0 ? cfg.lib_disp2 : 60.0) * bohr;
    const double cut3 = (cfg.lib_disp3 > 0.0 ? cfg.lib_disp3 : 40.0) * bohr;
    const double cutn = (cfg.lib_cn > 0.0 ? cfg.lib_cn : 40.0) * bohr;
    d3bj_evaluate(in.pos, in.atmnrs, in.nAtoms, in.box, cfg.periodic && in.box != nullptr, pairs, bj,
                  cut2, cut3, cutn, energy, F, strain);
    *has_stress = 1;
  }
};
#endif

#if defined(RGPOT_HAS_DFTD4)
class D4Model final : public DispersionModel {
public:
  D4Model() {
    err_ = dftd4_new_error();
    if (!err_) {
      throw std::runtime_error("dftd4_new_error failed");
    }
  }
  ~D4Model() override {
    if (param_) dftd4_delete_param(&param_);
    if (model_) dftd4_delete_model(&model_);
    if (mol_) dftd4_delete_structure(&mol_);
    if (err_) dftd4_delete_error(&err_);
  }
  [[nodiscard]] const char *name() const override { return "D4"; }
  [[nodiscard]] DispersionKind kind() const override { return DispersionKind::D4; }
  void evaluate(const ForceInput &in, const std::vector<Pair> &, const std::vector<double> &,
                const std::vector<double> *, const DispersionConfig &cfg, double *energy, double *F,
                double strain[6], int *has_stress) const override {
    const bool periodic = cfg.periodic && in.box != nullptr;
    // The dense dftd4 factorization is the reference for the gas-phase sets.
    // Periodic and large systems use the cutoff conjugate-gradient kernel.
    if (periodic || in.nAtoms > 256) {
      D4BjParam bj;
      if (!d4bj_param(cfg.functional.c_str(), &bj)) {
        throw std::invalid_argument("dftd4 load param failed");
      }
      if (!cfg.atm) {
        bj.s9 = 0.0;
      }
      const double bohr = units::BOHR_TO_ANGSTROM;
      const double cut2 = (cfg.lib_disp2 > 0.0 ? cfg.lib_disp2 : 60.0) * bohr;
      const double cut3 = (cfg.lib_disp3 > 0.0 ? cfg.lib_disp3 : 40.0) * bohr;
      const double cutn = (cfg.lib_cn > 0.0 ? cfg.lib_cn : 30.0) * bohr;
      d4_evaluate(in.pos, in.atmnrs, in.nAtoms, in.box, periodic, cfg.charge, bj, cut2, cut3, cutn,
                  energy, F, strain);
      *has_stress = 1;
      return;
    }
    if (!param_ || functional_ != cfg.functional || atm_ != cfg.atm) {
      if (param_) dftd4_delete_param(&param_);
      std::vector<char> method(cfg.functional.begin(), cfg.functional.end());
      method.push_back('\0');
      param_ = dftd4_load_rational_damping(err_, method.data(), cfg.atm);
      if (dftd4_check_error(err_) != 0 || !param_) {
        throw std::runtime_error("dftd4 load param failed");
      }
      functional_ = cfg.functional;
      atm_ = cfg.atm;
    }
    const int nat = static_cast<int>(in.nAtoms);
    pos_.resize(static_cast<std::size_t>(nat) * 3);
    for (std::size_t i = 0; i < pos_.size(); ++i) {
      pos_[i] = in.pos[i] * units::ANGSTROM_TO_BOHR;
    }
    double box[9] = {};
    const bool pbc[3] = {periodic, periodic, periodic};
    for (int i = 0; i < 9; ++i) {
      box[i] = periodic ? in.box[i] * units::ANGSTROM_TO_BOHR : (i % 4 == 0 ? 100.0 : 0.0);
    }
    if (!mol_ || nat_ != nat || charge_ != cfg.charge) {
      if (model_) dftd4_delete_model(&model_);
      if (mol_) dftd4_delete_structure(&mol_);
      mol_ = dftd4_new_structure(err_, nat, in.atmnrs, pos_.data(), &cfg.charge, box, pbc);
      model_ = dftd4_new_d4_model(err_, mol_);
      nat_ = nat;
      charge_ = cfg.charge;
    } else {
      dftd4_update_structure(err_, mol_, pos_.data(), box);
    }
    if (dftd4_check_error(err_) != 0) {
      throw std::runtime_error("dftd4 structure update failed");
    }
    if (cfg.lib_disp2 > 0.0 && cfg.lib_disp3 > 0.0 && cfg.lib_cn > 0.0) {
      dftd4_set_model_realspace_cutoff(err_, model_, cfg.lib_disp2, cfg.lib_disp3, cfg.lib_cn);
      if (dftd4_check_error(err_) != 0) {
        throw std::runtime_error("dftd4 realspace cutoff failed");
      }
    }
    double eh = 0.0;
    std::vector<double> grad(static_cast<std::size_t>(nat) * 3);
    double sigma[9] = {};
    dftd4_get_dispersion(err_, mol_, model_, param_, &eh, grad.data(), sigma);
    if (dftd4_check_error(err_) != 0) {
      throw std::runtime_error("dftd4 dispersion failed");
    }
    *energy = eh * kHartree;
    if (F != nullptr) {
      for (std::size_t i = 0; i < grad.size(); ++i) {
        F[i] += grad[i] * units::NEG_GRAD_TO_FORCE;
      }
    }
    strain[0] = sigma[0] * kHartree;
    strain[1] = sigma[4] * kHartree;
    strain[2] = sigma[8] * kHartree;
    strain[3] = sigma[5] * kHartree;
    strain[4] = sigma[2] * kHartree;
    strain[5] = sigma[1] * kHartree;
    *has_stress = 1;
  }

private:
  mutable dftd4_error err_ = nullptr;
  mutable dftd4_structure mol_ = nullptr;
  mutable dftd4_model model_ = nullptr;
  mutable dftd4_param param_ = nullptr;
  mutable std::string functional_;
  mutable bool atm_ = true;
  mutable int nat_ = 0;
  mutable double charge_ = 0.0;
  mutable std::vector<double> pos_;
};
#endif

class MissingModel final : public DispersionModel {
public:
  explicit MissingModel(DispersionKind kind) : kind_(kind) {}
  [[nodiscard]] const char *name() const override { return "unavailable"; }
  [[nodiscard]] DispersionKind kind() const override { return kind_; }
  void evaluate(const ForceInput &, const std::vector<Pair> &, const std::vector<double> &,
                const std::vector<double> *, const DispersionConfig &, double *, double *, double *,
                int *) const override {
    throw std::runtime_error("this dispersion library was not linked");
  }

private:
  DispersionKind kind_;
};

} // namespace

void UnitRatios::load(const ForceInput &in, std::vector<double> *h, std::vector<double> *c6) const {
  h->assign(in.nAtoms, 1.0);
  c6->assign(in.nAtoms, 1.0);
}

FixedRatios::FixedRatios(std::vector<double> h, std::vector<double> c6)
    : h_(std::move(h)), c6_(std::move(c6)), have_c6_(!c6_.empty()) {}

void FixedRatios::load(const ForceInput &in, std::vector<double> *h, std::vector<double> *c6) const {
  if (h_.size() != in.nAtoms) {
    throw std::invalid_argument("ratio vector does not match the atom count");
  }
  *h = h_;
  if (have_c6_) {
    if (c6_.size() != in.nAtoms) {
      throw std::invalid_argument("C6 ratio vector does not match the atom count");
    }
    *c6 = c6_;
  } else {
    c6->clear();
  }
}

std::unique_ptr<DispersionModel> make_dispersion(DispersionKind kind) {
  switch (kind) {
  case DispersionKind::MBD:
    return std::make_unique<MbdModel>();
  case DispersionKind::QDO:
    return std::make_unique<QdoModel>();
  case DispersionKind::D3BJ:
#if defined(RGPOT_HAS_DFTD3)
    return std::make_unique<D3Model>();
#else
    return std::make_unique<MissingModel>(kind);
#endif
  case DispersionKind::D4:
#if defined(RGPOT_HAS_DFTD4)
    return std::make_unique<D4Model>();
#else
    return std::make_unique<MissingModel>(kind);
#endif
  }
  throw std::invalid_argument("unknown dispersion kind");
}

DispersionPot::DispersionPot(DispersionConfig cfg, std::shared_ptr<RatioSource> ratios,
                             std::shared_ptr<RatioResponse> response)
    : Potential(PotType::LrDispersion), cfg_(std::move(cfg)), ratios_(std::move(ratios)),
      response_(std::move(response)),
      lists_(std::make_shared<DualList>(cfg_.short_cutoff, cfg_.cutoff, cfg_.skin)),
      model_(make_dispersion(cfg_.kind)) {
  if (!ratios_) {
    ratios_ = std::make_shared<UnitRatios>();
  }
  Fnv1a fp;
  fp.u64(2);
  fp.u64(static_cast<uint64_t>(cfg_.kind));
  fp.f64(cfg_.charge);
  fp.f64(cfg_.mbd_beta);
  fp.f64(cfg_.mbd_a);
  fp.i64(cfg_.mbd_k[0]);
  fp.i64(cfg_.mbd_k[1]);
  fp.i64(cfg_.mbd_k[2]);
  fp.f64(cfg_.qdo_scale);
  fp.f64(cfg_.cutoff);
  fp.u64(cfg_.qdo_legacy ? 1 : 0);
  fp.u64(cfg_.atm ? 1 : 0);
  params_ = fp.h;
}

void DispersionPot::forceImpl(const ForceInput &in, ForceOut *out) const {
  std::fill(out->F, out->F + 3 * in.nAtoms, 0.0);
  out->energy = 0.0;
  out->variance = 0.0;
  out->has_stress = 0;
  if (in.nAtoms > 0 && in.atmnrs == nullptr) {
    throw std::invalid_argument("dispersion needs atomic numbers");
  }
  std::vector<double> h;
  std::vector<double> c6;
  ratios_->load(in, &h, &c6);
  const std::vector<double> *c6ptr = c6.empty() ? nullptr : &c6;
  lists_->update(in.pos, in.nAtoms, in.box, cfg_.periodic);
  double strain[6] = {};
  int has = 0;
  model_->evaluate(in, lists_->long_pairs(), h, c6ptr, cfg_, &out->energy, out->F, strain, &has);
  if (response_ && in.nAtoms > 0) {
    const double *dh = nullptr;
    const double *dc = nullptr;
    response_->jacobians(in, &dh, &dc);
    constexpr double eps = 1e-6;
    for (std::size_t i = 0; i < in.nAtoms; ++i) {
      auto energy_of = [&](std::vector<double> hh, std::vector<double> cc) {
        double e = 0.0;
        double st[6] = {};
        int hs = 0;
        const std::vector<double> *cp = cc.empty() ? nullptr : &cc;
        model_->evaluate(in, lists_->long_pairs(), hh, cp, cfg_, &e, nullptr, st, &hs);
        return e;
      };
      if (dh != nullptr) {
        auto hp = h;
        auto hm = h;
        hp[i] += eps;
        hm[i] -= eps;
        const double dedh = (energy_of(hp, c6) - energy_of(hm, c6)) / (2.0 * eps);
        for (int c = 0; c < 3; ++c) {
          out->F[3 * i + static_cast<std::size_t>(c)] += -dedh * dh[3 * i + static_cast<std::size_t>(c)];
        }
      }
      if (dc != nullptr && c6ptr != nullptr) {
        auto cp = c6;
        auto cm = c6;
        cp[i] += eps;
        cm[i] -= eps;
        const double dedc = (energy_of(h, cp) - energy_of(h, cm)) / (2.0 * eps);
        for (int c = 0; c < 3; ++c) {
          out->F[3 * i + static_cast<std::size_t>(c)] += -dedc * dc[3 * i + static_cast<std::size_t>(c)];
        }
      }
    }
  }
  if (has) {
    const double volume = cellVolume(in.box);
    if (volume > 0.0) {
      publishCauchyStress(out, strain, volume);
    }
  }
}

} // namespace rgpot::lr
