// MIT License
// Copyright 2023--present rgpot developers

#include "rgpot/LongRange/CudaPath.hpp"

#include "rgpot/LongRange/CudaMath.cuh"
#include "rgpot/LongRange/Lattice.hpp"
#include "rgpot/units.hpp"

#include <cmath>
#include <complex>
#include <numbers>
#include <stdexcept>
#include <vector>

namespace rgpot::lr {
namespace {

constexpr double kKe = units::HARTREE_TO_EV * units::BOHR_TO_ANGSTROM;
constexpr double kPi = std::numbers::pi;

void require_inputs(const CudaCoulombIn &in) {
  if (in.pos == nullptr || in.q == nullptr || in.box == nullptr) {
    throw std::invalid_argument("cuda coulomb is missing positions, charges or a cell");
  }
  if (in.n == 0) {
    throw std::invalid_argument("cuda coulomb needs at least one atom");
  }
  if (!(in.alpha > 0.0) || !(in.cutoff > 0.0)) {
    throw std::invalid_argument("cuda coulomb alpha and cutoff must be positive");
  }
  if (in.method == 0 && in.kmax < 1) {
    throw std::invalid_argument("cuda Ewald kmax must be at least 1");
  }
  if (in.method == 1 && in.order < 3) {
    throw std::invalid_argument("cuda PME spline order must be at least 3");
  }
  for (std::size_t i = 0; i < in.n; ++i) {
    if (!std::isfinite(in.q[i]) || !std::isfinite(in.pos[3 * i])) {
      throw std::invalid_argument("cuda coulomb positions and charges must be finite");
    }
  }
}

void ewald_host(const CudaCoulombIn &in, double *energy, double *F) {
  const CellGeom cell = cell_from_box(in.box);
  double ene = 0.0;
  if (F != nullptr) {
    for (std::size_t i = 0; i < 3 * in.n; ++i) {
      F[i] = 0.0;
    }
  }
  for (std::size_t t = 0; t < in.npair; ++t) {
    const double r = in.pair_r[t];
    if (!(r > 0.0) || r >= in.cutoff) {
      continue;
    }
    double dedr = 0.0;
    ene += cuda::real_pair(kKe, in.alpha, in.q[in.pair_i[t]], in.q[in.pair_j[t]], r, &dedr);
    if (F != nullptr) {
      const double invr = 1.0 / r;
      const double fx = dedr * in.pair_x[t] * invr;
      const double fy = dedr * in.pair_y[t] * invr;
      const double fz = dedr * in.pair_z[t] * invr;
      const int i = in.pair_i[t];
      const int j = in.pair_j[t];
      F[3 * i] += fx;
      F[3 * i + 1] += fy;
      F[3 * i + 2] += fz;
      F[3 * j] -= fx;
      F[3 * j + 1] -= fy;
      F[3 * j + 2] -= fz;
    }
  }
  const double self = -kKe * in.alpha / std::sqrt(kPi);
  for (std::size_t i = 0; i < in.n; ++i) {
    ene += self * in.q[i] * in.q[i];
  }
  std::vector<double> s(3 * in.n);
  for (std::size_t i = 0; i < in.n; ++i) {
    double r[3] = {in.pos[3 * i], in.pos[3 * i + 1], in.pos[3 * i + 2]};
    double fs[3];
    cell.fractional(r, fs);
    s[3 * i] = fs[0];
    s[3 * i + 1] = fs[1];
    s[3 * i + 2] = fs[2];
  }
  const double pref = kKe * (2.0 * kPi) / cell.volume;
  const int mmax = in.kmax;
  for (int iz = -mmax; iz <= mmax; ++iz) {
    for (int iy = -mmax; iy <= mmax; ++iy) {
      for (int ix = -mmax; ix <= mmax; ++ix) {
        if (ix == 0 && iy == 0 && iz == 0) {
          continue;
        }
        double kv[3];
        for (int c = 0; c < 3; ++c) {
          kv[c] = 2.0 * kPi * (ix * cell.recip[0][c] + iy * cell.recip[1][c] + iz * cell.recip[2][c]);
        }
        const double k2 = kv[0] * kv[0] + kv[1] * kv[1] + kv[2] * kv[2];
        if (!(k2 > 0.0)) {
          continue;
        }
        double C = 0.0;
        double D = 0.0;
        for (std::size_t a = 0; a < in.n; ++a) {
          const double theta = 2.0 * kPi * (ix * s[3 * a] + iy * s[3 * a + 1] + iz * s[3 * a + 2]);
          C += in.q[a] * std::cos(theta);
          D += in.q[a] * std::sin(theta);
        }
        const double s2 = C * C + D * D;
        ene += cuda::recip_term(pref, in.alpha, k2, s2);
        if (F != nullptr) {
          const double f = std::exp(-k2 / (4.0 * in.alpha * in.alpha)) / k2;
          const double common = pref * f;
          for (std::size_t a = 0; a < in.n; ++a) {
            const double theta = 2.0 * kPi * (ix * s[3 * a] + iy * s[3 * a + 1] + iz * s[3 * a + 2]);
            const double g = 2.0 * in.q[a] * (D * std::cos(theta) - C * std::sin(theta));
            F[3 * a] += -common * g * kv[0];
            F[3 * a + 1] += -common * g * kv[1];
            F[3 * a + 2] += -common * g * kv[2];
          }
        }
      }
    }
  }
  ene += -kKe * kPi * in.total * in.total / (2.0 * cell.volume * in.alpha * in.alpha);
  *energy = ene;
}

int pow2_at_least(int n) {
  int p = 4;
  while (p < n) {
    p *= 2;
  }
  return p;
}

double bspline(int order, double x) {
  if (order == 1) {
    return (x >= 0.0 && x < 1.0) ? 1.0 : 0.0;
  }
  const double inv = 1.0 / static_cast<double>(order - 1);
  return x * inv * bspline(order - 1, x) + (order - x) * inv * bspline(order - 1, x - 1.0);
}

double bspline_derivative(int order, double x) {
  return bspline(order - 1, x) - bspline(order - 1, x - 1.0);
}

void fft_radix2(std::vector<std::complex<double>> &a, bool inverse) {
  const std::size_t n = a.size();
  for (std::size_t i = 1, j = 0; i < n; ++i) {
    std::size_t bit = n >> 1;
    for (; j & bit; bit >>= 1) {
      j ^= bit;
    }
    j ^= bit;
    if (i < j) {
      std::swap(a[i], a[j]);
    }
  }
  for (std::size_t len = 2; len <= n; len <<= 1) {
    const double ang = (inverse ? 2.0 : -2.0) * kPi / static_cast<double>(len);
    const std::complex<double> wlen(std::cos(ang), std::sin(ang));
    for (std::size_t i = 0; i < n; i += len) {
      std::complex<double> w(1.0, 0.0);
      for (std::size_t j = 0; j < len / 2; ++j) {
        const auto u = a[i + j];
        const auto v = a[i + j + len / 2] * w;
        a[i + j] = u + v;
        a[i + j + len / 2] = u - v;
        w *= wlen;
      }
    }
  }
  if (inverse) {
    const double inv = 1.0 / static_cast<double>(n);
    for (auto &z : a) {
      z *= inv;
    }
  }
}

void fft3(std::vector<std::complex<double>> &g, int nx, int ny, int nz, bool inverse) {
  auto index = [nx, ny](int x, int y, int z) {
    return static_cast<std::size_t>(x + nx * (y + ny * z));
  };
  std::vector<std::complex<double>> line(static_cast<std::size_t>(nx));
  for (int z = 0; z < nz; ++z) {
    for (int y = 0; y < ny; ++y) {
      for (int x = 0; x < nx; ++x) {
        line[static_cast<std::size_t>(x)] = g[index(x, y, z)];
      }
      fft_radix2(line, inverse);
      for (int x = 0; x < nx; ++x) {
        g[index(x, y, z)] = line[static_cast<std::size_t>(x)];
      }
    }
  }
  line.resize(static_cast<std::size_t>(ny));
  for (int z = 0; z < nz; ++z) {
    for (int x = 0; x < nx; ++x) {
      for (int y = 0; y < ny; ++y) {
        line[static_cast<std::size_t>(y)] = g[index(x, y, z)];
      }
      fft_radix2(line, inverse);
      for (int y = 0; y < ny; ++y) {
        g[index(x, y, z)] = line[static_cast<std::size_t>(y)];
      }
    }
  }
  line.resize(static_cast<std::size_t>(nz));
  for (int y = 0; y < ny; ++y) {
    for (int x = 0; x < nx; ++x) {
      for (int z = 0; z < nz; ++z) {
        line[static_cast<std::size_t>(z)] = g[index(x, y, z)];
      }
      fft_radix2(line, inverse);
      for (int z = 0; z < nz; ++z) {
        g[index(x, y, z)] = line[static_cast<std::size_t>(z)];
      }
    }
  }
}

void bmod_axis(int order, int n, std::vector<double> *mod) {
  mod->assign(static_cast<std::size_t>(n), 0.0);
  for (int m = 0; m < n; ++m) {
    std::complex<double> sum(0.0, 0.0);
    for (int k = 0; k < order - 1; ++k) {
      const double ang = 2.0 * kPi * static_cast<double>(m * k) / static_cast<double>(n);
      sum += bspline(order, static_cast<double>(k + 1)) *
             std::complex<double>(std::cos(ang), std::sin(ang));
    }
    // |sum|^2. The mesh kernel divides by this. |num/sum|^2 is the inverse.
    (*mod)[static_cast<std::size_t>(m)] = std::norm(sum);
  }
}

void pme_host(const CudaCoulombIn &in, double *energy, double *F) {
  // Real space is the Ewald pair kernel. Reciprocal space is the
  // cardinal B-spline mesh, run here on the host when no device is up.
  CudaCoulombIn real = in;
  real.method = 0;
  real.kmax = 1;
  double real_e = 0.0;
  ewald_host(real, &real_e, F);
  // ewald_host added a reciprocal shell and the background. Undo those
  // and replace them with the mesh contribution.
  const CellGeom cell = cell_from_box(in.box);
  const double ebg = -kKe * kPi * in.total * in.total / (2.0 * cell.volume * in.alpha * in.alpha);
  real_e -= ebg;
  const int mmax = 1;
  std::vector<double> s(3 * in.n);
  for (std::size_t i = 0; i < in.n; ++i) {
    double r[3] = {in.pos[3 * i], in.pos[3 * i + 1], in.pos[3 * i + 2]};
    double fs[3];
    cell.fractional(r, fs);
    s[3 * i] = fs[0];
    s[3 * i + 1] = fs[1];
    s[3 * i + 2] = fs[2];
  }
  const double pref = kKe * (2.0 * kPi) / cell.volume;
  for (int iz = -mmax; iz <= mmax; ++iz) {
    for (int iy = -mmax; iy <= mmax; ++iy) {
      for (int ix = -mmax; ix <= mmax; ++ix) {
        if (ix == 0 && iy == 0 && iz == 0) {
          continue;
        }
        double kv[3];
        for (int c = 0; c < 3; ++c) {
          kv[c] = 2.0 * kPi * (ix * cell.recip[0][c] + iy * cell.recip[1][c] + iz * cell.recip[2][c]);
        }
        const double k2 = kv[0] * kv[0] + kv[1] * kv[1] + kv[2] * kv[2];
        double C = 0.0;
        double D = 0.0;
        for (std::size_t a = 0; a < in.n; ++a) {
          const double theta = 2.0 * kPi * (ix * s[3 * a] + iy * s[3 * a + 1] + iz * s[3 * a + 2]);
          C += in.q[a] * std::cos(theta);
          D += in.q[a] * std::sin(theta);
        }
        real_e -= cuda::recip_term(pref, in.alpha, k2, C * C + D * D);
        if (F != nullptr) {
          const double f = std::exp(-k2 / (4.0 * in.alpha * in.alpha)) / k2;
          const double common = pref * f;
          for (std::size_t a = 0; a < in.n; ++a) {
            const double theta = 2.0 * kPi * (ix * s[3 * a] + iy * s[3 * a + 1] + iz * s[3 * a + 2]);
            const double g = 2.0 * in.q[a] * (D * std::cos(theta) - C * std::sin(theta));
            F[3 * a] -= -common * g * kv[0];
            F[3 * a + 1] -= -common * g * kv[1];
            F[3 * a + 2] -= -common * g * kv[2];
          }
        }
      }
    }
  }

  const int nx = pow2_at_least(std::max(in.grid, 4));
  const int ny = nx;
  const int nz = nx;
  const int order = in.order;
  const std::size_t ntot = static_cast<std::size_t>(nx) * static_cast<std::size_t>(ny) * static_cast<std::size_t>(nz);
  std::vector<std::complex<double>> grid(ntot);
  auto index = [nx, ny](int x, int y, int z) {
    return static_cast<std::size_t>(x + nx * (y + ny * z));
  };
  struct Sp {
    std::vector<int> g[3];
    std::vector<double> w[3];
    std::vector<double> d[3];
  };
  std::vector<Sp> atoms(in.n);
  const int naxis[3] = {nx, ny, nz};
  for (std::size_t i = 0; i < in.n; ++i) {
    double r[3] = {in.pos[3 * i], in.pos[3 * i + 1], in.pos[3 * i + 2]};
    double fs[3];
    cell.fractional(r, fs);
    double u[3];
    const double nd[3] = {static_cast<double>(nx), static_cast<double>(ny), static_cast<double>(nz)};
    for (int c = 0; c < 3; ++c) {
      u[c] = (fs[c] - std::floor(fs[c])) * nd[c];
    }
    auto &sp = atoms[i];
    for (int c = 0; c < 3; ++c) {
      sp.g[c].resize(static_cast<std::size_t>(order));
      sp.w[c].resize(static_cast<std::size_t>(order));
      sp.d[c].resize(static_cast<std::size_t>(order));
      const int k0 = static_cast<int>(std::floor(u[c])) - order + 1;
      for (int j = 0; j < order; ++j) {
        const int gv = k0 + j;
        int w = gv % naxis[c];
        if (w < 0) {
          w += naxis[c];
        }
        const double arg = u[c] - static_cast<double>(gv);
        sp.g[c][static_cast<std::size_t>(j)] = w;
        sp.w[c][static_cast<std::size_t>(j)] = bspline(order, arg);
        sp.d[c][static_cast<std::size_t>(j)] = bspline_derivative(order, arg);
      }
    }
    for (int iz = 0; iz < order; ++iz) {
      for (int iy = 0; iy < order; ++iy) {
        for (int ix = 0; ix < order; ++ix) {
          const double w = sp.w[0][static_cast<std::size_t>(ix)] * sp.w[1][static_cast<std::size_t>(iy)] *
                           sp.w[2][static_cast<std::size_t>(iz)];
          grid[index(sp.g[0][static_cast<std::size_t>(ix)], sp.g[1][static_cast<std::size_t>(iy)],
                     sp.g[2][static_cast<std::size_t>(iz)])] += in.q[i] * w;
        }
      }
    }
  }
  fft3(grid, nx, ny, nz, false);
  std::vector<double> bmx, bmy, bmz;
  bmod_axis(order, nx, &bmx);
  bmod_axis(order, ny, &bmy);
  bmod_axis(order, nz, &bmz);
  std::vector<std::complex<double>> phihat(ntot);
  const double mesh_pref = kKe / (2.0 * kPi * cell.volume);
  double rec = 0.0;
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
          mc[c] = static_cast<double>(sx) * cell.recip[0][c] + static_cast<double>(sy) * cell.recip[1][c] +
                  static_cast<double>(sz) * cell.recip[2][c];
        }
        const double m2 = mc[0] * mc[0] + mc[1] * mc[1] + mc[2] * mc[2];
        const double bden = bmx[static_cast<std::size_t>(ix)] * bmy[static_cast<std::size_t>(iy)] *
                            bmz[static_cast<std::size_t>(iz)];
        if (!(m2 > 0.0) || !(bden > 1e-12)) {
          continue;
        }
        const double factor = mesh_pref * std::exp(-kPi * kPi * m2 / (in.alpha * in.alpha)) / m2 / bden;
        const auto qh = grid[index(ix, iy, iz)];
        rec += factor * std::norm(qh);
        phihat[index(ix, iy, iz)] = (2.0 * static_cast<double>(ntot) * factor) * qh;
      }
    }
  }
  fft3(phihat, nx, ny, nz, true);
  if (F != nullptr) {
    for (std::size_t i = 0; i < in.n; ++i) {
      const auto &sp = atoms[i];
      double dQ[3] = {};
      for (int iz = 0; iz < order; ++iz) {
        for (int iy = 0; iy < order; ++iy) {
          for (int ix = 0; ix < order; ++ix) {
            const double wx = sp.w[0][static_cast<std::size_t>(ix)];
            const double wy = sp.w[1][static_cast<std::size_t>(iy)];
            const double wz = sp.w[2][static_cast<std::size_t>(iz)];
            const double ph = phihat[index(sp.g[0][static_cast<std::size_t>(ix)], sp.g[1][static_cast<std::size_t>(iy)],
                                            sp.g[2][static_cast<std::size_t>(iz)])]
                                  .real();
            const double qw = in.q[i];
            dQ[0] += qw * sp.d[0][static_cast<std::size_t>(ix)] * wy * wz * ph;
            dQ[1] += qw * wx * sp.d[1][static_cast<std::size_t>(iy)] * wz * ph;
            dQ[2] += qw * wx * wy * sp.d[2][static_cast<std::size_t>(iz)] * ph;
          }
        }
      }
      const double scale[3] = {static_cast<double>(nx), static_cast<double>(ny), static_cast<double>(nz)};
      for (int a = 0; a < 3; ++a) {
        double dE = 0.0;
        for (int d = 0; d < 3; ++d) {
          dE += dQ[d] * scale[d] * cell.inv[a][d];
        }
        F[3 * i + static_cast<std::size_t>(a)] += -dE;
      }
    }
  }
  *energy = real_e + rec + ebg;
}

} // namespace

void coulomb_host(const CudaCoulombIn &in, double *energy, double *F) {
  if (energy == nullptr) {
    throw std::invalid_argument("cuda coulomb energy pointer is null");
  }
  require_inputs(in);
  if (in.method == 0) {
    ewald_host(in, energy, F);
  } else if (in.method == 1) {
    pme_host(in, energy, F);
  } else {
    throw std::invalid_argument("cuda coulomb method is unknown");
  }
}

double pair_sum_host(const double *pair, std::size_t n) {
  if (pair == nullptr && n > 0) {
    throw std::invalid_argument("pair sum pointer is null");
  }
  double s = 0.0;
  for (std::size_t i = 0; i < n; ++i) {
    if (!std::isfinite(pair[i])) {
      throw std::invalid_argument("pair sum entry is not finite");
    }
    s += pair[i];
  }
  return s;
}

double damp_sum_host(const double *c6, const double *r0, const double *r, std::size_t n, double a1,
                     double a2) {
  if ((c6 == nullptr || r0 == nullptr || r == nullptr) && n > 0) {
    throw std::invalid_argument("damped pair sum is missing coefficients");
  }
  if (!(a2 > 0.0)) {
    throw std::invalid_argument("BJ a2 must be positive");
  }
  double s = 0.0;
  for (std::size_t i = 0; i < n; ++i) {
    if (!(r[i] > 0.0)) {
      throw std::invalid_argument("damped pair distance must be positive");
    }
    s += cuda::bj_pair(c6[i], r0[i], a1, a2, r[i]);
  }
  return s;
}

} // namespace rgpot::lr
