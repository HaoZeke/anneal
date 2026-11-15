// MIT License
// Copyright 2023--present rgpot developers

/**
 * @brief Tabulated EAM energy and forces.
 *
 * The resampling, spline and lookup follow PairEAM::file2array,
 * PairEAM::interpolate and PairEAM::compute of LAMMPS
 * (src/MANYBODY/pair_eam.cpp). The force on atom i along d = r_i - r_j is
 * fscale * d with fscale = -(F'_a rho'_ab + F'_b rho'_ba + phi'_ab) / r,
 * and with EAMTableConfig::stress the same pass sums -fscale * d (x) d.
 */

// clang-format off
#include <algorithm>
#include <array>
#include <cmath>
#include <cstddef>
#include <fstream>
#include <istream>
#include <stdexcept>
#include <string>
#include <vector>
// clang-format on

#include "rgpot/EAMTable/EAMTablePot.hpp"
#include "rgpot/nlist/NeighbourList.hpp"
#include "rgpot/stress.hpp"

namespace {

/// Hartree in eV times Bohr in Angstrom, as pair_eam.cpp writes it.
constexpr double kHartreeBohr = 27.2 * 0.529;

[[noreturn]] void bad(const std::string &what) {
  throw std::invalid_argument("EAM table: " + what);
}

rgpot::EAMTable load(const rgpot::EAMTableConfig &c) {
  std::ifstream in(c.path);
  if (!in) {
    bad("cannot open " + c.path);
  }
  return rgpot::read_eam_table(in, c.format);
}

std::vector<double> read_block(std::istream &in, int n) {
  std::vector<double> v(static_cast<std::size_t>(n));
  for (double &x : v) {
    in >> x;
  }
  if (!in) {
    bad("short or unreadable array");
  }
  return v;
}

/// Four-point Lagrange resampling of f (spacing fd) onto n points of
/// spacing d, the funcfl step of PairEAM::file2array.
std::vector<double> resample(const std::vector<double> &f, double fd, int n,
                             double d) {
  const double sixth = 1.0 / 6.0;
  const int nf = static_cast<int>(f.size());
  std::vector<double> out(static_cast<std::size_t>(n));
  for (int m = 1; m <= n; ++m) {
    double p = (m - 1) * d / fd + 1.0;
    const int k = std::max(std::min(static_cast<int>(p), nf - 2), 2);
    p = std::min(p - k, 2.0);
    const double c1 = -sixth * p * (p - 1.0) * (p - 2.0);
    const double c2 = 0.5 * (p * p - 1.0) * (p - 2.0);
    const double c3 = -0.5 * p * (p + 1.0) * (p - 2.0);
    const double c4 = sixth * p * (p * p - 1.0);
    out[m - 1] = c1 * f[k - 2] + c2 * f[k - 1] + c3 * f[k] + c4 * f[k + 1];
  }
  return out;
}

/// PairEAM::interpolate. Point m holds the cubic on [m, m + 1] in the
/// local coordinate p: c[3..6] for the value, c[0..2] for its derivative
/// in x.
std::vector<double> spline(const std::vector<double> &f, double delta) {
  const int n = static_cast<int>(f.size());
  std::vector<double> s(7 * f.size());
  auto c = [&](int m, int k) -> double & {
    return s[static_cast<std::size_t>(7 * (m - 1) + k)];
  };
  for (int m = 1; m <= n; ++m) {
    c(m, 6) = f[m - 1];
  }
  c(1, 5) = c(2, 6) - c(1, 6);
  c(2, 5) = 0.5 * (c(3, 6) - c(1, 6));
  c(n - 1, 5) = 0.5 * (c(n, 6) - c(n - 2, 6));
  c(n, 5) = c(n, 6) - c(n - 1, 6);
  for (int m = 3; m <= n - 2; ++m) {
    c(m, 5) =
        ((c(m - 2, 6) - c(m + 2, 6)) + 8.0 * (c(m + 1, 6) - c(m - 1, 6))) /
        12.0;
  }
  for (int m = 1; m <= n - 1; ++m) {
    c(m, 4) = 3.0 * (c(m + 1, 6) - c(m, 6)) - 2.0 * c(m, 5) - c(m + 1, 5);
    c(m, 3) = c(m, 5) + c(m + 1, 5) - 2.0 * (c(m + 1, 6) - c(m, 6));
  }
  c(n, 4) = 0.0;
  c(n, 3) = 0.0;
  for (int m = 1; m <= n; ++m) {
    c(m, 2) = c(m, 5) / delta;
    c(m, 1) = 2.0 * c(m, 4) / delta;
    c(m, 0) = 3.0 * c(m, 3) / delta;
  }
  return s;
}

/// The cubic on [m, m + 1] of a PairEAM::interpolate table and the local
/// coordinate p in it.
struct Knot {
  const double *c;
  double p;
  [[nodiscard]] double value() const {
    return ((c[3] * p + c[4]) * p + c[5]) * p + c[6];
  }
  [[nodiscard]] double slope() const { return (c[0] * p + c[1]) * p + c[2]; }
};

Knot lookup(const std::vector<double> &s, double x, double rdx, int n) {
  const double q = x * rdx + 1.0;
  const int m = std::max(1, std::min(static_cast<int>(q), n - 1));
  return {s.data() + 7 * static_cast<std::size_t>(m - 1), std::min(q - m, 1.0)};
}

} // namespace

namespace rgpot {

EAMTable read_eam_table(std::istream &in, EAMTableFormat format) {
  EAMTable t;
  std::string rest;
  auto element = [&] {
    int z = 0;
    double mass = 0.0;
    if (!(in >> z >> mass)) {
      bad("unreadable element line");
    }
    std::getline(in, rest);
    t.z.push_back(z);
  };
  auto grid = [&] {
    if (!(in >> t.nrho >> t.drho >> t.nr >> t.dr >> t.cutoff) || t.nrho < 5 ||
        t.nr < 5 || !(t.drho > 0.0) || !(t.dr > 0.0) || !(t.cutoff > 0.0)) {
      bad("unreadable grid line");
    }
    std::getline(in, rest);
  };

  if (format == EAMTableFormat::Funcfl) {
    std::getline(in, rest);
    element();
    grid();
    const auto f = read_block(in, t.nrho);
    const auto zr = read_block(in, t.nr);
    const auto rho = read_block(in, t.nr);
    t.rhomax = (t.nrho - 1) * t.drho;
    const int nrho = static_cast<int>(std::lround(t.rhomax / t.drho));
    const int nr = static_cast<int>(std::lround((t.nr - 1) * t.dr / t.dr));
    t.frho = {resample(f, t.drho, nrho, t.drho)};
    t.rhor = {resample(rho, t.dr, nr, t.dr)};
    auto z2 = resample(zr, t.dr, nr, t.dr);
    for (double &v : z2) {
      v = kHartreeBohr * v * v;
    }
    t.z2r = {z2};
    t.nrho = nrho;
    t.nr = nr;
    return t;
  }

  for (int k = 0; k < 3; ++k) {
    std::getline(in, rest);
  }
  int nel = 0;
  if (!(in >> nel) || nel < 1) {
    bad("unreadable element count");
  }
  std::getline(in, rest);
  grid();
  const int nd = (format == EAMTableFormat::FinnisSinclair) ? nel : 1;
  std::vector<std::vector<double>> dens;
  for (int b = 0; b < nel; ++b) {
    element();
    t.frho.push_back(read_block(in, t.nrho));
    for (int k = 0; k < nd; ++k) {
      dens.push_back(read_block(in, t.nr));
    }
  }
  for (int a = 0; a < nel; ++a) {
    for (int b = 0; b <= a; ++b) {
      t.z2r.push_back(read_block(in, t.nr));
    }
  }
  for (int b = 0; b < nel; ++b) {
    for (int a = 0; a < nel; ++a) {
      t.rhor.push_back(dens[static_cast<std::size_t>(b * nd + a % nd)]);
    }
  }
  t.rhomax = (t.nrho - 1) * t.drho;
  return t;
}

EAMTablePot::EAMTablePot(const EAMTableConfig &c)
    : EAMTablePot(load(c), c.periodic, c.stress) {}

EAMTablePot::EAMTablePot(const EAMTable &table, bool periodic, bool stress)
    : Potential(PotType::EAMTable), m_table{table}, m_periodic{periodic},
      m_stress{stress} {
  const std::size_t n = table.z.size();
  if (n == 0 || table.frho.size() != n || table.rhor.size() != n * n ||
      table.z2r.size() != n * (n + 1) / 2) {
    bad("array count does not match");
  }
  Fnv1a fp;
  fp.u64(kKernelVersion);
  fp.f64(table.drho);
  fp.f64(table.dr);
  fp.f64(table.cutoff);
  fp.f64(table.rhomax);
  for (const int z : table.z) {
    fp.i64(z);
  }
  auto add = [&](const std::vector<std::vector<double>> &src, int len, double d,
                 std::vector<std::vector<double>> &dst) {
    for (const auto &f : src) {
      if (static_cast<int>(f.size()) != len) {
        bad("array length does not match");
      }
      fp.u64(f.size());
      fp.bytes(f.data(), f.size() * sizeof(double));
      dst.push_back(spline(f, d));
    }
  };
  add(table.frho, table.nrho, table.drho, m_frho);
  add(table.rhor, table.nr, table.dr, m_rhor);
  add(table.z2r, table.nr, table.dr, m_z2r);
  fp.b8(periodic);
  if (stress) {
    fp.u64(1);
  }
  m_paramsKey = fp.h;
}

void EAMTablePot::forceImpl(const ForceInput &in, ForceOut *out) const {
  const auto n_size = in.nAtoms;
  double *F = out->F;
  std::fill(F, F + 3 * n_size, 0.0);
  out->energy = 0.0;
  out->variance = 0.0;
  const EAMTable &t = m_table;
  const auto nel = t.z.size();
  std::vector<std::size_t> el(n_size);
  for (std::size_t i = 0; i < n_size; ++i) {
    const auto it = std::find(t.z.begin(), t.z.end(), in.atmnrs[i]);
    if (it == t.z.end()) {
      bad("no element for atomic number " + std::to_string(in.atmnrs[i]));
    }
    el[i] = static_cast<std::size_t>(it - t.z.begin());
  }

  static constexpr double free_box[9] = {1.0e6, 0.0, 0.0, 0.0,  1.0e6,
                                         0.0,   0.0, 0.0, 1.0e6};
  const double *box = in.box;
  if (m_periodic) {
    const double vol = cellVolume(box);
    if (!(vol > 0.0)) {
      bad("periodic evaluation needs a cell");
    }
    // The pair list folds to one image per partner.
    for (int k = 0; k < 3; ++k) {
      const double *u = box + 3 * ((k + 1) % 3);
      const double *v = box + 3 * ((k + 2) % 3);
      const double cx = u[1] * v[2] - u[2] * v[1];
      const double cy = u[2] * v[0] - u[0] * v[2];
      const double cz = u[0] * v[1] - u[1] * v[0];
      if (!(2.0 * t.cutoff < vol / std::sqrt(cx * cx + cy * cy + cz * cz))) {
        bad("cutoff exceeds half a face spacing");
      }
    }
  } else {
    box = free_box;
  }
  // A local list keeps one instance safe to share across threads.
  nlist::Neighbours cells;
  const double cut2 = t.cutoff * t.cutoff;
  const double rdr = 1.0 / t.dr;
  const double rdrho = 1.0 / t.drho;

  const bool pairs = n_size >= 2;
  if (pairs) {
    nlist::updateCells(cells, in.pos, n_size, box, t.cutoff, 0.0,
                       nlist::Image::Round, nlist::Mode::Auto,
                       {m_periodic, m_periodic, m_periodic});
  }

  std::vector<double> rho(n_size, 0.0);
  if (pairs) {
    nlist::visitHalf(cells, 0, static_cast<std::int64_t>(n_size),
                  [&](int32_t i, int32_t j, double, double, double, double r2) {
                    if (!(r2 > 0.0) || !(r2 < cut2)) {
                      return;
                    }
                    const auto a = el[static_cast<std::size_t>(i)];
                    const auto b = el[static_cast<std::size_t>(j)];
                    const double r = std::sqrt(r2);
                    rho[static_cast<std::size_t>(i)] +=
                        lookup(m_rhor[b * nel + a], r, rdr, t.nr).value();
                    rho[static_cast<std::size_t>(j)] +=
                        lookup(m_rhor[a * nel + b], r, rdr, t.nr).value();
                  });
  }

  std::vector<double> fprime(n_size, 0.0);
  double energy = 0.0;
  for (std::size_t i = 0; i < n_size; ++i) {
    const Knot k = lookup(m_frho[el[i]], rho[i], rdrho, t.nrho);
    fprime[i] = k.slope();
    energy += k.value();
    if (rho[i] > t.rhomax) {
      energy += fprime[i] * (rho[i] - t.rhomax);
    }
  }

  double strain[6] = {};
  if (pairs) {
    nlist::visitHalf(
        cells, 0, static_cast<std::int64_t>(n_size),
        [&](int32_t i, int32_t j, double dx, double dy, double dz, double r2) {
          if (!(r2 > 0.0) || !(r2 < cut2)) {
            return;
          }
          const auto ui = static_cast<std::size_t>(i);
          const auto uj = static_cast<std::size_t>(j);
          const auto a = el[ui];
          const auto b = el[uj];
          const double r = std::sqrt(r2);
          const double rhoip =
              lookup(m_rhor[a * nel + b], r, rdr, t.nr).slope();
          const double rhojp =
              lookup(m_rhor[b * nel + a], r, rdr, t.nr).slope();
          const auto hi = std::max(a, b);
          const Knot z2 =
              lookup(m_z2r[hi * (hi + 1) / 2 + std::min(a, b)], r, rdr, t.nr);
          const double recip = 1.0 / r;
          const double phi = z2.value() * recip;
          const double phip = z2.slope() * recip - phi * recip;
          const double fscale =
              -(fprime[ui] * rhojp + fprime[uj] * rhoip + phip) * recip;
          energy += phi;
          F[3 * ui] += fscale * dx;
          F[3 * ui + 1] += fscale * dy;
          F[3 * ui + 2] += fscale * dz;
          F[3 * uj] -= fscale * dx;
          F[3 * uj + 1] -= fscale * dy;
          F[3 * uj + 2] -= fscale * dz;
          if (m_stress) {
            accumulatePairStrain(strain, fscale, dx, dy, dz);
          }
        });
  }

  out->energy = energy;
  if (m_stress) {
    publishCauchyStress(out, strain, cellVolume(in.box));
  }
}

} // namespace rgpot
