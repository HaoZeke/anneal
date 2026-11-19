#pragma once
// MIT License
// Copyright 2023--present rgpot developers

// Pair terms for the image walk. The stored neighbour walk keeps its
// own copy in EAMTablePot.cc, so that translation unit is compiled
// without this header.

#include <algorithm>
#include <cstddef>
#include <stdexcept>
#include <string>
#include <vector>

namespace {

[[noreturn]] void bad(const std::string &what) {
  throw std::invalid_argument("EAM table: " + what);
}

/// Where x falls in a PairEAM::interpolate table of n knots 1/rdx apart:
/// the offset of the coefficients of its cubic and the local coordinate in
/// it. The tables on one grid share it.
struct Place {
  std::size_t offset;
  double p;
};

Place place(double x, double rdx, int n) {
  const double q = x * rdx + 1.0;
  const int m = std::max(1, std::min(static_cast<int>(q), n - 1));
  return {7 * static_cast<std::size_t>(m - 1), std::min(q - m, 1.0)};
}

/// The cubic of a table at a Place.
struct Knot {
  const double *c;
  double p;
  Knot(const double *s, Place at) : c(s + at.offset), p(at.p) {}
  [[nodiscard]] double value() const {
    return ((c[3] * p + c[4]) * p + c[5]) * p + c[6];
  }
  [[nodiscard]] double slope() const { return (c[0] * p + c[1]) * p + c[2]; }
};

/// The radial tables of a pair of an a atom i and a b atom j: the density
/// j puts at i, the density i puts at j, and r phi_ab.
struct Couple {
  const double *atI;
  const double *atJ;
  const double *z2;
};

/// The terms of one EAMTablePot that compute() and the stages share.
struct Terms {
  const rgpot::EAMTable &t;
  const std::vector<std::vector<double>> &frho;
  const std::vector<std::vector<double>> &rhor;
  const std::vector<std::vector<double>> &z2r;
  std::size_t nel = t.z.size();
  double rdr = 1.0 / t.dr;
  double rdrho = 1.0 / t.drho;
  /// The Couple of an a atom and a b atom at a * nel + b.
  std::vector<Couple> couples = coupled();

  [[nodiscard]] std::vector<Couple> coupled() const {
    std::vector<Couple> out(nel * nel);
    for (std::size_t a = 0; a < nel; ++a) {
      for (std::size_t b = 0; b < nel; ++b) {
        const auto hi = std::max(a, b);
        out[a * nel + b] = {rhor[b * nel + a].data(), rhor[a * nel + b].data(),
                            z2r[hi * (hi + 1) / 2 + std::min(a, b)].data()};
      }
    }
    return out;
  }

  [[nodiscard]] const Couple &couple(std::size_t a, std::size_t b) const {
    return couples[a * nel + b];
  }

  /// Where r falls in the radial tables, which share their grid.
  [[nodiscard]] Place radial(double r) const { return place(r, rdr, t.nr); }

  /// F'_a(rho).
  [[nodiscard]] double slope(std::size_t a, double rho) const {
    return Knot(frho[a].data(), place(rho, rdrho, t.nrho)).slope();
  }

  /// F_a(rho), extended linearly above rhomax as pair_style eam does, and
  /// F'_a(rho) in @p slope.
  double embedding(std::size_t a, double rho, double &slope) const {
    const Knot k(frho[a].data(), place(rho, rdrho, t.nrho));
    slope = k.slope();
    double e = k.value();
    if (rho > t.rhomax) {
      e += slope * (rho - t.rhomax);
    }
    return e;
  }

  /// phi_ab(r), and the force scale of the pair @p c at distance r, with
  /// fi = F'_a(rho_i) and fj = F'_b(rho_j).
  double pair(const Couple &c, double r, double fi, double fj, double &phi) const {
    const Place k = radial(r);
    const double rhoip = Knot(c.atJ, k).slope();
    const double rhojp = Knot(c.atI, k).slope();
    const Knot z2(c.z2, k);
    const double recip = 1.0 / r;
    phi = z2.value() * recip;
    const double phip = z2.slope() * recip - phi * recip;
    return -(fi * rhojp + fj * rhoip + phip) * recip;
  }
};

/// Element index of atomic number z, -1 when the table lacks it.
int element_of(const std::vector<int> &element, int z) {
  return z >= 0 && static_cast<std::size_t>(z) < element.size()
             ? element[static_cast<std::size_t>(z)]
             : -1;
}

} // namespace
