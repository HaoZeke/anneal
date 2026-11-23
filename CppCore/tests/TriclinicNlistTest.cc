// MIT License
// Copyright 2023--present rgpot developers
//
// The pair search on strongly skewed and large triclinic cells, and on
// atoms outside the cell, against a sum over every periodic image. The
// oracle names no rgpot code: it takes the integer shifts of each pair from
// the reciprocal vectors of the cell, so the bound on a shift along an axis
// is the search radius over the perpendicular face height of that axis.
//
// A single image per pair is the minimum image, and it is the only image
// inside a cutoff of at most half the smallest face height. A cutoff past
// that is checked on its own: the search keeps the nearest image of a pair.

#include <algorithm>
#include <array>
#include <cmath>
#include <cstdint>
#include <cstdlib>
#include <map>
#include <random>
#include <set>
#include <string>
#include <tuple>
#include <vector>

#include <catch2/catch_all.hpp>

#include "rgpot/LennardJones/LJPot.hpp"
#include "rgpot/Morse/MorsePot.hpp"
#include "rgpot/nlist/MinimumImage.hpp"
#include "rgpot/nlist/NeighbourList.hpp"
#include "rgpot/nlist/Neighbours.hpp"
#include "rgpot/types/AtomMatrix.hpp"

namespace {

using Box = std::array<double, 9>;
using Periodic = std::array<bool, 3>;

constexpr Periodic kAll{{true, true, true}};

double det3(const Box &b) {
  return b[0] * (b[4] * b[8] - b[5] * b[7]) -
         b[1] * (b[3] * b[8] - b[5] * b[6]) +
         b[2] * (b[3] * b[7] - b[4] * b[6]);
}

void cross(const double *u, const double *v, double *o) {
  o[0] = u[1] * v[2] - u[2] * v[1];
  o[1] = u[2] * v[0] - u[0] * v[2];
  o[2] = u[0] * v[1] - u[1] * v[0];
}

double norm(const double *v) {
  return std::sqrt(v[0] * v[0] + v[1] * v[1] + v[2] * v[2]);
}

/// Perpendicular face heights: volume over the area of the other two vectors.
std::array<double, 3> faceHeights(const Box &b) {
  const double volume = std::fabs(det3(b));
  std::array<double, 3> h{};
  for (int k = 0; k < 3; ++k) {
    double c[3];
    cross(&b[3 * ((k + 1) % 3)], &b[3 * ((k + 2) % 3)], c);
    h[k] = volume / norm(c);
  }
  return h;
}

double minHeight(const Box &b) {
  const auto h = faceHeights(b);
  return std::min({h[0], h[1], h[2]});
}

/// Rows k of the inverse of the cell: reciprocal vectors with b_k . a_m =
/// delta.
std::array<std::array<double, 3>, 3> reciprocal(const Box &b) {
  const double volume = det3(b);
  std::array<std::array<double, 3>, 3> rec{};
  for (int k = 0; k < 3; ++k) {
    double c[3];
    cross(&b[3 * ((k + 1) % 3)], &b[3 * ((k + 2) % 3)], c);
    for (int x = 0; x < 3; ++x) {
      rec[k][x] = c[x] / volume;
    }
  }
  return rec;
}

struct Image {
  int i;
  int j;
  int s[3];
  double d[3]; // r_i - r_j - S.H
  double r2;
};

/// Every image of every pair, and of every atom with itself (S != 0), with
/// |r| <= rc. i <= j.
std::vector<Image> oracle(const std::vector<double> &p, const Box &b, double rc,
                          const Periodic &periodic = kAll) {
  const std::size_t n = p.size() / 3;
  const auto rec = reciprocal(b);
  double reach[3];
  for (int k = 0; k < 3; ++k) {
    reach[k] =
        periodic[static_cast<std::size_t>(k)] ? rc * norm(rec[k].data()) : 0.0;
  }
  std::vector<Image> out;
  const double rc2 = rc * rc;
  for (std::size_t i = 0; i < n; ++i) {
    for (std::size_t j = i; j < n; ++j) {
      const double d0[3] = {p[3 * i] - p[3 * j], p[3 * i + 1] - p[3 * j + 1],
                            p[3 * i + 2] - p[3 * j + 2]};
      int lo[3];
      int hi[3];
      for (int k = 0; k < 3; ++k) {
        if (!periodic[static_cast<std::size_t>(k)]) {
          lo[k] = hi[k] = 0;
          continue;
        }
        const double f =
            rec[k][0] * d0[0] + rec[k][1] * d0[1] + rec[k][2] * d0[2];
        lo[k] = static_cast<int>(std::ceil(f - reach[k] - 1e-9));
        hi[k] = static_cast<int>(std::floor(f + reach[k] + 1e-9));
      }
      for (int s0 = lo[0]; s0 <= hi[0]; ++s0) {
        for (int s1 = lo[1]; s1 <= hi[1]; ++s1) {
          for (int s2 = lo[2]; s2 <= hi[2]; ++s2) {
            if (i == j && s0 == 0 && s1 == 0 && s2 == 0) {
              continue;
            }
            double d[3];
            for (int x = 0; x < 3; ++x) {
              d[x] = d0[x] - (s0 * b[x] + s1 * b[3 + x] + s2 * b[6 + x]);
            }
            const double r2 = d[0] * d[0] + d[1] * d[1] + d[2] * d[2];
            if (r2 <= rc2) {
              out.push_back(Image{static_cast<int>(i),
                                  static_cast<int>(j),
                                  {s0, s1, s2},
                                  {d[0], d[1], d[2]},
                                  r2});
            }
          }
        }
      }
    }
  }
  return out;
}

/// Atoms at least `dmin` apart through every image. Unwrapped places each
/// atom in a random image within two cells of the one inside the cell.
std::vector<double> atoms(const Box &b, std::size_t n, unsigned seed,
                          bool unwrapped, double dmin = 0.9) {
  std::mt19937 g(seed);
  std::uniform_real_distribution<double> u(0.0, 1.0);
  std::uniform_int_distribution<int> k(-2, 2);
  const auto rec = reciprocal(b);
  double reach[3];
  for (int q = 0; q < 3; ++q) {
    reach[q] = dmin * norm(rec[q].data());
  }
  std::vector<double> p;
  p.reserve(3 * n);
  std::size_t tries = 0;
  while (p.size() < 3 * n && tries++ < 200 * n) {
    const double f[3] = {u(g), u(g), u(g)};
    double r[3];
    for (int x = 0; x < 3; ++x) {
      r[x] = f[0] * b[x] + f[1] * b[3 + x] + f[2] * b[6 + x];
    }
    bool ok = true;
    for (std::size_t j = 0; j < p.size() / 3 && ok; ++j) {
      const double d0[3] = {r[0] - p[3 * j], r[1] - p[3 * j + 1],
                            r[2] - p[3 * j + 2]};
      double ff[3];
      for (int q = 0; q < 3; ++q) {
        ff[q] = rec[q][0] * d0[0] + rec[q][1] * d0[1] + rec[q][2] * d0[2];
      }
      for (int s0 = static_cast<int>(std::ceil(ff[0] - reach[0]));
           s0 <= static_cast<int>(std::floor(ff[0] + reach[0])) && ok; ++s0) {
        for (int s1 = static_cast<int>(std::ceil(ff[1] - reach[1]));
             s1 <= static_cast<int>(std::floor(ff[1] + reach[1])) && ok; ++s1) {
          for (int s2 = static_cast<int>(std::ceil(ff[2] - reach[2]));
               s2 <= static_cast<int>(std::floor(ff[2] + reach[2])) && ok;
               ++s2) {
            double d[3];
            for (int x = 0; x < 3; ++x) {
              d[x] = d0[x] - (s0 * b[x] + s1 * b[3 + x] + s2 * b[6 + x]);
            }
            if (norm(d) < dmin) {
              ok = false;
            }
          }
        }
      }
    }
    if (!ok) {
      continue;
    }
    if (unwrapped) {
      const int s0 = k(g);
      const int s1 = k(g);
      const int s2 = k(g);
      for (int x = 0; x < 3; ++x) {
        r[x] += s0 * b[x] + s1 * b[3 + x] + s2 * b[6 + x];
      }
    }
    p.insert(p.end(), r, r + 3);
  }
  return p;
}

struct Cell {
  const char *name;
  Box box;
  std::size_t atoms;
};

// Rows are the lattice vectors a, b, c. Every tilt below is past half the
// length it tilts against, so each cell reduces to a different basis.
const std::vector<Cell> kSkewed = {
    {"half tilt", {10, 0, 0, 5, 10, 0, 0, 0, 10}, 150},
    {"skew above half", {10, 0, 0, 14, 9, 0, -6, 3.5, 10}, 150},
    {"hex skew", {12, 0, 0, 30, 10.392304845413264, 0, 0, 0, 12}, 200},
    {"cube 30, tilts 3x", {30, 0, 0, 90, 30, 0, -60, 60, 30}, 600},
    {"cube 30, tilts 4x", {30, 0, 0, 120, 30, 0, -90, 90, 30}, 600},
};

const std::vector<Cell> kLarge = {
    {"large triclinic", {44, 0, 0, 25, 40, 0, -13, 11, 38}, 800},
    {"large triclinic b", {52, 0, 0, 40, 50, 0, 30, -20, 48}, 800},
};

std::vector<Cell> allCells() {
  std::vector<Cell> cells = kSkewed;
  cells.insert(cells.end(), kLarge.begin(), kLarge.end());
  return cells;
}

using Pair = std::pair<int, int>;

struct Visit {
  double d[3];
  double r2;
};

/// Pairs with r2 <= rc^2 from one half walk. The vector is r_i - r_j.
std::map<Pair, std::vector<Visit>>
halfVisits(const rgpot::nlist::Neighbours &cells, std::size_t n, double rc) {
  std::map<Pair, std::vector<Visit>> out;
  cells.forHalf(0, static_cast<std::int64_t>(n),
                [&](std::int32_t i, std::int32_t j, double dx, double dy,
                    double dz, double r2) {
                  if (r2 <= rc * rc) {
                    REQUIRE(i < j);
                    out[{i, j}].push_back(Visit{{dx, dy, dz}, r2});
                  }
                });
  return out;
}

rgpot::nlist::Neighbours build(const std::vector<double> &p, const Box &box,
                               double rc, rgpot::nlist::Mode mode,
                               const Periodic &periodic = kAll) {
  rgpot::nlist::Spec spec;
  spec.cutoff = rc;
  spec.skin = mode == rgpot::nlist::Mode::Stream ? 0.0 : 0.2 * rc;
  spec.mode = mode;
  spec.periodic = periodic;
  rgpot::nlist::Neighbours cells;
  cells.update(p.data(), p.size() / 3, box.data(), spec);
  return cells;
}

const char *modeName(rgpot::nlist::Mode m) {
  return m == rgpot::nlist::Mode::Stored ? "stored" : "stream";
}

/// The walk of `cells` holds exactly the images of `want` that are the
/// nearest of their pair, with the oracle's vector.
void requireNearestImages(const rgpot::nlist::Neighbours &cells, std::size_t n,
                          double rc, const std::vector<Image> &want) {
  std::map<Pair, Image> nearest;
  for (const auto &m : want) {
    if (m.i == m.j) {
      continue;
    }
    auto it = nearest.find({m.i, m.j});
    if (it == nearest.end() || m.r2 < it->second.r2) {
      nearest[{m.i, m.j}] = m;
    }
  }
  const auto got = halfVisits(cells, n, rc);
  std::size_t missed = 0;
  std::size_t extra = 0;
  for (const auto &item : nearest) {
    if (got.count(item.first) == 0) {
      ++missed;
    }
  }
  for (const auto &item : got) {
    if (nearest.count(item.first) == 0) {
      ++extra;
    }
  }
  REQUIRE(missed == 0);
  REQUIRE(extra == 0);
  for (const auto &item : got) {
    REQUIRE(item.second.size() == 1);
    const Image &m = nearest.at(item.first);
    for (int x = 0; x < 3; ++x) {
      REQUIRE(item.second[0].d[x] == Catch::Approx(m.d[x]).margin(1e-9));
    }
    REQUIRE(item.second[0].r2 == Catch::Approx(m.r2).epsilon(1e-9));
  }
}

/// A directed walk visits each pair twice, with opposite vectors bit for bit.
void requireOpposite(const rgpot::nlist::Neighbours &cells, std::size_t n,
                     double rc) {
  std::map<Pair, std::array<double, 3>> seen;
  std::size_t visits = 0;
  cells.forDirected(0, static_cast<std::int64_t>(n),
                    [&](std::int32_t i, std::int32_t j, double dx, double dy,
                        double dz, double r2) {
                      if (r2 <= rc * rc) {
                        seen[{i, j}] = {dx, dy, dz};
                        ++visits;
                      }
                    });
  REQUIRE(seen.size() == visits);
  for (const auto &item : seen) {
    const auto back = seen.find({item.first.second, item.first.first});
    REQUIRE(back != seen.end());
    for (int x = 0; x < 3; ++x) {
      REQUIRE(back->second[x] == -item.second[x]);
    }
  }
}

} // namespace

TEST_CASE("every pair inside half the smallest face height is visited on "
          "skewed and large cells",
          "[nlist][triclinic][oracle]") {
  for (const Cell &cell : allCells()) {
    const double hmin = minHeight(cell.box);
    for (const bool unwrapped : {false, true}) {
      for (const double fraction : {0.5, 0.4}) {
        const double rc = std::min(fraction * hmin, 9.0);
        INFO(cell.name << " rc " << rc << " unwrapped " << unwrapped);
        REQUIRE(2.0 * rc <= hmin * (1.0 + 1e-12));
        const auto p = atoms(cell.box, cell.atoms, 12345, unwrapped,
                             std::min(0.9, 0.3 * rc));
        const auto want = oracle(p, cell.box, rc);
        std::size_t pairs = 0;
        for (const auto &m : want) {
          pairs += m.i == m.j ? 0 : 1;
          REQUIRE(m.i != m.j);
        }
        // One image per pair inside the cutoff, and some pairs to find.
        std::set<Pair> distinct;
        for (const auto &m : want) {
          distinct.insert({m.i, m.j});
        }
        REQUIRE(distinct.size() == pairs);
        if (fraction == 0.5) {
          REQUIRE(pairs > 5);
        }
        for (const auto mode :
             {rgpot::nlist::Mode::Stored, rgpot::nlist::Mode::Stream}) {
          INFO("mode " << modeName(mode));
          const auto cells = build(p, cell.box, rc, mode);
          requireNearestImages(cells, p.size() / 3, rc, want);
          requireOpposite(cells, p.size() / 3, rc);
        }
      }
    }
  }
}

TEST_CASE("a cutoff past half the smallest face height keeps the nearest image "
          "of a pair",
          "[nlist][triclinic][oracle]") {
  // One image per pair is what the pair kernels sum. Past 2 rc <= h it is
  // the nearest image, not every image, that a walk visits.
  for (const Cell &cell : kSkewed) {
    const double hmin = minHeight(cell.box);
    for (const double fraction : {0.75, 1.25}) {
      const double rc = std::min(fraction * hmin, 9.0);
      INFO(cell.name << " rc " << rc);
      const auto p = atoms(cell.box, cell.atoms, 777, false);
      const auto want = oracle(p, cell.box, rc);
      for (const auto mode :
           {rgpot::nlist::Mode::Stored, rgpot::nlist::Mode::Stream}) {
        INFO("mode " << modeName(mode));
        const auto cells = build(p, cell.box, rc, mode);
        requireNearestImages(cells, p.size() / 3, rc, want);
        requireOpposite(cells, p.size() / 3, rc);
      }
    }
  }
}

namespace {

/// Energy and forces of a pair potential summed over every image in `want`.
template <typename Phi>
void oracleSum(const std::vector<Image> &want, std::size_t n, double shift,
               Phi phi, double &energy, std::vector<double> &forces) {
  energy = 0.0;
  forces.assign(3 * n, 0.0);
  for (const auto &m : want) {
    double fs = 0.0;
    const double w = m.i == m.j ? 0.5 : 1.0;
    energy += w * (phi(std::sqrt(m.r2), fs) - shift);
    if (m.i != m.j) {
      for (int x = 0; x < 3; ++x) {
        forces[3 * m.i + x] += fs * m.d[x];
        forces[3 * m.j + x] -= fs * m.d[x];
      }
    }
  }
}

std::array<std::array<double, 3>, 3> rows(const Box &b) {
  return {{{b[0], b[1], b[2]}, {b[3], b[4], b[5]}, {b[6], b[7], b[8]}}};
}

void requireSameForces(const rgpot::types::AtomMatrix &f,
                       const std::vector<double> &want, std::size_t n) {
  double scale = 0.0;
  for (const double v : want) {
    scale = std::max(scale, std::fabs(v));
  }
  for (std::size_t i = 0; i < n; ++i) {
    for (int x = 0; x < 3; ++x) {
      REQUIRE(std::fabs(f(i, x) - want[3 * i + x]) <= 1e-8 * (1.0 + scale));
    }
  }
}

} // namespace

TEST_CASE(
    "evaluate sums every image inside the cutoff on skewed and large cells",
    "[nlist][triclinic][oracle][evaluate]") {
  for (const Cell &cell : allCells()) {
    if (std::string(cell.name) == "half tilt" ||
        std::string(cell.name) == "cube 30, tilts 4x") {
      continue;
    }
    const double hmin = minHeight(cell.box);
    for (const bool unwrapped : {false, true}) {
      const double rc = std::min(0.5 * hmin, 9.0);
      INFO(cell.name << " rc " << rc << " unwrapped " << unwrapped);
      const auto p = atoms(cell.box, cell.atoms, 12345, unwrapped);
      const std::size_t n = p.size() / 3;
      const auto want = oracle(p, cell.box, rc);
      REQUIRE_FALSE(want.empty());
      rgpot::types::AtomMatrix pos(n, 3);
      for (std::size_t i = 0; i < n; ++i) {
        for (int x = 0; x < 3; ++x) {
          pos(i, x) = p[3 * i + x];
        }
      }
      const std::vector<int> z(n, 18);
      const auto bx = rows(cell.box);
      {
        rgpot::LJConfig cfg;
        cfg.cutoff = rc;
        cfg.psi = 1.0;
        rgpot::LJPot pot(cfg);
        auto [e, f, v] = pot(pos, z, bx);
        (void)v;
        auto phi = [](double r, double &fs) {
          const double a = std::pow(1.0 / r, 6.0);
          fs = 24.0 * (2.0 * a * a - a) / (r * r);
          return 4.0 * (a * a - a);
        };
        double ignored = 0.0;
        const double shift = phi(rc, ignored);
        double eo = 0.0;
        std::vector<double> fo;
        oracleSum(want, n, shift, phi, eo, fo);
        REQUIRE(e == Catch::Approx(eo).epsilon(1e-9).margin(1e-9));
        requireSameForces(f, fo, n);
      }
      {
        rgpot::MorseConfig cfg;
        cfg.cutoff = rc;
        cfg.re = 1.1;
        cfg.De = 1.0;
        cfg.a = 1.5;
        rgpot::MorsePot pot(cfg);
        auto [e, f, v] = pot(pos, z, bx);
        (void)v;
        auto phi = [&](double r, double &fs) {
          const double ex = std::exp(-cfg.a * (r - cfg.re));
          fs = -2.0 * cfg.De * cfg.a * ex * (1.0 - ex) / r;
          return cfg.De * (1.0 - ex) * (1.0 - ex) - cfg.De;
        };
        double ignored = 0.0;
        const double shift = phi(rc, ignored);
        double eo = 0.0;
        std::vector<double> fo;
        oracleSum(want, n, shift, phi, eo, fo);
        REQUIRE(e == Catch::Approx(eo).epsilon(1e-9).margin(1e-9));
        requireSameForces(f, fo, n);
      }
    }
  }
}

namespace {

using Key = std::tuple<int, int, int, int, int>;

std::set<Key> fullImages(const std::vector<Image> &want) {
  std::set<Key> out;
  for (const auto &m : want) {
    out.insert({m.i, m.j, m.s[0], m.s[1], m.s[2]});
    out.insert({m.j, m.i, -m.s[0], -m.s[1], -m.s[2]});
  }
  return out;
}

} // namespace

TEST_CASE("the neighbour table lists every image with its shift on skewed and "
          "large cells",
          "[nlist][triclinic][oracle][table]") {
  struct Case {
    Cell cell;
    double fraction;
    bool unwrapped;
  };
  std::vector<Case> cases;
  for (const Cell &cell : allCells()) {
    cases.push_back({cell, 0.5, false});
    cases.push_back({cell, 0.5, true});
  }
  // Narrower than twice the cutoff, where a pair has several images and an
  // atom has images of itself.
  for (const Cell &cell : kSkewed) {
    cases.push_back({cell, 1.25, false});
    cases.push_back({cell, 2.0, true});
  }
  for (const Case &c : cases) {
    const double hmin = minHeight(c.cell.box);
    const double rc = std::min(c.fraction * hmin, 9.0);
    INFO(c.cell.name << " rc " << rc << " unwrapped " << c.unwrapped);
    const auto p = atoms(c.cell.box, c.cell.atoms, 4321, c.unwrapped);
    const std::size_t n = p.size() / 3;
    const auto want = oracle(p, c.cell.box, rc);
    const auto expected = fullImages(want);
    REQUIRE_FALSE(expected.empty());

    rgpot::nlist::Query query;
    query.cutoff = rc;
    query.cut = rgpot::nlist::Cut::Loose;
    query.shape = rgpot::nlist::Pairs{rgpot::nlist::Shell::Full};
    rgpot::nlist::Table table;
    rgpot::nlist::build(table, p.data(), n, c.cell.box.data(), query);
    std::set<Key> got;
    REQUIRE(table.size() == expected.size());
    for (std::size_t q = 0; q < table.size(); ++q) {
      const Key key{table.first[q], table.second[q], table.shift[3 * q],
                    table.shift[3 * q + 1], table.shift[3 * q + 2]};
      REQUIRE(got.insert(key).second);
      // r_j - r_i + S.H
      for (int x = 0; x < 3; ++x) {
        const double v = p[3 * static_cast<std::size_t>(table.second[q]) + x] -
                         p[3 * static_cast<std::size_t>(table.first[q]) + x] +
                         table.shift[3 * q] * c.cell.box[x] +
                         table.shift[3 * q + 1] * c.cell.box[3 + x] +
                         table.shift[3 * q + 2] * c.cell.box[6 + x];
        REQUIRE(table.vector[3 * q + x] == Catch::Approx(v).margin(1e-9));
      }
    }
    REQUIRE(got == expected);

    query.shape = rgpot::nlist::Pairs{rgpot::nlist::Shell::Half};
    rgpot::nlist::Table half;
    rgpot::nlist::build(half, p.data(), n, c.cell.box.data(), query);
    std::set<Key> halved;
    for (std::size_t q = 0; q < half.size(); ++q) {
      const Key key{half.first[q], half.second[q], half.shift[3 * q],
                    half.shift[3 * q + 1], half.shift[3 * q + 2]};
      REQUIRE(expected.count(key) == 1);
      REQUIRE(halved.insert(key).second);
    }
    REQUIRE(2 * half.size() == expected.size());
    for (const auto &key : halved) {
      REQUIRE(
          halved.count({std::get<1>(key), std::get<0>(key), -std::get<2>(key),
                        -std::get<3>(key), -std::get<4>(key)}) == 0);
    }
  }
}

TEST_CASE("an orthogonal cell finds the pairs of atoms outside it",
          "[nlist][triclinic][oracle]") {
  // Atoms in other images of an orthogonal cell, with many bins along each
  // axis. The bins hold the coordinate folded into the cell.
  struct Ortho {
    Box box;
    std::size_t atoms;
    double rc;
  };
  const std::vector<Ortho> cells = {
      {{30, 0, 0, 0, 30, 0, 0, 0, 30}, 800, 3.0},
      {{40, 0, 0, 0, 45, 0, 0, 0, 50}, 1200, 4.0},
  };
  for (const Ortho &cell : cells) {
    for (const bool unwrapped : {false, true}) {
      INFO("cell " << cell.box[0] << " rc " << cell.rc << " unwrapped "
                   << unwrapped);
      auto p = atoms(cell.box, cell.atoms, 99, unwrapped);
      if (!unwrapped) {
        // Atoms exactly on the upper face, and a hair below the lower one.
        p[0] = cell.box[0];
        p[4] = -1e-12;
      }
      const auto want = oracle(p, cell.box, cell.rc);
      REQUIRE(want.size() > 100);
      for (const auto mode :
           {rgpot::nlist::Mode::Stored, rgpot::nlist::Mode::Stream}) {
        INFO("mode " << modeName(mode));
        const auto cells_built = build(p, cell.box, cell.rc, mode);
        std::set<Pair> expected;
        for (const auto &m : want) {
          expected.insert({m.i, m.j});
        }
        const auto got = halfVisits(cells_built, p.size() / 3, cell.rc);
        std::set<Pair> found;
        for (const auto &item : got) {
          found.insert(item.first);
        }
        REQUIRE(found == expected);
        requireOpposite(cells_built, p.size() / 3, cell.rc);
      }
    }
  }
}

TEST_CASE(
    "a list on a sheared cell is kept within the skin of the nearest image",
    "[nlist][triclinic][skin]") {
  const Cell cell = kSkewed[1];
  const double rc = 0.5 * minHeight(cell.box);
  const double skin = 0.5;
  auto p = atoms(cell.box, cell.atoms, 31, false);
  const std::size_t n = p.size() / 3;
  rgpot::nlist::Spec spec;
  spec.cutoff = rc;
  spec.skin = skin;
  spec.mode = rgpot::nlist::Mode::Stored;
  rgpot::nlist::Neighbours cells;
  cells.update(p.data(), n, cell.box.data(), spec);
  REQUIRE_FALSE(cells.reused());

  auto requireMatches = [&](const char *what) {
    INFO(what);
    const auto want = oracle(p, cell.box, rc);
    requireNearestImages(cells, n, rc, want);
  };
  requireMatches("first build");

  std::mt19937 g(5);
  std::uniform_real_distribution<double> u(-1.0, 1.0);
  for (double &v : p) {
    v += 0.1 * skin * u(g);
  }
  cells.update(p.data(), n, cell.box.data(), spec);
  REQUIRE(cells.reused());
  requireMatches("moved inside half the skin");

  // A whole lattice step is inside the skin of the nearest image, so the
  // stored partners stay and the walk reads the image from the new
  // coordinates.
  p[0] += cell.box[0];
  p[1] += cell.box[1];
  p[2] += cell.box[2];
  cells.update(p.data(), n, cell.box.data(), spec);
  REQUIRE(cells.reused());
  requireMatches("one atom through a lattice vector");

  p[3] += 0.75 * skin;
  cells.update(p.data(), n, cell.box.data(), spec);
  REQUIRE_FALSE(cells.reused());
  requireMatches("one atom past half the skin");
}

TEST_CASE("a sheared slab with an open axis keeps its pairs",
          "[nlist][triclinic][oracle]") {
  // The c axis is open and perpendicular to the sheared a, b plane.
  const Box box = {10, 0, 0, 14, 9, 0, 0, 0, 12};
  const Periodic periodic{{true, true, false}};
  std::mt19937 g(17);
  std::uniform_real_distribution<double> u(0.0, 1.0);
  std::vector<double> p;
  for (int a = 0; a < 140; ++a) {
    const double f0 = u(g);
    const double f1 = u(g);
    const double z = 12.0 * u(g);
    p.push_back(f0 * box[0] + f1 * box[3]);
    p.push_back(f1 * box[4]);
    p.push_back(z);
  }
  const double rc = 2.2;
  const auto want = oracle(p, box, rc, periodic);
  REQUIRE(want.size() > 50);
  for (const auto mode :
       {rgpot::nlist::Mode::Stored, rgpot::nlist::Mode::Stream}) {
    INFO("mode " << modeName(mode));
    const auto cells = build(p, box, rc, mode, periodic);
    requireNearestImages(cells, p.size() / 3, rc, want);
  }
}

TEST_CASE(
    "MinimumImage folds a displacement to its nearest image on skewed cells",
    "[nlist][triclinic][oracle]") {
  std::mt19937 g(3);
  std::uniform_real_distribution<double> u(-1.0, 1.0);
  for (const Cell &cell : allCells()) {
    const rgpot::nlist::MinimumImage image(cell.box.data(), kAll);
    std::size_t checked = 0;
    for (int trial = 0; trial < 400; ++trial) {
      double d[3] = {0.0, 0.0, 0.0};
      for (int x = 0; x < 3; ++x) {
        const double f = u(g) * 1.5;
        for (int y = 0; y < 3; ++y) {
          d[y] += f * cell.box[3 * x + y];
        }
      }
      // The nearest image is at most |d| away, so the shifts along an axis lie
      // within |d| over the face height of the axis, about the fractional
      // coordinate.
      double best = norm(d);
      const auto rec = reciprocal(cell.box);
      int lo[3];
      int hi[3];
      for (int k = 0; k < 3; ++k) {
        const double f = rec[k][0] * d[0] + rec[k][1] * d[1] + rec[k][2] * d[2];
        const double reach = norm(d) * norm(rec[k].data());
        lo[k] = static_cast<int>(std::ceil(f - reach));
        hi[k] = static_cast<int>(std::floor(f + reach));
      }
      for (int s0 = lo[0]; s0 <= hi[0]; ++s0) {
        for (int s1 = lo[1]; s1 <= hi[1]; ++s1) {
          for (int s2 = lo[2]; s2 <= hi[2]; ++s2) {
            double v[3];
            for (int y = 0; y < 3; ++y) {
              v[y] = d[y] - (s0 * cell.box[y] + s1 * cell.box[3 + y] +
                             s2 * cell.box[6 + y]);
            }
            best = std::min(best, norm(v));
          }
        }
      }
      double folded[3] = {d[0], d[1], d[2]};
      image.fold(folded[0], folded[1], folded[2]);
      REQUIRE(norm(folded) <= best * (1.0 + 1e-12) + 1e-12);
      ++checked;
    }
    REQUIRE(checked == 400);
  }
}
