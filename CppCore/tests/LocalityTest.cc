// MIT License
// Copyright 2023--present rgpot developers
//
// locality() of the finite-range potentials. Half of a periodic box,
// evaluated with every periodic image within the force range around it
// in a padded orthogonal box, gives that half the forces, site energies
// and per-atom virials of the periodic box. StagedTest.cc covers the
// staged evaluation, which needs the images within one cutoff only.

#include <algorithm>
#include <array>
#include <cmath>
#include <vector>

#include <catch2/catch_all.hpp>

#include "rgpot/LennardJones/LJPot.hpp"
#include "rgpot/Locality.hpp"
#include "rgpot/Morse/MorsePot.hpp"
#include "rgpot/PerAtom.hpp"
#include "rgpot/SuttonChen/SuttonChenPot.hpp"
#include "rgpot/SuttonChen/SuttonChenScales.hpp"
#ifdef RGPOT_HAS_EAM_TABLE
#include "rgpot/EAMTable/EAMTablePot.hpp"
#endif

namespace {

struct Box {
  std::vector<double> pos;
  std::vector<int> z;
  double cell[9] = {};
};

/// n^3 fcc cells of side a, displaced by a function of the index and
/// optionally sheared. Every second atom takes z2.
Box fcc(double a, int n, bool triclinic, int z1, int z2) {
  Box b;
  const double basis[4][3] = {{0, 0, 0}, {0.5, 0.5, 0}, {0.5, 0, 0.5}, {0, 0.5, 0.5}};
  const double side = a * n;
  const double h[9] = {side, 0, 0, triclinic ? 0.1 * side : 0.0, side, 0,
                       triclinic ? -0.08 * side : 0.0, triclinic ? 0.05 * side : 0.0, side};
  std::copy(h, h + 9, b.cell);
  int k = 0;
  for (int i = 0; i < n; ++i) {
    for (int j = 0; j < n; ++j) {
      for (int l = 0; l < n; ++l) {
        for (const auto &s : basis) {
          const double f[3] = {(i + s[0]) / n, (j + s[1]) / n, (l + s[2]) / n};
          for (int d = 0; d < 3; ++d) {
            const double r = f[0] * h[d] + f[1] * h[3 + d] + f[2] * h[6 + d];
            b.pos.push_back(r + 0.06 * a * std::sin(1.37 * k + 0.9 * d));
          }
          b.z.push_back(k % 2 == 0 ? z1 : z2);
          ++k;
        }
      }
    }
  }
  return b;
}

struct PerAtom {
  std::vector<double> forces, energy, virial;
};

template <typename P> PerAtom evaluate(const P &pot, const Box &b) {
  const std::size_t n = b.z.size();
  PerAtom r;
  r.forces.assign(3 * n, 0.0);
  r.energy.assign(n, 0.0);
  r.virial.assign(6 * n, 0.0);
  rgpot::ForceOut out{r.forces.data(), 0.0, 0.0, {}, 0};
  const rgpot::ForceInput in{n, b.pos.data(), b.z.data(), b.cell};
  pot.forcePerAtom(in, &out, {r.energy.data(), r.virial.data()});
  return r;
}

/// The atoms whose first fractional coordinate is under one half, then
/// every image of every atom within reach of one of them, in an
/// orthogonal box wide enough that no image of the cluster is within
/// the cutoff. Returns the cluster and the count of owned atoms, which
/// come first; owned[k] is the index in b of cluster atom k < count.
Box cluster(const Box &b, double reach, double cutoff, std::vector<std::size_t> &owned) {
  const double *h = b.cell;
  const std::size_t n = b.z.size();
  auto image = [&](std::size_t j, int i0, int i1, int i2) {
    std::array<double, 3> r{};
    for (int d = 0; d < 3; ++d) {
      r[static_cast<std::size_t>(d)] =
          b.pos[3 * j + static_cast<std::size_t>(d)] + i0 * h[d] + i1 * h[3 + d] + i2 * h[6 + d];
    }
    return r;
  };
  owned.clear();
  std::vector<std::array<double, 3>> pos;
  std::vector<int> z;
  for (std::size_t j = 0; j < n; ++j) {
    // Lower-triangular rows: a = (h0, 0, 0) carries the only x of row a.
    const double y = b.pos[3 * j + 1];
    const double zz = b.pos[3 * j + 2];
    const double s2 = zz / h[8];
    const double s1 = (y - s2 * h[7]) / h[4];
    const double s0 = (b.pos[3 * j] - s1 * h[3] - s2 * h[6]) / h[0];
    if (s0 < 0.5) {
      owned.push_back(j);
      pos.push_back(image(j, 0, 0, 0));
      z.push_back(b.z[j]);
    }
  }
  const std::size_t nowned = owned.size();
  const int m = 2;
  for (std::size_t j = 0; j < n; ++j) {
    for (int i0 = -m; i0 <= m; ++i0) {
      for (int i1 = -m; i1 <= m; ++i1) {
        for (int i2 = -m; i2 <= m; ++i2) {
          if (i0 == 0 && i1 == 0 && i2 == 0 &&
              std::find(owned.begin(), owned.end(), j) != owned.end()) {
            continue;
          }
          const auto r = image(j, i0, i1, i2);
          bool near = false;
          for (std::size_t k = 0; k < nowned && !near; ++k) {
            const double dx = r[0] - pos[k][0];
            const double dy = r[1] - pos[k][1];
            const double dz = r[2] - pos[k][2];
            near = dx * dx + dy * dy + dz * dz <= reach * reach;
          }
          if (near) {
            pos.push_back(r);
            z.push_back(b.z[j]);
          }
        }
      }
    }
  }
  Box c;
  for (int d = 0; d < 3; ++d) {
    double lo = pos[0][static_cast<std::size_t>(d)];
    double hi = lo;
    for (const auto &r : pos) {
      lo = std::min(lo, r[static_cast<std::size_t>(d)]);
      hi = std::max(hi, r[static_cast<std::size_t>(d)]);
    }
    c.cell[4 * d] = hi - lo + 3.0 * cutoff;
    for (std::size_t k = 0; k < pos.size(); ++k) {
      if (d == 0) {
        c.pos.resize(3 * pos.size());
      }
      c.pos[3 * k + static_cast<std::size_t>(d)] =
          pos[k][static_cast<std::size_t>(d)] - lo + 1.5 * cutoff;
    }
  }
  c.z = z;
  return c;
}

double max_abs(const std::vector<double> &v) {
  double m = 0.0;
  for (double x : v) {
    m = std::max(m, std::abs(x));
  }
  return m;
}

struct Gap {
  double force = 0.0, energy = 0.0, virial = 0.0;
};

template <typename P> Gap owned_gap(const P &pot, const Box &b, double reach) {
  const double cutoff = pot.locality().cutoff;
  const PerAtom whole = evaluate(pot, b);
  std::vector<std::size_t> owned;
  const Box c = cluster(b, reach, cutoff, owned);
  REQUIRE(c.z.size() > owned.size());
  const PerAtom part = evaluate(pot, c);
  Gap g;
  for (std::size_t k = 0; k < owned.size(); ++k) {
    const std::size_t j = owned[k];
    for (std::size_t d = 0; d < 3; ++d) {
      g.force = std::max(g.force, std::abs(part.forces[3 * k + d] - whole.forces[3 * j + d]));
    }
    g.energy = std::max(g.energy, std::abs(part.energy[k] - whole.energy[j]));
    for (std::size_t c6 = 0; c6 < 6; ++c6) {
      g.virial = std::max(g.virial, std::abs(part.virial[6 * k + c6] - whole.virial[6 * j + c6]));
    }
  }
  g.force /= max_abs(whole.forces);
  g.energy /= max_abs(whole.energy);
  g.virial /= max_abs(whole.virial);
  return g;
}

template <typename P> void check_owned(const P &pot, const Box &b) {
  static_assert(rgpot::LocalPotential<P>);
  const rgpot::Locality range = pot.locality();
  REQUIRE(range.finite());
  const Gap g = owned_gap(pot, b, range.forceRange());
  CHECK(g.force < 1e-12);
  CHECK(g.energy < 1e-12);
  CHECK(g.virial < 1e-12);
}

} // namespace

TEST_CASE("Pair potentials declare their cutoff as a pairwise range", "[Locality]") {
  const rgpot::Locality lj = rgpot::LJPot({.u0 = 0.0104, .cutoff = 8.5, .psi = 3.40}).locality();
  CHECK(lj.cutoff == 8.5);
  CHECK(lj.pairwise);
  CHECK(lj.forceRange() == 8.5);
  CHECK_FALSE(lj.full);
  CHECK(lj.reach == 1);
  CHECK(lj.ghostReach() == 8.5);
  const rgpot::Locality morse = rgpot::MorsePot({.cutoff = 7.0}).locality();
  CHECK(morse.cutoff == 7.0);
  CHECK(morse.pairwise);
  CHECK_FALSE(rgpot::LJPot({.cutoff = 0.0}).locality().finite());
}

TEST_CASE("Many-body potentials reach twice their cutoff", "[Locality]") {
  rgpot::SuttonChenConfig c = rgpot::sutton_chen_platinum();
  c.cutoff = 7.0;
  const rgpot::Locality sc = rgpot::SuttonChenPot(c).locality();
  CHECK(sc.cutoff == 7.0);
  CHECK_FALSE(sc.pairwise);
  CHECK(sc.forceRange() == 14.0);
  CHECK_FALSE(sc.full);
  CHECK(sc.reach == 1);
  CHECK(sc.ghostReach() == 7.0);
  c.cutoff = 0.0;
  CHECK_FALSE(rgpot::SuttonChenPot(c).locality().finite());
}

TEST_CASE("A potential without locality() is not a LocalPotential", "[Locality]") {
  struct Plain {
    void forceImpl(const rgpot::ForceInput &, rgpot::ForceOut *) const {}
  };
  STATIC_REQUIRE_FALSE(rgpot::LocalPotential<Plain>);
  STATIC_REQUIRE(rgpot::LocalPotential<rgpot::LJPot>);
}

TEST_CASE("Owned atoms of a cluster match the periodic box", "[Locality]") {
  const bool triclinic = GENERATE(false, true);
  SECTION("LJ") {
    check_owned(rgpot::LJPot({.u0 = 0.0104, .cutoff = 8.5, .psi = 3.40}),
                fcc(5.26, 4, triclinic, 18, 18));
  }
  SECTION("Morse") { check_owned(rgpot::MorsePot({.cutoff = 7.0}), fcc(3.92, 4, triclinic, 78, 78)); }
  SECTION("SuttonChen") {
    rgpot::SuttonChenConfig c = rgpot::sutton_chen_platinum();
    c.cutoff = 7.0;
    c.periodic = true;
    c.stress = true;
    check_owned(rgpot::SuttonChenPot(c), fcc(3.92, 4, triclinic, 78, 78));
  }
#ifdef RGPOT_HAS_EAM_TABLE
  SECTION("EAMTable") {
    rgpot::EAMTableConfig c;
    c.path = "CppCore/tests/data/eam_table/CuNi.eam.alloy";
    c.format = rgpot::EAMTableFormat::Setfl;
    c.stress = true;
    check_owned(rgpot::EAMTablePot(c), fcc(3.6, 4, triclinic, 29, 28));
  }
#endif
}

#ifdef RGPOT_HAS_EAM_TABLE
TEST_CASE("EAM forces need neighbours of neighbours", "[Locality]") {
  rgpot::EAMTableConfig c;
  c.path = "CppCore/tests/data/eam_table/CuNi.eam.alloy";
  c.format = rgpot::EAMTableFormat::Setfl;
  const rgpot::EAMTablePot pot(c);
  const Gap g = owned_gap(pot, fcc(3.6, 4, false, 29, 28), pot.locality().cutoff);
  CHECK(g.energy < 1e-12);
  CHECK(g.force > 1e-6);
}
#endif
