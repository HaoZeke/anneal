// MIT License
// Copyright 2023--present rgpot developers
//
// forcePerAtom of the pair and EAM potentials: the same forces, energy
// and stress as forceImpl, per-atom energies that sum to the energy,
// per-atom virials that sum to -V sigma (or to sum r (x) F for a free
// cluster), and the same split from one thread and from four.

#include <algorithm>
#include <cmath>
#include <cstdlib>
#include <string>
#include <vector>

#include <catch2/catch_all.hpp>

#include "lsan_libomp.hpp"
#include "rgpot/LennardJones/LJClusterPot.hpp"
#include "rgpot/LennardJones/LJPot.hpp"
#include "rgpot/Morse/MorsePot.hpp"
#include "rgpot/PerAtom.hpp"
#include "rgpot/SuttonChen/SuttonChenPot.hpp"
#include "rgpot/SuttonChen/SuttonChenScales.hpp"
#include "rgpot/ZBL/ZBLPot.hpp"
#include "rgpot/stress.hpp"
#ifdef RGPOT_HAS_EAM_TABLE
#include "rgpot/EAMTable/EAMTablePot.hpp"
#endif

namespace {

void set_omp_threads(const char *value) {
#if defined(_MSC_VER)
  _putenv_s("OMP_NUM_THREADS", value);
#else
  setenv("OMP_NUM_THREADS", value, 1);
#endif
}

struct Box {
  std::vector<double> pos;
  std::vector<int> z;
  double cell[9] = {};
};

/// n^3 fcc cells of side a, atoms displaced by a function of their index,
/// optionally sheared. Every second atom takes z2.
Box fcc(double a, int n, bool triclinic, int z1, int z2) {
  Box b;
  const double basis[4][3] = {{0, 0, 0}, {0.5, 0.5, 0}, {0.5, 0, 0.5}, {0, 0.5, 0.5}};
  const double side = a * n;
  const double tilt[3] = {triclinic ? 0.1 * side : 0.0, triclinic ? -0.08 * side : 0.0,
                          triclinic ? 0.05 * side : 0.0};
  const double h[9] = {side, 0, 0, tilt[0], side, 0, tilt[1], tilt[2], side};
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

struct Result {
  std::vector<double> forces, energy, virial;
  rgpot::ForceOut out{};
};

template <typename P> Result run(const P &pot, const Box &b, bool per_atom) {
  const std::size_t n = b.z.size();
  Result r;
  r.forces.assign(3 * n, 0.0);
  r.out.F = r.forces.data();
  const rgpot::ForceInput in{n, b.pos.data(), b.z.data(), b.cell};
  if (per_atom) {
    r.energy.assign(n, 1.0);
    r.virial.assign(6 * n, 1.0);
    pot.forcePerAtom(in, &r.out, {r.energy.data(), r.virial.data()});
  } else {
    pot.forceImpl(in, &r.out);
  }
  return r;
}

double max_abs(const std::vector<double> &v) {
  double m = 0.0;
  for (double x : v) {
    m = std::max(m, std::abs(x));
  }
  return m;
}

double max_diff(const std::vector<double> &a, const std::vector<double> &b) {
  REQUIRE(a.size() == b.size());
  double m = 0.0;
  for (std::size_t k = 0; k < a.size(); ++k) {
    m = std::max(m, std::abs(a[k] - b[k]));
  }
  return m;
}

/// stress_free: compare the virial sum with sum_i r_i (x) F_i instead of
/// -V sigma. exact: forcePerAtom repeats forceImpl's sums bit for bit.
template <typename P>
void check(const P &pot, const Box &b, bool stress_free = false, bool exact = true) {
  static_assert(rgpot::PerAtomPotential<P>);
  set_omp_threads("1");
  const Result plain = run(pot, b, false);
  const Result split = run(pot, b, true);
  const std::size_t n = b.z.size();

  if (exact) {
    CHECK(split.out.energy == plain.out.energy);
    CHECK(max_diff(split.forces, plain.forces) == 0.0);
  } else {
    CHECK_THAT(split.out.energy, Catch::Matchers::WithinRel(plain.out.energy, 1e-12));
    CHECK(max_diff(split.forces, plain.forces) < 1e-12 * max_abs(plain.forces));
  }
  CHECK(split.out.has_stress == plain.out.has_stress);

  double esum = 0.0;
  double wsum[6] = {};
  for (std::size_t a = 0; a < n; ++a) {
    esum += split.energy[a];
    for (int c = 0; c < 6; ++c) {
      wsum[c] += split.virial[6 * a + static_cast<std::size_t>(c)];
    }
  }
  CHECK_THAT(esum, Catch::Matchers::WithinAbs(plain.out.energy,
                                              1e-12 * std::max(1.0, std::abs(plain.out.energy))));

  // Voigt xx yy zz yz xz xy.
  const int rows[6] = {0, 1, 2, 1, 0, 0};
  const int cols[6] = {0, 1, 2, 2, 2, 1};
  double want[6] = {};
  if (stress_free) {
    for (std::size_t a = 0; a < n; ++a) {
      for (int c = 0; c < 6; ++c) {
        want[c] += b.pos[3 * a + static_cast<std::size_t>(rows[c])] *
                   plain.forces[3 * a + static_cast<std::size_t>(cols[c])];
      }
    }
  } else {
    REQUIRE(plain.out.has_stress == 1);
    const double volume = rgpot::cellVolume(b.cell);
    for (int c = 0; c < 6; ++c) {
      want[c] = -volume * plain.out.stress[3 * rows[c] + cols[c]];
    }
  }
  double scale = 0.0;
  for (double w : want) {
    scale = std::max(scale, std::abs(w));
  }
  REQUIRE(scale > 0.0);
  for (int c = 0; c < 6; ++c) {
    CHECK_THAT(wsum[c], Catch::Matchers::WithinAbs(want[c], 1e-10 * scale));
  }

  set_omp_threads("4");
  const Result team = run(pot, b, true);
  set_omp_threads("1");
  CHECK_THAT(team.out.energy, Catch::Matchers::WithinRel(plain.out.energy, 1e-12));
  CHECK(max_diff(team.forces, plain.forces) < 1e-12 * max_abs(plain.forces));
  CHECK(max_diff(team.energy, split.energy) < 1e-12 * max_abs(split.energy));
  CHECK(max_diff(team.virial, split.virial) < 1e-12 * max_abs(split.virial));
}

} // namespace

TEST_CASE("forcePerAtom splits pair potentials half to each atom", "[PerAtom]") {
  const bool triclinic = GENERATE(false, true);
  const Box ar = fcc(5.26, 4, triclinic, 18, 18);
  SECTION("LJ") { check(rgpot::LJPot({.u0 = 0.0104, .cutoff = 8.5, .psi = 3.40}), ar); }
  SECTION("LJ with a switch") {
    check(rgpot::LJPot({.u0 = 0.0104, .cutoff = 8.5, .psi = 3.40, .switch_width = 1.0}), ar);
  }
  SECTION("Morse") { check(rgpot::MorsePot({.cutoff = 7.0}), fcc(3.92, 4, triclinic, 78, 78)); }
}

TEST_CASE("forcePerAtom on free-boundary pair potentials", "[PerAtom]") {
  Box cluster = fcc(5.26, 3, false, 18, 18);
  std::fill(cluster.cell, cluster.cell + 9, 0.0);
  SECTION("LJCluster") {
    // The cell walk replaces the direct all-pairs sum, so not bit for bit.
    check(rgpot::LJClusterPot({.u0 = 0.0104, .cutoff = 8.5, .psi = 3.40}), cluster, true,
          false);
  }
  SECTION("ZBL") {
    Box dense = fcc(2.6, 3, false, 14, 79);
    const double cell[9] = {100, 0, 0, 0, 100, 0, 0, 0, 100};
    std::copy(cell, cell + 9, dense.cell);
    check(rgpot::ZBLPot({.cut_inner = 1.5, .cut_global = 2.4}), dense);
  }
}

TEST_CASE("forcePerAtom gives Sutton-Chen atoms their embedding", "[PerAtom]") {
  const bool triclinic = GENERATE(false, true);
  rgpot::SuttonChenConfig c = rgpot::sutton_chen_platinum();
  c.cutoff = 7.0;
  c.periodic = true;
  c.stress = true;
  check(rgpot::SuttonChenPot(c), fcc(3.92, 4, triclinic, 78, 78));
}

#ifdef RGPOT_HAS_EAM_TABLE
TEST_CASE("forcePerAtom gives EAM atoms their embedding", "[PerAtom]") {
  const bool triclinic = GENERATE(false, true);
  rgpot::EAMTableConfig c;
  c.path = "CppCore/tests/data/eam_table/CuNi.eam.alloy";
  c.format = rgpot::EAMTableFormat::Setfl;
  c.stress = true;
  check(rgpot::EAMTablePot(c), fcc(3.6, 4, triclinic, 29, 28));
}
#endif

TEST_CASE("A potential without forcePerAtom is not a PerAtomPotential", "[PerAtom]") {
  struct Plain {
    void forceImpl(const rgpot::ForceInput &, rgpot::ForceOut *) const {}
  };
  STATIC_REQUIRE_FALSE(rgpot::PerAtomPotential<Plain>);
  STATIC_REQUIRE(rgpot::PerAtomPotential<rgpot::LJPot>);
}
