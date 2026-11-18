// MIT License
// Copyright 2023--present rgpot developers
//
// Multi-image EAM checks against an independent supercell reference.

#include <array>
#include <cmath>
#include <cstdint>
#include <cstring>
#include <sstream>
#include <string>
#include <vector>

#include <catch2/catch_all.hpp>

#include "rgpot/EAMTable/EAMTablePot.hpp"
#include "rgpot/PerAtom.hpp"

using Catch::Matchers::WithinAbs;
using rgpot::EAMTableFormat;

namespace {

const std::string kData = "CppCore/tests/data/";

rgpot::EAMTablePot cu_u3() {
  return rgpot::EAMTablePot({.path = kData + "eam_table/Cu_u3.eam",
                             .format = EAMTableFormat::Funcfl,
                             .periodic = true,
                             .stress = true});
}

std::string pair_table() {
  // phi(r) = (5 - r)^2 on knots one angstrom apart, stored as r phi.
  // Embedding and density are zero, so the energy is only the pair sum.
  std::ostringstream s;
  s << "pair\npair\npair\n1 Cu\n5 1 6 1 5\n29 63.546 1 fcc\n";
  for (int m = 0; m < 5; ++m) {
    s << "0 ";
  }
  s << '\n';
  for (int m = 0; m < 6; ++m) {
    s << "0 ";
  }
  s << '\n';
  for (int m = 0; m < 6; ++m) {
    const double r = static_cast<double>(m);
    s << (r * (5.0 - r) * (5.0 - r)) << ' ';
  }
  s << '\n';
  return s.str();
}

rgpot::EAMTablePot pair_only() {
  std::istringstream in(pair_table());
  return rgpot::EAMTablePot(
      rgpot::read_eam_table(in, EAMTableFormat::Setfl), true, true);
}

struct Eval {
  double energy = 0.0;
  std::array<double, 9> stress{};
  int has_stress = 0;
  std::vector<double> forces;
  std::vector<double> eatom;
  std::vector<double> vatom;
};

Eval eval_full(const rgpot::EAMTablePot &pot, const std::vector<double> &pos,
               const std::vector<int> &z, const double *box) {
  Eval e;
  e.forces.assign(pos.size(), 0.0);
  e.eatom.assign(z.size(), 0.0);
  e.vatom.assign(6 * z.size(), 0.0);
  rgpot::ForceOut out{};
  out.F = e.forces.data();
  pot.forcePerAtom(rgpot::ForceInput{z.size(), pos.data(), z.data(), box}, &out,
                   rgpot::PerAtomOut{e.eatom.data(), e.vatom.data()});
  e.energy = out.energy;
  e.has_stress = out.has_stress;
  std::copy(out.stress, out.stress + 9, e.stress.begin());
  return e;
}

double face_spacing(const double *box, int axis) {
  const double *u = box + 3 * ((axis + 1) % 3);
  const double *v = box + 3 * ((axis + 2) % 3);
  const double cx = u[1] * v[2] - u[2] * v[1];
  const double cy = u[2] * v[0] - u[0] * v[2];
  const double cz = u[0] * v[1] - u[1] * v[0];
  const double area = std::sqrt(cx * cx + cy * cy + cz * cz);
  const double vol = std::abs(box[0] * (box[4] * box[8] - box[5] * box[7]) -
                              box[1] * (box[3] * box[8] - box[5] * box[6]) +
                              box[2] * (box[3] * box[7] - box[4] * box[6]));
  REQUIRE(area > 0.0);
  REQUIRE(vol > 0.0);
  return vol / area;
}

std::array<int, 3> replication(const double *box, double cutoff) {
  std::array<int, 3> rep{1, 1, 1};
  for (int axis = 0; axis < 3; ++axis) {
    const double spacing = face_spacing(box, axis);
    while (!(2.0 * cutoff < rep[static_cast<std::size_t>(axis)] * spacing)) {
      ++rep[static_cast<std::size_t>(axis)];
    }
  }
  return rep;
}

struct Replicated {
  std::array<double, 9> box{};
  std::vector<double> pos;
  std::vector<int> z;
  std::size_t nrep = 1;
};

Replicated replicate(const std::vector<double> &pos, const std::vector<int> &z,
                     const double *box, const std::array<int, 3> &rep) {
  Replicated out;
  out.box = {rep[0] * box[0], rep[0] * box[1], rep[0] * box[2], rep[1] * box[3],
             rep[1] * box[4], rep[1] * box[5], rep[2] * box[6], rep[2] * box[7],
             rep[2] * box[8]};
  const std::size_t n = z.size();
  out.nrep = static_cast<std::size_t>(rep[0] * rep[1] * rep[2]);
  out.pos.reserve(3 * n * out.nrep);
  out.z.reserve(n * out.nrep);
  for (int h = 0; h < rep[0]; ++h) {
    for (int k = 0; k < rep[1]; ++k) {
      for (int l = 0; l < rep[2]; ++l) {
        const double sx = h * box[0] + k * box[3] + l * box[6];
        const double sy = h * box[1] + k * box[4] + l * box[7];
        const double sz = h * box[2] + k * box[5] + l * box[8];
        for (std::size_t i = 0; i < n; ++i) {
          out.pos.push_back(pos[3 * i] + sx);
          out.pos.push_back(pos[3 * i + 1] + sy);
          out.pos.push_back(pos[3 * i + 2] + sz);
          out.z.push_back(z[i]);
        }
      }
    }
  }
  return out;
}

Eval supercell_reference(const rgpot::EAMTablePot &pot, const std::vector<double> &pos,
                         const std::vector<int> &z, const double *box) {
  const std::array<int, 3> rep = replication(box, pot.locality().cutoff);
  const Replicated sc = replicate(pos, z, box, rep);
  const Eval big = eval_full(pot, sc.pos, sc.z, sc.box.data());
  const std::size_t n = z.size();
  const double inv = 1.0 / static_cast<double>(sc.nrep);
  Eval ref;
  ref.energy = big.energy * inv;
  ref.stress = big.stress;
  ref.has_stress = big.has_stress;
  ref.forces.assign(3 * n, 0.0);
  ref.eatom.assign(n, 0.0);
  ref.vatom.assign(6 * n, 0.0);
  for (std::size_t t = 0; t < sc.nrep; ++t) {
    for (std::size_t i = 0; i < n; ++i) {
      const std::size_t src = t * n + i;
      ref.eatom[i] += big.eatom[src] * inv;
      for (int c = 0; c < 3; ++c) {
        ref.forces[3 * i + static_cast<std::size_t>(c)] +=
            big.forces[3 * src + static_cast<std::size_t>(c)] * inv;
      }
      for (int v = 0; v < 6; ++v) {
        ref.vatom[6 * i + static_cast<std::size_t>(v)] +=
            big.vatom[6 * src + static_cast<std::size_t>(v)] * inv;
      }
    }
  }
  return ref;
}

std::array<double, 6> total_virial(const Eval &e) {
  std::array<double, 6> v{};
  for (std::size_t i = 0; i < e.vatom.size() / 6; ++i) {
    for (int k = 0; k < 6; ++k) {
      v[static_cast<std::size_t>(k)] +=
          e.vatom[6 * i + static_cast<std::size_t>(k)];
    }
  }
  return v;
}

void check(const rgpot::EAMTablePot &pot, const std::vector<double> &pos,
           const std::vector<int> &z, const double *box, double etol, double ftol,
           double stol, double atom_tol) {
  const Eval got = eval_full(pot, pos, z, box);
  const Eval want = supercell_reference(pot, pos, z, box);
  REQUIRE(got.has_stress == 1);
  REQUIRE(want.has_stress == 1);
  REQUIRE_THAT(got.energy, WithinAbs(want.energy, etol));
  for (std::size_t c = 0; c < got.forces.size(); ++c) {
    REQUIRE_THAT(got.forces[c], WithinAbs(want.forces[c], ftol));
  }
  for (int c = 0; c < 9; ++c) {
    REQUIRE_THAT(got.stress[static_cast<std::size_t>(c)],
                 WithinAbs(want.stress[static_cast<std::size_t>(c)], stol));
  }
  for (std::size_t i = 0; i < got.eatom.size(); ++i) {
    REQUIRE_THAT(got.eatom[i], WithinAbs(want.eatom[i], atom_tol));
  }
  const std::array<double, 6> gv = total_virial(got);
  const std::array<double, 6> wv = total_virial(want);
  for (int c = 0; c < 6; ++c) {
    REQUIRE_THAT(gv[static_cast<std::size_t>(c)],
                 WithinAbs(wv[static_cast<std::size_t>(c)], atom_tol));
  }
}

std::uint64_t hash_mix(std::uint64_t h, std::uint64_t x) {
  constexpr std::uint64_t p = 1099511628211ull;
  h ^= x;
  h *= p;
  return h;
}

std::uint64_t hash_double(std::uint64_t h, double x) {
  std::uint64_t bits = 0;
  std::memcpy(&bits, &x, sizeof(bits));
  return hash_mix(h, bits);
}

std::uint64_t hash_eval(std::uint64_t h, const Eval &e) {
  h = hash_double(h, e.energy);
  for (double x : e.forces) {
    h = hash_double(h, x);
  }
  for (double x : e.stress) {
    h = hash_double(h, x);
  }
  for (double x : e.eatom) {
    h = hash_double(h, x);
  }
  for (double x : e.vatom) {
    h = hash_double(h, x);
  }
  return h;
}

} // namespace

TEST_CASE("EAM multi-image matches an independent supercell reference",
          "[EAMTable][nlist]") {
  SECTION("pair-only one-atom cubic 4A cell") {
    const rgpot::EAMTablePot pot = pair_only();
    const std::vector<double> pos{0.4, 0.5, 0.6};
    const std::vector<int> z{29};
    const double box[9] = {4.0, 0.0, 0.0, 0.0, 4.0, 0.0, 0.0, 0.0, 4.0};
    const Eval got = eval_full(pot, pos, z, box);
    const Eval want = supercell_reference(pot, pos, z, box);
    REQUIRE_THAT(want.energy, WithinAbs(3.0, 1e-12));
    REQUIRE_THAT(got.energy, WithinAbs(3.0, 1e-12));
    check(pot, pos, z, box, 1e-12, 1e-10, 1e-12, 1e-12);
  }

  SECTION("pair-only one-atom triclinic cell") {
    const rgpot::EAMTablePot pot = pair_only();
    const std::vector<double> pos{0.2, 0.3, 0.4};
    const std::vector<int> z{29};
    const double box[9] = {3.6, 0.0, 0.0, 1.8, 3.1, 0.0, 1.8, 1.0, 2.9};
    check(pot, pos, z, box, 1e-11, 1e-9, 1e-11, 1e-11);
  }

  SECTION("Cu_u3 one-atom cubic cell") {
    const rgpot::EAMTablePot pot = cu_u3();
    const std::vector<double> pos{0.2, 0.3, 0.4};
    const std::vector<int> z{29};
    const double box[9] = {8.0, 0.0, 0.0, 0.0, 8.0, 0.0, 0.0, 0.0, 8.0};
    check(pot, pos, z, box, 1e-8, 1e-7, 1e-10, 1e-8);
  }

  SECTION("Cu_u3 vacancy cell") {
    const rgpot::EAMTablePot pot = cu_u3();
    const double a = 3.615;
    const double box[9] = {a, 0.0, 0.0, 0.0, a, 0.0, 0.0, 0.0, a};
    // One missing site in a conventional FCC cell.
    const std::vector<double> pos{
        0.0, 0.0, 0.0, 0.0, 0.5 * a, 0.5 * a, 0.5 * a, 0.0, 0.5 * a};
    const std::vector<int> z{29, 29, 29};
    check(pot, pos, z, box, 2e-8, 2e-7, 2e-10, 2e-8);
  }
}

TEST_CASE("EAM minimum-image cells keep the develop bits", "[EAMTable][nlist]") {
  const rgpot::EAMTablePot pot = cu_u3();
  constexpr std::size_t n = 4;
  const std::vector<int> z(n, 29);
  std::uint64_t state = 0x9e3779b97f4a7c15ull;
  auto next = [&]() {
    state = state * 6364136223846793005ull + 1442695040888963407ull;
    return state;
  };
  auto uniform = [&](double lo, double hi) {
    const double u = static_cast<double>(next() >> 11) *
                     (1.0 / static_cast<double>(1ull << 53));
    return lo + (hi - lo) * u;
  };
  std::uint64_t hash = 1469598103934665603ull;
  for (int sample = 0; sample < 300; ++sample) {
    const double ax = uniform(11.5, 14.5);
    const double by = uniform(11.5, 14.5);
    const double cz = uniform(11.5, 14.5);
    const double b0 = uniform(-0.7, 0.7);
    const double c0 = uniform(-0.7, 0.7);
    const double c1 = uniform(-0.7, 0.7);
    const double box[9] = {ax, 0.0, 0.0, b0, by, 0.0, c0, c1, cz};
    std::vector<double> pos(3 * n);
    for (std::size_t i = 0; i < n; ++i) {
      const double s0 = uniform(0.0, 1.0);
      const double s1 = uniform(0.0, 1.0);
      const double s2 = uniform(0.0, 1.0);
      pos[3 * i] = s0 * box[0] + s1 * box[3] + s2 * box[6];
      pos[3 * i + 1] = s1 * box[4] + s2 * box[7];
      pos[3 * i + 2] = s2 * box[8];
    }
    hash = hash_eval(hash, eval_full(pot, pos, z, box));
  }
  CAPTURE(hash);
  REQUIRE(hash == 17112356085923370801ull);
}

TEST_CASE("EAM multi-image rejects unbounded image extent", "[EAMTable][nlist]") {
  const rgpot::EAMTablePot pot = pair_only();
  const std::vector<double> pos{0.4, 0.5, 0.6};
  const std::vector<int> z{29};
  const double bad_box[9] = {1.0e-6, 0.0, 0.0, 0.0, 4.0, 0.0, 0.0, 0.0, 4.0};
  REQUIRE_THROWS_WITH(eval_full(pot, pos, z, bad_box),
                      Catch::Matchers::ContainsSubstring("image extent on axis"));
}
