// MIT License
// Copyright 2023--present rgpot developers
//
// Multi-image EAM checks against an independent supercell reference.

#include <array>
#include <cmath>
#include <sstream>
#include <string>
#include <vector>

#include <catch2/catch_all.hpp>

#include "rgpot/EAMTable/EAMTablePot.hpp"
#include "rgpot/PerAtom.hpp"
#include "rgpot/nlist/MultiImage.hpp"

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
  return rgpot::EAMTablePot(rgpot::read_eam_table(in, EAMTableFormat::Setfl),
                            true, true);
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
  out.box = {rep[0] * box[0], rep[0] * box[1], rep[0] * box[2],
             rep[1] * box[3], rep[1] * box[4], rep[1] * box[5],
             rep[2] * box[6], rep[2] * box[7], rep[2] * box[8]};
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

Eval supercell_reference(const rgpot::EAMTablePot &pot,
                         const std::vector<double> &pos,
                         const std::vector<int> &z, const double *box,
                         const std::array<int, 3> &rep);

Eval supercell_reference(const rgpot::EAMTablePot &pot,
                         const std::vector<double> &pos,
                         const std::vector<int> &z, const double *box) {
  return supercell_reference(pot, pos, z, box,
                             replication(box, pot.locality().cutoff));
}

Eval supercell_reference(const rgpot::EAMTablePot &pot,
                         const std::vector<double> &pos,
                         const std::vector<int> &z, const double *box,
                         const std::array<int, 3> &rep) {
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
           const std::vector<int> &z, const double *box, double etol,
           double ftol, double stol, double atom_tol) {
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
    const std::array<double, 6> got_v = total_virial(got);
    REQUIRE_THAT(got_v[0], WithinAbs(7.0, 1e-12));
    REQUIRE_THAT(got.stress[0], WithinAbs(-7.0 / 64.0, 1e-12));
    check(pot, pos, z, box, 1e-12, 1e-10, 1e-12, 1e-12);
  }

  SECTION("pair-only one-atom triclinic cell") {
    const rgpot::EAMTablePot pot = pair_only();
    const std::vector<double> pos{0.2, 0.3, 0.4};
    const std::vector<int> z{29};
    const double box[9] = {3.6, 0.0, 0.0, 1.8, 3.1, 0.0, 1.8, 1.0, 2.9};
    check(pot, pos, z, box, 1e-11, 1e-9, 1e-11, 1e-11);
  }

  SECTION("Cu_u3 one-atom cubic cell a=3.615A (3x3x3 supercell/N)") {
    const rgpot::EAMTablePot pot = cu_u3();
    const std::vector<double> pos{0.3, 0.5, 0.7};
    const std::vector<int> z{29};
    const double a = 3.615;
    const double box[9] = {a, 0.0, 0.0, 0.0, a, 0.0, 0.0, 0.0, a};
    REQUIRE(face_spacing(box, 0) < pot.locality().cutoff);
    const Eval got = eval_full(pot, pos, z, box);
    const Eval want = supercell_reference(pot, pos, z, box, {3, 3, 3});
    REQUIRE_THAT(want.energy, WithinAbs(-0.802510540848, 5e-10));
    REQUIRE_THAT(want.stress[0], WithinAbs(0.0321577087, 5e-10));
    REQUIRE(std::abs(want.energy) > 1e-6);
    REQUIRE(got.has_stress == 1);
    REQUIRE(want.has_stress == 1);
    REQUIRE_THAT(got.energy, WithinAbs(want.energy, 1e-8));
    for (std::size_t c = 0; c < got.forces.size(); ++c) {
      REQUIRE_THAT(got.forces[c], WithinAbs(want.forces[c], 1e-7));
    }
    for (int c = 0; c < 9; ++c) {
      REQUIRE_THAT(got.stress[static_cast<std::size_t>(c)],
                   WithinAbs(want.stress[static_cast<std::size_t>(c)], 1e-10));
    }
    for (std::size_t i = 0; i < got.eatom.size(); ++i) {
      REQUIRE_THAT(got.eatom[i], WithinAbs(want.eatom[i], 1e-8));
    }
    const std::array<double, 6> gv = total_virial(got);
    const std::array<double, 6> wv = total_virial(want);
    for (int c = 0; c < 6; ++c) {
      REQUIRE_THAT(gv[static_cast<std::size_t>(c)],
                   WithinAbs(wv[static_cast<std::size_t>(c)], 1e-8));
    }
  }

  SECTION("Cu_u3 one-atom cubic cell a=3.0A (4x4x4 supercell/N)") {
    const rgpot::EAMTablePot pot = cu_u3();
    const std::vector<double> pos{0.25, 0.5, 0.75};
    const std::vector<int> z{29};
    const double a = 3.0;
    const double box[9] = {a, 0.0, 0.0, 0.0, a, 0.0, 0.0, 0.0, a};
    REQUIRE(face_spacing(box, 0) < pot.locality().cutoff);
    const Eval got = eval_full(pot, pos, z, box);
    const Eval want = supercell_reference(pot, pos, z, box, {4, 4, 4});
    REQUIRE_THAT(want.energy, WithinAbs(-1.941358963766, 5e-10));
    REQUIRE_THAT(want.stress[0], WithinAbs(0.0855559087, 5e-10));
    REQUIRE(std::abs(want.energy) > 1e-6);
    REQUIRE(got.has_stress == 1);
    REQUIRE(want.has_stress == 1);
    REQUIRE_THAT(got.energy, WithinAbs(want.energy, 2e-8));
    for (std::size_t c = 0; c < got.forces.size(); ++c) {
      REQUIRE_THAT(got.forces[c], WithinAbs(want.forces[c], 2e-7));
    }
    for (int c = 0; c < 9; ++c) {
      REQUIRE_THAT(got.stress[static_cast<std::size_t>(c)],
                   WithinAbs(want.stress[static_cast<std::size_t>(c)], 2e-10));
    }
    for (std::size_t i = 0; i < got.eatom.size(); ++i) {
      REQUIRE_THAT(got.eatom[i], WithinAbs(want.eatom[i], 2e-8));
    }
    const std::array<double, 6> gv = total_virial(got);
    const std::array<double, 6> wv = total_virial(want);
    for (int c = 0; c < 6; ++c) {
      REQUIRE_THAT(gv[static_cast<std::size_t>(c)],
                   WithinAbs(wv[static_cast<std::size_t>(c)], 2e-8));
    }
  }

  SECTION("Cu_u3 vacancy cell") {
    const rgpot::EAMTablePot pot = cu_u3();
    const double a = 3.615;
    const double box[9] = {a, 0.0, 0.0, 0.0, a, 0.0, 0.0, 0.0, a};
    // One missing site in a conventional FCC cell.
    const std::vector<double> pos{0.0,     0.0,     0.0, 0.0,    0.5 * a,
                                  0.5 * a, 0.5 * a, 0.0, 0.5 * a};
    const std::vector<int> z{29, 29, 29};
    check(pot, pos, z, box, 2e-8, 2e-7, 2e-10, 2e-8);
  }
}

namespace {

void expect_pin(const Eval &got, double energy,
                const std::vector<double> &forces,
                const std::vector<double> &stress, double tol) {
  REQUIRE(got.has_stress == 1);
  REQUIRE(got.forces.size() == forces.size());
  REQUIRE(stress.size() == 9);
  REQUIRE_THAT(got.energy, WithinAbs(energy, tol));
  for (std::size_t i = 0; i < forces.size(); ++i) {
    REQUIRE_THAT(got.forces[i], WithinAbs(forces[i], tol));
  }
  for (std::size_t i = 0; i < stress.size(); ++i) {
    REQUIRE_THAT(got.stress[i], WithinAbs(stress[i], tol));
  }
}

} // namespace

TEST_CASE("EAM minimum-image cells match the stored neighbour values",
          "[EAMTable][nlist]") {
  // Face spacing above twice the cutoff, so the image walk is not entered.
  // The literals are the stored neighbour walk on these cells.
  const rgpot::EAMTablePot pot = cu_u3();
  const double cutoff = pot.locality().cutoff;
  const std::vector<int> z{29, 29, 29, 29};

  SECTION("orthogonal 12A cell") {
    const double box[9] = {12.0, 0.0, 0.0, 0.0, 12.0, 0.0, 0.0, 0.0, 12.0};
    REQUIRE(face_spacing(box, 0) > 2.0 * cutoff);
    const std::vector<double> pos{1.0, 2.0, 3.0, 3.2, 2.4, 3.5,
                                  2.2, 4.5, 3.2, 4.0, 3.5, 5.0};
    const Eval got = eval_full(pot, pos, z, box);
    expect_pin(got, -6.4302145542936842,
               {0.4875496499688981, 0.76465695323661698, 0.13023723466897327,
                -1.0982647766643991, -1.2376673869681145, -1.8736236435097737,
                0.18288849580104144, -1.0908290140632739, 0.4250492434355394,
                0.42782663089445966, 1.5638394477947717, 1.3183371654052611},
               {0.00052849329919968152, -0.0003817449509135031,
                -0.00019855241470887036, -0.0003817449509135031,
                0.00050716453602619993, -0.0013256246523685362,
                -0.00019855241470887036, -0.0013256246523685362,
                -0.0010329122440640875},
               1e-12);
  }

  SECTION("triclinic cell") {
    const double box[9] = {12.0, 0.0, 0.0, 0.5, 12.4, 0.0, -0.4, 0.6, 13.0};
    for (int axis = 0; axis < 3; ++axis) {
      REQUIRE(face_spacing(box, axis) > 2.0 * cutoff);
    }
    const std::vector<double> pos{1.0, 1.5, 2.0, 3.4, 1.8, 2.6,
                                  1.6, 3.6, 2.4, 3.0, 3.2, 4.2};
    const Eval got = eval_full(pot, pos, z, box);
    expect_pin(
        got, -7.1238524269291768,
        {0.57276282635237352, -0.13871962187670273, 0.23900655285138978,
         -0.68106950559316604, -0.28231772988798709, -0.78836734278112719,
         0.418382145660189, 0.012399136073383834, 0.092924877428047101,
         -0.31007546641939648, 0.40863821569130593, 0.45643591250169036},
        {0.001035819095774544, -7.607183689799067e-05, 0.0004773856861117124,
         -7.607183689799067e-05, -0.00032879850778687444,
         -0.00037974208592712698, 0.0004773856861117124,
         -0.00037974208592712698, -0.00029379060835724836},
        1e-12);
  }
}

TEST_CASE(
    "EAM multi-image counts a second image between cutoff and twice cutoff",
    "[EAMTable][nlist]") {
  const rgpot::EAMTablePot pot = cu_u3();
  const double a = 8.0;
  const double cutoff = pot.locality().cutoff;
  const double box[9] = {a, 0.0, 0.0, 0.0, a, 0.0, 0.0, 0.0, a};
  REQUIRE(face_spacing(box, 0) > cutoff);
  REQUIRE(face_spacing(box, 0) < 2.0 * cutoff);
  // Atom 1 sits 4 A from atom 0, so both the home image and the neighbour
  // cell image fall inside the 4.95 A cutoff.
  const std::vector<double> pos{0.2, 0.3, 0.4, 4.2, 0.3, 0.5, 0.3, 4.5, 0.4};
  const std::vector<int> z{29, 29, 29};
  REQUIRE(replication(box, cutoff) == std::array<int, 3>{2, 2, 2});
  check(pot, pos, z, box, 1e-8, 1e-7, 1e-10, 1e-8);
}

TEST_CASE("EAM multi-image recenters pairs outside the home cell",
          "[EAMTable][nlist]") {
  const rgpot::EAMTablePot pot = cu_u3();
  const double a = 8.0;
  const double cutoff = pot.locality().cutoff;
  const double box[9] = {a, 0.0, 0.0, 0.0, a, 0.0, 0.0, 0.0, a};
  REQUIRE(face_spacing(box, 0) < 2.0 * cutoff);
  // 80.2, 17.1, 24.4 and 184.5, 16.3, 25.2 are the images of
  // 0.2, 1.1, 0.4 and 0.5, 0.3, 1.2. The raw separation is many cells.
  const std::vector<double> unwrapped{80.2, 17.1, 24.4, 184.5, 16.3, 25.2};
  const std::vector<double> wrapped{0.2, 1.1, 0.4, 0.5, 0.3, 1.2};
  const std::vector<int> z{29, 29};
  const Eval got = eval_full(pot, unwrapped, z, box);
  const Eval want = supercell_reference(pot, wrapped, z, box);
  REQUIRE(std::abs(want.energy) > 1e-6);
  REQUIRE(got.has_stress == 1);
  REQUIRE(want.has_stress == 1);
  REQUIRE_THAT(got.energy, WithinAbs(want.energy, 1e-8));
  for (std::size_t c = 0; c < got.forces.size(); ++c) {
    REQUIRE_THAT(got.forces[c], WithinAbs(want.forces[c], 1e-7));
  }
  for (int c = 0; c < 9; ++c) {
    REQUIRE_THAT(got.stress[static_cast<std::size_t>(c)],
                 WithinAbs(want.stress[static_cast<std::size_t>(c)], 1e-10));
  }
  const std::array<double, 6> gv = total_virial(got);
  const std::array<double, 6> wv = total_virial(want);
  for (int c = 0; c < 6; ++c) {
    REQUIRE_THAT(gv[static_cast<std::size_t>(c)],
                 WithinAbs(wv[static_cast<std::size_t>(c)], 1e-8));
  }
}

TEST_CASE("EAM multi-image rejects unbounded image extent",
          "[EAMTable][nlist]") {
  const rgpot::EAMTablePot pot = pair_only();
  const std::vector<double> pos{0.4, 0.5, 0.6};
  const std::vector<int> z{29};
  const double bad_box[9] = {1.0e-6, 0.0, 0.0, 0.0, 4.0, 0.0, 0.0, 0.0, 4.0};
  REQUIRE_THROWS_WITH(
      eval_full(pot, pos, z, bad_box),
      Catch::Matchers::ContainsSubstring("image extent on axis"));
}

TEST_CASE("EAM multi-image rejects a large image count", "[EAMTable][nlist]") {
  const rgpot::EAMTablePot pot = pair_only();
  const double cutoff = pot.locality().cutoff;
  // extent = ceil(cutoff / spacing) + 1. Stay under the per-axis cap and
  // over the product of the shift counts (2 * extent + 1).
  constexpr int extent = 40;
  REQUIRE(extent < rgpot::nlist::detail::kMaxImageExtent);
  const auto span = static_cast<std::int64_t>(2 * extent + 1);
  REQUIRE(span * span * span > rgpot::nlist::detail::kMaxImageCount);
  const double spacing = cutoff / (static_cast<double>(extent - 1) - 0.25);
  const double box[9] = {spacing, 0.0, 0.0, 0.0,    spacing,
                         0.0,     0.0, 0.0, spacing};
  const std::vector<double> pos{0.2 * spacing, 0.3 * spacing, 0.4 * spacing};
  const std::vector<int> z{29};
  REQUIRE_THROWS_WITH(eval_full(pot, pos, z, box),
                      Catch::Matchers::ContainsSubstring("image count"));
}
