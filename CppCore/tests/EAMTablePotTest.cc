// MIT License
// Copyright 2023--present rgpot developers
//
// Tabulated EAM checks against the Large-scale Atomic/Molecular Massively
// Parallel Simulator (LAMMPS): Cu_u3.eam through pair_style eam,
// CuNi.eam.alloy through eam/alloy (data/eam_table/make_lammps_ref.py),
// and the Sutton-Chen eam/fs run of data/sutton_chen/make_lammps_ref.py.

#include <algorithm>
#include <array>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <fstream>
#include <sstream>
#include <stdexcept>
#include <string>
#include <vector>

#include <catch2/catch_all.hpp>

#include "rgpot/EAMTable/EAMTablePot.hpp"

using Catch::Matchers::WithinAbs;
using rgpot::EAMTableFormat;

namespace {

const std::string kData = "CppCore/tests/data/";

/// One LAMMPS run. Stress is xx yy zz yz xz xy in eV/Angstrom^3. Rows
/// are x y z fx fy fz, led by the atomic number when `with_z`.
struct LammpsRef {
  double energy = 0.0, box[9] = {}, stress[6] = {};
  std::vector<double> pos, forces;
  std::vector<int> z;
};

LammpsRef load_ref(const std::string &path, bool with_z) {
  std::ifstream in(kData + path);
  REQUIRE(in.good());
  LammpsRef ref;
  std::string key;
  std::size_t n = 0;
  while (in >> key) {
    if (key == "energy") {
      in >> ref.energy;
    } else if (key == "box") {
      for (double &v : ref.box)
        in >> v;
    } else if (key == "stress") {
      for (double &v : ref.stress)
        in >> v;
    } else if (key == "atoms" && in >> n) {
      for (std::size_t i = 0; i < n; ++i) {
        int z = 78;
        if (with_z)
          in >> z;
        ref.z.push_back(z);
        for (int k = 0; k < 6; ++k) {
          double v = 0.0;
          in >> v;
          (k < 3 ? ref.pos : ref.forces).push_back(v);
        }
      }
    } else {
      std::getline(in, key);
    }
  }
  return ref;
}

/// Row-major indices of the Voigt order xx yy zz yz xz xy.
constexpr int kVoigt[6][2] = {{0, 0}, {1, 1}, {2, 2}, {1, 2}, {0, 2}, {0, 1}};

struct Eval {
  double energy = 0.0, stress[9] = {};
  int has_stress = 0;
  std::vector<double> forces;
};

Eval eval(const rgpot::EAMTablePot &pot, const std::vector<double> &pos,
          const std::vector<int> &z, const double *box) {
  Eval e;
  e.forces.assign(pos.size(), 0.0);
  rgpot::ForceOut out{};
  out.F = e.forces.data();
  std::fill(out.stress, out.stress + 9, 7.0);
  pot.forceImpl(rgpot::ForceInput{z.size(), pos.data(), z.data(), box}, &out);
  e.energy = out.energy;
  std::copy(out.stress, out.stress + 9, e.stress);
  e.has_stress = out.has_stress;
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

Eval supercell_reference(const rgpot::EAMTablePot &pot,
                         const std::vector<double> &pos,
                         const std::vector<int> &z, const double *box) {
  const std::array<int, 3> rep = replication(box, pot.locality().cutoff);
  const std::size_t n = z.size();
  const std::size_t nrep = static_cast<std::size_t>(rep[0] * rep[1] * rep[2]);
  const double big_box[9] = {rep[0] * box[0], rep[0] * box[1], rep[0] * box[2],
                             rep[1] * box[3], rep[1] * box[4], rep[1] * box[5],
                             rep[2] * box[6], rep[2] * box[7], rep[2] * box[8]};
  std::vector<double> big_pos;
  std::vector<int> big_z;
  big_pos.reserve(3 * n * nrep);
  big_z.reserve(n * nrep);
  for (int h = 0; h < rep[0]; ++h) {
    for (int k = 0; k < rep[1]; ++k) {
      for (int l = 0; l < rep[2]; ++l) {
        const double sx = h * box[0] + k * box[3] + l * box[6];
        const double sy = h * box[1] + k * box[4] + l * box[7];
        const double sz = h * box[2] + k * box[5] + l * box[8];
        for (std::size_t i = 0; i < n; ++i) {
          big_pos.push_back(pos[3 * i] + sx);
          big_pos.push_back(pos[3 * i + 1] + sy);
          big_pos.push_back(pos[3 * i + 2] + sz);
          big_z.push_back(z[i]);
        }
      }
    }
  }
  const Eval big = eval(pot, big_pos, big_z, big_box);
  Eval ref;
  ref.energy = big.energy / static_cast<double>(nrep);
  std::copy(big.stress, big.stress + 9, ref.stress);
  ref.has_stress = big.has_stress;
  ref.forces.assign(3 * n, 0.0);
  for (std::size_t t = 0; t < nrep; ++t) {
    for (std::size_t i = 0; i < n; ++i) {
      const std::size_t src = t * n + i;
      for (int c = 0; c < 3; ++c) {
        ref.forces[3 * i + static_cast<std::size_t>(c)] +=
            big.forces[3 * src + static_cast<std::size_t>(c)] /
            static_cast<double>(nrep);
      }
    }
  }
  return ref;
}

void check_against(const rgpot::EAMTablePot &pot, const LammpsRef &ref,
                   double etol, double ftol, double stol) {
  const Eval e = eval(pot, ref.pos, ref.z, ref.box);
  REQUIRE_THAT(e.energy, WithinAbs(ref.energy, etol));
  for (std::size_t k = 0; k < ref.forces.size(); ++k) {
    REQUIRE_THAT(e.forces[k], WithinAbs(ref.forces[k], ftol));
  }
  REQUIRE(e.has_stress == 1);
  for (int v = 0; v < 6; ++v) {
    REQUIRE_THAT(e.stress[3 * kVoigt[v][0] + kVoigt[v][1]],
                 WithinAbs(ref.stress[v], stol));
  }
}

/// The eam/fs table write_table in data/sutton_chen/make_lammps_ref.py
/// gives LAMMPS: platinum, n = 10, m = 8, 50001 points.
std::string sutton_chen_table() {
  const double eps = 1.98330e-2, c = 34.408, a = 3.92, cut = 5.5;
  const int nr = 50001;
  const double dr = cut / (nr - 1), drho = 1000.0 / (nr - 1);
  std::string s = "Sutton-Chen platinum\nn 10 m 8\nmake_lammps_ref.py\n1 Pt\n";
  char buf[64];
  std::snprintf(buf, sizeof buf, "%d %.17g %d %.17g %.17g\n", nr, drho, nr, dr,
                cut);
  s += buf;
  s += "78 195.084 3.92 fcc\n";
  for (int col = 0; col < 3; ++col) {
    for (int m = 0; m < nr; ++m) {
      const double r = std::max(m * dr, 1.0);
      const double v = col == 0   ? -eps * c * std::sqrt(m * drho)
                       : col == 1 ? std::pow(a / r, 8)
                                  : r * eps * std::pow(a / r, 10);
      std::snprintf(buf, sizeof buf, "%.17g\n", v);
      s += buf;
    }
  }
  return s;
}

/// Two-element eam/fs table of constants: F_A = -rho, F_B = -2 rho, no
/// pair term, rho_AA = 1, rho_BA = 3, rho_AB = 5, rho_BB = 7, with
/// rho_ab the density a b neighbor puts at an a site.
std::string two_element_fs() {
  std::ostringstream s;
  auto row = [&](auto f) {
    for (int m = 0; m < 11; ++m)
      s << f(m) << ' ';
    s << '\n';
  };
  s << "c\nc\nc\n2 A B\n11 1.0 11 0.5 4.0\n";
  s << "13 1.0 1.0 fcc\n";
  row([](int m) { return -1.0 * m; });
  row([](int) { return 1.0; });
  row([](int) { return 3.0; });
  s << "28 1.0 1.0 fcc\n";
  row([](int m) { return -2.0 * m; });
  row([](int) { return 5.0; });
  row([](int) { return 7.0; });
  for (int k = 0; k < 3; ++k)
    row([](int) { return 0.0; });
  return s.str();
}

rgpot::EAMTablePot cu_u3(bool periodic, bool stress) {
  return rgpot::EAMTablePot({.path = kData + "eam_table/Cu_u3.eam",
                             .format = EAMTableFormat::Funcfl,
                             .periodic = periodic,
                             .stress = stress});
}

} // namespace

TEST_CASE("funcfl Cu_u3 matches LAMMPS pair_style eam", "[EAMTable]") {
  check_against(cu_u3(true, true), load_ref("eam_table/cu_u3_cubic.ref", true),
                1e-10, 1e-10, 1e-13);
}

TEST_CASE("setfl CuNi matches LAMMPS pair_style eam/alloy", "[EAMTable]") {
  const rgpot::EAMTablePot pot(
      {.path = kData + "eam_table/CuNi.eam.alloy", .stress = true});
  check_against(pot, load_ref("eam_table/cuni_sheared.ref", true), 1e-10, 1e-10,
                1e-13);
}

TEST_CASE("eam/fs reader matches the LAMMPS Sutton-Chen run", "[EAMTable]") {
  std::istringstream in(sutton_chen_table());
  const rgpot::EAMTablePot pot(
      rgpot::read_eam_table(in, EAMTableFormat::FinnisSinclair), true, true);
  check_against(pot, load_ref("sutton_chen/pt_cubic.ref", false), 1e-10, 1e-10,
                1e-13);
}

TEST_CASE("eam/fs density blocks follow the LAMMPS order", "[EAMTable]") {
  std::istringstream in(two_element_fs());
  const rgpot::EAMTablePot pot(
      rgpot::read_eam_table(in, EAMTableFormat::FinnisSinclair), false, false);
  const std::vector<double> pos{0.0, 0.0, 0.0, 2.0, 0.0, 0.0};
  // A at an A site 1, B at an A site 5, A at a B site 3, B at a B site 7.
  REQUIRE(eval(pot, pos, {13, 13}, nullptr).energy == -2.0);
  REQUIRE(eval(pot, pos, {13, 28}, nullptr).energy == -5.0 - 2.0 * 3.0);
  REQUIRE(eval(pot, pos, {28, 28}, nullptr).energy == -28.0);
}

TEST_CASE("EAM table stress flag leaves energy and forces alone",
          "[EAMTable]") {
  const LammpsRef ref = load_ref("eam_table/cu_u3_cubic.ref", true);
  const Eval off = eval(cu_u3(true, false), ref.pos, ref.z, ref.box);
  const Eval on = eval(cu_u3(true, true), ref.pos, ref.z, ref.box);
  REQUIRE(off.energy == on.energy);
  REQUIRE(off.forces == on.forces);
  REQUIRE(off.has_stress == 0);
  REQUIRE(std::all_of(off.stress, off.stress + 9,
                      [](double v) { return v == 7.0; }));
  REQUIRE(cu_u3(true, false).paramsKey() != cu_u3(true, true).paramsKey());
}

TEST_CASE("EAM table rejects bad input", "[EAMTable]") {
  const LammpsRef ref = load_ref("eam_table/cu_u3_cubic.ref", true);
  std::vector<int> nickel(ref.z.size(), 28);
  REQUIRE_THROWS_AS(eval(cu_u3(true, false), ref.pos, nickel, ref.box),
                    std::invalid_argument);
  std::ifstream file(kData + "eam_table/Cu_u3.eam");
  std::string head, line;
  for (int k = 0; k < 40 && std::getline(file, line); ++k)
    head += line + "\n";
  std::istringstream cut(head);
  REQUIRE_THROWS_AS(rgpot::read_eam_table(cut, EAMTableFormat::Funcfl),
                    std::invalid_argument);
}

TEST_CASE("EAM table sums images for a 9 angstrom cell", "[EAMTable]") {
  const LammpsRef ref = load_ref("eam_table/cu_u3_cubic.ref", true);
  const double small[9] = {9.0, 0.0, 0.0, 0.0, 9.0, 0.0, 0.0, 0.0, 9.0};
  const rgpot::EAMTablePot pot = cu_u3(true, true);
  REQUIRE(face_spacing(small, 0) > pot.locality().cutoff);
  REQUIRE(face_spacing(small, 0) < 2.0 * pot.locality().cutoff);
  const Eval folded = eval(pot, ref.pos, ref.z, small);
  const Eval want = supercell_reference(pot, ref.pos, ref.z, small);
  REQUIRE_THAT(folded.energy, WithinAbs(want.energy, 1e-8));
  for (std::size_t c = 0; c < folded.forces.size(); ++c) {
    REQUIRE_THAT(folded.forces[c], WithinAbs(want.forces[c], 1e-7));
  }
  REQUIRE(folded.has_stress == 1);
  REQUIRE(want.has_stress == 1);
  for (int c = 0; c < 9; ++c) {
    REQUIRE_THAT(folded.stress[c], WithinAbs(want.stress[c], 1e-10));
  }
}

namespace {

struct Full {
  Eval e;
  std::vector<double> eatom, vatom;
};

/// forceImpl, then forcePerAtom on the same atoms. `reused` is
/// listReused() after the first of the two.
Full eval_full(const rgpot::EAMTablePot &pot, const std::vector<double> &pos,
               const std::vector<int> &z, const double *box,
               bool *reused = nullptr) {
  Full f;
  f.e = eval(pot, pos, z, box);
  if (reused != nullptr) {
    *reused = pot.listReused();
  }
  const std::uint64_t builds = pot.listBuilds();
  std::vector<double> forces(pos.size(), 0.0);
  f.eatom.assign(z.size(), 0.0);
  f.vatom.assign(6 * z.size(), 0.0);
  rgpot::ForceOut out{};
  out.F = forces.data();
  pot.forcePerAtom(rgpot::ForceInput{z.size(), pos.data(), z.data(), box}, &out,
                   rgpot::PerAtomOut{f.eatom.data(), f.vatom.data()});
  REQUIRE(pot.listBuilds() == builds);
  REQUIRE(out.energy == f.e.energy);
  REQUIRE(forces == f.e.forces);
  return f;
}

/// Positions sent into a lower-triangular cell whose rows are a, b, c.
void wrap_cell(std::vector<double> &pos, const double *box) {
  for (std::size_t i = 0; i < pos.size() / 3; ++i) {
    double *r = pos.data() + 3 * i;
    double s2 = r[2] / box[8];
    double s1 = (r[1] - s2 * box[7]) / box[4];
    double s0 = (r[0] - s1 * box[3] - s2 * box[6]) / box[0];
    s0 -= std::floor(s0);
    s1 -= std::floor(s1);
    s2 -= std::floor(s2);
    r[0] = s0 * box[0] + s1 * box[3] + s2 * box[6];
    r[1] = s1 * box[4] + s2 * box[7];
    r[2] = s2 * box[8];
  }
}

} // namespace

TEST_CASE("EAM table pair list reuse gives the bits of a rebuilt list",
          "[EAMTable][skin]") {
  // The cached list keeps its pairs out to the cutoff plus 1 Angstrom; a
  // skin of 0 rebuilds the list whenever an atom moves, so it is the
  // evaluation without the cache. Both visit the pairs in the same order.
  const LammpsRef ref = load_ref("eam_table/cuni_sheared.ref", true);
  const std::string path = kData + "eam_table/CuNi.eam.alloy";
  const rgpot::EAMTablePot warm({.path = path, .stress = true});
  const rgpot::EAMTablePot cold({.path = path, .stress = true, .skin = 0.0});
  REQUIRE(warm.paramsKey() == cold.paramsKey());
  REQUIRE(warm.caps().reentrancy == rgpot::Reentrancy::PerInstance);

  std::vector<double> pos = ref.pos;
  std::vector<int> z = ref.z;
  std::vector<double> box(ref.box, ref.box + 9);
  wrap_cell(pos, box.data());
  std::uint64_t builds = 0;

  auto step = [&](bool reused) {
    bool kept = false;
    const Full got = eval_full(warm, pos, z, box.data(), &kept);
    builds += reused ? 0 : 1;
    REQUIRE(kept == reused);
    REQUIRE(warm.listBuilds() == builds);
    const Full want = eval_full(cold, pos, z, box.data());
    REQUIRE(got.e.energy == want.e.energy);
    REQUIRE(got.e.forces == want.e.forces);
    REQUIRE(std::equal(got.e.stress, got.e.stress + 9, want.e.stress));
    REQUIRE(got.eatom == want.eatom);
    REQUIRE(got.vatom == want.vatom);
    return got;
  };

  const Full first = step(false);
  REQUIRE_THAT(first.e.energy, WithinAbs(ref.energy, 1e-10));

  // Steps of at most 0.12 Angstrom in all, under half the skin.
  for (int k = 0; k < 3; ++k) {
    for (std::size_t c = 0; c < pos.size(); ++c) {
      pos[c] +=
          0.04 / std::sqrt(3.0) * std::sin(0.37 * static_cast<double>(c) + k);
    }
    step(true);
  }

  // The same atoms seen through other images, and wrapped back.
  for (std::size_t i = 0; i < z.size(); i += 3) {
    pos[3 * i] += box[0] - box[3];
    pos[3 * i + 1] -= box[4];
  }
  step(true);
  wrap_cell(pos, box.data());
  step(true);

  // Another element on some sites leaves the list alone.
  for (std::size_t i = 0; i < z.size(); i += 5) {
    z[i] = z[i] == 29 ? 28 : 29;
  }
  step(true);

  // One atom past half the skin.
  pos[0] += 0.55;
  step(false);
  step(true);

  // A strained cell.
  for (double &v : box) {
    v *= 1.003;
  }
  for (double &v : pos) {
    v *= 1.003;
  }
  step(false);

  // One atom fewer.
  pos.resize(pos.size() - 3);
  z.pop_back();
  step(false);
}

TEST_CASE("EAM table pair list follows a free cluster and a stream",
          "[EAMTable][skin]") {
  const LammpsRef ref = load_ref("eam_table/cu_u3_cubic.ref", true);
  const std::string path = kData + "eam_table/Cu_u3.eam";
  const rgpot::EAMTablePot warm(
      {.path = path, .format = EAMTableFormat::Funcfl, .periodic = false});
  rgpot::EAMTablePot stream(
      {.path = path, .format = EAMTableFormat::Funcfl, .periodic = false});
  stream.setNeighbourMode(rgpot::nlist::Mode::Stream);
  const rgpot::EAMTablePot cold({.path = path,
                                 .format = EAMTableFormat::Funcfl,
                                 .periodic = false,
                                 .skin = 0.0});
  std::vector<double> pos = ref.pos;
  std::vector<double> built;
  int reuses = 0;
  for (int k = 0; k < 6; ++k) {
    for (std::size_t c = 0; c < pos.size(); ++c) {
      pos[c] += 0.08 * std::cos(0.53 * static_cast<double>(c) + k);
    }
    // No cell, so the move is the plain displacement.
    const bool keep =
        !built.empty() &&
        rgpot::nlist::withinSkin(built.data(), pos.data(), pos.size() / 3, 1.0);
    const Eval got = eval(warm, pos, ref.z, nullptr);
    const Eval want = eval(cold, pos, ref.z, nullptr);
    REQUIRE(warm.listReused() == keep);
    if (!keep) {
      built = pos;
    }
    reuses += keep ? 1 : 0;
    REQUIRE(got.energy == want.energy);
    REQUIRE(got.forces == want.forces);
    const Eval streamed = eval(stream, pos, ref.z, nullptr);
    REQUIRE(stream.listBuilds() == 0);
    REQUIRE_THAT(streamed.energy, WithinAbs(want.energy, 1e-10));
    for (std::size_t c = 0; c < pos.size(); ++c) {
      REQUIRE_THAT(streamed.forces[c], WithinAbs(want.forces[c], 1e-10));
    }
  }
  REQUIRE(reuses > 0);
  REQUIRE(warm.listBuilds() < 6);
}

TEST_CASE("EAM table rejects a negative skin", "[EAMTable][skin]") {
  REQUIRE_THROWS_AS(rgpot::EAMTablePot({.path = kData + "eam_table/Cu_u3.eam",
                                        .format = EAMTableFormat::Funcfl,
                                        .skin = -0.5}),
                    std::invalid_argument);
}
