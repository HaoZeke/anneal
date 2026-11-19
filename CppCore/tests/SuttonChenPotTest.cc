// MIT License
// Copyright 2023--present rgpot developers
//
// Sutton-Chen checks: an analytic dimer, a finite-difference force, and
// the Doye and Wales 10-8 cluster energies.

#include <algorithm>
#include <array>
#include <cmath>
#include <cstdlib>
#include <cstring>
#include <fstream>
#include <sstream>
#include <stdexcept>
#include <string>
#include <thread>
#include <tuple>
#include <vector>

#include <catch2/catch_all.hpp>

#include "lsan_libomp.hpp"
#include "rgpot/KeptPotential.hpp"
#include "rgpot/PairTerms.hpp"
#include "rgpot/SuttonChen/SuttonChenPot.hpp"
#include "rgpot/SuttonChen/SuttonChenScales.hpp"
#include "rgpot/nlist/NeighbourList.hpp"
#include "rgpot/stress.hpp"
#include "rgpot/types/AtomMatrix.hpp"

using Catch::Matchers::WithinAbs;
using rgpot::types::AtomMatrix;

namespace {

struct SuttonChenPotDevelopLayout : rgpot::Potential<rgpot::SuttonChenPot> {
  rgpot::SuttonChenConfig m_config;
  std::uint64_t m_paramsKey;
  rgpot::nlist::Mode mode_;
  std::size_t term_limit_;
  mutable rgpot::nlist::Neighbours neighbours_;
  mutable std::vector<rgpot::PairTerms<3>> terms_;
  mutable std::vector<double> rho_;
  mutable std::vector<double> weight_;
};

AtomMatrix load_points(const std::string &path, double *epsilon, double *c,
                       double *a, int *nexp, int *mexp) {
  std::ifstream in(path);
  if (!in) {
    throw std::runtime_error("cannot open " + path);
  }
  double nraw = 0.0;
  double mraw = 0.0;
  in >> *epsilon >> *c >> *a >> nraw >> mraw;
  *nexp = static_cast<int>(nraw);
  *mexp = static_cast<int>(mraw);
  std::vector<double> xyz;
  double x = 0.0;
  double y = 0.0;
  double z = 0.0;
  while (in >> x >> y >> z) {
    xyz.push_back(x);
    xyz.push_back(y);
    xyz.push_back(z);
  }
  AtomMatrix pos(xyz.size() / 3, 3);
  for (size_t i = 0; i < xyz.size(); ++i) {
    pos(i / 3, i % 3) = xyz[i];
  }
  return pos;
}

std::array<std::array<double, 3>, 3> cube(double length) {
  return {{{length, 0.0, 0.0}, {0.0, length, 0.0}, {0.0, 0.0, length}}};
}

/// Independent dimer energy and dE/dr. Not the library kernel.
void dimer_oracle(double r, double epsilon, double c, double a, int nexp,
                  int mexp, double *energy, double *dEdr) {
  const double s = a / r;
  const double rep = std::pow(s, nexp);
  const double phi = std::pow(s, mexp);
  *energy = epsilon * rep - 2.0 * epsilon * c * std::sqrt(phi);
  *dEdr = -epsilon * nexp * rep / r +
          epsilon * c * mexp * std::sqrt(phi) / r;
}

/// Energy per atom of an fcc crystal at nearest-neighbor `radius`. The
/// sites are the conventional cubic cell and its four-site basis within
/// a sphere of 12 cells. Not fcc_power_sum.
double direct_fcc_energy(const rgpot::SuttonChenConfig &cfg, double radius) {
  constexpr int kCells = 12;
  const double basis[4][3] = {
      {0.0, 0.0, 0.0}, {0.5, 0.5, 0.0}, {0.5, 0.0, 0.5}, {0.0, 0.5, 0.5}};
  const double cell = radius * std::sqrt(2.0);
  double pair = 0.0;
  double rho = 0.0;
  for (int i = -kCells; i <= kCells; ++i) {
    for (int j = -kCells; j <= kCells; ++j) {
      for (int k = -kCells; k <= kCells; ++k) {
        for (const auto &b : basis) {
          const double x = i + b[0];
          const double y = j + b[1];
          const double z = k + b[2];
          const double d2 = x * x + y * y + z * z;
          if (d2 == 0.0 || d2 > kCells * kCells) {
            continue;
          }
          const double s = cfg.a / (cell * std::sqrt(d2));
          pair += std::pow(s, cfg.n);
          rho += std::pow(s, cfg.m);
        }
      }
    }
  }
  return cfg.epsilon * (0.5 * pair - cfg.c * std::sqrt(rho));
}

/// One run of the Large-scale Atomic/Molecular Massively Parallel Simulator
/// (LAMMPS) with eam/fs, written by data/sutton_chen/make_lammps_ref.py.
/// Stress is xx yy zz yz xz xy in eV/Angstrom^3; rows are x y z fx fy fz.
struct LammpsRef {
  double cutoff = 0.0, energy = 0.0, box[9] = {}, stress[6] = {};
  std::vector<double> pos, forces;
};

LammpsRef load_lammps_ref(const std::string &name) {
  std::ifstream in("CppCore/tests/data/sutton_chen/" + name + ".ref");
  REQUIRE(in.good());
  LammpsRef ref;
  std::string key;
  std::size_t n = 0;
  while (in >> key) {
    if (key == "cutoff")
      in >> ref.cutoff;
    else if (key == "energy")
      in >> ref.energy;
    else if (key == "box")
      for (double &v : ref.box)
        in >> v;
    else if (key == "stress")
      for (double &v : ref.stress)
        in >> v;
    else if (key == "atoms" && in >> n) {
      for (std::size_t i = 0; i < 6 * n; ++i) {
        double v = 0.0;
        in >> v;
        (i % 6 < 3 ? ref.pos : ref.forces).push_back(v);
      }
    } else
      std::getline(in, key);
  }
  return ref;
}

/// Row-major indices of the Voigt order xx yy zz yz xz xy.
constexpr int kVoigt[6][2] = {{0, 0}, {1, 1}, {2, 2}, {1, 2}, {0, 2}, {0, 1}};

struct StressEval {
  double energy = 0.0, stress[9] = {};
  int has_stress = 0;
  std::vector<double> forces;
};

/// forceImpl with the stress slots preset to 7, so an untouched tensor shows.
StressEval eval_stress(const rgpot::SuttonChenPot &pot,
                       const std::vector<double> &pos, const double *box) {
  StressEval e;
  e.forces.assign(pos.size(), 0.0);
  rgpot::ForceOut out{};
  out.F = e.forces.data();
  std::fill(out.stress, out.stress + 9, 7.0);
  pot.forceImpl(rgpot::ForceInput{pos.size() / 3, pos.data(), nullptr, box},
                &out);
  e.energy = out.energy;
  std::copy(out.stress, out.stress + 9, e.stress);
  e.has_stress = out.has_stress;
  return e;
}

rgpot::SuttonChenPot platinum_periodic(double cutoff, bool stress) {
  rgpot::SuttonChenConfig cfg = rgpot::sutton_chen_platinum();
  cfg.cutoff = cutoff;
  cfg.periodic = true;
  cfg.stress = stress;
  return rgpot::SuttonChenPot(cfg);
}

/// Perfect fcc crystal of cells^3 conventional cells in a cubic box.
std::vector<double> fcc_cell(int cells, double lattice, double *box) {
  const double basis[4][3] = {
      {0, 0, 0}, {0.5, 0.5, 0}, {0.5, 0, 0.5}, {0, 0.5, 0.5}};
  std::fill(box, box + 9, 0.0);
  box[0] = box[4] = box[8] = cells * lattice;
  std::vector<double> pos;
  for (int i = 0; i < cells * cells * cells; ++i) {
    for (const auto &b : basis) {
      pos.push_back((i % cells + b[0]) * lattice);
      pos.push_back((i / cells % cells + b[1]) * lattice);
      pos.push_back((i / (cells * cells) + b[2]) * lattice);
    }
  }
  return pos;
}

} // namespace

TEST_CASE("Sutton-Chen layout matches develop baseline", "[SuttonChen][layout]") {
  REQUIRE(sizeof(rgpot::SuttonChenPot) == sizeof(SuttonChenPotDevelopLayout));
}

TEST_CASE("Sutton-Chen dimer matches an independent derivative",
          "[SuttonChen]") {
  const rgpot::SuttonChenConfig cfg{.cutoff = 0.0};
  rgpot::SuttonChenPot pot(cfg);
  const double r = 0.8;
  const AtomMatrix pos{{0.0, 0.0, 0.0}, {r, 0.0, 0.0}};
  const std::vector<int> types{78, 78};
  auto [energy, forces, variance] = pot(pos, types, cube(40.0));
  (void)variance;

  double expect_e = 0.0;
  double dEdr = 0.0;
  dimer_oracle(r, cfg.epsilon, cfg.c, cfg.a, cfg.n, cfg.m, &expect_e, &dEdr);
  REQUIRE_THAT(energy, WithinAbs(expect_e, 1e-9));
  // Atom 0 sits at the origin and atom 1 at +r. Attraction pushes atom 0
  // toward +x, so F_0x = dE/dr.
  REQUIRE_THAT(forces(0, 0), WithinAbs(dEdr, 1e-8));
  REQUIRE_THAT(forces(1, 0), WithinAbs(-dEdr, 1e-8));
  REQUIRE_THAT(forces(0, 1), WithinAbs(0.0, 1e-12));
  REQUIRE_THAT(forces(0, 2), WithinAbs(0.0, 1e-12));
}

TEST_CASE("Sutton-Chen forces match a central difference", "[SuttonChen]") {
  rgpot::SuttonChenPot pot(rgpot::SuttonChenConfig{.cutoff = 0.0});
  const int n = 5;
  AtomMatrix pos(static_cast<size_t>(n), 3);
  for (int i = 0; i < n; ++i) {
    const double t = 0.37 * i;
    pos(static_cast<size_t>(i), 0) = 0.55 * std::cos(t);
    pos(static_cast<size_t>(i), 1) = 0.55 * std::sin(t);
    pos(static_cast<size_t>(i), 2) = 0.21 * (i % 3);
  }
  const std::vector<int> types(static_cast<size_t>(n), 78);
  const auto box = cube(30.0);
  auto [energy, forces, variance] = pot(pos, types, box);
  (void)energy;
  (void)variance;

  const double h = 1e-6;
  for (int i = 0; i < n; ++i) {
    for (int k = 0; k < 3; ++k) {
      AtomMatrix plus = pos;
      AtomMatrix minus = pos;
      plus(static_cast<size_t>(i), static_cast<size_t>(k)) += h;
      minus(static_cast<size_t>(i), static_cast<size_t>(k)) -= h;
      const double e_plus = std::get<0>(pot(plus, types, box));
      const double e_minus = std::get<0>(pot(minus, types, box));
      const double numeric = (e_plus - e_minus) / (2.0 * h);
      const double analytic = -forces(static_cast<size_t>(i),
                                      static_cast<size_t>(k));
      REQUIRE_THAT(analytic, WithinAbs(numeric, 1e-4 * (1.0 + std::abs(numeric))));
    }
  }
}

TEST_CASE("Sutton-Chen reproduces the Doye 10-8 cluster energies",
          "[SuttonChen]") {
  const char *files[] = {"CppCore/tests/data/sc108/pt38",
                         "CppCore/tests/data/sc108/pt75"};
  const double published[] = {-10117.2454, -20431.3452};
  for (int which = 0; which < 2; ++which) {
    double epsilon = 0.0;
    double c = 0.0;
    double a = 0.0;
    int nexp = 0;
    int mexp = 0;
    const AtomMatrix pos =
        load_points(files[which], &epsilon, &c, &a, &nexp, &mexp);
    REQUIRE(epsilon == 1.0);
    REQUIRE(c == 34.408);
    REQUIRE(a == 1.0);
    REQUIRE(nexp == 10);
    REQUIRE(mexp == 8);

    const rgpot::SuttonChenConfig cfg{
        .epsilon = epsilon, .c = c, .a = a, .n = nexp, .m = mexp, .cutoff = 40.0};
    rgpot::SuttonChenPot pot(cfg);
    const std::vector<int> types(pos.rows(), 78);
    auto [energy, forces, variance] = pot(pos, types, cube(80.0));
    (void)variance;
    // The published tables print four digits. The geometry is the
    // deposited point file for that row.
    REQUIRE_THAT(energy, WithinAbs(published[which], 5e-5));

    rgpot::SuttonChenPot full(rgpot::SuttonChenConfig{
        .epsilon = epsilon, .c = c, .a = a, .n = nexp, .m = mexp, .cutoff = 0.0});
    const double energy_full = std::get<0>(full(pos, types, cube(80.0)));
    REQUIRE_THAT(energy_full, WithinAbs(energy, 1e-9));

    double net[3] = {0.0, 0.0, 0.0};
    for (size_t i = 0; i < forces.rows(); ++i) {
      net[0] += forces(i, 0);
      net[1] += forces(i, 1);
      net[2] += forces(i, 2);
    }
    for (double component : net) {
      REQUIRE_THAT(component, WithinAbs(0.0, 1e-6));
    }
  }
}

TEST_CASE("Sutton-Chen with no cutoff keeps the bits of a one-bin search",
          "[SuttonChen][nlist]") {
  // cutoff <= 0 once stood for a 1e6 search, which bins a cluster in one
  // cell. The unbounded walk takes the pairs of that cell in its order.
  double epsilon = 0.0;
  double c = 0.0;
  double a = 0.0;
  int nexp = 0;
  int mexp = 0;
  const AtomMatrix pt38 = load_points("CppCore/tests/data/sc108/pt38", &epsilon,
                                      &c, &a, &nexp, &mexp);
  std::vector<double> pos;
  for (size_t i = 0; i < pt38.rows(); ++i) {
    for (size_t k = 0; k < 3; ++k) {
      pos.push_back(pt38(i, k) *
                    (1.0 + 0.01 * std::sin(1.7 * static_cast<double>(3 * i + k))));
    }
  }
  std::vector<double> scaled = pos;
  for (double &x : scaled) {
    x *= rgpot::sutton_chen_platinum().a;
  }
  const rgpot::SuttonChenConfig rows[] = {rgpot::SuttonChenConfig{},
                                          rgpot::sutton_chen_platinum()};
  for (const auto &row : rows) {
    const std::vector<double> &xyz = row.a == 1.0 ? pos : scaled;
    for (const auto mode : {rgpot::nlist::Mode::Auto, rgpot::nlist::Mode::Stored,
                            rgpot::nlist::Mode::Stream}) {
      INFO("a " << row.a << " mode " << static_cast<int>(mode));
      rgpot::SuttonChenConfig none = row;
      none.cutoff = 0.0;
      rgpot::SuttonChenConfig wide = row;
      wide.cutoff = 1.0e6;
      rgpot::SuttonChenPot unbounded(none);
      rgpot::SuttonChenPot one_bin(wide);
      unbounded.setNeighbourMode(mode);
      one_bin.setNeighbourMode(mode);
      const StressEval got = eval_stress(unbounded, xyz, nullptr);
      const StressEval want = eval_stress(one_bin, xyz, nullptr);
      REQUIRE(std::memcmp(&got.energy, &want.energy, sizeof(double)) == 0);
      REQUIRE(std::memcmp(got.forces.data(), want.forces.data(),
                          got.forces.size() * sizeof(double)) == 0);
    }
  }
}

TEST_CASE("periodic Sutton-Chen folds a pair and agrees in a wide cell",
          "[SuttonChen]") {
  const AtomMatrix pos{{0.1, 0.0, 0.0}, {4.9, 0.0, 0.0}};
  const std::vector<int> types{78, 78};
  rgpot::SuttonChenPot cluster(rgpot::SuttonChenConfig{.cutoff = 3.0});
  rgpot::SuttonChenPot periodic(
      rgpot::SuttonChenConfig{.cutoff = 3.0, .periodic = true});
  const double e_cluster = std::get<0>(cluster(pos, types, cube(5.0)));
  const double e_periodic = std::get<0>(periodic(pos, types, cube(5.0)));
  // Direct separation is 4.8, outside the cutoff. The periodic image is
  // 0.2. That image is on the repulsive wall, so the energy is large
  // and positive, and it matches an unwrapped pair at the same distance.
  REQUIRE_THAT(e_cluster, WithinAbs(0.0, 1e-15));
  const AtomMatrix close{{0.0, 0.0, 0.0}, {0.2, 0.0, 0.0}};
  const double e_close = std::get<0>(cluster(close, types, cube(40.0)));
  REQUIRE_THAT(e_periodic, WithinAbs(e_close, 1e-6));

  const AtomMatrix wide{{0.0, 0.0, 0.0}, {0.9, 0.2, -0.1}};
  const auto box = cube(40.0);
  const double e_free = std::get<0>(cluster(wide, types, box));
  const double e_boxed = std::get<0>(periodic(wide, types, box));
  REQUIRE_THAT(e_boxed, WithinAbs(e_free, 1e-9));
}

TEST_CASE("the C entry point matches the C++ potential", "[SuttonChen]") {
  const rgpot_sc_config cfg{1.0, 34.408, 1.0, 10, 8, 40.0, 0};
  const double pos[] = {0.0, 0.0, 0.0, 0.7, 0.1, -0.2, -0.4, 0.5, 0.3};
  const double box[] = {20, 0, 0, 0, 20, 0, 0, 0, 20};
  double energy = 0.0;
  double forces[9] = {};
  REQUIRE(rgpot_sutton_chen_energy_forces(&cfg, 3, pos, box, &energy, forces) == 0);

  AtomMatrix m(3, 3);
  for (int i = 0; i < 9; ++i) {
    m(static_cast<size_t>(i / 3), static_cast<size_t>(i % 3)) = pos[i];
  }
  rgpot::SuttonChenPot pot;
  auto [e2, f2, v2] = pot(m, std::vector<int>{78, 78, 78}, cube(20.0));
  (void)v2;
  REQUIRE_THAT(energy, WithinAbs(e2, 1e-12));
  for (int i = 0; i < 3; ++i) {
    for (int k = 0; k < 3; ++k) {
      REQUIRE_THAT(forces[3 * i + k], WithinAbs(f2(static_cast<size_t>(i),
                                                   static_cast<size_t>(k)),
                                                1e-12));
    }
  }
}

TEST_CASE("Sutton-Chen parameter fingerprint tracks the config",
          "[SuttonChen]") {
  rgpot::SuttonChenPot defaults;
  rgpot::SuttonChenPot same{rgpot::SuttonChenConfig{}};
  rgpot::SuttonChenPot other{rgpot::SuttonChenConfig{.c = 39.432, .n = 9, .m = 6}};
  REQUIRE(defaults.paramsKey() == same.paramsKey());
  REQUIRE(defaults.paramsKey() != other.paramsKey());
  REQUIRE(defaults.getType() == rgpot::PotType::SuttonChen);
  REQUIRE_THROWS(rgpot::SuttonChenPot(
      rgpot::SuttonChenConfig{.cutoff = 0.0, .periodic = true}));
}

TEST_CASE("Sutton-Chen bulk scales and the gold and platinum rows",
          "[SuttonChen]") {
  const rgpot::SuttonChenConfig reduced;
  const auto scales = rgpot::sutton_chen_scales(reduced);
  const double here =
      rgpot::sutton_chen_fcc_energy(reduced, scales.nearest_neighbor);
  const double left = rgpot::sutton_chen_fcc_energy(
      reduced, scales.nearest_neighbor * (1.0 - 1.0e-4));
  const double right = rgpot::sutton_chen_fcc_energy(
      reduced, scales.nearest_neighbor * (1.0 + 1.0e-4));
  REQUIRE(here < left);
  REQUIRE(here < right);
  REQUIRE_THAT(here, WithinAbs(-scales.well_depth, 1e-9));

  const auto pt = rgpot::sutton_chen_platinum();
  const auto au = rgpot::sutton_chen_gold();
  REQUIRE(pt.epsilon == 1.98330e-2);
  REQUIRE(pt.a == 3.92);
  REQUIRE(pt.c == 34.408);
  REQUIRE(pt.n == 10);
  REQUIRE(pt.m == 8);
  REQUIRE(au.epsilon == 1.27930e-2);
  REQUIRE(au.a == 4.08);
  REQUIRE(au.c == 34.408);
  REQUIRE(au.n == 10);
  REQUIRE(au.m == 8);
  const auto pt_scales = rgpot::sutton_chen_scales(pt);
  const auto au_scales = rgpot::sutton_chen_scales(au);
  REQUIRE_THAT(au_scales.nearest_neighbor / pt_scales.nearest_neighbor,
               WithinAbs(au.a / pt.a, 1e-12));

  rgpot_sc_config raw{1.0, 34.408, 1.0, 10, 8, 40.0, 0};
  rgpot_sc_scales exported{};
  REQUIRE(rgpot_sutton_chen_scales(&raw, &exported) == 0);
  REQUIRE_THAT(exported.nearest_neighbor,
               WithinAbs(scales.nearest_neighbor, 0.0));
  REQUIRE_THAT(exported.well_depth, WithinAbs(scales.well_depth, 0.0));
  REQUIRE(rgpot_sutton_chen_scales(nullptr, &exported) == 1);
}

TEST_CASE("Sutton-Chen bulk scales match a direct fcc lattice sum",
          "[SuttonChen]") {
  // c was fitted for zero pressure at the lattice parameter a, so the
  // minimum sits at a / sqrt(2) to the digits c is printed with.
  const rgpot::SuttonChenConfig rows[] = {
      rgpot::SuttonChenConfig{}, rgpot::sutton_chen_platinum(),
      rgpot::sutton_chen_gold()};
  for (const auto &cfg : rows) {
    const auto scales = rgpot::sutton_chen_scales(cfg);
    REQUIRE_THAT(scales.nearest_neighbor / cfg.a,
                 WithinAbs(1.0 / std::sqrt(2.0), 1.0e-5));
    const double here = direct_fcc_energy(cfg, scales.nearest_neighbor);
    REQUIRE_THAT(here / cfg.epsilon,
                 WithinAbs(-scales.well_depth / cfg.epsilon, 1.0e-4));
    for (double step : {1.0e-3, -1.0e-3, 2.0e-2, -2.0e-2}) {
      REQUIRE(direct_fcc_energy(cfg, scales.nearest_neighbor * (1.0 + step)) >
              here);
    }
  }
  const double reduced =
      direct_fcc_energy(rgpot::SuttonChenConfig{}, 1.0 / std::sqrt(2.0));
  REQUIRE_THAT(reduced, WithinAbs(-295.4645, 1.0e-3));
}

TEST_CASE("gold clusters relax from the rescaled platinum geometries",
          "[SuttonChen]") {
  const auto pt = rgpot::sutton_chen_platinum();
  const auto au = rgpot::sutton_chen_gold();
  const char *files[] = {"CppCore/tests/data/sc108/pt38",
                         "CppCore/tests/data/sc108/pt75"};
  for (const char *file : files) {
    double epsilon = 0.0;
    double c = 0.0;
    double a = 0.0;
    int nexp = 0;
    int mexp = 0;
    const AtomMatrix deposited =
        load_points(file, &epsilon, &c, &a, &nexp, &mexp);
    const std::vector<int> types(deposited.rows(), 79);
    rgpot::SuttonChenPot reduced(rgpot::SuttonChenConfig{
        .epsilon = epsilon, .c = c, .a = a, .n = nexp, .m = mexp, .cutoff = 0.0});
    const double e_reduced = std::get<0>(reduced(deposited, types, cube(200.0)));

    // The deposited file is in units of a = 1. Platinum in Angstrom is
    // that file times a_Pt, and the gold copy is that geometry times
    // a_Au / a_Pt.
    const double factor = au.a / pt.a;
    AtomMatrix pos = deposited;
    for (size_t i = 0; i < pos.rows(); ++i) {
      for (size_t k = 0; k < 3; ++k) {
        pos(i, k) *= factor;
      }
    }
    rgpot::SuttonChenConfig cfg = au;
    cfg.cutoff = 0.0;
    rgpot::SuttonChenPot pot(cfg);
    auto [energy, forces, variance] = pot(pos, types, cube(200.0));
    (void)variance;
    const double move_cap = 0.05 * cfg.a;
    double step = 1.0e-3 * cfg.a;
    for (int iter = 0; iter < 2000; ++iter) {
      double mf = 0.0;
      for (size_t i = 0; i < forces.rows(); ++i) {
        for (size_t k = 0; k < 3; ++k) {
          mf = std::max(mf, std::abs(forces(i, k)));
        }
      }
      if (mf < 1.0e-8) {
        break;
      }
      double disp = step;
      if (disp * mf > move_cap) {
        disp = move_cap / mf;
      }
      AtomMatrix trial = pos;
      for (size_t i = 0; i < trial.rows(); ++i) {
        for (size_t k = 0; k < 3; ++k) {
          trial(i, k) += disp * forces(i, k);
        }
      }
      auto [next, next_forces, next_variance] = pot(trial, types, cube(200.0));
      (void)next_variance;
      if (next < energy) {
        pos = trial;
        forces = next_forces;
        energy = next;
        step = disp * 1.2;
      } else {
        step = disp * 0.5;
      }
    }
    REQUIRE_THAT(energy, WithinAbs(au.epsilon * e_reduced, 1e-6));
  }
}

namespace {

void setThreads(const char *value) {
#if defined(_WIN32)
  _putenv_s("OMP_NUM_THREADS", value);
#else
  setenv("OMP_NUM_THREADS", value, 1);
#endif
}

bool same_bits(const std::vector<double> &a, const std::vector<double> &b) {
  return a.size() == b.size() &&
         std::memcmp(a.data(), b.data(), a.size() * sizeof(double)) == 0;
}

/// Energy, forces, stress and the per-atom terms, with the pair terms of
/// the density pass kept for at most `limit` pairs.
struct TermEval {
  double energy = 0.0;
  std::vector<double> forces, stress, atom_energy, atom_virial;
};

TermEval eval_terms(const rgpot::SuttonChenConfig &cfg, rgpot::nlist::Mode mode,
                    const std::size_t *limit, const std::vector<double> &pos,
                    const double *box, int calls) {
  rgpot::SuttonChenPot pot(cfg);
  pot.setNeighbourMode(mode);
  if (limit != nullptr) {
    pot.setPairTermLimit(*limit);
  }
  const std::size_t n = pos.size() / 3;
  TermEval e;
  for (int call = 0; call < calls; ++call) {
    TermEval now;
    now.forces.assign(pos.size(), 0.0);
    now.stress.assign(9, 0.0);
    now.atom_energy.assign(n, 0.0);
    now.atom_virial.assign(6 * n, 0.0);
    rgpot::ForceOut out{};
    out.F = now.forces.data();
    const rgpot::ForceInput in{n, pos.data(), nullptr, box};
    pot.forcePerAtom(in, &out, {now.atom_energy.data(), now.atom_virial.data()});
    now.energy = out.energy;
    std::copy(out.stress, out.stress + 9, now.stress.begin());
    if (call > 0) {
      // The same potential again, on the update and the block it kept.
      REQUIRE(std::memcmp(&now.energy, &e.energy, sizeof(double)) == 0);
      REQUIRE(same_bits(now.forces, e.forces));
    }
    e = std::move(now);
  }
  return e;
}

} // namespace

TEST_CASE("Sutton-Chen force terms read back keep the bits of computing them again",
          "[SuttonChen][terms]") {
  double epsilon = 0.0;
  double c = 0.0;
  double a = 0.0;
  int nexp = 0;
  int mexp = 0;
  const AtomMatrix pt38 = load_points("CppCore/tests/data/sc108/pt38", &epsilon,
                                      &c, &a, &nexp, &mexp);
  std::vector<double> reduced;
  for (size_t i = 0; i < pt38.rows(); ++i) {
    for (size_t k = 0; k < 3; ++k) {
      reduced.push_back(pt38(i, k) *
                        (1.0 + 0.01 * std::cos(0.7 * static_cast<double>(3 * i + k))));
    }
  }
  const double a_pt = rgpot::sutton_chen_platinum().a;
  std::vector<double> scaled = reduced;
  for (double &x : scaled) {
    x *= a_pt;
  }
  const auto sheared = load_lammps_ref("pt_sheared");

  struct Case {
    const char *name;
    rgpot::SuttonChenConfig cfg;
    const std::vector<double> *pos;
    const double *box;
  };
  rgpot::SuttonChenConfig unit_all{.cutoff = 0.0};
  rgpot::SuttonChenConfig unit_cut{.cutoff = 1.3};
  rgpot::SuttonChenConfig pt_all = rgpot::sutton_chen_platinum();
  pt_all.cutoff = 0.0;
  rgpot::SuttonChenConfig pt_cut = rgpot::sutton_chen_platinum();
  pt_cut.cutoff = 1.3 * a_pt;
  rgpot::SuttonChenConfig pt_cell = rgpot::sutton_chen_platinum();
  pt_cell.cutoff = sheared.cutoff;
  pt_cell.periodic = true;
  pt_cell.stress = true;
  const Case cases[] = {
      {"a = 1, no cutoff", unit_all, &reduced, nullptr},
      {"a = 1, cutoff", unit_cut, &reduced, nullptr},
      {"platinum, no cutoff", pt_all, &scaled, nullptr},
      {"platinum, cutoff", pt_cut, &scaled, nullptr},
      {"platinum, sheared cell", pt_cell, &sheared.pos, sheared.box},
  };
  // 0 computes every force term again. 40 keeps some pairs and computes
  // the rest; across three threads each keeps 13.
  const std::size_t none = 0;
  const std::size_t some = 40;
  for (const char *threads : {"1", "3"}) {
    setThreads(threads);
    for (const auto &item : cases) {
      for (const auto mode : {rgpot::nlist::Mode::Auto, rgpot::nlist::Mode::Stored,
                              rgpot::nlist::Mode::Stream}) {
        INFO(item.name << ", threads " << threads << ", mode " << static_cast<int>(mode));
        const TermEval want = eval_terms(item.cfg, mode, &none, *item.pos, item.box, 1);
        for (const std::size_t *limit : {static_cast<const std::size_t *>(nullptr), &some}) {
          const TermEval got = eval_terms(item.cfg, mode, limit, *item.pos, item.box, 2);
          REQUIRE(std::memcmp(&got.energy, &want.energy, sizeof(double)) == 0);
          REQUIRE(same_bits(got.forces, want.forces));
          REQUIRE(same_bits(got.stress, want.stress));
          REQUIRE(same_bits(got.atom_energy, want.atom_energy));
          REQUIRE(same_bits(got.atom_virial, want.atom_virial));
        }
      }
    }
  }
  setThreads("1");
}

TEST_CASE("pair-term limit keeps threaded row walks when three-phase is skipped",
          "[SuttonChen][terms][nlist]") {
  double box[9] = {};
  rgpot::SuttonChenConfig cfg = rgpot::sutton_chen_platinum();
  cfg.cutoff = 1.6 * cfg.a;
  cfg.periodic = true;
  const auto pos = fcc_cell(13, cfg.a, box);
  REQUIRE(pos.size() / 3 >= rgpot::kTeamMinAtoms);
  const auto n = pos.size() / 3;

  rgpot::nlist::Neighbours cells;
  rgpot::nlist::updateCells(cells, pos.data(), n, box, cfg.cutoff, cfg.cutoff / 5.0,
                            rgpot::nlist::Image::Round, rgpot::nlist::Mode::Stored,
                            std::array<bool, 3>{true, true, true});
  const auto &off = cells.offsets();
  const auto &nbr = cells.neighbours();
  constexpr std::size_t kMinParallelHalfPairs = 4096;
  std::size_t total_up = 0;
  for (std::size_t i = 0; i < n; ++i) {
    const auto lo = off[i];
    const auto hi = off[i + 1];
    const auto self = static_cast<std::int32_t>(i);
    const auto mid = static_cast<std::int32_t>(
        std::upper_bound(nbr.begin() + lo, nbr.begin() + hi, self) -
        nbr.begin());
    total_up += static_cast<std::size_t>(hi - mid);
  }
  REQUIRE(total_up > kMinParallelHalfPairs);
  const std::size_t low_limit = 3;
  REQUIRE(low_limit < 4);
  const std::size_t mid_limit =
      kMinParallelHalfPairs + (total_up - kMinParallelHalfPairs) / 2;
  REQUIRE(mid_limit > kMinParallelHalfPairs);
  REQUIRE(mid_limit < total_up);

  auto run_eval = [&](const char *threads, std::size_t limit) {
    setThreads(threads);
    rgpot::SuttonChenPot pot(cfg);
    pot.setNeighbourMode(rgpot::nlist::Mode::Stored);
    pot.setPairTermLimit(limit);
    std::vector<double> forces(pos.size(), 0.0);
    rgpot::ForceOut out{};
    out.F = forces.data();
    // First call builds the stored list. Measure the reused call so any team
    // open comes from the force path itself.
    pot.forceImpl({pos.size() / 3, pos.data(), nullptr, box}, &out);
    std::fill(forces.begin(), forces.end(), 0.0);
    const auto before = rgpot::teamsOpened();
    pot.forceImpl({pos.size() / 3, pos.data(), nullptr, box}, &out);
    return std::tuple<double, std::vector<double>, std::uint64_t>{
        out.energy, std::move(forces), rgpot::teamsOpened() - before};
  };

  auto [ref_low_energy, ref_low_forces, ref_low_teams] = run_eval("1", low_limit);
  auto [ref_mid_energy, ref_mid_forces, ref_mid_teams] = run_eval("1", mid_limit);
  REQUIRE(ref_low_teams == 0);
  REQUIRE(ref_mid_teams == 0);

  auto [low_energy, low_forces, low_teams] = run_eval("4", low_limit);
  REQUIRE(std::memcmp(&low_energy, &ref_low_energy, sizeof(double)) == 0);
  REQUIRE(same_bits(low_forces, ref_low_forces));
  REQUIRE(low_teams > 0);

  auto [mid_energy, mid_forces, mid_teams] = run_eval("4", mid_limit);
  REQUIRE(std::memcmp(&mid_energy, &ref_mid_energy, sizeof(double)) == 0);
  REQUIRE(same_bits(mid_forces, ref_mid_forces));
  REQUIRE(mid_teams > 0);

  setThreads("1");
}

TEST_CASE("pair terms past the room are left for the second pass", "[SuttonChen][terms]") {
  // Pushes `pairs` pairs and reads back what the pass kept.
  auto pass = [](rgpot::PairTerms<2> &terms, int pairs) {
    for (int k = 0; k < pairs; ++k) {
      double *slot = terms.push();
      if (slot != nullptr) {
        slot[0] = 0.5 * k;
        slot[1] = -k;
      }
    }
    const auto kept = static_cast<int>(terms.size());
    terms.replay();
    for (int k = 0; k < kept; ++k) {
      const double *back = terms.next();
      REQUIRE(back != nullptr);
      REQUIRE(back[0] == 0.5 * k);
      REQUIRE(back[1] == -k);
    }
    REQUIRE(terms.next() == nullptr);
    return kept;
  };

  rgpot::PairTerms<2> terms;
  terms.setLimit(3);
  terms.record(10);
  REQUIRE(pass(terms, 5) == 3);

  // A pass allocates nothing, so one with no room keeps no pair. The next
  // record makes room for all that the last pass saw, up to the limit.
  rgpot::PairTerms<2> grown;
  grown.setLimit(1000);
  grown.record(0);
  REQUIRE(pass(grown, 700) == 0);
  grown.record(0);
  REQUIRE(pass(grown, 700) == 700);
  grown.record(0);
  REQUIRE(pass(grown, 900) == 700);
  grown.record(0);
  REQUIRE(pass(grown, 900) == 900);
  grown.record(0);
  REQUIRE(pass(grown, 1200) == 900);
  grown.record(0);
  REQUIRE(pass(grown, 1200) == 1000);

  // A lower limit gives the room back.
  grown.setLimit(2);
  grown.record(0);
  REQUIRE(pass(grown, 5) == 2);

  terms.setLimit(0);
  terms.record(5);
  REQUIRE(terms.push() == nullptr);
  terms.replay();
  REQUIRE(terms.next() == nullptr);
  terms.release();
  REQUIRE(terms.size() == 0);
}

namespace {

struct EntryResult {
  int status = 1;
  double energy = 0.0;
  std::vector<double> forces;
};

EntryResult call_entry(const rgpot_sc_config &cfg, const std::vector<double> &xyz,
                       const double *box) {
  EntryResult r;
  r.forces.assign(xyz.size(), 0.0);
  r.status = rgpot_sutton_chen_energy_forces(&cfg, static_cast<int32_t>(xyz.size() / 3),
                                             xyz.data(), box, &r.energy, r.forces.data());
  return r;
}

/// A potential built for this call alone, as the C entry once did.
EntryResult call_fresh(const rgpot_sc_config &raw, const std::vector<double> &xyz,
                       const double *box) {
  rgpot::SuttonChenConfig cfg;
  cfg.epsilon = raw.epsilon;
  cfg.c = raw.c;
  cfg.a = raw.a;
  cfg.n = raw.n;
  cfg.m = raw.m;
  cfg.cutoff = raw.cutoff;
  cfg.periodic = raw.periodic != 0;
  const rgpot::SuttonChenPot pot(cfg);
  EntryResult r;
  r.forces.assign(xyz.size(), 0.0);
  rgpot::ForceOut out{};
  out.F = r.forces.data();
  pot.forceImpl({xyz.size() / 3, xyz.data(), nullptr, box}, &out);
  r.status = 0;
  r.energy = out.energy;
  return r;
}

bool same_result(const EntryResult &a, const EntryResult &b) {
  return a.status == b.status && std::memcmp(&a.energy, &b.energy, sizeof(double)) == 0 &&
         same_bits(a.forces, b.forces);
}

} // namespace

TEST_CASE("the C entry point keeps its potential until the parameters or atoms change",
          "[SuttonChen][kept]") {
  using Kept = rgpot::Kept<rgpot::SuttonChenPot>;
  Kept::release();
  double epsilon = 0.0;
  double c = 0.0;
  double a = 0.0;
  int nexp = 0;
  int mexp = 0;
  const AtomMatrix pt38 = load_points("CppCore/tests/data/sc108/pt38", &epsilon,
                                      &c, &a, &nexp, &mexp);
  std::vector<double> pos;
  std::vector<double> moved;
  std::vector<double> spread;
  for (size_t i = 0; i < pt38.rows(); ++i) {
    for (size_t k = 0; k < 3; ++k) {
      const double phase = 1.3 * static_cast<double>(3 * i + k);
      pos.push_back(pt38(i, k) * (1.0 + 0.01 * std::sin(phase)));
      moved.push_back(pt38(i, k) * (1.0 + 0.01 * std::cos(phase)));
      // Past the skin of a 1.3 cutoff, so a kept list is built again.
      spread.push_back(1.1 * pt38(i, k));
    }
  }
  const std::vector<double> fewer(pos.begin(), pos.begin() + 3 * 13);

  std::uint64_t builds = Kept::counts().builds;
  std::uint64_t reuses = Kept::counts().reuses;
  // Each call has the bits of a potential built for it alone, and either
  // builds the kept potential or reuses it.
  auto check = [&](const rgpot_sc_config &cfg, const std::vector<double> &xyz,
                   const double *box, bool build) {
    REQUIRE(same_result(call_entry(cfg, xyz, box), call_fresh(cfg, xyz, box)));
    const rgpot::KeptCounts now = Kept::counts();
    REQUIRE(now.builds == builds + (build ? 1 : 0));
    REQUIRE(now.reuses == reuses + (build ? 0 : 1));
    builds = now.builds;
    reuses = now.reuses;
  };

  const rgpot_sc_config all{1.0, 34.408, 1.0, 10, 8, 0.0, 0};
  check(all, pos, nullptr, true);
  check(all, moved, nullptr, false);
  check(all, pos, nullptr, false);
  rgpot_sc_config deeper = all;
  deeper.epsilon = 2.0;
  check(deeper, pos, nullptr, true);
  // One potential is kept, so going back builds again.
  check(all, pos, nullptr, true);
  check(all, fewer, nullptr, true);
  check(all, pos, nullptr, true);
  rgpot_sc_config nine_six = all;
  nine_six.n = 9;
  nine_six.m = 6;
  check(nine_six, pos, nullptr, true);
  // The key compares bits, so -0.0 is another cutoff with the same result.
  rgpot_sc_config negative = all;
  negative.cutoff = -0.0;
  check(negative, pos, nullptr, true);

  rgpot_sc_config cut = all;
  cut.cutoff = 1.3;
  check(cut, pos, nullptr, true);
  check(cut, moved, nullptr, false);
  check(cut, spread, nullptr, false);
  check(cut, pos, nullptr, false);

  const double box[9] = {20, 0, 0, 0, 20, 0, 0, 0, 20};
  const double wider[9] = {24, 0, 0, 0, 22, 0, 0, 0, 20};
  rgpot_sc_config cell = cut;
  cell.periodic = 1;
  check(cell, pos, box, true);
  check(cell, moved, box, false);
  // A new cell keeps the potential, whose neighbour update follows it.
  check(cell, pos, wider, false);

  // A call that fails drops the potential it ran on.
  double energy = 0.0;
  std::vector<double> forces(pos.size(), 0.0);
  REQUIRE(rgpot_sutton_chen_energy_forces(&cell, 38, pos.data(), nullptr, &energy,
                                          forces.data()) == 1);
  REQUIRE(Kept::counts().reuses == reuses + 1);
  reuses += 1;
  check(cell, pos, box, true);
  // Parameters the constructor rejects build nothing and keep nothing.
  rgpot_sc_config flat = all;
  flat.n = 0;
  REQUIRE(rgpot_sutton_chen_energy_forces(&flat, 38, pos.data(), nullptr, &energy,
                                          forces.data()) == 1);
  REQUIRE(Kept::counts().builds == builds);
  check(cell, pos, box, true);

  // Another thread keeps its own potential.
  EntryResult there;
  EntryResult again;
  rgpot::KeptCounts counts;
  std::thread worker([&] {
    there = call_entry(all, pos, nullptr);
    again = call_entry(all, moved, nullptr);
    counts = Kept::counts();
  });
  worker.join();
  REQUIRE(same_result(there, call_fresh(all, pos, nullptr)));
  REQUIRE(same_result(again, call_fresh(all, moved, nullptr)));
  REQUIRE(counts.builds == 1);
  REQUIRE(counts.reuses == 1);
  check(cell, pos, box, false);
}

TEST_CASE("Sutton-Chen with no cutoff keeps the one-bin bits on any thread count",
          "[SuttonChen][nlist]") {
  // 38 atoms stay on the calling thread whatever OMP_NUM_THREADS asks for,
  // and the unbounded walk takes the pairs in the order of the one bin that
  // a search wider than the cluster gives.
  double epsilon = 0.0;
  double c = 0.0;
  double a = 0.0;
  int nexp = 0;
  int mexp = 0;
  const AtomMatrix pt38 = load_points("CppCore/tests/data/sc108/pt38", &epsilon,
                                      &c, &a, &nexp, &mexp);
  std::vector<double> pos;
  for (size_t i = 0; i < pt38.rows(); ++i) {
    for (size_t k = 0; k < 3; ++k) {
      pos.push_back(pt38(i, k) *
                    (1.0 + 0.01 * std::sin(0.9 * static_cast<double>(3 * i + k))));
    }
  }
  std::vector<double> scaled = pos;
  for (double &x : scaled) {
    x *= rgpot::sutton_chen_platinum().a;
  }
  const rgpot::SuttonChenConfig rows[] = {rgpot::SuttonChenConfig{},
                                          rgpot::sutton_chen_platinum()};
  for (const auto &row : rows) {
    const std::vector<double> &xyz = row.a == 1.0 ? pos : scaled;
    rgpot::SuttonChenConfig none = row;
    none.cutoff = 0.0;
    rgpot::SuttonChenConfig wide = row;
    wide.cutoff = 1.0e6;
    setThreads("1");
    const StressEval want = eval_stress(rgpot::SuttonChenPot(wide), xyz, nullptr);
    for (const char *threads : {"2", "3", "5"}) {
      INFO("a " << row.a << ", threads " << threads);
      setThreads(threads);
      const StressEval one_bin = eval_stress(rgpot::SuttonChenPot(wide), xyz, nullptr);
      const StressEval got = eval_stress(rgpot::SuttonChenPot(none), xyz, nullptr);
      REQUIRE(std::memcmp(&one_bin.energy, &want.energy, sizeof(double)) == 0);
      REQUIRE(same_bits(one_bin.forces, want.forces));
      REQUIRE(same_bits(got.forces, want.forces));
      REQUIRE(std::memcmp(&got.energy, &want.energy, sizeof(double)) == 0);
    }
  }
  setThreads("1");
}

TEST_CASE("Sutton-Chen stress is opt-in and keeps the energy and force bits",
          "[SuttonChen][stress]") {
  REQUIRE_FALSE(rgpot::SuttonChenConfig{}.stress);
  const auto ref = load_lammps_ref("pt_sheared");
  const auto plain = platinum_periodic(ref.cutoff, false);
  const auto with = platinum_periodic(ref.cutoff, true);
  REQUIRE_FALSE(plain.caps().stress);
  REQUIRE(with.caps().stress);
  REQUIRE(plain.paramsKey() != with.paramsKey());
  const StressEval a = eval_stress(plain, ref.pos, ref.box);
  const StressEval b = eval_stress(with, ref.pos, ref.box);
  REQUIRE(a.has_stress == 0);
  REQUIRE(
      std::all_of(a.stress, a.stress + 9, [](double v) { return v == 7.0; }));
  REQUIRE(b.has_stress == 1);
  REQUIRE(std::memcmp(&a.energy, &b.energy, sizeof(double)) == 0);
  REQUIRE(std::memcmp(a.forces.data(), b.forces.data(),
                      a.forces.size() * sizeof(double)) == 0);
  // A cluster with no cell has no volume, so no stress is reported.
  const rgpot::SuttonChenPot cluster({.cutoff = 0.0, .stress = true});
  const StressEval c = eval_stress(cluster, {0, 0, 0, 0.8, 0.1, 0}, nullptr);
  REQUIRE(c.has_stress == 0);
  REQUIRE(
      std::all_of(c.stress, c.stress + 9, [](double v) { return v == 0.0; }));
}

TEST_CASE("Sutton-Chen stress is the strain derivative of the energy",
          "[SuttonChen][stress]") {
  const auto ref = load_lammps_ref("pt_sheared");
  const auto pot = platinum_periodic(ref.cutoff, true);
  const StressEval here = eval_stress(pot, ref.pos, ref.box);
  const double h = 1.0e-6;
  for (const auto &ab : kVoigt) {
    const int a = ab[0];
    const int b = ab[1];
    double energy[2] = {};
    for (int side = 0; side < 2; ++side) {
      // Rows of the positions and the cell strain as r (I + eps), with eps
      // symmetric and eps_ab + eps_ba = 2 h or -2 h on the diagonal.
      const double step = (side == 0 ? h : -h) * (a == b ? 1.0 : 0.5);
      std::vector<double> rows(ref.pos);
      rows.insert(rows.end(), ref.box, ref.box + 9);
      for (std::size_t r = 0; r < rows.size(); r += 3) {
        const double ra = rows[r + a];
        const double rb = rows[r + b];
        rows[r + b] += step * ra;
        if (a != b)
          rows[r + a] += step * rb;
      }
      const std::vector<double> pos(rows.begin(), rows.end() - 9);
      energy[side] = eval_stress(pot, pos, rows.data() + pos.size()).energy;
    }
    const double numeric =
        (energy[0] - energy[1]) / (2.0 * h * rgpot::cellVolume(ref.box));
    INFO("component " << a << b);
    REQUIRE_THAT(here.stress[3 * a + b], WithinAbs(numeric, 1.0e-9));
    REQUIRE(here.stress[3 * b + a] == here.stress[3 * a + b]);
  }
}

TEST_CASE("Sutton-Chen stress matches LAMMPS eam/fs", "[SuttonChen][stress]") {
  // Tables of 12501 and 50001 points give the same LAMMPS energy to
  // 2e-12 eV, so these tolerances are not set by the table.
  for (const char *name : {"pt_cubic", "pt_sheared"}) {
    const auto ref = load_lammps_ref(name);
    REQUIRE(ref.pos.size() == 3 * 108);
    const StressEval e =
        eval_stress(platinum_periodic(ref.cutoff, true), ref.pos, ref.box);
    INFO(name);
    REQUIRE(e.has_stress == 1);
    REQUIRE_THAT(e.energy, WithinAbs(ref.energy, 1.0e-9));
    for (std::size_t i = 0; i < e.forces.size(); ++i) {
      REQUIRE_THAT(e.forces[i], WithinAbs(ref.forces[i], 1.0e-8));
    }
    for (int v = 0; v < 6; ++v) {
      REQUIRE_THAT(e.stress[3 * kVoigt[v][0] + kVoigt[v][1]],
                   WithinAbs(ref.stress[v], 1.0e-11));
    }
  }
}

TEST_CASE("Sutton-Chen pressure vanishes at the fitted lattice parameter",
          "[SuttonChen][stress]") {
  // Sutton and Chen's c = 34.408 is the zero-pressure value 34.408374
  // rounded to three decimals, which leaves about 0.0015 GPa in the
  // infinite fcc crystal at a.
  // A truncated sum adds a tail that shrinks as the cutoff grows.
  const double a = rgpot::sutton_chen_platinum().a;
  double box[9];
  double last = 1.0;
  for (int cells : {4, 6, 8}) {
    const StressEval e = eval_stress(platinum_periodic(0.49 * cells * a, true),
                                     fcc_cell(cells, a, box), box);
    INFO("cells " << cells);
    for (int k = 0; k < 9; ++k) {
      REQUIRE_THAT(e.stress[k],
                   WithinAbs(k % 4 == 0 ? e.stress[0] : 0.0, 1e-12));
    }
    REQUIRE(std::abs(e.stress[0]) < last);
    last = std::abs(e.stress[0]);
  }
  const double squeezed = eval_stress(platinum_periodic(0.49 * 8 * a, true),
                                      fcc_cell(8, 0.99 * a, box), box)
                              .stress[0];
  REQUIRE(squeezed < 0.0);
  REQUIRE(last < 1.0e-3 * std::abs(squeezed));
}

namespace {

struct ScThreadGuard {
  std::string prev;
  bool had = false;

  ScThreadGuard() {
    if (const char *v = std::getenv("OMP_NUM_THREADS")) {
      had = true;
      prev = v;
    }
  }

  ~ScThreadGuard() {
#if defined(_WIN32)
    _putenv_s("OMP_NUM_THREADS", had ? prev.c_str() : "");
#else
    if (had) {
      setenv("OMP_NUM_THREADS", prev.c_str(), 1);
    } else {
      unsetenv("OMP_NUM_THREADS");
    }
#endif
  }
};

struct FullScEval {
  double energy = 0.0;
  std::vector<double> forces;
  std::vector<double> stress;
  int has_stress = 0;
  double pa_energy = 0.0;
  std::vector<double> pa_forces;
  std::vector<double> pa_stress;
  std::vector<double> atom_energy;
  std::vector<double> atom_virial;
};

FullScEval run_full_sc_eval(const rgpot::SuttonChenPot &pot,
                            const std::vector<double> &pos, const double *box) {
  const std::size_t n = pos.size() / 3;
  FullScEval out;
  out.forces.assign(3 * n, 0.0);
  out.stress.assign(9, 0.0);
  rgpot::ForceOut fo{};
  fo.F = out.forces.data();
  const rgpot::ForceInput in{n, pos.data(), nullptr, box};
  pot.forceImpl(in, &fo);
  out.energy = fo.energy;
  std::copy(fo.stress, fo.stress + 9, out.stress.begin());
  out.has_stress = fo.has_stress;

  out.pa_forces.assign(3 * n, 0.0);
  out.pa_stress.assign(9, 0.0);
  out.atom_energy.assign(n, 0.0);
  out.atom_virial.assign(6 * n, 0.0);
  rgpot::ForceOut pao{};
  pao.F = out.pa_forces.data();
  pot.forcePerAtom(in, &pao, {out.atom_energy.data(), out.atom_virial.data()});
  out.pa_energy = pao.energy;
  std::copy(pao.stress, pao.stress + 9, out.pa_stress.begin());
  return out;
}

std::size_t count_sc_bit_diffs(const FullScEval &a, const FullScEval &b) {
  std::size_t diffs = 0;
  auto cmp = [&diffs](double x, double y) {
    if (std::memcmp(&x, &y, sizeof(double)) != 0) {
      ++diffs;
    }
  };
  cmp(a.energy, b.energy);
  cmp(a.pa_energy, b.pa_energy);
  if (a.has_stress != b.has_stress) {
    ++diffs;
  }
  auto cmp_vec = [&](const std::vector<double> &va, const std::vector<double> &vb) {
    if (va.size() != vb.size()) {
      diffs += std::max(va.size(), vb.size());
      return;
    }
    for (std::size_t i = 0; i < va.size(); ++i) {
      cmp(va[i], vb[i]);
    }
  };
  cmp_vec(a.forces, b.forces);
  cmp_vec(a.stress, b.stress);
  cmp_vec(a.pa_forces, b.pa_forces);
  cmp_vec(a.pa_stress, b.pa_stress);
  cmp_vec(a.atom_energy, b.atom_energy);
  cmp_vec(a.atom_virial, b.atom_virial);
  return diffs;
}

std::vector<double> jitter_positions(std::vector<double> pos, double amp) {
  for (std::size_t i = 0; i < pos.size(); ++i) {
    pos[i] += amp * std::sin(1.7 * static_cast<double>(i) + 0.3);
  }
  return pos;
}

} // namespace

TEST_CASE("Sutton-Chen matches one-thread bits at 1, 2, 4 and 8 threads across Verlet reuse and rebuild",
          "[SuttonChen][threads]") {
  ScThreadGuard guard;

  // Below kTeamMinAtoms: 6x6x6 fcc = 864 atoms stays on one thread across
  // 1, 2, 4 and 8 threads and reuses the Verlet list inside half the skin.
  {
    double box[9] = {};
    rgpot::SuttonChenConfig cfg = rgpot::sutton_chen_platinum();
    cfg.cutoff = 5.5;
    cfg.periodic = true;
    cfg.stress = true;
    const auto base = fcc_cell(6, cfg.a, box);
    const auto step0 = jitter_positions(base, 0.04);
    const auto step1 = jitter_positions(step0, 0.08);
    auto step2 = step1;
    step2[0] += 0.70;
    REQUIRE(step0.size() / 3 < rgpot::kTeamMinAtoms);

    setThreads("1");
    rgpot::SuttonChenPot ref_pot(cfg);
    ref_pot.setNeighbourMode(rgpot::nlist::Mode::Stored);
    const auto ref0 = run_full_sc_eval(ref_pot, step0, box);
    const auto ref1 = run_full_sc_eval(ref_pot, step1, box);
    REQUIRE(ref_pot.listReused());
    std::vector<double> fbuf(step2.size(), 0.0);
    rgpot::ForceOut probe{};
    probe.F = fbuf.data();
    ref_pot.forceImpl({step2.size() / 3, step2.data(), nullptr, box}, &probe);
    REQUIRE_FALSE(ref_pot.listReused());
    const auto ref2 = run_full_sc_eval(ref_pot, step2, box);
    REQUIRE(ref_pot.listReused());

    for (const char *th : {"1", "2", "4", "8"}) {
      INFO("below kTeamMinAtoms OMP_NUM_THREADS=" << th);
      setThreads(th);
      const auto before = rgpot::teamsOpened();
      rgpot::SuttonChenPot pot(cfg);
      pot.setNeighbourMode(rgpot::nlist::Mode::Stored);
      CHECK(count_sc_bit_diffs(run_full_sc_eval(pot, step0, box), ref0) == 0);
      CHECK(count_sc_bit_diffs(run_full_sc_eval(pot, step1, box), ref1) == 0);
      CHECK(pot.listReused());
      pot.forceImpl({step2.size() / 3, step2.data(), nullptr, box}, &probe);
      CHECK_FALSE(pot.listReused());
      CHECK(count_sc_bit_diffs(run_full_sc_eval(pot, step2, box), ref2) == 0);
      CHECK(pot.listReused());
      CHECK(rgpot::teamsOpened() == before);
    }
  }

  // Above kTeamMinAtoms: 13x13x13 fcc = 8788 atoms >= 8192 for both unit
  // lattice parameter (a = 1) and platinum (a = 3.92), orthogonal and sheared.
  for (const bool unit_a : {true, false}) {
    for (const bool sheared : {false, true}) {
      double box[9] = {};
      rgpot::SuttonChenConfig cfg =
          unit_a ? rgpot::SuttonChenConfig{.cutoff = 1.6, .periodic = true, .stress = true}
                 : rgpot::sutton_chen_platinum();
      if (!unit_a) {
        cfg.cutoff = 1.6 * cfg.a;
        cfg.periodic = true;
        cfg.stress = true;
      }
      const auto base = fcc_cell(13, cfg.a, box);
      if (sheared) {
        box[3] = 0.08 * box[0];
        box[6] = 0.04 * box[0];
        box[7] = 0.06 * box[4];
      }
      const std::size_t n = base.size() / 3;
      REQUIRE(n >= rgpot::kTeamMinAtoms);

      // Skin is cutoff / 5; half-skin is cutoff / 10.
      const double half_skin = cfg.cutoff / 10.0;
      const auto step0 = jitter_positions(base, 0.01 * cfg.a);
      const auto within_skin = jitter_positions(step0, 0.15 * half_skin);
      auto past_skin = within_skin;
      past_skin[0] += 1.5 * half_skin;

      for (const auto mode : {rgpot::nlist::Mode::Stored, rgpot::nlist::Mode::Auto,
                              rgpot::nlist::Mode::Stream}) {
        if (sheared && mode != rgpot::nlist::Mode::Stored) {
          continue;
        }
        INFO("unit_a=" << unit_a << " sheared=" << sheared
                       << " mode=" << static_cast<int>(mode));
        setThreads("1");
        rgpot::SuttonChenPot ref_pot(cfg);
        ref_pot.setNeighbourMode(mode);
        const auto ref0 = run_full_sc_eval(ref_pot, step0, box);
        const auto ref1 = run_full_sc_eval(ref_pot, within_skin, box);
        if (mode != rgpot::nlist::Mode::Stream) {
          REQUIRE(ref_pot.listReused());
        }
        std::vector<double> fbuf(past_skin.size(), 0.0);
        rgpot::ForceOut probe{};
        probe.F = fbuf.data();
        ref_pot.forceImpl({n, past_skin.data(), nullptr, box}, &probe);
        if (mode != rgpot::nlist::Mode::Stream) {
          REQUIRE_FALSE(ref_pot.listReused());
        }
        const auto ref2 = run_full_sc_eval(ref_pot, past_skin, box);
        if (mode != rgpot::nlist::Mode::Stream) {
          REQUIRE(ref_pot.listReused());
        }

        const bool full_matrix = mode != rgpot::nlist::Mode::Stream && !sheared;
        const std::vector<const char *> thread_list =
            full_matrix ? std::vector<const char *>{"1", "2", "4", "8"}
                        : std::vector<const char *>{"1", "8"};
        for (const char *th : thread_list) {
          INFO("OMP_NUM_THREADS=" << th);
          setThreads(th);
          rgpot::SuttonChenPot pot(cfg);
          pot.setNeighbourMode(mode);
          CHECK(count_sc_bit_diffs(run_full_sc_eval(pot, step0, box), ref0) == 0);

          const auto before_reuse = rgpot::teamsOpened();
          CHECK(count_sc_bit_diffs(run_full_sc_eval(pot, within_skin, box), ref1) == 0);
          if (mode != rgpot::nlist::Mode::Stream) {
            CHECK(pot.listReused());
          }
#if defined(_OPENMP)
          if (std::string(th) != "1") {
            CHECK(rgpot::teamsOpened() > before_reuse);
          } else {
            CHECK(rgpot::teamsOpened() == before_reuse);
          }
#endif

          pot.forceImpl({n, past_skin.data(), nullptr, box}, &probe);
          if (mode != rgpot::nlist::Mode::Stream) {
            CHECK_FALSE(pot.listReused());
          }
          CHECK(count_sc_bit_diffs(run_full_sc_eval(pot, past_skin, box), ref2) == 0);
          if (mode != rgpot::nlist::Mode::Stream) {
            CHECK(pot.listReused());
          }
        }
      }
    }
  }

  // Dilute stored system above kTeamMinAtoms (< 4096 upper pairs). The
  // force kernel stays on the calling thread, as Morse does for a stored
  // list below kMinParallelHalfPairs. The shared gather may still open
  // one region: buildStored threads when the 27-cell stencil bound is at
  // least 4096 and greater than the atom count. Cutoff 5.5 with a skin of
  // one fifth searches to 6.6, and atoms 12 apart fall in that stencil
  // even though they fail the cutoff, so the bound exceeds the atom count.
  // Morse's dilute cutoff of 5.3 does not. A reused call builds no list,
  // so a team there would be the force kernel.
  {
    const std::size_t n = rgpot::kTeamMinAtoms + 16;
    const double spacing = 12.0;
    const int grid = 22;
    const double side = static_cast<double>(grid) * spacing;
    const double box[9] = {side, 0.0, 0.0, 0.0, side, 0.0, 0.0, 0.0, side};
    std::vector<double> pos;
    pos.reserve(3 * n);
    for (std::size_t a = 0; a < n; ++a) {
      const std::size_t gx = a % grid;
      const std::size_t gy = (a / grid) % grid;
      const std::size_t gz = a / (static_cast<std::size_t>(grid) * static_cast<std::size_t>(grid));
      pos.push_back((static_cast<double>(gx) + 0.5) * spacing);
      pos.push_back((static_cast<double>(gy) + 0.5) * spacing);
      pos.push_back((static_cast<double>(gz) + 0.5) * spacing);
    }
    for (std::size_t p = 0; p < 10; ++p) {
      pos[3 * (2 * p + 1)] = pos[3 * (2 * p)] + 2.6;
      pos[3 * (2 * p + 1) + 1] = pos[3 * (2 * p) + 1];
      pos[3 * (2 * p + 1) + 2] = pos[3 * (2 * p) + 2];
    }
    rgpot::SuttonChenConfig cfg = rgpot::sutton_chen_platinum();
    cfg.cutoff = 5.5;
    cfg.periodic = true;
    cfg.stress = true;

    setThreads("1");
    rgpot::SuttonChenPot ref_pot(cfg);
    ref_pot.setNeighbourMode(rgpot::nlist::Mode::Stored);
    const auto ref = run_full_sc_eval(ref_pot, pos, box);

    for (const char *th : {"2", "4", "8"}) {
      setThreads(th);
      rgpot::SuttonChenPot pot(cfg);
      pot.setNeighbourMode(rgpot::nlist::Mode::Stored);
      const auto before = rgpot::teamsOpened();
      CHECK(count_sc_bit_diffs(run_full_sc_eval(pot, pos, box), ref) == 0);
      const auto built = rgpot::teamsOpened();
      // At most the stored-list gather. The force walk itself is serial.
      CHECK(built - before <= 1);
      CHECK(count_sc_bit_diffs(run_full_sc_eval(pot, pos, box), ref) == 0);
      CHECK(pot.listReused());
      CHECK(rgpot::teamsOpened() == built);
    }
  }
}
