// MIT License
// Copyright 2023--present rgpot developers
//
// Morse pot checks: the analytic two-atom limit, a 13-atom regression pin,
// and the parameter fingerprint.

#include <algorithm>
#include <array>
#include <cmath>
#include <cstdint>
#include <cstdlib>
#include <cstring>
#include <limits>
#include <string>
#include <vector>

#include <catch2/catch_all.hpp>

#include "rgpot/ForceStructs.hpp"
#include "rgpot/Morse/MorsePot.hpp"
#include "rgpot/PerAtom.hpp"
#include "rgpot/Slices.hpp"
#include "rgpot/types/AtomMatrix.hpp"

using Catch::Matchers::WithinAbs;
using Catch::Matchers::WithinRel;
using rgpot::types::AtomMatrix;

namespace {

/// eOn's neb_morse reactant geometry (13 hydrogens, client/unit_tests/data/
/// systems/neb_morse/reactant.con), inlined because rgpot reads no .con.
AtomMatrix nebMorseCluster() {
  return AtomMatrix{
      {50.594227, 52.017165, 52.277482}, {51.578540, 52.642148, 51.195222},
      {51.243758, 52.919086, 52.218383}, {50.490127, 52.736378, 51.417663},
      {51.522643, 50.884535, 51.656125}, {50.927263, 51.743087, 51.272643},
      {51.240676, 50.960436, 50.573042}, {51.275727, 52.061877, 50.284160},
      {50.434110, 50.976423, 51.879062}, {51.695007, 51.919472, 52.038680},
      {51.363596, 52.199101, 53.064923}, {52.017483, 51.639074, 51.008389},
      {51.187843, 51.165217, 52.678225}};
}

/// The cell that ships with the geometry above.
std::array<std::array<double, 3>, 3> nebMorseCell() {
  return {
      {{101.942400, 0.0, 0.0}, {0.0, 103.142600, 0.0}, {0.0, 0.0, 102.605500}}};
}

double maxForceNorm(const AtomMatrix &forces) {
  double worst = 0.0;
  for (size_t i = 0; i < forces.rows(); ++i) {
    const double n2 = forces(i, 0) * forces(i, 0) +
                      forces(i, 1) * forces(i, 1) + forces(i, 2) * forces(i, 2);
    worst = std::max(worst, std::sqrt(n2));
  }
  return worst;
}

double columnSum(const AtomMatrix &forces, size_t col) {
  double acc = 0.0;
  for (size_t i = 0; i < forces.rows(); ++i) {
    acc += forces(i, col);
  }
  return acc;
}

} // namespace

TEST_CASE("MorsePot two-atom well depth", "[MorsePot]") {
  rgpot::MorsePot pot;
  const auto cfg = pot.config();

  // Cutoff shift: the pair term at the minimum is -De, shifted by the
  // unshifted energy at the cutoff. Two terms near De cancel to 3.6e-5, so
  // the comparison is absolute at the operands' rounding scale: a relative
  // pin on the cancelled result asks for agreement below one ulp of De, and
  // exp() and multiply-add contraction differ by that much between an x86
  // and an aarch64 build of the same expression.
  const double d = 1.0 - std::exp(-cfg.a * (cfg.cutoff - cfg.re));
  const double shift = cfg.De * d * d - cfg.De;
  REQUIRE_THAT(pot.energyShift(),
               WithinAbs(shift, 8 * std::numeric_limits<double>::epsilon()));
  REQUIRE_THAT(pot.energyShift(), WithinAbs(-3.5537997279178057e-05, 1e-14));

  const AtomMatrix positions{{10.0, 10.0, 10.0}, {10.0 + cfg.re, 10.0, 10.0}};
  const std::vector<int> atmtypes{78, 78};
  // Cutoff 9.5 A needs a cell wider than 19 A for the minimum image
  // convention to leave a single pair.
  const std::array<std::array<double, 3>, 3> box{
      {{40.0, 0.0, 0.0}, {0.0, 40.0, 0.0}, {0.0, 0.0, 40.0}}};

  auto [energy, forces, variance] = pot(positions, atmtypes, box);
  (void)variance;

  REQUIRE_THAT(energy, WithinRel(-cfg.De - shift, 1e-12));
  REQUIRE_THAT(energy, WithinAbs(-0.71016446200272088, 1e-12));
  // r == re is the minimum of the pair term: the force vanishes there.
  for (size_t i = 0; i < forces.rows(); ++i) {
    for (size_t j = 0; j < forces.cols(); ++j) {
      REQUIRE_THAT(forces(i, j), WithinAbs(0.0, 1e-12));
    }
  }
}

TEST_CASE("MorsePot energy and forces on the neb_morse cluster", "[MorsePot]") {
  // Reference produced by an independent double-precision evaluation of
  // this kernel (eOn pins only finiteness for Morse on this geometry).
  // Platinum parameters on hydrogens: deterministic, not physical.
  rgpot::MorsePot pot;
  const auto positions = nebMorseCluster();
  const std::vector<int> atmtypes(13, 1);

  auto [energy, forces, variance] = pot(positions, atmtypes, nebMorseCell());
  (void)variance;

  REQUIRE_THAT(energy, WithinRel(7483.1166203812, 1e-9));

  const std::array<double, 3> expected_row0{-2056.95286055812, 333.678513150438,
                                            1022.83140169994};
  for (size_t j = 0; j < 3; ++j) {
    REQUIRE_THAT(forces(0, j), WithinRel(expected_row0[j], 1e-9));
  }
  REQUIRE_THAT(maxForceNorm(forces), WithinRel(2480.28362300107, 1e-9));

  // Pair forces are equal and opposite, so the net force vanishes.
  for (size_t j = 0; j < 3; ++j) {
    REQUIRE_THAT(columnSum(forces, j), WithinAbs(0.0, 1e-9));
  }
}

TEST_CASE("MorsePot parameter fingerprint tracks the config", "[MorsePot]") {
  const rgpot::MorsePot defaults;
  const rgpot::MorsePot deeper{rgpot::MorseConfig{.De = 0.8}};
  const rgpot::MorsePot sameAsDefaults{rgpot::MorseConfig{}};

  REQUIRE(defaults.paramsKey() == sameAsDefaults.paramsKey());
  REQUIRE(defaults.paramsKey() != deeper.paramsKey());
  REQUIRE(defaults.getType() == rgpot::PotType::Morse);
}

namespace {

struct ThreadGuard {
  std::string saved;
  bool had = false;

  ThreadGuard() {
    if (const char *v = std::getenv("OMP_NUM_THREADS")) {
      saved = v;
      had = true;
    }
  }

  ~ThreadGuard() { set(had ? saved.c_str() : nullptr); }

  static void set(const char *v) {
#if defined(_WIN32)
    _putenv_s("OMP_NUM_THREADS", v == nullptr ? "" : v);
#else
    if (v == nullptr || v[0] == '\0') {
      unsetenv("OMP_NUM_THREADS");
    } else {
      setenv("OMP_NUM_THREADS", v, 1);
    }
#endif
  }
};

std::uint64_t f64Bits(double v) {
  std::uint64_t out = 0;
  std::memcpy(&out, &v, sizeof(out));
  return out;
}

struct VacancyLattice {
  std::vector<double> pos;
  std::vector<int> z;
  std::array<double, 9> box{};
};

/// Jittered FCC supercell with one vacancy at site 0 (Makri/Ortner/Kermode
/// Cu vacancy geometry when side = 6, lattice = 2.55 * sqrt(2)).
VacancyLattice fccVacancy(int side, double lattice, int z_num, bool sheared = false) {
  static constexpr double kBasis[4][3] = {
      {0.0, 0.0, 0.0}, {0.0, 0.5, 0.5}, {0.5, 0.0, 0.5}, {0.5, 0.5, 0.0}};
  const double L = static_cast<double>(side) * lattice;
  VacancyLattice out;
  if (sheared) {
    out.box = {L, 0.0, 0.0, 0.06 * L, L, 0.0, 0.03 * L, -0.04 * L, L};
  } else {
    out.box = {L, 0.0, 0.0, 0.0, L, 0.0, 0.0, 0.0, L};
  }
  bool dropped_first = false;
  std::size_t idx = 0;
  for (int ix = 0; ix < side; ++ix) {
    for (int iy = 0; iy < side; ++iy) {
      for (int iz = 0; iz < side; ++iz) {
        for (const auto &b : kBasis) {
          if (!dropped_first) {
            dropped_first = true;
            continue;
          }
          const double fx = (static_cast<double>(ix) + b[0]) / static_cast<double>(side);
          const double fy = (static_cast<double>(iy) + b[1]) / static_cast<double>(side);
          const double fz = (static_cast<double>(iz) + b[2]) / static_cast<double>(side);
          const double jx = 0.015 * lattice * std::sin(1.7 * static_cast<double>(idx) + 0.3);
          const double jy = 0.015 * lattice * std::cos(2.1 * static_cast<double>(idx) + 0.5);
          const double jz = 0.015 * lattice * std::sin(1.3 * static_cast<double>(idx) + 1.1);
          out.pos.push_back(fx * out.box[0] + fy * out.box[3] + fz * out.box[6] + jx);
          out.pos.push_back(fx * out.box[1] + fy * out.box[4] + fz * out.box[7] + jy);
          out.pos.push_back(fx * out.box[2] + fy * out.box[5] + fz * out.box[8] + jz);
          out.z.push_back(z_num);
          ++idx;
        }
      }
    }
  }
  return out;
}

std::vector<double> perturbed(const std::vector<double> &pos, double step) {
  std::vector<double> out = pos;
  for (std::size_t k = 0; k < out.size(); ++k) {
    out[k] += step * std::sin(0.73 * static_cast<double>(k) + 0.4);
  }
  return out;
}

struct FullEval {
  double energy = 0.0;
  std::vector<double> forces;
  std::array<double, 9> stress{};
  double atom_energy_total = 0.0;
  std::vector<double> atom_forces;
  std::array<double, 9> atom_stress{};
  std::vector<double> atom_e;
  std::vector<double> atom_v;
};

FullEval runFullEval(rgpot::MorsePot &pot, const std::vector<double> &pos,
                     const std::vector<int> &z, const std::array<double, 9> &box) {
  const std::size_t n = z.size();
  const rgpot::ForceInput in{n, pos.data(), z.data(), box.data()};
  FullEval res;
  res.forces.assign(3 * n, 0.0);
  rgpot::ForceOut out{};
  out.F = res.forces.data();
  pot.forceImpl(in, &out);
  res.energy = out.energy;
  std::copy(out.stress, out.stress + 9, res.stress.begin());

  res.atom_forces.assign(3 * n, 0.0);
  res.atom_e.assign(n, 0.0);
  res.atom_v.assign(6 * n, 0.0);
  rgpot::ForceOut pa_out{};
  pa_out.F = res.atom_forces.data();
  pot.forcePerAtom(in, &pa_out, rgpot::PerAtomOut{res.atom_e.data(), res.atom_v.data()});
  res.atom_energy_total = pa_out.energy;
  std::copy(pa_out.stress, pa_out.stress + 9, res.atom_stress.begin());
  return res;
}

std::size_t countBitDiffs(const FullEval &a, const FullEval &b) {
  std::size_t diffs = 0;
  diffs += (f64Bits(a.energy) == f64Bits(b.energy)) ? 0 : 1;
  diffs += (f64Bits(a.atom_energy_total) == f64Bits(b.atom_energy_total)) ? 0 : 1;
  for (std::size_t k = 0; k < 9; ++k) {
    diffs += (f64Bits(a.stress[k]) == f64Bits(b.stress[k])) ? 0 : 1;
    diffs += (f64Bits(a.atom_stress[k]) == f64Bits(b.atom_stress[k])) ? 0 : 1;
  }
  for (std::size_t k = 0; k < a.forces.size(); ++k) {
    diffs += (f64Bits(a.forces[k]) == f64Bits(b.forces[k])) ? 0 : 1;
    diffs += (f64Bits(a.atom_forces[k]) == f64Bits(b.atom_forces[k])) ? 0 : 1;
  }
  for (std::size_t k = 0; k < a.atom_e.size(); ++k) {
    diffs += (f64Bits(a.atom_e[k]) == f64Bits(b.atom_e[k])) ? 0 : 1;
  }
  for (std::size_t k = 0; k < a.atom_v.size(); ++k) {
    diffs += (f64Bits(a.atom_v[k]) == f64Bits(b.atom_v[k])) ? 0 : 1;
  }
  return diffs;
}

} // namespace

TEST_CASE("MorsePot OpenMP threaded and serial paths agree across build, Verlet reuse and rebuild",
          "[MorsePot][threads]") {
  ThreadGuard guard;
  const rgpot::MorseConfig cu_vacancy{
      .De = 1.0, .a = 4.0 / 2.55, .re = 2.55, .cutoff = 5.3, .switch_width = 0.0};
  const rgpot::MorseConfig cu_switched{
      .De = 1.0, .a = 4.0 / 2.55, .re = 2.55, .cutoff = 5.3, .switch_width = 0.8};

  // Below kTeamMinAtoms (6x6x6 Cu vacancy = 863 atoms): stays on one thread
  // with zero OpenMP regions opened and identical bits across thread settings.
  {
    const auto sys = fccVacancy(6, 2.55 * std::sqrt(2.0), 29);
    REQUIRE(sys.z.size() == 863);
    const auto step2 = perturbed(sys.pos, 0.05);
    ThreadGuard::set("1");
    rgpot::MorsePot ref_pot(cu_vacancy);
    const auto ref1 = runFullEval(ref_pot, sys.pos, sys.z, sys.box);
    const auto ref2 = runFullEval(ref_pot, step2, sys.z, sys.box);
    REQUIRE(ref_pot.listReused());

    for (const char *th : {"2", "4", "8"}) {
      ThreadGuard::set(th);
      const auto before_teams = rgpot::teamsOpened();
      rgpot::MorsePot pot(cu_vacancy);
      CHECK(countBitDiffs(runFullEval(pot, sys.pos, sys.z, sys.box), ref1) == 0);
      CHECK(countBitDiffs(runFullEval(pot, step2, sys.z, sys.box), ref2) == 0);
      CHECK(pot.listReused());
      CHECK(rgpot::teamsOpened() == before_teams);
    }
  }

  // Above kTeamMinAtoms (13x13x13 Cu vacancy = 8787 atoms): shifted, switched
  // and sheared cells across Stored, Auto and Stream modes. Stored and Auto
  // cover 1, 2, 4 and 8 threads; Stream covers the largest requested team.
  for (const bool sheared : {false, true}) {
    const auto sys = fccVacancy(13, 2.55 * std::sqrt(2.0), 29, sheared);
    REQUIRE(sys.z.size() >= rgpot::kTeamMinAtoms);
    // Inside half of neighbourSkin(5.3) = 0.53 A -> Verlet reuse.
    const auto within_skin = perturbed(sys.pos, 0.08);
    // Past half of neighbourSkin(5.3) -> forces a Verlet rebuild.
    auto past_skin = within_skin;
    past_skin[0] += 0.75;

    for (const auto &cfg : {cu_vacancy, cu_switched}) {
      if (sheared && cfg.switch_width != 0.0) {
        continue;
      }
      for (const auto mode :
           {rgpot::nlist::Mode::Stored, rgpot::nlist::Mode::Auto, rgpot::nlist::Mode::Stream}) {
        if (sheared && mode != rgpot::nlist::Mode::Stored) {
          continue;
        }
        if (cfg.switch_width != 0.0 && mode != rgpot::nlist::Mode::Stored) {
          continue;
        }
        INFO("sheared=" << sheared << " switch=" << cfg.switch_width
                        << " mode=" << static_cast<int>(mode));

        const bool full_matrix =
            !sheared && cfg.switch_width == 0.0 && mode == rgpot::nlist::Mode::Stored;
        ThreadGuard::set("1");
        rgpot::MorsePot serial_pot(cfg);
        serial_pot.setNeighbourMode(mode);
        const auto ref0 = runFullEval(serial_pot, sys.pos, sys.z, sys.box);

        ThreadGuard::set(full_matrix ? "1" : "8");
        rgpot::MorsePot pot(cfg);
        pot.setNeighbourMode(mode);
        CHECK(countBitDiffs(runFullEval(pot, sys.pos, sys.z, sys.box), ref0) == 0);
        if (!full_matrix) {
          continue;
        }

        // The stored neighbour build is serial and independent of the requested
        // team. Build it once, then exercise every team on the same geometry.
        for (const char *th : {"2", "4", "8"}) {
          INFO("OMP_NUM_THREADS=" << th);
          ThreadGuard::set(th);
          CHECK(countBitDiffs(runFullEval(pot, sys.pos, sys.z, sys.box), ref0) == 0);
        }

        ThreadGuard::set("1");
        const auto ref1 = runFullEval(serial_pot, within_skin, sys.z, sys.box);
        const auto ref2 = runFullEval(serial_pot, past_skin, sys.z, sys.box);
        for (const char *th : {"1", "2", "4", "8"}) {
          INFO("OMP_NUM_THREADS=" << th);
          ThreadGuard::set(th);
          const auto teams_before_reuse = rgpot::teamsOpened();
          CHECK(countBitDiffs(runFullEval(pot, within_skin, sys.z, sys.box), ref1) == 0);
          CHECK(pot.listReused());
          if (std::string(th) != "1") {
#if defined(_OPENMP)
            // forceImpl + forcePerAtom each open one two-phase region.
            CHECK(rgpot::teamsOpened() == teams_before_reuse + 2);
#endif
          }
        }

        ThreadGuard::set("8");
        CHECK(countBitDiffs(runFullEval(pot, past_skin, sys.z, sys.box), ref2) == 0);
        rgpot::ForceOut probe{};
        std::vector<double> fbuf(3 * sys.z.size(), 0.0);
        probe.F = fbuf.data();
        auto moved = past_skin;
        moved[0] += 0.75;
        pot.forceImpl({sys.z.size(), moved.data(), sys.z.data(), sys.box.data()}, &probe);
        CHECK_FALSE(pot.listReused());
      }
    }
  }

  // Dilute system above kTeamMinAtoms with fewer than kMinParallelHalfPairs
  // candidate pairs stays on the calling thread without opening a region.
  {
    const std::size_t n = rgpot::kTeamMinAtoms + 16;
    const double spacing = 12.0;
    const int grid = 22;
    const double side = static_cast<double>(grid) * spacing;
    const std::array<double, 9> box = {side, 0.0, 0.0, 0.0, side, 0.0, 0.0, 0.0, side};
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
    // Place 10 close pairs within cutoff 5.3 A; total upper pairs = 10 < 4096.
    for (std::size_t p = 0; p < 10; ++p) {
      pos[3 * (2 * p + 1)] = pos[3 * (2 * p)] + 2.6;
      pos[3 * (2 * p + 1) + 1] = pos[3 * (2 * p) + 1];
      pos[3 * (2 * p + 1) + 2] = pos[3 * (2 * p) + 2];
    }
    const std::vector<int> z(n, 29);

    ThreadGuard::set("1");
    rgpot::MorsePot serial_pot(cu_vacancy);
    serial_pot.setNeighbourMode(rgpot::nlist::Mode::Stored);
    const auto ref = runFullEval(serial_pot, pos, z, box);
    REQUIRE(serial_pot.retainedPairs() == 20);

    ThreadGuard::set("4");
    rgpot::MorsePot par_pot(cu_vacancy);
    par_pot.setNeighbourMode(rgpot::nlist::Mode::Stored);
    const auto before = rgpot::teamsOpened();
    const auto got = runFullEval(par_pot, pos, z, box);
    CHECK(rgpot::teamsOpened() == before);
    CHECK(countBitDiffs(got, ref) == 0);
  }
}
