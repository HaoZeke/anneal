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
#include <thread>
#include <vector>

#include <catch2/catch_all.hpp>

#include "lsan_libomp.hpp"
#include "rgpot/ForceStructs.hpp"
#include "rgpot/Morse/MorsePot.hpp"
#include "rgpot/Morse/detail/MorseSplitSnapshot.hpp"
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

    for (const char *th : {"2", "4", "8", "16"}) {
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
  // cover 1, 2, 4, 8 and 16 threads; Stream covers the largest requested team.
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

        ThreadGuard::set(full_matrix ? "1" : "16");
        rgpot::MorsePot pot(cfg);
        pot.setNeighbourMode(mode);
        CHECK(countBitDiffs(runFullEval(pot, sys.pos, sys.z, sys.box), ref0) == 0);
        if (!full_matrix) {
          continue;
        }

        // The stored neighbour build is serial and independent of the requested
        // team. Build it once, then exercise every team on the same geometry.
        for (const char *th : {"2", "4", "8", "16"}) {
          INFO("OMP_NUM_THREADS=" << th);
          ThreadGuard::set(th);
          CHECK(countBitDiffs(runFullEval(pot, sys.pos, sys.z, sys.box), ref0) == 0);
        }

        ThreadGuard::set("1");
        const auto ref1 = runFullEval(serial_pot, within_skin, sys.z, sys.box);
        const auto ref2 = runFullEval(serial_pot, past_skin, sys.z, sys.box);
        for (const char *th : {"1", "2", "4", "8", "16"}) {
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

        ThreadGuard::set("16");
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

  // Stored-half rebuild that preserves n_atoms, slices, total_up and total_inc
  // while shifting the pair distribution from the low atom indices to the high
  // atom indices must recompute p0_splits/p1_splits and stay bit-exact.
  {
    const std::size_t n = rgpot::kTeamMinAtoms;
    const auto cluster = fccVacancy(6, 2.55 * std::sqrt(2.0), 29);
    const std::size_t m = cluster.z.size();
    REQUIRE(m == 863);
    const double spacing = 12.0;
    const int grid = 22;
    const double side = static_cast<double>(grid) * spacing;
    const std::array<double, 9> box = {side, 0.0, 0.0, 0.0, side, 0.0, 0.0, 0.0, side};

    std::vector<double> sparse_pos;
    sparse_pos.reserve(3 * (n - m));
    for (std::size_t idx = 0; idx < n - m; ++idx) {
      const std::size_t slot = idx + static_cast<std::size_t>(grid * grid * 4);
      const std::size_t gx = slot % grid;
      const std::size_t gy = (slot / grid) % grid;
      const std::size_t gz =
          slot / (static_cast<std::size_t>(grid) * static_cast<std::size_t>(grid));
      sparse_pos.push_back((static_cast<double>(gx) + 0.5) * spacing);
      sparse_pos.push_back((static_cast<double>(gy) + 0.5) * spacing);
      sparse_pos.push_back((static_cast<double>(gz) + 0.5) * spacing);
    }

    std::vector<double> pos_front;
    pos_front.reserve(3 * n);
    for (std::size_t i = 0; i < m; ++i) {
      pos_front.push_back(cluster.pos[3 * i] + 12.0);
      pos_front.push_back(cluster.pos[3 * i + 1] + 12.0);
      pos_front.push_back(cluster.pos[3 * i + 2] + 12.0);
    }
    pos_front.insert(pos_front.end(), sparse_pos.begin(), sparse_pos.end());

    std::vector<double> pos_back;
    pos_back.reserve(3 * n);
    pos_back.insert(pos_back.end(), sparse_pos.begin(), sparse_pos.end());
    for (std::size_t i = 0; i < m; ++i) {
      pos_back.push_back(cluster.pos[3 * i] + 12.0);
      pos_back.push_back(cluster.pos[3 * i + 1] + 12.0);
      pos_back.push_back(cluster.pos[3 * i + 2] + 12.0);
    }
    const std::vector<int> z(n, 29);

    ThreadGuard::set("1");
    rgpot::MorsePot serial_pot(cu_vacancy);
    serial_pot.setNeighbourMode(rgpot::nlist::Mode::Stored);
    const auto ref_front = runFullEval(serial_pot, pos_front, z, box);
    const auto ref_back = runFullEval(serial_pot, pos_back, z, box);

    for (const int slices : {4, 16}) {
      INFO("slices = " << slices);
      ThreadGuard::set(std::to_string(slices).c_str());
      rgpot::MorsePot par_pot(cu_vacancy);
      par_pot.setNeighbourMode(rgpot::nlist::Mode::Stored);
      const auto got_front = runFullEval(par_pot, pos_front, z, box);
      CHECK(par_pot.listReused());
      const std::size_t retained_front = par_pot.retainedPairs();
      const auto snap_front = rgpot::detail::morseStoredSplitSnapshot();
      REQUIRE(snap_front.atoms == n);
      REQUIRE(snap_front.slices == slices);
      REQUIRE(snap_front.total_up >= 4096);
      REQUIRE(snap_front.total_inc == 2 * snap_front.total_up);
      REQUIRE(snap_front.p0_splits.size() ==
              static_cast<std::size_t>(slices + 1));
      REQUIRE(snap_front.p1_splits.size() ==
              static_cast<std::size_t>(slices + 1));
      for (std::size_t s = 1; s < static_cast<std::size_t>(slices); ++s) {
        CHECK(snap_front.p0_splits[s] > 0);
        CHECK(snap_front.p0_splits[s] <= m);
        CHECK(snap_front.p1_splits[s] > 0);
        CHECK(snap_front.p1_splits[s] <= m);
      }
      CHECK(countBitDiffs(got_front, ref_front) == 0);

      // First call on pos_back forces a Verlet rebuild with identical totals
      // and the dense cluster moved to [n - m, n).
      std::vector<double> probe_f(3 * n, 0.0);
      rgpot::ForceOut probe_out{};
      probe_out.F = probe_f.data();
      par_pot.forceImpl({n, pos_back.data(), z.data(), box.data()}, &probe_out);
      CHECK_FALSE(par_pot.listReused());
      const auto got_back = runFullEval(par_pot, pos_back, z, box);
      CHECK(par_pot.listReused());
      CHECK(par_pot.retainedPairs() == retained_front);

      const auto snap_back = rgpot::detail::morseStoredSplitSnapshot();
      CHECK(snap_back.owner == snap_front.owner);
      CHECK(snap_back.build_id == snap_front.build_id + 1);
      CHECK(snap_back.atoms == snap_front.atoms);
      CHECK(snap_back.slices == snap_front.slices);
      CHECK(snap_back.total_up == snap_front.total_up);
      CHECK(snap_back.total_inc == snap_front.total_inc);
      CHECK(snap_back.p0_splits != snap_front.p0_splits);
      CHECK(snap_back.p1_splits != snap_front.p1_splits);
      for (std::size_t s = 1; s < static_cast<std::size_t>(slices); ++s) {
        CHECK(snap_back.p0_splits[s] >= n - m);
        CHECK(snap_back.p0_splits[s] < n);
        CHECK(snap_back.p1_splits[s] >= n - m);
        CHECK(snap_back.p1_splits[s] < n);
      }
      CHECK(countBitDiffs(got_back, ref_back) == 0);
    }
  }
}

TEST_CASE(
    "MorsePot boundary active-bit merge is loss-free under concurrent updates",
    "[MorsePot][threads][atomic]") {
  constexpr std::size_t kThreads = 8;
  constexpr std::size_t kWords = 4096;
  std::vector<std::uint64_t> words(kWords, 0ULL);
  std::atomic<bool> start{false};
  std::atomic<std::size_t> ready{0};

  std::vector<std::thread> workers;
  workers.reserve(kThreads);
  for (std::size_t t = 0; t < kThreads; ++t) {
    workers.emplace_back([&, t]() {
      const std::uint64_t mask = 0x0101010101010101ULL << t;
      ready.fetch_add(1, std::memory_order_release);
      while (!start.load(std::memory_order_acquire)) {
      }
      for (std::size_t w = 0; w < kWords; ++w) {
        rgpot::detail::morseAtomicFetchOrWord(words[w], mask);
      }
    });
  }
  while (ready.load(std::memory_order_acquire) < kThreads) {
  }
  start.store(true, std::memory_order_release);
  for (auto &th : workers) {
    th.join();
  }

  std::size_t lost_words = 0;
  for (std::size_t w = 0; w < kWords; ++w) {
    if (words[w] != ~0ULL) {
      ++lost_words;
    }
  }
  CHECK(lost_words == 0);

  // Also stress Phase 0 boundary-word merging in MorsePot::compute on 16
  // threads with an atom count where interior p0_splits land inside shared
  // 64-pair active_bits words (last_w of slice s equals first_w of slice s +
  // 1).
  ThreadGuard guard;
  const rgpot::MorseConfig cu_vacancy{.De = 1.0,
                                      .a = 4.0 / 2.55,
                                      .re = 2.55,
                                      .cutoff = 5.3,
                                      .switch_width = 0.0};
  const auto sys = fccVacancy(13, 2.55 * std::sqrt(2.0), 29);
  const std::size_t n = sys.z.size();
  ThreadGuard::set("1");
  rgpot::MorsePot ref_pot(cu_vacancy);
  ref_pot.setNeighbourMode(rgpot::nlist::Mode::Stored);
  const auto want = runFullEval(ref_pot, sys.pos, sys.z, sys.box);

  ThreadGuard::set("16");
  rgpot::MorsePot par_pot(cu_vacancy);
  par_pot.setNeighbourMode(rgpot::nlist::Mode::Stored);
  // Build stored list on 1st call, then run 32 back-to-back stored-half
  // evaluations.
  std::vector<double> fbuf(3 * n, 0.0);
  rgpot::ForceOut out{};
  out.F = fbuf.data();
  par_pot.forceImpl({n, sys.pos.data(), sys.z.data(), sys.box.data()}, &out);
  std::size_t bad_evals = 0;
  for (int iter = 0; iter < 32; ++iter) {
    par_pot.forceImpl({n, sys.pos.data(), sys.z.data(), sys.box.data()}, &out);
    if (f64Bits(out.energy) != f64Bits(want.energy)) {
      ++bad_evals;
    }
  }
  CHECK(bad_evals == 0);
}

TEST_CASE("MorsePot workspace shrinks retained capacities when a smaller "
          "evaluation follows a large one",
          "[MorsePot][threads][shrink]") {
  ThreadGuard guard;
  const rgpot::MorseConfig cu_vacancy{.De = 1.0,
                                      .a = 4.0 / 2.55,
                                      .re = 2.55,
                                      .cutoff = 5.3,
                                      .switch_width = 0.0};

  // 1. Evaluate a large 16x16x16 Cu vacancy system (16383 atoms, ~703k upper
  // pairs) on 4 threads with per-atom energy and virial so all workspace
  // buffers grow well above 4 * kShrinkFloor (16384 elements).
  const auto large_sys = fccVacancy(16, 2.55 * std::sqrt(2.0), 29);
  REQUIRE(large_sys.z.size() == 16383);
  ThreadGuard::set("4");
  rgpot::MorsePot pot(cu_vacancy);
  pot.setNeighbourMode(rgpot::nlist::Mode::Stored);
  const auto large_got =
      runFullEval(pot, large_sys.pos, large_sys.z, large_sys.box);
  (void)large_got;

  const auto snap_large = rgpot::detail::morseStoredSplitSnapshot();
  REQUIRE(snap_large.total_up > 500000);
  REQUIRE(snap_large.up_off_cap >= large_sys.z.size() + 1);
  REQUIRE(snap_large.lo_off_cap >= large_sys.z.size() + 1);
  REQUIRE(snap_large.lo_count_cap >= large_sys.z.size());
  REQUIRE(snap_large.lo_u_cap >= snap_large.total_up);
  REQUIRE(snap_large.pair_forces_cap >= snap_large.total_up);
  REQUIRE(snap_large.pair_he_cap >= snap_large.total_up);
  REQUIRE(snap_large.pair_hv_cap >= 6 * snap_large.total_up);
  REQUIRE(snap_large.row_e_cap >= large_sys.z.size());
  REQUIRE(snap_large.row_s_cap >= 6 * large_sys.z.size());

  // Calling forceImpl alone (without per-atom energy or virial) on the same
  // large system must release pair_he and pair_hv down to zero capacity while
  // keeping pair_forces at the high-water mark.
  std::vector<double> fbuf(3 * large_sys.z.size(), 0.0);
  rgpot::ForceOut out{};
  out.F = fbuf.data();
  pot.forceImpl({large_sys.z.size(), large_sys.pos.data(), large_sys.z.data(),
                 large_sys.box.data()},
                &out);
  const auto snap_no_pa = rgpot::detail::morseStoredSplitSnapshot();
  CHECK(snap_no_pa.pair_he_cap == 0);
  CHECK(snap_no_pa.pair_hv_cap == 0);
  CHECK(snap_no_pa.pair_forces_cap == snap_large.pair_forces_cap);

  // Restore pair_he and pair_hv to the large high-water mark before step 2.
  (void)runFullEval(pot, large_sys.pos, large_sys.z, large_sys.box);

  // 2. Evaluate a medium system at kTeamMinAtoms = 8192 atoms (a compact
  // 6x6x6 Cu cluster of ~37k upper pairs < 703k / 4, plus sparse atoms) on 4
  // threads. Every pair buffer (lo_u, active_bits, pair_forces, pair_he,
  // pair_hv) and row_s (6 * 8192 = 49152 > 4096) requires less than a quarter
  // of its retained capacity and must shrink.
  const std::size_t n_med = rgpot::kTeamMinAtoms;
  const auto cluster = fccVacancy(6, 2.55 * std::sqrt(2.0), 29);
  const std::size_t m = cluster.z.size();
  const double spacing = 12.0;
  const int grid = 22;
  const double side = static_cast<double>(grid) * spacing;
  const std::array<double, 9> med_box = {side, 0.0, 0.0, 0.0, side,
                                         0.0,  0.0, 0.0, side};
  std::vector<double> med_pos;
  med_pos.reserve(3 * n_med);
  for (std::size_t i = 0; i < m; ++i) {
    med_pos.push_back(cluster.pos[3 * i] + 12.0);
    med_pos.push_back(cluster.pos[3 * i + 1] + 12.0);
    med_pos.push_back(cluster.pos[3 * i + 2] + 12.0);
  }
  for (std::size_t idx = 0; idx < n_med - m; ++idx) {
    const std::size_t slot = idx + static_cast<std::size_t>(grid * grid * 4);
    const std::size_t gx = slot % grid;
    const std::size_t gy = (slot / grid) % grid;
    const std::size_t gz = slot / (static_cast<std::size_t>(grid) *
                                   static_cast<std::size_t>(grid));
    med_pos.push_back((static_cast<double>(gx) + 0.5) * spacing);
    med_pos.push_back((static_cast<double>(gy) + 0.5) * spacing);
    med_pos.push_back((static_cast<double>(gz) + 0.5) * spacing);
  }
  const std::vector<int> med_z(n_med, 29);

  ThreadGuard::set("1");
  rgpot::MorsePot med_ref_pot(cu_vacancy);
  med_ref_pot.setNeighbourMode(rgpot::nlist::Mode::Stored);
  const auto med_want = runFullEval(med_ref_pot, med_pos, med_z, med_box);

  // Restore the calling thread's workspace to the 16x16x16 high-water mark
  // (after med_ref_pot ran on 1 thread), then rebuild pot on med_pos on 4
  // threads.
  ThreadGuard::set("4");
  (void)runFullEval(pot, large_sys.pos, large_sys.z, large_sys.box);
  const auto med_got = runFullEval(pot, med_pos, med_z, med_box);
  const auto snap_med = rgpot::detail::morseStoredSplitSnapshot();
  CHECK(countBitDiffs(med_got, med_want) == 0);
  REQUIRE(snap_med.total_up >= 4096);
  REQUIRE(snap_med.total_up * 4 < snap_large.total_up);
  CHECK(snap_med.lo_u_cap < snap_large.lo_u_cap);
  CHECK(snap_med.active_bits_cap < snap_large.active_bits_cap);
  CHECK(snap_med.pair_forces_cap < snap_large.pair_forces_cap);
  CHECK(snap_med.pair_he_cap < snap_large.pair_he_cap);
  CHECK(snap_med.pair_hv_cap < snap_large.pair_hv_cap);

  // 3. Also verify that the atom-indexed buffers (up_off, lo_off, lo_count,
  // row_e, row_s) shrink when an evaluation with > 4 * kTeamMinAtoms atoms
  // (34000 atoms: the 6x6x6 Cu cluster plus sparse atoms) is followed by the
  // 8192-atom system on 4 threads.
  const std::size_t n_wide = 34000;
  const int wide_grid = 36;
  const double wide_side = static_cast<double>(wide_grid) * spacing;
  const std::array<double, 9> wide_box = {
      -wide_side, 0.0, 0.0, 0.0, -wide_side, 0.0, 0.0, 0.0, -wide_side};
  std::vector<double> wide_pos;
  wide_pos.reserve(3 * n_wide);
  for (std::size_t i = 0; i < m; ++i) {
    wide_pos.push_back(cluster.pos[3 * i] + 12.0);
    wide_pos.push_back(cluster.pos[3 * i + 1] + 12.0);
    wide_pos.push_back(cluster.pos[3 * i + 2] + 12.0);
  }
  for (std::size_t idx = 0; idx < n_wide - m; ++idx) {
    const std::size_t slot =
        idx + static_cast<std::size_t>(wide_grid * wide_grid * 4);
    const std::size_t gx = slot % wide_grid;
    const std::size_t gy = (slot / wide_grid) % wide_grid;
    const std::size_t gz = slot / (static_cast<std::size_t>(wide_grid) *
                                   static_cast<std::size_t>(wide_grid));
    wide_pos.push_back((static_cast<double>(gx) + 0.5) * spacing);
    wide_pos.push_back((static_cast<double>(gy) + 0.5) * spacing);
    wide_pos.push_back((static_cast<double>(gz) + 0.5) * spacing);
  }
  const std::vector<int> wide_z(n_wide, 29);
  (void)runFullEval(pot, wide_pos, wide_z, wide_box);
  const auto snap_wide = rgpot::detail::morseStoredSplitSnapshot();
  REQUIRE(snap_wide.up_off_cap >= n_wide + 1);
  REQUIRE(snap_wide.lo_off_cap >= n_wide + 1);
  REQUIRE(snap_wide.lo_count_cap >= n_wide);
  REQUIRE(snap_wide.row_e_cap >= n_wide);
  REQUIRE(snap_wide.row_s_cap >= 6 * n_wide);

  (void)runFullEval(pot, med_pos, med_z, med_box);
  const auto snap_after_wide = rgpot::detail::morseStoredSplitSnapshot();
  CHECK(snap_after_wide.up_off_cap < snap_wide.up_off_cap);
  CHECK(snap_after_wide.lo_off_cap < snap_wide.lo_off_cap);
  CHECK(snap_after_wide.lo_count_cap < snap_wide.lo_count_cap);
  CHECK(snap_after_wide.row_e_cap < snap_wide.row_e_cap);
  CHECK(snap_after_wide.row_s_cap < snap_wide.row_s_cap);

  // 4. Also verify that the serial fallback path shrinks row_e and row_s when
  // a dilute 8192-atom system (0 close pairs < kMinParallelHalfPairs) follows
  // the 34000-atom system on 1 thread.
  ThreadGuard::set("1");
  (void)runFullEval(pot, wide_pos, wide_z, wide_box);
  const auto snap_1t_wide = rgpot::detail::morseStoredSplitSnapshot();
  REQUIRE(snap_1t_wide.row_e_cap >= n_wide);
  REQUIRE(snap_1t_wide.row_s_cap >= 6 * n_wide);

  std::vector<double> dilute_pos;
  dilute_pos.reserve(3 * n_med);
  for (std::size_t idx = 0; idx < n_med; ++idx) {
    const std::size_t gx = idx % grid;
    const std::size_t gy = (idx / grid) % grid;
    const std::size_t gz =
        idx / (static_cast<std::size_t>(grid) * static_cast<std::size_t>(grid));
    dilute_pos.push_back((static_cast<double>(gx) + 0.5) * spacing);
    dilute_pos.push_back((static_cast<double>(gy) + 0.5) * spacing);
    dilute_pos.push_back((static_cast<double>(gz) + 0.5) * spacing);
  }
  pot.forceImpl({n_med, dilute_pos.data(), med_z.data(), med_box.data()}, &out);
  const auto snap_1t_dilute = rgpot::detail::morseStoredSplitSnapshot();
  CHECK(snap_1t_dilute.row_e_cap < snap_1t_wide.row_e_cap);
  CHECK(snap_1t_dilute.row_s_cap < snap_1t_wide.row_s_cap);
}
