// MIT License
// Copyright 2023--present rgpot developers
//
// Copper FCC cluster pin from eOn EMTCuTest / point_emt_cu.dat.

#include <array>
#include <cmath>
#include <cstdint>
#include <cstdlib>
#include <cstring>
#include <string>
#include <vector>

#include <catch2/catch_all.hpp>

#include "rgpot/ForceStructs.hpp"
#include "rgpot/EMT/EMTPot.hpp"
#include "rgpot/Slices.hpp"

#include "rgpot/EMT/Asap/Atoms.h"
#include "rgpot/EMT/Asap/EMT.h"
#include "rgpot/EMT/Asap/SuperCell.h"
#include "rgpot/EMT/Asap/Vec.h"

using Catch::Matchers::WithinAbs;

TEST_CASE("EMT energy matches the copper cluster pin", "[emt]") {
  const std::vector<std::array<double, 3>> xyz = {{
      {8.1925, 8.1925, 8.1925},
      {10.0, 10.0, 8.1925},
      {10.0, 8.1925, 10.0},
      {8.1925, 10.0, 10.0},
  }};
  rgpot::types::AtomMatrix positions(4, 3);
  std::vector<int> types(4, 29);
  for (int i = 0; i < 4; ++i) {
    for (int k = 0; k < 3; ++k) {
      positions(i, k) = xyz[static_cast<size_t>(i)][static_cast<size_t>(k)];
    }
  }
  const std::array<std::array<double, 3>, 3> box = {{
      {20.0, 0.0, 0.0},
      {0.0, 20.0, 0.0},
      {0.0, 0.0, 20.0},
  }};
  rgpot::EMTPot pot(rgpot::EMTConfig{.rasmussen = false});
  const auto [energy, forces, variance] = pot(positions, types, box);
  REQUIRE_THAT(energy, WithinAbs(5.129167, 5e-4));
  double worst = 0.0;
  for (int i = 0; i < 4; ++i) {
    const double n2 = forces(i, 0) * forces(i, 0) + forces(i, 1) * forces(i, 1) +
                      forces(i, 2) * forces(i, 2);
    worst = std::max(worst, std::sqrt(n2));
  }
  REQUIRE_THAT(worst, WithinAbs(1.914263, 2e-3));
  REQUIRE(pot.config().rasmussen == false);
  REQUIRE(pot.paramsKey() != 0);
}

TEST_CASE("EMT reads the species again when only the species change",
          "[emt]") {
  rgpot::types::AtomMatrix positions(4, 3);
  const std::array<std::array<double, 3>, 4> xyz = {{
      {8.1925, 8.1925, 8.1925},
      {10.0, 10.0, 8.1925},
      {10.0, 8.1925, 10.0},
      {8.1925, 10.0, 10.0},
  }};
  for (int i = 0; i < 4; ++i) {
    for (int k = 0; k < 3; ++k) {
      positions(i, k) = xyz[static_cast<size_t>(i)][static_cast<size_t>(k)];
    }
  }
  const std::array<std::array<double, 3>, 3> box = {{
      {20.0, 0.0, 0.0},
      {0.0, 20.0, 0.0},
      {0.0, 0.0, 20.0},
  }};
  const std::vector<int> copper(4, 29);
  const std::vector<int> mixed = {29, 79, 29, 79};

  rgpot::EMTPot reused;
  (void)reused(positions, copper, box);
  const auto [e_reused, f_reused, v_reused] = reused(positions, mixed, box);

  rgpot::EMTPot fresh;
  const auto [e_fresh, f_fresh, v_fresh] = fresh(positions, mixed, box);

  REQUIRE_THAT(e_reused, WithinAbs(e_fresh, 1e-12));
  for (int i = 0; i < 4; ++i) {
    for (int k = 0; k < 3; ++k) {
      REQUIRE_THAT(f_reused(i, k), WithinAbs(f_fresh(i, k), 1e-12));
    }
  }
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

struct Eval {
  double energy = 0.0;
  std::vector<double> forces;
  std::array<double, 9> stress{};
  int has_stress = 0;
};

Eval runEval(rgpot::EMTPot &pot, const std::vector<double> &pos,
             const std::vector<int> &z, const std::array<double, 9> &box) {
  Eval outv;
  outv.forces.assign(pos.size(), 0.0);
  rgpot::ForceOut out{};
  out.F = outv.forces.data();
  std::fill(std::begin(out.stress), std::end(out.stress), 0.0);
  pot.forceImpl({z.size(), pos.data(), z.data(), box.data()}, &out);
  outv.energy = out.energy;
  outv.has_stress = out.has_stress;
  std::copy(std::begin(out.stress), std::end(out.stress), outv.stress.begin());
  return outv;
}

std::size_t countBitDiffs(const Eval &a, const Eval &b) {
  std::size_t diffs = 0;
  diffs += (f64Bits(a.energy) == f64Bits(b.energy)) ? 0 : 1;
  diffs += (a.has_stress == b.has_stress) ? 0 : 1;
  for (std::size_t k = 0; k < 9; ++k) {
    diffs += (f64Bits(a.stress[k]) == f64Bits(b.stress[k])) ? 0 : 1;
  }
  for (std::size_t k = 0; k < a.forces.size(); ++k) {
    diffs += (f64Bits(a.forces[k]) == f64Bits(b.forces[k])) ? 0 : 1;
  }
  return diffs;
}

struct FccBox {
  std::vector<double> pos;
  std::vector<int> z;
  std::array<double, 9> box{};
};

FccBox fccCu(int side, double lattice) {
  static constexpr double kBasis[4][3] = {
      {0.0, 0.0, 0.0}, {0.0, 0.5, 0.5}, {0.5, 0.0, 0.5}, {0.5, 0.5, 0.0}};
  FccBox out;
  const double L = static_cast<double>(side) * lattice;
  out.box = {L, 0.0, 0.0, 0.0, L, 0.0, 0.0, 0.0, L};
  for (int ix = 0; ix < side; ++ix) {
    for (int iy = 0; iy < side; ++iy) {
      for (int iz = 0; iz < side; ++iz) {
        for (const auto &b : kBasis) {
          const double fx =
              (static_cast<double>(ix) + b[0]) / static_cast<double>(side);
          const double fy =
              (static_cast<double>(iy) + b[1]) / static_cast<double>(side);
          const double fz =
              (static_cast<double>(iz) + b[2]) / static_cast<double>(side);
          const std::size_t idx = out.z.size();
          const double jx = 0.01 * lattice * std::sin(0.23 * static_cast<double>(idx) + 0.1);
          const double jy = 0.01 * lattice * std::cos(0.31 * static_cast<double>(idx) + 0.2);
          const double jz = 0.01 * lattice * std::sin(0.29 * static_cast<double>(idx) + 0.3);
          out.pos.push_back(fx * L + jx);
          out.pos.push_back(fy * L + jy);
          out.pos.push_back(fz * L + jz);
          out.z.push_back(29);
        }
      }
    }
  }
  return out;
}

std::vector<double> shifted(const std::vector<double> &pos, double step) {
  std::vector<double> out = pos;
  for (std::size_t i = 0; i < out.size(); ++i) {
    out[i] += step * std::sin(0.71 * static_cast<double>(i) + 0.4);
  }
  return out;
}

constexpr bool kPeriodic[3] = {true, true, true};

// Writes a marker over the force buffer EMT keeps. A later call that
// returns the marker came from the cache, and one that overwrites it
// recomputed.
class MarkableEMT : public EMT {
public:
  MarkableEMT() : EMT(nullptr) {}

  static constexpr double kMarker = 1234.5;

  void markForces() {
    for (Vec &f : force) {
      f[0] = f[1] = f[2] = kMarker;
    }
  }
};

struct DirectEmt {
  Vec basis[3];
  SuperCell cell;
  Atoms atoms;
  MarkableEMT emt;

  explicit DirectEmt(const FccBox &sys)
      : basis{Vec(sys.box[0], sys.box[1], sys.box[2]),
              Vec(sys.box[3], sys.box[4], sys.box[5]),
              Vec(sys.box[6], sys.box[7], sys.box[8])},
        cell(basis, kPeriodic),
        atoms(reinterpret_cast<const Vec *>(sys.pos.data()),
              static_cast<int>(sys.z.size()), &cell) {
    atoms.SetAtomicNumbers(sys.z.data());
    atoms.SetUnitCell(basis, true);
    atoms.SetCalculator(&emt);
  }

  void move(const std::vector<double> &pos) {
    atoms.SetCartesianPositions(reinterpret_cast<const Vec *>(pos.data()));
    atoms.SetUnitCell(basis, true);
  }

  std::vector<double> forces(std::size_t n) {
    const Vec *f = emt.GetCartesianForces();
    std::vector<double> out(3 * n);
    for (std::size_t i = 0; i < n; ++i) {
      for (int k = 0; k < 3; ++k) {
        out[3 * i + static_cast<std::size_t>(k)] = f[i][k];
      }
    }
    return out;
  }
};

std::size_t countBitDiffs(const std::vector<double> &a,
                          const std::vector<double> &b) {
  std::size_t diffs = a.size() == b.size() ? 0 : 1;
  for (std::size_t k = 0; k < std::min(a.size(), b.size()); ++k) {
    diffs += (f64Bits(a[k]) == f64Bits(b[k])) ? 0 : 1;
  }
  return diffs;
}

} // namespace

TEST_CASE("EMT direct repeated force calls reuse the cached forces",
          "[emt][threads]") {
  ThreadGuard guard;
  const auto large = fccCu(13, 3.615 / std::sqrt(2.0));
  const std::size_t n = large.z.size();
  REQUIRE(n >= rgpot::kTeamMinAtoms);
  const auto moved = shifted(large.pos, 0.02);

  std::vector<double> ref_forces;
  std::vector<double> ref_moved;
  for (const char *th : {"1", "4"}) {
    INFO("OMP_NUM_THREADS=" << th);
    ThreadGuard::set(th);

    DirectEmt direct(large);
    const auto first = direct.forces(n);

    direct.emt.markForces();
    const double marker = MarkableEMT::kMarker;
    const Vec *again = direct.emt.GetCartesianForces();
    bool cached = true;
    for (std::size_t i = 0; i < n; ++i) {
      for (int k = 0; k < 3; ++k) {
        cached = cached && f64Bits(again[i][k]) == f64Bits(marker);
      }
    }
    CHECK(cached);
    const auto before = rgpot::teamsOpened();
    CHECK(direct.emt.GetCartesianForces() == again);
    CHECK(rgpot::teamsOpened() == before);

    // A moved atom set is a new evaluation, not the cached one.
    direct.move(moved);
    const auto after_move = direct.forces(n);
    CHECK(countBitDiffs(after_move, first) != 0);

    // The same forces, bit for bit, as an evaluation through EMTPot.
    rgpot::EMTPot pot;
    const auto through_pot = runEval(pot, large.pos, large.z, large.box);
    CHECK(countBitDiffs(first, through_pot.forces) == 0);
    const auto through_pot_moved = runEval(pot, moved, large.z, large.box);
    CHECK(countBitDiffs(after_move, through_pot_moved.forces) == 0);

    if (ref_forces.empty()) {
      ref_forces = first;
      ref_moved = after_move;
    } else {
      CHECK(countBitDiffs(first, ref_forces) == 0);
      CHECK(countBitDiffs(after_move, ref_moved) == 0);
    }
  }
}

TEST_CASE("EMT OpenMP force path is bit-identical across threads", "[emt][threads]") {
  ThreadGuard guard;
  rgpot::EMTPot sanity;

  const auto small = fccCu(12, 3.615 / std::sqrt(2.0));
  REQUIRE(small.z.size() < rgpot::kTeamMinAtoms);
  const auto small_reuse = shifted(small.pos, 0.02);
  const auto small_rebuild = shifted(small.pos, 0.35);

  ThreadGuard::set("1");
  auto ref_small = runEval(sanity, small.pos, small.z, small.box);
  auto ref_small_reuse = runEval(sanity, small_reuse, small.z, small.box);
  auto ref_small_rebuild = runEval(sanity, small_rebuild, small.z, small.box);
  REQUIRE(ref_small.has_stress == 0);

  for (const char *th : {"2", "4", "8"}) {
    INFO("small OMP_NUM_THREADS=" << th);
    ThreadGuard::set(th);
    const auto before = rgpot::teamsOpened();
    rgpot::EMTPot pot;
    CHECK(countBitDiffs(runEval(pot, small.pos, small.z, small.box), ref_small) == 0);
    CHECK(countBitDiffs(runEval(pot, small_reuse, small.z, small.box), ref_small_reuse) == 0);
    CHECK(countBitDiffs(runEval(pot, small_rebuild, small.z, small.box), ref_small_rebuild) == 0);
    CHECK(rgpot::teamsOpened() == before);
  }

  const auto large = fccCu(13, 3.615 / std::sqrt(2.0));
  REQUIRE(large.z.size() >= rgpot::kTeamMinAtoms);
  const auto large_reuse = shifted(large.pos, 0.02);
  const auto large_rebuild = shifted(large.pos, 0.35);

  ThreadGuard::set("1");
  rgpot::EMTPot serial;
  auto ref_large = runEval(serial, large.pos, large.z, large.box);
  auto ref_large_reuse = runEval(serial, large_reuse, large.z, large.box);
  auto ref_large_rebuild = runEval(serial, large_rebuild, large.z, large.box);

  for (const char *th : {"2", "4", "8"}) {
    INFO("large OMP_NUM_THREADS=" << th);
    ThreadGuard::set(th);
    rgpot::EMTPot pot;
    CHECK(countBitDiffs(runEval(pot, large.pos, large.z, large.box), ref_large) == 0);
    const auto before = rgpot::teamsOpened();
    CHECK(countBitDiffs(runEval(pot, large_reuse, large.z, large.box), ref_large_reuse) == 0);
#if defined(_OPENMP)
    CHECK(rgpot::teamsOpened() == before + 1);
#endif
    CHECK(countBitDiffs(runEval(pot, large_rebuild, large.z, large.box), ref_large_rebuild) == 0);
  }
}
