// MIT License
// Copyright 2023--present rgpot developers
//
// Pot-level microbenchmarks that gate potential migrations.
//
// Every case carries the hidden tag pair "[.][benchmark]", so a plain run
// of this binary reports zero cases; the benchmarks execute only when the
// tag is requested explicitly (meson passes "[benchmark]").
//
// API choice: the timed loops call forceImpl with caller-owned flat
// buffers, so the measurement covers the physics and the neighbour
// bookkeeping without a per-iteration force-matrix allocation. One case
// per pot goes through operator() instead, which prices the wrapper eOn
// actually calls: AtomMatrix construction, box flattening, and (in
// RGPOT_HAS_CACHE builds) the XXH3 key hash.

#include <array>
#include <cmath>
#include <cstddef>
#include <vector>

#include <catch2/catch_all.hpp>

#include "rgpot/ForceStructs.hpp"
#include "rgpot/HessianVector.hpp"
#include "rgpot/LennardJones/LJClusterPot.hpp"
#include "rgpot/EAM/EAMPot.hpp"
#include "rgpot/EMT/EMTPot.hpp"
#include "rgpot/LennardJones/LJPot.hpp"
#include "rgpot/Morse/MorsePot.hpp"
#include "rgpot/ZBL/ZBLPot.hpp"
#include "rgpot/types/AtomMatrix.hpp"

#ifdef RGPOT_BENCH_CUH2
#include "rgpot/fortran/FortranPots.hpp"
#endif

using rgpot::types::AtomMatrix;

namespace {

constexpr std::size_t kCellsPerSide = 5;
constexpr std::size_t kNAtoms = 4 * kCellsPerSide * kCellsPerSide *
                                kCellsPerSide; // 500, FCC has 4 sites per cell
// LJPot truncates at 15 A. The neighbour layer only stores a list when the
// cutoff fits within half the smallest periodic width, so a 32 A cube keeps
// the benchmark inside the cached regime with margin.
constexpr double kBoxSide = 32.0;
constexpr int kAtomicNumber = 18;
// A warm geometry stays inside the Verlet skin.
constexpr double kColdStride = 1.0;
constexpr int kColdCycle = 128;
constexpr double kWarmOffset = 500.0;
constexpr double kOperatorOffset = 900.0;
constexpr double kWarmAmplitude = 0.02; // per-atom, well inside skin/2

// ZBL truncates at 2.5 A, shorter than the nearest-neighbour distance of the
// 32 A lattice, so its case packs the same atom count into a denser cube.
constexpr double kDenseBoxSide = 12.0;

/// Analytic FCC lattice filling the cube, with a deterministic sub-angstrom
/// modulation so no two pair distances coincide exactly.
std::vector<double> fccPositions(double side = kBoxSide) {
  const double a = side / static_cast<double>(kCellsPerSide);
  static const double basis[4][3] = {
      {0.0, 0.0, 0.0}, {0.0, 0.5, 0.5}, {0.5, 0.0, 0.5}, {0.5, 0.5, 0.0}};
  std::vector<double> R;
  R.reserve(3 * kNAtoms);
  for (std::size_t ix = 0; ix < kCellsPerSide; ++ix) {
    for (std::size_t iy = 0; iy < kCellsPerSide; ++iy) {
      for (std::size_t iz = 0; iz < kCellsPerSide; ++iz) {
        for (const auto &b : basis) {
          const auto idx = static_cast<double>(R.size() / 3);
          R.push_back((static_cast<double>(ix) + b[0]) * a +
                      0.05 * std::sin(1.7 * idx));
          R.push_back((static_cast<double>(iy) + b[1]) * a +
                      0.05 * std::sin(2.3 * idx + 1.0));
          R.push_back((static_cast<double>(iz) + b[2]) * a +
                      0.05 * std::sin(3.1 * idx + 2.0));
        }
      }
    }
  }
  return R;
}

std::array<double, 9> cubicBox(double side) {
  return {side, 0.0, 0.0, 0.0, side, 0.0, 0.0, 0.0, side};
}

void translate(const std::vector<double> &src, double offsetX,
               std::vector<double> &dst) {
  for (std::size_t k = 0; k < src.size(); k += 3) {
    dst[k] = src[k] + offsetX;
    dst[k + 1] = src[k + 1];
    dst[k + 2] = src[k + 2];
  }
}

/// Sub-0.1 A wobble around a fixed base: successive geometries stay well
/// inside the Verlet skin, so the pool matches the same slot every call.
void wobble(const std::vector<double> &src, double offsetX, int step,
            std::vector<double> &dst) {
  const auto phase = static_cast<double>(step);
  for (std::size_t k = 0; k < src.size(); k += 3) {
    const auto idx = static_cast<double>(k / 3);
    dst[k] = src[k] + offsetX + kWarmAmplitude * std::sin(0.37 * phase + idx);
    dst[k + 1] =
        src[k + 1] + kWarmAmplitude * std::sin(0.41 * phase + 1.3 * idx);
    dst[k + 2] =
        src[k + 2] + kWarmAmplitude * std::sin(0.53 * phase + 2.1 * idx);
  }
}

template <typename Pot>
double evalPot(const Pot &pot, const std::vector<double> &pos,
               const std::vector<int> &types, const std::array<double, 9> &box,
               std::vector<double> &forces) {
  rgpot::ForceInput fi{.nAtoms = pos.size() / 3,
                       .pos = pos.data(),
                       .atmnrs = types.data(),
                       .box = box.data()};
  rgpot::ForceOut fo{.F = forces.data(),
                      .energy = 0.0,
                      .variance = 0.0,
                      .stress = {},
                      .has_stress = 0};
  pot.forceImpl(fi, &fo);
  return fo.energy;
}

} // namespace

TEST_CASE("Benchmark: LJPot force, cold pair list", "[.][benchmark][ljpot]") {
  const auto base = fccPositions();
  const std::vector<int> types(kNAtoms, kAtomicNumber);
  const auto box = cubicBox(kBoxSide);
  std::vector<double> pos(base.size());
  std::vector<double> forces(base.size(), 0.0);
  rgpot::LJPot pot;
  int step = 0;

  // Each iteration lands on a rigid translation 1 A away from the previous
  // one, past the 0.5 A proximity window of every retained slot, so the call
  // takes the fused scan path. The offset cycles over 128 steps, far more
  // than the 8 slots the pool keeps, which bounds the coordinate drift.
  BENCHMARK("LJ force, 500 atoms, pair list miss") {
    const double offset =
        static_cast<double>(step++ % kColdCycle) * kColdStride;
    translate(base, offset, pos);
    return evalPot(pot, pos, types, box, forces);
  };
}

TEST_CASE("Benchmark: LJPot force, warm pair list", "[.][benchmark][ljpot]") {
  const auto base = fccPositions();
  const std::vector<int> types(kNAtoms, kAtomicNumber);
  const auto box = cubicBox(kBoxSide);
  std::vector<double> pos(base.size());
  std::vector<double> forces(base.size(), 0.0);
  rgpot::LJPot pot;
  int step = 0;

  // Promotion takes two sightings of the geometry family: the first stamps a
  // phantom reference, the second captures the candidate list at cutoff+skin.
  for (int warmup = 0; warmup < 3; ++warmup) {
    wobble(base, kWarmOffset, step++, pos);
    evalPot(pot, pos, types, box, forces);
  }

  BENCHMARK("LJ force, 500 atoms, pair list hit") {
    wobble(base, kWarmOffset, step++, pos);
    return evalPot(pot, pos, types, box, forces);
  };
}

TEST_CASE("Benchmark: LJPot force via operator()",
          "[.][benchmark][ljpot][api]") {
  const auto base = fccPositions();
  const std::vector<int> types(kNAtoms, kAtomicNumber);
  const std::array<std::array<double, 3>, 3> box{
      {{kBoxSide, 0.0, 0.0}, {0.0, kBoxSide, 0.0}, {0.0, 0.0, kBoxSide}}};
  AtomMatrix positions(kNAtoms, 3);
  for (std::size_t i = 0; i < kNAtoms; ++i) {
    positions(i, 0) = base[3 * i] + kOperatorOffset;
    positions(i, 1) = base[3 * i + 1];
    positions(i, 2) = base[3 * i + 2];
  }
  rgpot::LJPot pot;

  for (int warmup = 0; warmup < 3; ++warmup) {
    pot(positions, types, box);
  }

  BENCHMARK("LJ operator(), 500 atoms, pair list hit") {
    auto [energy, forces, variance] = pot(positions, types, box);
    return energy;
  };
}

TEST_CASE("Benchmark: EMTPot force, Cu tetrahedron", "[.][benchmark][emt]") {
  const std::vector<double> base = {
      8.1925, 8.1925, 8.1925, 10.0, 10.0, 8.1925,
      10.0,   8.1925, 10.0,   8.1925, 10.0, 10.0,
  };
  const std::vector<int> types(4, 29);
  const std::array<double, 9> box = {20, 0, 0, 0, 20, 0, 0, 0, 20};
  std::vector<double> forces(base.size(), 0.0);
  rgpot::EMTPot pot;
  BENCHMARK("EMT force, 4 Cu") {
    return evalPot(pot, base, types, box, forces);
  };
}

TEST_CASE("Benchmark: EAMPot force, two aluminium atoms",
          "[.][benchmark][eam]") {
  const std::vector<double> base = {0.0, 0.0, 0.0, 2.5, 0.0, 0.0};
  const std::vector<int> types(2, 13);
  const std::array<double, 9> box = {20, 0, 0, 0, 20, 0, 0, 0, 20};
  std::vector<double> forces(base.size(), 0.0);
  rgpot::EAMPot pot;
  BENCHMARK("EAM cell-list force, 2 Al") {
    return evalPot(pot, base, types, box, forces);
  };
}

TEST_CASE("Benchmark: MorsePot force, warm pair list",
          "[.][benchmark][morse]") {
  const auto base = fccPositions();
  const std::vector<int> types(kNAtoms, kAtomicNumber);
  const auto box = cubicBox(kBoxSide);
  std::vector<double> pos(base.size());
  std::vector<double> forces(base.size(), 0.0);
  rgpot::MorsePot pot;
  int step = 0;

  for (int warmup = 0; warmup < 3; ++warmup) {
    wobble(base, kWarmOffset, step++, pos);
    evalPot(pot, pos, types, box, forces);
  }

  BENCHMARK("Morse force, 500 atoms, pair list hit") {
    wobble(base, kWarmOffset, step++, pos);
    return evalPot(pot, pos, types, box, forces);
  };
}

TEST_CASE("Benchmark: MorsePot Hessian-vector products at one centre",
          "[.][benchmark][morse][hvp]") {
  // The work of a minimum-mode search between two steps: kProducts
  // products at one geometry along different directions. Forward against
  // the cached centre makes kProducts + 1 force calls, the naive loop and
  // central 2 kProducts, the analytic product none.
  constexpr std::size_t kProducts = 16;
  const auto pos = fccPositions();
  const std::vector<int> types(kNAtoms, kAtomicNumber);
  const auto box = cubicBox(kBoxSide);
  const rgpot::ForceInput in{.nAtoms = kNAtoms,
                             .pos = pos.data(),
                             .atmnrs = types.data(),
                             .box = box.data()};
  std::vector<std::vector<double>> directions(kProducts,
                                              std::vector<double>(pos.size()));
  for (std::size_t d = 0; d < kProducts; ++d) {
    for (std::size_t k = 0; k < pos.size(); ++k) {
      directions[d][k] = std::sin(0.61 * static_cast<double>(k) +
                                  1.9 * static_cast<double>(d));
    }
  }
  std::vector<double> hv(pos.size());
  rgpot::MorsePot pot;
  std::vector<double> forces(pos.size(), 0.0);
  evalPot(pot, pos, types, box, forces);

  auto loop = [&](rgpot::HvpScheme scheme, bool fresh_centre) {
    rgpot::HessianVectorCache cache;
    double acc = 0.0;
    for (const auto &v : directions) {
      if (fresh_centre) {
        cache.invalidate();
      }
      cache.apply(pot, in, v.data(), hv.data(), 0.0, scheme);
      acc += hv[0];
    }
    return acc;
  };
  BENCHMARK("Morse 16 HVPs, 500 atoms, forward, cached centre") {
    return loop(rgpot::HvpScheme::Forward, false);
  };
  BENCHMARK("Morse 16 HVPs, 500 atoms, forward, centre per product") {
    return loop(rgpot::HvpScheme::Forward, true);
  };
  BENCHMARK("Morse 16 HVPs, 500 atoms, central") {
    return loop(rgpot::HvpScheme::Central, false);
  };
  BENCHMARK("Morse 16 HVPs, 500 atoms, analytic") {
    return loop(rgpot::HvpScheme::Analytic, false);
  };
}

TEST_CASE("Benchmark: LJClusterPot force, warm pair list",
          "[.][benchmark][ljcluster]") {
  const auto base = fccPositions();
  const std::vector<int> types(kNAtoms, kAtomicNumber);
  const auto box = cubicBox(kBoxSide);
  std::vector<double> pos(base.size());
  std::vector<double> forces(base.size(), 0.0);
  rgpot::LJClusterPot pot;
  int step = 0;

  // Free boundaries: the lattice is a finite block here, not a periodic
  // crystal, so the pair count sits below the LJPot case at equal N.
  for (int warmup = 0; warmup < 3; ++warmup) {
    wobble(base, kWarmOffset, step++, pos);
    evalPot(pot, pos, types, box, forces);
  }

  BENCHMARK("LJCluster force, 500 atoms, pair list hit") {
    wobble(base, kWarmOffset, step++, pos);
    return evalPot(pot, pos, types, box, forces);
  };
}

TEST_CASE("Benchmark: ZBLPot force, warm pair list", "[.][benchmark][zbl]") {
  const auto base = fccPositions(kDenseBoxSide);
  std::vector<int> types(kNAtoms, 14);
  for (std::size_t i = 1; i < kNAtoms; i += 2) {
    types[i] = 79; // two species exercise the pair-coefficient tables
  }
  const auto box = cubicBox(kDenseBoxSide);
  std::vector<double> pos(base.size());
  std::vector<double> forces(base.size(), 0.0);
  rgpot::ZBLPot pot;
  int step = 0;

  for (int warmup = 0; warmup < 3; ++warmup) {
    wobble(base, kWarmOffset, step++, pos);
    evalPot(pot, pos, types, box, forces);
  }

  BENCHMARK("ZBL force, 500 atoms, pair list hit") {
    wobble(base, kWarmOffset, step++, pos);
    return evalPot(pot, pos, types, box, forces);
  };
}

#ifdef RGPOT_BENCH_CUH2
namespace {

constexpr std::size_t kCuCellsXY = 4;
constexpr std::size_t kCuLayers = 2;
constexpr double kCuLattice = 3.615;
constexpr std::size_t kNCu = 4 * kCuCellsXY * kCuCellsXY * kCuLayers; // 128
constexpr std::size_t kNH = 2;
constexpr double kCuSlabSide = kCuLattice * static_cast<double>(kCuCellsXY);
constexpr double kCuVacuum = 30.0;

/// FCC Cu slab with vacuum along z and an H2 molecule above the top layer.
/// Cu comes first: the Fortran EAM bridge takes species counts, not a
/// per-atom type array.
std::vector<double> cuh2Positions() {
  static const double basis[4][3] = {
      {0.0, 0.0, 0.0}, {0.0, 0.5, 0.5}, {0.5, 0.0, 0.5}, {0.5, 0.5, 0.0}};
  std::vector<double> R;
  R.reserve(3 * (kNCu + kNH));
  for (std::size_t ix = 0; ix < kCuCellsXY; ++ix) {
    for (std::size_t iy = 0; iy < kCuCellsXY; ++iy) {
      for (std::size_t iz = 0; iz < kCuLayers; ++iz) {
        for (const auto &b : basis) {
          R.push_back((static_cast<double>(ix) + b[0]) * kCuLattice);
          R.push_back((static_cast<double>(iy) + b[1]) * kCuLattice);
          R.push_back((static_cast<double>(iz) + b[2]) * kCuLattice);
        }
      }
    }
  }
  const double slabTop = kCuLattice * static_cast<double>(kCuLayers);
  const double zH = slabTop + 2.3;
  const double xC = 0.5 * kCuSlabSide;
  const double yC = 0.5 * kCuSlabSide;
  R.insert(R.end(), {xC - 0.37, yC, zH});
  R.insert(R.end(), {xC + 0.37, yC, zH});
  return R;
}

} // namespace

TEST_CASE("Benchmark: CuH2Pot force", "[.][benchmark][cuh2]") {
  const auto pos = cuh2Positions();
  std::vector<int> types(kNCu, 29);
  types.insert(types.end(), kNH, 1);
  const std::array<double, 9> box{kCuSlabSide, 0.0, 0.0, 0.0,      kCuSlabSide,
                                  0.0,         0.0, 0.0, kCuVacuum};
  std::vector<double> forces(pos.size(), 0.0);
  rgpot::fortranpots::CuH2Pot pot;

  BENCHMARK("CuH2 force, 128 Cu + 2 H") {
    rgpot::ForceInput fi{.nAtoms = kNCu + kNH,
                         .pos = pos.data(),
                         .atmnrs = types.data(),
                         .box = box.data()};
    rgpot::ForceOut fo{.F = forces.data(),
                      .energy = 0.0,
                      .variance = 0.0,
                      .stress = {},
                      .has_stress = 0};
    pot.forceImpl(fi, &fo);
    return fo.energy;
  };
}
#endif
