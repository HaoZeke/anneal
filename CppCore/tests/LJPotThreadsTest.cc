// MIT License
// Copyright 2023--present rgpot developers
//
// Concurrent const evaluation of one shared LJPot instance across many
// threads: energy, forces, stress, per-atom terms and Hessian-vector
// products must match serial evaluations bit for bit without locks or races.

#include <algorithm>
#include <array>
#include <atomic>
#include <bit>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <new>
#include <thread>
#include <vector>

#include <catch2/catch_all.hpp>

#include "rgpot/ForceStructs.hpp"
#include "rgpot/HessianVector.hpp"
#include "rgpot/LennardJones/LJPot.hpp"
#include "rgpot/PerAtom.hpp"

namespace {

struct Frame {
  std::vector<double> pos;
  std::vector<double> nudged;
  std::vector<int> z;
  std::array<double, 9> box{};
  std::vector<double> direction;
};

struct Expected {
  double energy = 0.0;
  std::vector<double> forces;
  std::array<double, 9> stress{};
  int has_stress = 0;

  double nudged_energy = 0.0;
  std::vector<double> nudged_forces;

  double per_atom_energy = 0.0;
  std::vector<double> per_atom_forces;
  std::vector<double> atom_energies;
  std::vector<double> atom_virials;

  std::vector<double> hvp_forward;
  std::vector<double> hvp_central;
};

bool sameBits(double a, double b) noexcept {
  return std::bit_cast<std::uint64_t>(a) == std::bit_cast<std::uint64_t>(b);
}

bool sameVectorBits(const std::vector<double> &a,
                    const std::vector<double> &b) noexcept {
  if (a.size() != b.size()) {
    return false;
  }
  for (std::size_t i = 0; i < a.size(); ++i) {
    if (!sameBits(a[i], b[i])) {
      return false;
    }
  }
  return true;
}

Frame makeFccFrame(int nx, int ny, int nz, double a, bool sheared,
                   double phase_seed) {
  constexpr double basis[4][3] = {
      {0.0, 0.0, 0.0},
      {0.0, 0.5, 0.5},
      {0.5, 0.0, 0.5},
      {0.5, 0.5, 0.0},
  };
  const double lx = static_cast<double>(nx) * a;
  const double ly = static_cast<double>(ny) * a;
  const double lz = static_cast<double>(nz) * a;

  Frame f;
  if (sheared) {
    f.box = {lx, 0.0, 0.0, 0.0625 * lx, ly, 0.0, 0.03125 * lx, -0.03125 * ly,
             lz};
  } else {
    f.box = {lx, 0.0, 0.0, 0.0, ly, 0.0, 0.0, 0.0, lz};
  }

  for (int ix = 0; ix < nx; ++ix) {
    for (int iy = 0; iy < ny; ++iy) {
      for (int iz = 0; iz < nz; ++iz) {
        for (const auto &b : basis) {
          const double fx = (static_cast<double>(ix) + b[0]) /
                            static_cast<double>(nx);
          const double fy = (static_cast<double>(iy) + b[1]) /
                            static_cast<double>(ny);
          const double fz = (static_cast<double>(iz) + b[2]) /
                            static_cast<double>(nz);
          for (std::size_t c = 0; c < 3; ++c) {
            const auto slot = static_cast<double>(f.pos.size());
            const double cell_coord =
                fx * f.box[c] + fy * f.box[3 + c] + fz * f.box[6 + c];
            const double jitter =
                0.03125 * std::sin(1.7 * slot + phase_seed);
            const double small_step =
                0.0625 * std::cos(1.1 * slot + phase_seed);
            f.pos.push_back(cell_coord + jitter);
            f.nudged.push_back(cell_coord + jitter + small_step);
            f.direction.push_back(std::sin(0.9 * slot + 0.5 + phase_seed));
          }
          f.z.push_back(18);
        }
      }
    }
  }
  return f;
}

Expected computeSerialReference(const rgpot::LJConfig &cfg,
                                rgpot::nlist::Mode mode, const Frame &frame,
                                double hvp_step) {
  rgpot::LJPot pot_init(cfg);
  pot_init.setNeighbourMode(mode);
  const rgpot::LJPot &pot = pot_init;

  const std::size_t n = frame.z.size();
  const rgpot::ForceInput in{
      .nAtoms = n,
      .pos = frame.pos.data(),
      .atmnrs = frame.z.data(),
      .box = frame.box.data(),
  };
  const rgpot::ForceInput in_nudged{
      .nAtoms = n,
      .pos = frame.nudged.data(),
      .atmnrs = frame.z.data(),
      .box = frame.box.data(),
  };

  Expected ref;
  ref.forces.assign(3 * n, 0.0);
  rgpot::ForceOut out{
      .F = ref.forces.data(),
      .energy = 0.0,
      .variance = 0.0,
      .stress = {},
      .has_stress = 0,
  };
  pot.forceImpl(in, &out);
  ref.energy = out.energy;
  ref.has_stress = out.has_stress;
  std::copy(std::begin(out.stress), std::end(out.stress), ref.stress.begin());

  ref.nudged_forces.assign(3 * n, 0.0);
  rgpot::ForceOut out_nudged{
      .F = ref.nudged_forces.data(),
      .energy = 0.0,
      .variance = 0.0,
      .stress = {},
      .has_stress = 0,
  };
  pot.forceImpl(in_nudged, &out_nudged);
  ref.nudged_energy = out_nudged.energy;

  ref.per_atom_forces.assign(3 * n, 0.0);
  ref.atom_energies.assign(n, 0.0);
  ref.atom_virials.assign(6 * n, 0.0);
  rgpot::ForceOut out_atoms{
      .F = ref.per_atom_forces.data(),
      .energy = 0.0,
      .variance = 0.0,
      .stress = {},
      .has_stress = 0,
  };
  pot.forcePerAtom(
      in, &out_atoms,
      rgpot::PerAtomOut{ref.atom_energies.data(), ref.atom_virials.data()});
  ref.per_atom_energy = out_atoms.energy;

  rgpot::HessianVectorCache hvp;
  ref.hvp_forward.assign(3 * n, 0.0);
  ref.hvp_central.assign(3 * n, 0.0);
  hvp.apply(pot, in, frame.direction.data(), ref.hvp_forward.data(), hvp_step,
            rgpot::HvpScheme::Forward);
  hvp.apply(pot, in, frame.direction.data(), ref.hvp_central.data(), hvp_step,
            rgpot::HvpScheme::Central);
  return ref;
}

void runConcurrentSharedLj(const rgpot::LJConfig &cfg,
                           rgpot::nlist::Mode mode) {
  constexpr std::size_t kThreads = 12;
  constexpr int kRounds = 10;
  const double hvp_step = 1.0 / 1024.0;
  const double a_lj = std::pow(2.0, 1.0 / 6.0) * std::sqrt(2.0);

  // Distinct atom counts, aspect ratios and shear so concurrent updates on a
  // shared neighbour list would reallocate vectors and corrupt xyz_ pointers.
  const std::vector<Frame> frames = {
      makeFccFrame(2, 2, 2, a_lj, false, 0.1),
      makeFccFrame(3, 2, 2, a_lj, false, 0.4),
      makeFccFrame(3, 3, 2, a_lj, true, 0.7),
      makeFccFrame(3, 3, 3, a_lj, false, 1.1),
      makeFccFrame(4, 3, 3, a_lj, true, 1.5),
      makeFccFrame(4, 4, 4, a_lj, false, 1.9),
  };

  std::vector<Expected> refs;
  refs.reserve(frames.size());
  for (const Frame &f : frames) {
    refs.push_back(computeSerialReference(cfg, mode, f, hvp_step));
  }

  rgpot::LJPot shared_mut(cfg);
  shared_mut.setNeighbourMode(mode);
  const rgpot::LJPot &shared = shared_mut;

  rgpot::HessianVectorCache shared_hvp;
  std::atomic<std::size_t> ready{0};
  std::atomic<bool> go{false};
  std::atomic<std::size_t> mismatches{0};

  std::vector<std::thread> workers;
  workers.reserve(kThreads);
  for (std::size_t tid = 0; tid < kThreads; ++tid) {
    workers.emplace_back([&, tid] {
      rgpot::HessianVectorCache local_hvp;
      ready.fetch_add(1, std::memory_order_acq_rel);
      while (!go.load(std::memory_order_acquire)) {
        std::this_thread::yield();
      }

      for (int round = 0; round < kRounds; ++round) {
        const std::size_t idx =
            (tid + static_cast<std::size_t>(round)) % frames.size();
        const Frame &frame = frames[idx];
        const Expected &want = refs[idx];
        const std::size_t n = frame.z.size();

        const rgpot::ForceInput in{
            .nAtoms = n,
            .pos = frame.pos.data(),
            .atmnrs = frame.z.data(),
            .box = frame.box.data(),
        };
        const rgpot::ForceInput in_nudged{
            .nAtoms = n,
            .pos = frame.nudged.data(),
            .atmnrs = frame.z.data(),
            .box = frame.box.data(),
        };

        std::vector<double> forces(3 * n, 0.0);
        rgpot::ForceOut out{
            .F = forces.data(),
            .energy = 0.0,
            .variance = 0.0,
            .stress = {},
            .has_stress = 0,
        };
        shared.forceImpl(in, &out);
        if (!sameBits(out.energy, want.energy) ||
            out.has_stress != want.has_stress ||
            !sameVectorBits(forces, want.forces)) {
          mismatches.fetch_add(1, std::memory_order_relaxed);
        }
        for (std::size_t s = 0; s < 9; ++s) {
          if (!sameBits(out.stress[s], want.stress[s])) {
            mismatches.fetch_add(1, std::memory_order_relaxed);
          }
        }

        // Second evaluation within half the 1 A skin exercises Verlet reuse on
        // this thread while other threads rebuild on different atom counts.
        std::vector<double> nudged_forces(3 * n, 0.0);
        rgpot::ForceOut out_nudged{
            .F = nudged_forces.data(),
            .energy = 0.0,
            .variance = 0.0,
            .stress = {},
            .has_stress = 0,
        };
        shared.forceImpl(in_nudged, &out_nudged);
        if (!sameBits(out_nudged.energy, want.nudged_energy) ||
            !sameVectorBits(nudged_forces, want.nudged_forces)) {
          mismatches.fetch_add(1, std::memory_order_relaxed);
        }

        std::vector<double> pa_forces(3 * n, 0.0);
        std::vector<double> atom_e(n, 0.0);
        std::vector<double> atom_v(6 * n, 0.0);
        rgpot::ForceOut out_pa{
            .F = pa_forces.data(),
            .energy = 0.0,
            .variance = 0.0,
            .stress = {},
            .has_stress = 0,
        };
        shared.forcePerAtom(in, &out_pa,
                            rgpot::PerAtomOut{atom_e.data(), atom_v.data()});
        if (!sameBits(out_pa.energy, want.per_atom_energy) ||
            !sameVectorBits(pa_forces, want.per_atom_forces) ||
            !sameVectorBits(atom_e, want.atom_energies) ||
            !sameVectorBits(atom_v, want.atom_virials)) {
          mismatches.fetch_add(1, std::memory_order_relaxed);
        }

        std::vector<double> hv_fwd(3 * n, 0.0);
        std::vector<double> hv_cen(3 * n, 0.0);
        local_hvp.invalidate();
        local_hvp.apply(shared, in, frame.direction.data(), hv_fwd.data(),
                        hvp_step, rgpot::HvpScheme::Forward);
        local_hvp.apply(shared, in, frame.direction.data(), hv_cen.data(),
                        hvp_step, rgpot::HvpScheme::Central);
        if (!sameVectorBits(hv_fwd, want.hvp_forward) ||
            !sameVectorBits(hv_cen, want.hvp_central)) {
          mismatches.fetch_add(1, std::memory_order_relaxed);
        }

        // Also exercise a shared HessianVectorCache over the shared const LJPot
        // on one common geometry (frames[0]).
        const Frame &common = frames[0];
        const Expected &common_want = refs[0];
        const rgpot::ForceInput common_in{
            .nAtoms = common.z.size(),
            .pos = common.pos.data(),
            .atmnrs = common.z.data(),
            .box = common.box.data(),
        };
        std::vector<double> common_hv(3 * common.z.size(), 0.0);
        shared_hvp.apply(shared, common_in, common.direction.data(),
                         common_hv.data(), hvp_step, rgpot::HvpScheme::Forward);
        if (!sameVectorBits(common_hv, common_want.hvp_forward)) {
          mismatches.fetch_add(1, std::memory_order_relaxed);
        }
      }
    });
  }

  while (ready.load(std::memory_order_acquire) < kThreads) {
    std::this_thread::yield();
  }
  go.store(true, std::memory_order_release);
  for (std::thread &w : workers) {
    w.join();
  }

  CHECK(shared.caps().reentrancy == rgpot::Reentrancy::SharedInstance);
  REQUIRE(mismatches.load() == 0);
}

} // namespace

TEST_CASE("shared const LJPot evaluates energy, forces and HVP across 12 threads",
          "[lj][threads]") {
  SECTION("shifted truncation, Auto") {
    runConcurrentSharedLj(
        rgpot::LJConfig{.u0 = 1.0, .cutoff = 3.2, .psi = 1.0},
        rgpot::nlist::Mode::Auto);
  }
  SECTION("shifted truncation, Stored") {
    runConcurrentSharedLj(
        rgpot::LJConfig{.u0 = 1.0, .cutoff = 3.2, .psi = 1.0},
        rgpot::nlist::Mode::Stored);
  }
  SECTION("shifted truncation, Stream") {
    runConcurrentSharedLj(
        rgpot::LJConfig{.u0 = 1.0, .cutoff = 3.2, .psi = 1.0},
        rgpot::nlist::Mode::Stream);
  }
  SECTION("quintic switch, Stored") {
    runConcurrentSharedLj(rgpot::LJConfig{.u0 = 0.85,
                                          .cutoff = 3.4,
                                          .psi = 1.05,
                                          .switch_width = 0.75},
                          rgpot::nlist::Mode::Stored);
  }
}

TEST_CASE("LJPot per-thread scratch isolates instances and releases on destroy",
          "[lj][threads][skin]") {
  const double a_lj = std::pow(2.0, 1.0 / 6.0) * std::sqrt(2.0);
  const Frame frame = makeFccFrame(2, 2, 2, a_lj, false, 0.25);
  const rgpot::ForceInput in{
      .nAtoms = frame.z.size(),
      .pos = frame.pos.data(),
      .atmnrs = frame.z.data(),
      .box = frame.box.data(),
  };
  const rgpot::ForceInput in_nudged{
      .nAtoms = frame.z.size(),
      .pos = frame.nudged.data(),
      .atmnrs = frame.z.data(),
      .box = frame.box.data(),
  };

  rgpot::LJConfig cfg{.u0 = 1.0, .cutoff = 3.0, .psi = 1.0};
  rgpot::LJPot stored_a(cfg);
  rgpot::LJPot stored_b(rgpot::LJConfig{.u0 = 1.0, .cutoff = 2.6, .psi = 1.0});
  rgpot::LJPot streamed(cfg);
  stored_a.setNeighbourMode(rgpot::nlist::Mode::Stored);
  stored_b.setNeighbourMode(rgpot::nlist::Mode::Stored);
  streamed.setNeighbourMode(rgpot::nlist::Mode::Stream);

  std::vector<double> f(3 * frame.z.size(), 0.0);
  rgpot::ForceOut out{
      .F = f.data(),
      .energy = 0.0,
      .variance = 0.0,
      .stress = {},
      .has_stress = 0,
  };

  stored_a.forceImpl(in, &out);
  REQUIRE_FALSE(stored_a.listReused());
  const std::size_t pairs_a = stored_a.retainedPairs();
  REQUIRE(pairs_a > 0);

  stored_b.forceImpl(in, &out);
  streamed.forceImpl(in, &out);
  REQUIRE(streamed.retainedPairs() == 0);
  REQUIRE(stored_a.retainedPairs() == pairs_a);

  stored_a.forceImpl(in_nudged, &out);
  REQUIRE(stored_a.listReused());

  // Destroying an instance and constructing another at the exact same address
  // must not inherit the destroyed instance's Verlet list.
  alignas(rgpot::LJPot) unsigned char storage[sizeof(rgpot::LJPot)];
  auto *first = new (storage) rgpot::LJPot(cfg);
  first->setNeighbourMode(rgpot::nlist::Mode::Stored);
  first->forceImpl(in, &out);
  first->forceImpl(in_nudged, &out);
  REQUIRE(first->listReused());
  first->~LJPot();

  auto *second = new (storage) rgpot::LJPot(cfg);
  second->setNeighbourMode(rgpot::nlist::Mode::Stored);
  REQUIRE_FALSE(second->listReused());
  REQUIRE(second->retainedPairs() == 0);
  second->forceImpl(in_nudged, &out);
  REQUIRE_FALSE(second->listReused());
  second->~LJPot();
}
