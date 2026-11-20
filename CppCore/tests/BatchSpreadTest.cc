// MIT License
// Copyright 2023--present rgpot developers
//
// Small systems stay below kTeamMinAtoms, so a one-thread loop leaves the
// rest of the team idle. A batch of those systems has to spread across
// the team and still match the one-thread bits.

#include <array>
#include <bit>
#include <chrono>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <sstream>
#include <string>
#include <vector>

#include <catch2/catch_all.hpp>

#include "RestoreEnv.hpp"
#include "lsan_libomp.hpp"
#include "rgpot/LennardJones/LJPot.hpp"
#include "rgpot/Slices.hpp"
#include "rgpot/StillingerWeber/SWSi.hpp"
#include "rgpot/Tersoff/TersoffSiC.hpp"

namespace {

constexpr std::size_t kSystems = 16;
constexpr std::size_t kAtoms = 4;

bool sameBits(double a, double b) noexcept {
  return std::bit_cast<std::uint64_t>(a) == std::bit_cast<std::uint64_t>(b);
}

void requireBits(const std::vector<double> &got,
                 const std::vector<double> &ref) {
  REQUIRE(got.size() == ref.size());
  for (std::size_t i = 0; i < got.size(); ++i) {
    INFO("index " << i);
    REQUIRE(sameBits(got[i], ref[i]));
  }
}

std::vector<double> makePositions(std::size_t systems, std::size_t atoms,
                                  double shift) {
  std::vector<double> pos(systems * atoms * 3, 0.0);
  for (std::size_t s = 0; s < systems; ++s) {
    for (std::size_t a = 0; a < atoms; ++a) {
      const std::size_t b = (s * atoms + a) * 3;
      pos[b] = 1.65 * static_cast<double>(a) + shift * static_cast<double>(s);
      pos[b + 1] =
          0.4 * static_cast<double>(a % 2) + 0.01 * static_cast<double>(s);
      pos[b + 2] = 0.25 * static_cast<double>((a * a) % 3);
    }
  }
  return pos;
}

const double kBox[9] = {250.0, 0.0, 0.0, 0.0, 250.0, 0.0, 0.0, 0.0, 250.0};

struct Grad {
  std::vector<double> energy;
  std::vector<double> gradient;
  std::uint64_t teams = 0;
};

template <typename Pot, typename Run>
Grad runGradient(const std::vector<double> &pos, const int *species,
                 const char *threads, Run &&run) {
  rgpot::test_support::RestoreEnv env("OMP_NUM_THREADS");
  env.set(threads);
#if defined(_OPENMP)
  REQUIRE(rgpot::threadsRequested() == std::stoi(threads));
#else
  REQUIRE(rgpot::threadsRequested() == 1);
#endif
  Pot pot;
  Grad out;
  out.energy.assign(kSystems, 0.0);
  out.gradient.assign(kSystems * kAtoms * 3, 0.0);
  const auto before = rgpot::teamsOpened();
  run(pot, pos, species, out);
  out.teams = rgpot::teamsOpened() - before;
  for (double e : out.energy) {
    REQUIRE(std::isfinite(e));
  }
  return out;
}

void requireTeams(const Grad &one, const Grad &more) {
  REQUIRE(one.teams == 0);
#if defined(_OPENMP)
  REQUIRE(more.teams == 1);
#else
  REQUIRE(more.teams == 0);
#endif
  requireBits(more.energy, one.energy);
  requireBits(more.gradient, one.gradient);
}

std::string ratioText(const char *label, std::chrono::nanoseconds serial,
                      std::chrono::nanoseconds threaded) {
  std::ostringstream os;
  os << "smoke wall time ratio for " << label << " (one thread over four) ";
  if (threaded.count() > 0) {
    os << static_cast<double>(serial.count()) /
              static_cast<double>(threaded.count());
  } else {
    os << "undefined";
  }
  os << " serial_ns " << serial.count() << " threaded_ns " << threaded.count();
  return os.str();
}

} // namespace

TEST_CASE("small Stillinger-Weber batches match across thread counts",
          "[batch]") {
  const int species[kAtoms] = {14, 14, 14, 14};
  for (const double shift : {3.0, 0.02}) {
    const std::vector<double> pos = makePositions(kSystems, kAtoms, shift);
    auto run = [&](rgpot::StillingerWeberSi &pot, const std::vector<double> &p,
                   const int *z, Grad &out) {
      pot.energyAndGradientBatch(kSystems, p.data(), z, kBox, kAtoms,
                                 out.energy.data(), out.gradient.data());
    };
    const Grad one =
        runGradient<rgpot::StillingerWeberSi>(pos, species, "1", run);
    const Grad two =
        runGradient<rgpot::StillingerWeberSi>(pos, species, "2", run);
    const Grad four =
        runGradient<rgpot::StillingerWeberSi>(pos, species, "4", run);
    requireTeams(one, two);
    requireTeams(one, four);

    for (std::size_t s = 0; s < kSystems; ++s) {
      rgpot::StillingerWeberSi fresh;
      double energy = 0.0;
      std::vector<double> gradient(kAtoms * 3, 0.0);
      fresh.energyAndGradient(pos.data() + s * kAtoms * 3, species, kBox,
                              kAtoms, &energy, gradient.data());
      INFO("system " << s);
      REQUIRE(sameBits(energy, one.energy[s]));
      for (std::size_t g = 0; g < gradient.size(); ++g) {
        REQUIRE(sameBits(gradient[g], one.gradient[s * kAtoms * 3 + g]));
      }
    }
  }

  const std::vector<double> pos = makePositions(kSystems, kAtoms, 3.0);
  auto timeAt = [&](const char *threads) {
    rgpot::test_support::RestoreEnv env("OMP_NUM_THREADS");
    env.set(threads);
    rgpot::StillingerWeberSi pot;
    std::vector<double> energy(kSystems, 0.0);
    std::vector<double> gradient(kSystems * kAtoms * 3, 0.0);
    const auto start = std::chrono::steady_clock::now();
    for (int rep = 0; rep < 20; ++rep) {
      pot.energyAndGradientBatch(kSystems, pos.data(), species, kBox, kAtoms,
                                 energy.data(), gradient.data());
    }
    return std::chrono::steady_clock::now() - start;
  };
  const auto serialTime = timeAt("1");
  const auto threadedTime = timeAt("4");
  WARN(ratioText("Stillinger-Weber", serialTime, threadedTime));
}

TEST_CASE("small Tersoff batches match across thread counts", "[batch]") {
  const int species[kAtoms] = {14, 14, 6, 14};
  for (const double shift : {3.0, 0.02}) {
    const std::vector<double> pos = makePositions(kSystems, kAtoms, shift);
    auto run = [&](rgpot::TersoffSiC &pot, const std::vector<double> &p,
                   const int *z, Grad &out) {
      pot.energyAndGradientBatch(kSystems, p.data(), z, kBox, kAtoms,
                                 out.energy.data(), out.gradient.data());
    };
    const Grad one = runGradient<rgpot::TersoffSiC>(pos, species, "1", run);
    const Grad two = runGradient<rgpot::TersoffSiC>(pos, species, "2", run);
    const Grad four = runGradient<rgpot::TersoffSiC>(pos, species, "4", run);
    requireTeams(one, two);
    requireTeams(one, four);
    for (std::size_t s = 0; s < kSystems; ++s) {
      rgpot::TersoffSiC fresh;
      double energy = 0.0;
      std::vector<double> gradient(kAtoms * 3, 0.0);
      fresh.energyAndGradient(pos.data() + s * kAtoms * 3, species, kBox,
                              kAtoms, &energy, gradient.data());
      INFO("system " << s);
      REQUIRE(sameBits(energy, one.energy[s]));
      for (std::size_t g = 0; g < gradient.size(); ++g) {
        REQUIRE(sameBits(gradient[g], one.gradient[s * kAtoms * 3 + g]));
      }
    }
  }
}

TEST_CASE("small default-path batches match across thread counts", "[batch]") {
  const std::vector<double> pos = makePositions(kSystems, kAtoms, 3.0);
  const std::vector<double> close = makePositions(kSystems, kAtoms, 0.02);
  auto run = [](const std::vector<double> &coordinates, const char *threads) {
    rgpot::test_support::RestoreEnv env("OMP_NUM_THREADS");
    env.set(threads);
#if defined(_OPENMP)
    REQUIRE(rgpot::threadsRequested() == std::stoi(threads));
#else
    REQUIRE(rgpot::threadsRequested() == 1);
#endif
    std::vector<int> species(kAtoms, 18);
    const std::array<double, 9> box = {250.0, 0.0, 0.0, 0.0,  250.0,
                                       0.0,   0.0, 0.0, 250.0};
    std::vector<double> forces(kSystems * kAtoms * 3, 0.0);
    std::vector<rgpot::ForceInput> inputs;
    std::vector<rgpot::ForceOut> outputs(kSystems);
    inputs.reserve(kSystems);
    for (std::size_t s = 0; s < kSystems; ++s) {
      inputs.push_back(rgpot::ForceInput{
          .nAtoms = kAtoms,
          .pos = coordinates.data() + s * kAtoms * 3,
          .atmnrs = species.data(),
          .box = box.data(),
      });
      outputs[s] = rgpot::ForceOut{
          .F = forces.data() + s * kAtoms * 3,
          .energy = 0.0,
          .variance = 0.0,
          .stress = {},
          .has_stress = 0,
      };
    }
    rgpot::LJPot pot;
    REQUIRE(pot.caps().reentrancy == rgpot::Reentrancy::SharedInstance);
    const auto before = rgpot::teamsOpened();
    pot.forceBatch(rgpot::ForceBatch{
        .nSystems = kSystems,
        .in = inputs.data(),
        .out = outputs.data(),
    });
    struct Got {
      std::vector<double> energy;
      std::vector<double> force;
      std::vector<double> stress;
      std::vector<double> variance;
      std::uint64_t teams = 0;
    } got;
    got.teams = rgpot::teamsOpened() - before;
    got.force = std::move(forces);
    got.energy.resize(kSystems);
    got.variance.resize(kSystems);
    got.stress.resize(kSystems * 9);
    for (std::size_t s = 0; s < kSystems; ++s) {
      got.energy[s] = outputs[s].energy;
      got.variance[s] = outputs[s].variance;
      REQUIRE(std::isfinite(got.energy[s]));
      REQUIRE(outputs[s].has_stress == 1);
      for (int c = 0; c < 9; ++c) {
        got.stress[s * 9 + static_cast<std::size_t>(c)] = outputs[s].stress[c];
      }
    }
    return got;
  };

  for (const std::vector<double> *coordinates : {&pos, &close}) {
    const auto one = run(*coordinates, "1");
    const auto two = run(*coordinates, "2");
    const auto four = run(*coordinates, "4");
    REQUIRE(one.teams == 0);
#if defined(_OPENMP)
    REQUIRE(two.teams == 1);
    REQUIRE(four.teams == 1);
#else
    REQUIRE(two.teams == 0);
    REQUIRE(four.teams == 0);
#endif
    requireBits(two.energy, one.energy);
    requireBits(four.energy, one.energy);
    requireBits(two.force, one.force);
    requireBits(four.force, one.force);
    requireBits(two.stress, one.stress);
    requireBits(four.stress, one.stress);
    requireBits(two.variance, one.variance);
    requireBits(four.variance, one.variance);

    rgpot::LJPot fresh;
    std::vector<int> species(kAtoms, 18);
    const std::array<double, 9> box = {250.0, 0.0, 0.0, 0.0,  250.0,
                                       0.0,   0.0, 0.0, 250.0};
    for (std::size_t s = 0; s < kSystems; ++s) {
      std::vector<double> force(kAtoms * 3, 0.0);
      rgpot::ForceOut out{
          .F = force.data(),
          .energy = 0.0,
          .variance = 0.0,
          .stress = {},
          .has_stress = 0,
      };
      fresh.forceImpl(
          rgpot::ForceInput{.nAtoms = kAtoms,
                            .pos = coordinates->data() + s * kAtoms * 3,
                            .atmnrs = species.data(),
                            .box = box.data()},
          &out);
      INFO("system " << s);
      REQUIRE(sameBits(out.energy, one.energy[s]));
      requireBits(force, std::vector<double>(
                             one.force.begin() +
                                 static_cast<std::ptrdiff_t>(s * kAtoms * 3),
                             one.force.begin() + static_cast<std::ptrdiff_t>(
                                                     (s + 1) * kAtoms * 3)));
    }
  }
}

TEST_CASE("batch spread stops at the in-system crossover", "[batch]") {
  rgpot::test_support::RestoreEnv env("OMP_NUM_THREADS");
  env.set("4");
#if defined(_OPENMP)
  REQUIRE(rgpot::threadsRequested() == 4);
#else
  REQUIRE(rgpot::threadsRequested() == 1);
#endif

  std::size_t calls = 0;
  const auto largeBefore = rgpot::teamsOpened();
  rgpot::forEachBatchSystem(
      8, [](std::size_t) { return rgpot::kTeamMinAtoms; },
      [&](std::size_t) { ++calls; });
  REQUIRE(calls == 8);
  REQUIRE(rgpot::teamsOpened() == largeBefore);

  std::vector<int> seen(kSystems, 0);
  const auto smallBefore = rgpot::teamsOpened();
  rgpot::forEachBatchSystem(
      kSystems, [](std::size_t) { return rgpot::kTeamMinAtoms - 1; },
      [&](std::size_t i) { seen[i] += 1; });
#if defined(_OPENMP)
  REQUIRE(rgpot::teamsOpened() == smallBefore + 1);
#else
  REQUIRE(rgpot::teamsOpened() == smallBefore);
#endif
  for (int hit : seen) {
    REQUIRE(hit == 1);
  }

  const auto oneBefore = rgpot::teamsOpened();
  rgpot::forEachBatchSystem(
      1, [](std::size_t) { return std::size_t{2}; }, [&](std::size_t) {});
  REQUIRE(rgpot::teamsOpened() == oneBefore);

  const auto mixedBefore = rgpot::teamsOpened();
  rgpot::forEachBatchSystem(
      4,
      [](std::size_t i) {
        return i == 2 ? rgpot::kTeamMinAtoms : std::size_t{3};
      },
      [&](std::size_t) {});
  REQUIRE(rgpot::teamsOpened() == mixedBefore);
}
