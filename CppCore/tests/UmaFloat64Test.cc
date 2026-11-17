// Copyright (c) rgpot developers
//
// A float64 package must resolve an energy difference of about 1e-7
// Angstrom on a three-atom line. Coordinates are exact float32 values
// (12.5, 11.5 and 13.5), so casting the positions to float32 makes the
// shifted geometry identical to the original and the energy difference
// is 0. Meson builds the packages and passes their paths in.

#include <cmath>
#include <cstdlib>
#include <string>
#include <vector>

#include <catch2/catch_all.hpp>

#include "rgpot/UmaPot/UmaPot.hpp"

namespace {

std::string package_path(const char *name) {
  const char *path = std::getenv(name);
  if (path == nullptr || path[0] == '\0') {
    FAIL(std::string(name) + " is not set");
  }
  return path;
}

struct Eval {
  double energy = 0.0;
  std::vector<double> force;
};

Eval run(rgpot::UmaPot &pot, std::vector<double> pos) {
  const std::vector<int> z = {6, 7, 1};
  const double box[9] = {25, 0, 0, 0, 25, 0, 0, 0, 25};
  Eval out;
  out.force.assign(pos.size(), 0.0);
  rgpot::ForceOut slot{out.force.data(), 0.0, 0.0};
  const rgpot::ForceInput in{pos.size() / 3, pos.data(), z.data(), box};
  pot.forceImpl(in, &slot);
  out.energy = slot.energy;
  return out;
}

std::vector<double> line_geometry() {
  return {12.5, 12.5, 12.5, 12.5, 12.5, 13.5, 12.5, 12.5, 11.5};
}

rgpot::UmaPot open_pot(const std::string &path) {
  rgpot::UmaConfig cfg;
  cfg.model_path = path;
  cfg.device = "cpu";
  cfg.task_name = "omol";
  cfg.charge = 0;
  cfg.spin = 1;
  return rgpot::UmaPot(cfg);
}

void expect_resolved(const char *env) {
  rgpot::UmaPot pot = open_pot(package_path(env));
  const std::vector<double> base = line_geometry();
  std::vector<double> shifted = base;
  shifted.back() += 1e-7;
  const Eval first = run(pot, base);
  const Eval second = run(pot, shifted);
  const double difference = second.energy - first.energy;
  REQUIRE_THAT(first.energy, Catch::Matchers::WithinAbs(112.5, 1e-9));
  REQUIRE(std::fabs(difference) > 1e-9);
  REQUIRE_THAT(difference, Catch::Matchers::WithinAbs(1e-7, 1e-9));
}

} // namespace

TEST_CASE("float64 package resolves a 1e-7 angstrom shift", "[uma][float64]") {
  expect_resolved("UMA_FLOAT64_PT2");
}

TEST_CASE("model_dtype alone resolves a 1e-7 angstrom shift",
          "[uma][float64]") {
  expect_resolved("UMA_FLOAT64_MODEL_ONLY_PT2");
}

TEST_CASE("float32 package collapses a 1e-7 angstrom shift",
          "[uma][float64]") {
  rgpot::UmaPot pot = open_pot(package_path("UMA_FLOAT32_PT2"));
  const std::vector<double> base = line_geometry();
  std::vector<double> shifted = base;
  shifted.back() += 1e-7;
  const Eval first = run(pot, base);
  const Eval second = run(pot, shifted);
  REQUIRE_THAT(first.energy, Catch::Matchers::WithinAbs(112.5, 1e-4));
  REQUIRE(second.energy - first.energy == 0.0);
}

TEST_CASE("a package with neither dtype key stays float32", "[uma][float64]") {
  rgpot::UmaPot pot = open_pot(package_path("UMA_FLOAT32_NOMETA_PT2"));
  const std::vector<double> base = line_geometry();
  std::vector<double> shifted = base;
  shifted.back() += 1e-7;
  const Eval first = run(pot, base);
  const Eval second = run(pot, shifted);
  REQUIRE(second.energy - first.energy == 0.0);
}

TEST_CASE("disagreeing dtype keys are refused", "[uma][float64]") {
  rgpot::UmaConfig cfg;
  cfg.model_path = package_path("UMA_DTYPE_MISMATCH_PT2");
  cfg.device = "cpu";
  cfg.task_name = "omol";
  cfg.charge = 0;
  cfg.spin = 1;
  rgpot::UmaPot pot(cfg);
  const std::vector<double> base = line_geometry();
  const std::vector<int> z = {6, 7, 1};
  const double box[9] = {25, 0, 0, 0, 25, 0, 0, 0, 25};
  std::vector<double> force(base.size(), 0.0);
  rgpot::ForceOut slot{force.data(), 0.0, 0.0};
  const rgpot::ForceInput in{base.size() / 3, base.data(), z.data(), box};
  REQUIRE_THROWS_WITH(
      pot.forceImpl(in, &slot),
      Catch::Matchers::ContainsSubstring("pos_dtype is float32") &&
          Catch::Matchers::ContainsSubstring("model_dtype is float64"));
}

TEST_CASE("finite difference step of 1e-6 matches the sum", "[uma][float64]") {
  rgpot::UmaPot pot = open_pot(package_path("UMA_FLOAT64_PT2"));
  const std::vector<double> base = line_geometry();
  std::vector<double> shifted = base;
  const double step = 1e-6;
  shifted.back() += step;
  const Eval first = run(pot, base);
  const Eval second = run(pot, shifted);
  const double slope = (second.energy - first.energy) / step;
  REQUIRE_THAT(slope, Catch::Matchers::WithinAbs(1.0, 1e-6));
}

TEST_CASE("a changed neighbour list names both edge shapes",
          "[uma][graph]") {
  // The three-atom line at 6 Å has six edges. Moving one atom past
  // the cutoff changes that count. A static package must not return
  // a force; the error names the shape we passed and the shape the
  // package was exported for.
  rgpot::UmaPot pot = open_pot(package_path("UMA_FLOAT64_PT2"));
  std::vector<double> pos = line_geometry();
  pos.back() = 30.0;
  bool threw = false;
  try {
    run(pot, pos);
  } catch (const std::exception &err) {
    threw = true;
    const std::string message = err.what();
    REQUIRE_THAT(message,
                 Catch::Matchers::ContainsSubstring("edge_index shape is"));
    REQUIRE_THAT(message, Catch::Matchers::ContainsSubstring("[2, 2]"));
    REQUIRE_THAT(message, Catch::Matchers::ContainsSubstring(
                              "the package was exported for"));
    REQUIRE_THAT(message, Catch::Matchers::ContainsSubstring("[2, 6]"));
  }
  REQUIRE(threw);
}

TEST_CASE("a single-system package evaluates a batch one system at a time",
          "[uma][float64][batch]") {
  // batch_max 0: forceBatch is the per-system loop, so two geometries
  // match two forceImpl calls exactly.
  rgpot::UmaPot pot = open_pot(package_path("UMA_FLOAT64_PT2"));
  const std::vector<int> z = {6, 7, 1};
  const double box[9] = {25, 0, 0, 0, 25, 0, 0, 0, 25};
  std::vector<double> first_pos = line_geometry();
  std::vector<double> second_pos = first_pos;
  second_pos[0] += 0.2;
  const Eval first = run(pot, first_pos);
  const Eval second = run(pot, second_pos);
  std::vector<double> force0(first_pos.size(), 0.0);
  std::vector<double> force1(second_pos.size(), 0.0);
  rgpot::ForceInput in[2] = {
      {3, first_pos.data(), z.data(), box},
      {3, second_pos.data(), z.data(), box},
  };
  rgpot::ForceOut out[2] = {
      {force0.data(), 0.0, 0.0},
      {force1.data(), 0.0, 0.0},
  };
  const rgpot::ForceBatch batch{2, in, out};
  pot.forceBatch(batch);
  REQUIRE(out[0].energy == first.energy);
  REQUIRE(out[1].energy == second.energy);
  for (size_t i = 0; i < force0.size(); ++i) {
    REQUIRE(force0[i] == first.force[i]);
    REQUIRE(force1[i] == second.force[i]);
  }
}

TEST_CASE("a band package chunks past batch_max and matches per-system calls",
          "[uma][float64][batch]") {
  // The float64 band package is traced at batch_max 2. Three systems
  // are one chunk of two and one padded chunk of one. Each system
  // matches the single-call entry, which pads a lone system the same way.
  rgpot::UmaPot pot = open_pot(package_path("UMA_FLOAT64_BATCH_PT2"));
  const std::vector<int> z = {6, 7, 1};
  const double box[9] = {25, 0, 0, 0, 25, 0, 0, 0, 25};
  std::vector<double> geoms[3] = {line_geometry(), line_geometry(),
                                  line_geometry()};
  geoms[1][0] += 0.05;
  geoms[2][3] -= 0.04;
  Eval singles[3];
  for (int i = 0; i < 3; ++i)
    singles[i] = run(pot, geoms[i]);
  std::vector<double> forces[3] = {
      std::vector<double>(9),
      std::vector<double>(9),
      std::vector<double>(9),
  };
  rgpot::ForceInput in[3] = {
      {3, geoms[0].data(), z.data(), box},
      {3, geoms[1].data(), z.data(), box},
      {3, geoms[2].data(), z.data(), box},
  };
  rgpot::ForceOut out[3] = {
      {forces[0].data(), 0.0, 0.0},
      {forces[1].data(), 0.0, 0.0},
      {forces[2].data(), 0.0, 0.0},
  };
  const rgpot::ForceBatch batch{3, in, out};
  pot.forceBatch(batch);
  for (int s = 0; s < 3; ++s) {
    REQUIRE_THAT(out[s].energy,
                 Catch::Matchers::WithinAbs(singles[s].energy, 1e-9));
    for (size_t i = 0; i < forces[s].size(); ++i)
      REQUIRE_THAT(forces[s][i],
                   Catch::Matchers::WithinAbs(singles[s].force[i], 1e-9));
  }
}

TEST_CASE("a packed graph accepts one, two and three systems",
          "[uma][float64][batch]") {
  // batch_packed: the real system count is sent. Three systems are one
  // call, not a padded chunk of four, and each matches forceImpl.
  rgpot::UmaPot pot = open_pot(package_path("UMA_PACKED_PT2"));
  const std::vector<int> z = {6, 7, 1};
  const double box[9] = {25, 0, 0, 0, 25, 0, 0, 0, 25};
  std::vector<double> geoms[3] = {line_geometry(), line_geometry(),
                                  line_geometry()};
  geoms[1][0] += 0.2;
  geoms[2].back() += 1e-6;
  Eval singles[3];
  for (int i = 0; i < 3; ++i)
    singles[i] = run(pot, geoms[i]);
  REQUIRE_THAT(singles[0].energy, Catch::Matchers::WithinAbs(112.5, 1e-9));
  REQUIRE_THAT(singles[1].energy - singles[0].energy,
               Catch::Matchers::WithinAbs(0.2, 1e-9));
  std::vector<double> forces[3] = {
      std::vector<double>(9),
      std::vector<double>(9),
      std::vector<double>(9),
  };
  rgpot::ForceInput in[3] = {
      {3, geoms[0].data(), z.data(), box},
      {3, geoms[1].data(), z.data(), box},
      {3, geoms[2].data(), z.data(), box},
  };
  rgpot::ForceOut out[3] = {
      {forces[0].data(), 0.0, 0.0},
      {forces[1].data(), 0.0, 0.0},
      {forces[2].data(), 0.0, 0.0},
  };
  const rgpot::ForceBatch batch{3, in, out};
  pot.forceBatch(batch);
  for (int s = 0; s < 3; ++s) {
    REQUIRE_THAT(out[s].energy,
                 Catch::Matchers::WithinAbs(singles[s].energy, 1e-9));
    for (size_t i = 0; i < 9; ++i)
      REQUIRE_THAT(forces[s][i],
                   Catch::Matchers::WithinAbs(singles[s].force[i], 1e-9));
  }
}

TEST_CASE("batch path resolves a 1e-7 angstrom shift", "[uma][float64]") {
  rgpot::UmaPot pot = open_pot(package_path("UMA_FLOAT64_BATCH_PT2"));
  const std::vector<int> z = {6, 7, 1};
  const double box[9] = {25, 0, 0, 0, 25, 0, 0, 0, 25};
  std::vector<double> base = line_geometry();
  std::vector<double> shifted = base;
  shifted.back() += 1e-7;
  std::vector<double> force0(9, 0.0), force1(9, 0.0);
  rgpot::ForceInput in[2] = {
      {3, base.data(), z.data(), box},
      {3, shifted.data(), z.data(), box},
  };
  rgpot::ForceOut out[2] = {
      {force0.data(), 0.0, 0.0},
      {force1.data(), 0.0, 0.0},
  };
  const rgpot::ForceBatch batch{2, in, out};
  pot.forceBatchImpl(batch);
  REQUIRE_THAT(out[0].energy, Catch::Matchers::WithinAbs(112.5, 1e-9));
  REQUIRE_THAT(out[1].energy - out[0].energy,
               Catch::Matchers::WithinAbs(1e-7, 1e-9));
}
