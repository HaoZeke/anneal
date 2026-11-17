// Copyright (c) rgpot developers
//
// A universal model for atoms package compiled for one neighbour list
// must not return a force when that list changes size. The error names
// the shape that was passed and the shape the package was exported for.
// A package exported with a symbolic edge count returns finite forces
// that match the eager gather.

#include <cmath>
#include <cstdlib>
#include <string>
#include <unordered_map>
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

rgpot::UmaPot open_pot(const std::string &path) {
  rgpot::UmaConfig cfg;
  cfg.model_path = path;
  cfg.device = "cpu";
  cfg.task_name = "omol";
  cfg.charge = 0;
  cfg.spin = 1;
  return rgpot::UmaPot(cfg);
}

struct Eval {
  double energy = 0.0;
  std::vector<double> force;
};

Eval run(rgpot::UmaPot &pot, const std::vector<double> &pos) {
  std::vector<int> z(pos.size() / 3, 6);
  const double box[9] = {40, 0, 0, 0, 40, 0, 0, 0, 40};
  Eval out;
  out.force.assign(pos.size(), 0.0);
  rgpot::ForceOut slot{out.force.data(), 0.0, 0.0};
  const rgpot::ForceInput in{pos.size() / 3, pos.data(), z.data(), box};
  pot.forceImpl(in, &slot);
  out.energy = slot.energy;
  return out;
}

std::vector<double> line(int n, double spacing) {
  std::vector<double> pos(static_cast<size_t>(n) * 3);
  for (int i = 0; i < n; ++i) {
    pos[static_cast<size_t>(3 * i)] = 12.5;
    pos[static_cast<size_t>(3 * i + 1)] = 12.5;
    pos[static_cast<size_t>(3 * i + 2)] = 12.5 + spacing * i;
  }
  return pos;
}

} // namespace

TEST_CASE("tetrazine-sized static graph names both edge shapes",
          "[uma][graph]") {
  // Eight atoms, every pair inside the cutoff: 8 * 7 = 56 edges.
  // Moving one atom past the cutoff changes the neighbour list.
  rgpot::UmaPot pot = open_pot(package_path("UMA_GATHER_STATIC_PT2"));
  std::vector<double> pos = line(8, 0.5);
  pos.back() = 40.0;
  bool threw = false;
  try {
    run(pot, pos);
  } catch (const rgpot::UmaShapeError &err) {
    threw = true;
    const std::string message = err.what();
    REQUIRE(err.field() == "edge_index");
    REQUIRE_THAT(message,
                 Catch::Matchers::ContainsSubstring("edge_index shape is"));
    REQUIRE_THAT(message, Catch::Matchers::ContainsSubstring("[2, 42]"));
    REQUIRE_THAT(message, Catch::Matchers::ContainsSubstring(
                              "the package was exported for [2, 56]"));
  }
  REQUIRE(threw);
}

TEST_CASE("dynamic gather matches eager forces when the neighbour list changes",
          "[uma][graph]") {
  rgpot::UmaPot pot = open_pot(package_path("UMA_GATHER_DYNAMIC_PT2"));
  const Eval compact = run(pot, line(4, 1.0));
  std::vector<double> stretched = line(4, 1.0);
  stretched.back() = 40.0;
  const Eval far = run(pot, stretched);

  REQUIRE(std::isfinite(compact.energy));
  REQUIRE(std::isfinite(far.energy));
  REQUIRE_THAT(compact.energy, Catch::Matchers::WithinAbs(468.0, 1e-6));
  REQUIRE_THAT(far.energy, Catch::Matchers::WithinAbs(231.0, 1e-6));
  for (int i = 0; i < 4; ++i) {
    for (int k = 0; k < 3; ++k) {
      REQUIRE_THAT(compact.force[static_cast<size_t>(3 * i + k)],
                   Catch::Matchers::WithinAbs(-3.0, 1e-6));
    }
  }
  for (int k = 0; k < 3; ++k) {
    REQUIRE_THAT(far.force[static_cast<size_t>(k)],
                 Catch::Matchers::WithinAbs(-2.0, 1e-6));
    REQUIRE_THAT(far.force[9 + static_cast<size_t>(k)],
                 Catch::Matchers::WithinAbs(0.0, 1e-6));
  }

  // A 1e-4 angstrom step does not cross the cutoff, so the compiled
  // edge count stays put. The reported force is minus dE/dx.
  std::vector<double> nudged = line(4, 1.0);
  const double step = 1e-4;
  nudged[0] += step;
  const Eval after = run(pot, nudged);
  const double finite = -(after.energy - compact.energy) / step;
  REQUIRE_THAT(finite, Catch::Matchers::WithinAbs(compact.force[0], 1e-3));
}

TEST_CASE("a molecular box rejects a periodic image", "[uma][graph]") {
  // After recentering, these three atoms span more than the 6 angstrom
  // cutoff and less than the 15 angstrom box, so the missing pair comes
  // back as a periodic image.
  rgpot::UmaPot pot = open_pot(package_path("UMA_MOLECULAR_BOX_PT2"));
  const std::vector<double> pos = {0, 0, 0, 0, 0, 1, 0, 0, 10};
  const std::vector<int> z = {6, 7, 1};
  const double box[9] = {15, 0, 0, 0, 15, 0, 0, 0, 15};
  std::vector<double> force(9, 0.0);
  rgpot::ForceOut slot{force.data(), 0.0, 0.0};
  const rgpot::ForceInput in{3, pos.data(), z.data(), box};
  REQUIRE_THROWS_WITH(
      pot.forceImpl(in, &slot),
      Catch::Matchers::ContainsSubstring("cell_offsets") &&
          Catch::Matchers::ContainsSubstring("nonzero"));
}

TEST_CASE("a fixed-shape package loads and computes a force", "[uma][graph]") {
  // The keyless tetrazine package records edge_index [2, 56] in shapes
  // and neither nedges_min nor nedges_max. Eight atoms at 0.5 angstrom
  // stay inside the 10 angstrom cutoff, so the list is the traced 56.
  rgpot::UmaPot pot = open_pot(package_path("UMA_TETRAZINE_KEYLESS_PT2"));
  const Eval out = run(pot, line(8, 0.5));
  REQUIRE(std::isfinite(out.energy));
  REQUIRE_THAT(out.energy, Catch::Matchers::WithinAbs(0.0, 1e-6));
  REQUIRE(out.force.size() == 24);
  for (double component : out.force)
    REQUIRE_THAT(component, Catch::Matchers::WithinAbs(-7.0, 1e-6));
}

TEST_CASE("a dynamic package still loads", "[uma][graph]") {
  rgpot::UmaPot pot = open_pot(package_path("UMA_GATHER_DYNAMIC_PT2"));
  const Eval out = run(pot, line(4, 1.0));
  REQUIRE(std::isfinite(out.energy));
  REQUIRE_THAT(out.energy, Catch::Matchers::WithinAbs(468.0, 1e-6));
}

TEST_CASE("one dynamic package takes several atom counts", "[uma][graph]") {
  // The gather energy of a complete graph is (n - 1) times the sum of
  // the coordinates, and each atom's force component is -(n - 1). Six
  // atoms one angstrom apart stay inside the 6 angstrom cutoff.
  rgpot::UmaPot pot = open_pot(package_path("UMA_GATHER_DYNAMIC_PT2"));
  const Eval four = run(pot, line(4, 1.0));
  const Eval six = run(pot, line(6, 1.0));
  REQUIRE_THAT(four.energy, Catch::Matchers::WithinAbs(468.0, 1e-6));
  REQUIRE_THAT(six.energy, Catch::Matchers::WithinAbs(1200.0, 1e-6));
  REQUIRE(six.force.size() == 18);
  for (double component : six.force)
    REQUIRE_THAT(component, Catch::Matchers::WithinAbs(-5.0, 1e-6));
}

TEST_CASE("a dynamic package refuses an atomic number past z_max",
          "[uma][graph]") {
  rgpot::UmaPot pot = open_pot(package_path("UMA_GATHER_DYNAMIC_PT2"));
  std::vector<double> pos = line(4, 1.0);
  std::vector<int> z = {6, 6, 6, 120};
  const double box[9] = {40, 0, 0, 0, 40, 0, 0, 0, 40};
  std::vector<double> force(pos.size(), 0.0);
  rgpot::ForceOut slot{force.data(), 0.0, 0.0};
  const rgpot::ForceInput in{4, pos.data(), z.data(), box};
  try {
    pot.forceImpl(in, &slot);
    FAIL("an atomic number past z_max reached the compiled graph");
  } catch (const rgpot::UmaContractError &err) {
    REQUIRE(err.field() == "z_max");
    REQUIRE(err.actual() == "120");
  }
}

TEST_CASE("a dynamic contract checks atoms and edges against its ranges",
          "[uma][contract]") {
  // The metadata scripts/export_uma_aoti.py writes for a dynamic
  // package that keeps the MoLE experts: no natoms, counts or z_set.
  auto meta = std::unordered_map<std::string, std::string>{
      {"pos_shape", "natoms,3"},
      {"natoms_min", "2"},
      {"natoms_max", "1024"},
      {"edge_index_shape", "2,nedges"},
      {"nedges_min", "1"},
      {"nedges_max", "307200"},
      {"z_max", "99"},
  };
  const auto contract = rgpot::UmaContract::fromMetadata(meta);
  REQUIRE_NOTHROW(contract.checkCallShapes(3, 6));
  REQUIRE_NOTHROW(contract.checkCallShapes(3, 4));
  REQUIRE_NOTHROW(contract.checkCallShapes(8, 56));
  REQUIRE_NOTHROW(contract.checkCallShapes(8, 24));
  REQUIRE_NOTHROW(contract.checkCallShapes(1024, 307200));
  REQUIRE_THROWS_WITH(contract.checkCallShapes(1, 1),
                      Catch::Matchers::ContainsSubstring("pos shape is [1, 3]") &&
                          Catch::Matchers::ContainsSubstring("[2, 1024]"));
  REQUIRE_THROWS_AS(contract.checkCallShapes(1025, 6), rgpot::UmaShapeError);
  REQUIRE_THROWS_AS(contract.checkCallShapes(8, 307201), rgpot::UmaShapeError);
  const int64_t one[1] = {24};
  REQUIRE_NOTHROW(contract.checkEachSystem(1, one));
  const int hcn[3] = {6, 7, 1};
  const int ethanol[9] = {6, 6, 8, 1, 1, 1, 1, 1, 1};
  REQUIRE_NOTHROW(contract.checkSystem(3, hcn));
  REQUIRE_NOTHROW(contract.checkSystem(9, ethanol));
  const int ghost[2] = {6, 0};
  REQUIRE_THROWS_AS(contract.checkSystem(2, ghost), rgpot::UmaContractError);
}

TEST_CASE("tetrazine past the 10 angstrom cutoff names the fixed edge shape",
          "[uma][graph]") {
  // Control packages omit nedges_min and nedges_max. shapes records
  // the traced complete pair list, [2, 56], cutoff 10 angstrom. Moving
  // one hydrogen by 12 angstroms crosses that cutoff. The named error
  // is raised before the compiled graph runs.
  std::vector<double> pos = {
      12.5, 13.83, 12.5, 12.5, 11.17, 12.5, 13.65, 13.165, 12.5,
      13.65, 11.835, 12.5, 11.35, 13.165, 12.5, 11.35, 11.835, 12.5,
      12.5, 14.91, 12.5, 12.5, 10.09, 12.5,
  };
  pos[7 * 3 + 1] += 12.0;
  const std::vector<int> z = {6, 6, 7, 7, 7, 7, 1, 1};
  const double box[9] = {25, 0, 0, 0, 25, 0, 0, 0, 25};
  rgpot::UmaPot pot = open_pot(package_path("UMA_TETRAZINE_KEYLESS_PT2"));
  std::vector<double> force(pos.size(), 0.0);
  rgpot::ForceOut slot{force.data(), 0.0, 0.0};
  const rgpot::ForceInput in{8, pos.data(), z.data(), box};
  REQUIRE_THROWS_WITH(
      pot.forceImpl(in, &slot),
      Catch::Matchers::ContainsSubstring("edge_index shape is") &&
          Catch::Matchers::ContainsSubstring("[2, 56]"));
}

TEST_CASE("a package with no edge range is refused", "[uma][graph]") {
  rgpot::UmaPot pot = open_pot(package_path("UMA_NO_EDGES_PT2"));
  REQUIRE_THROWS_WITH(run(pot, line(3, 1.0)),
                      Catch::Matchers::ContainsSubstring("nedges") &&
                          Catch::Matchers::ContainsSubstring("absent"));
}

TEST_CASE("tetrazine walked past 10 angstroms names the edge count",
          "[uma][graph]") {
  // s-tetrazine, eight atoms. The static package was traced at 56
  // edges. Moving one hydrogen by 12 angstroms changes that count.
  std::vector<double> pos = {
      12.5, 13.83, 12.5, 12.5, 11.17, 12.5, 13.65, 13.165, 12.5,
      13.65, 11.835, 12.5, 11.35, 13.165, 12.5, 11.35, 11.835, 12.5,
      12.5, 14.91, 12.5, 12.5, 10.09, 12.5,
  };
  pos[7 * 3 + 1] += 12.0;
  const std::vector<int> z = {6, 6, 7, 7, 7, 7, 1, 1};
  const double box[9] = {40, 0, 0, 0, 40, 0, 0, 0, 40};
  rgpot::UmaPot pot = open_pot(package_path("UMA_GATHER_STATIC_PT2"));
  std::vector<double> force(pos.size(), 0.0);
  rgpot::ForceOut slot{force.data(), 0.0, 0.0};
  const rgpot::ForceInput in{8, pos.data(), z.data(), box};
  try {
    pot.forceImpl(in, &slot);
    FAIL("the static package returned a force");
  } catch (const rgpot::UmaShapeError &err) {
    const std::string message = err.what();
    REQUIRE(err.field() == "edge_index");
    REQUIRE_THAT(message,
                 Catch::Matchers::ContainsSubstring("edge_index shape is"));
    REQUIRE_THAT(message, Catch::Matchers::ContainsSubstring("[2, 56]"));
    REQUIRE_THAT(message, !Catch::Matchers::ContainsSubstring("returned 2"));
  }
}

TEST_CASE("a band checks each system's edge count", "[uma][graph]") {
  rgpot::UmaPot pot = open_pot(package_path("UMA_BATCH_STATIC_PT2"));
  const std::vector<int> z = {6, 7, 1};
  const double box[9] = {80, 0, 0, 0, 80, 0, 0, 0, 80};
  std::vector<double> compact = {12.5, 12.5, 12.5, 12.5, 12.5, 13.5, 12.5, 12.5,
                                 11.5};
  std::vector<double> stretched = compact;
  stretched.back() = 40.0;
  std::vector<double> f0(9, 0.0), f1(9, 0.0);
  rgpot::ForceInput in[2] = {
      {3, compact.data(), z.data(), box},
      {3, stretched.data(), z.data(), box},
  };
  rgpot::ForceOut out[2] = {
      {f0.data(), 0.0, 0.0},
      {f1.data(), 0.0, 0.0},
  };
  const rgpot::ForceBatch batch{2, in, out};
  REQUIRE_THROWS_WITH(
      pot.forceBatchImpl(batch),
      Catch::Matchers::ContainsSubstring("edge_index shape is") &&
          Catch::Matchers::ContainsSubstring("[2, 2]") &&
          Catch::Matchers::ContainsSubstring("[2, 6]"));
}

TEST_CASE("each system of a static band is accepted", "[uma][contract]") {
  auto meta = std::unordered_map<std::string, std::string>{
      {"nedges_min", "12"},
      {"nedges_max", "12"},
  };
  const auto contract = rgpot::UmaContract::fromMetadata(meta);
  const int64_t even[2] = {6, 6};
  REQUIRE_NOTHROW(contract.checkEachSystem(2, even));
  const int64_t uneven[2] = {4, 8};
  REQUIRE_THROWS_AS(contract.checkEachSystem(2, uneven), rgpot::UmaShapeError);
}

TEST_CASE("a band range checks each system's edge count", "[uma][contract]") {
  auto meta = std::unordered_map<std::string, std::string>{
      {"nedges_min", "1"},
      {"nedges_max", "8"},
  };
  const auto contract = rgpot::UmaContract::fromMetadata(meta);
  const int64_t inside[2] = {6, 2};
  REQUIRE_NOTHROW(contract.checkEachSystem(2, inside));
  const int64_t outside[2] = {6, 9};
  REQUIRE_THROWS_AS(contract.checkEachSystem(2, outside), rgpot::UmaShapeError);
}

TEST_CASE("a molecular band requires the complete pair list", "[uma][contract]") {
  auto meta = std::unordered_map<std::string, std::string>{
      {"nedges_min", "1"},
      {"nedges_max", "100"},
  };
  const auto contract = rgpot::UmaContract::fromMetadata(meta);
  const int64_t complete[2] = {6, 6};
  REQUIRE_NOTHROW(contract.checkEachSystem(2, complete, 3, true));
  const int64_t dropped[2] = {6, 4};
  REQUIRE_THROWS_WITH(contract.checkEachSystem(2, dropped, 3, true),
                      Catch::Matchers::ContainsSubstring("[2, 4]") &&
                          Catch::Matchers::ContainsSubstring("[2, 6]"));
}

TEST_CASE("a molecular band refuses a changed edge count", "[uma][graph]") {
  // batch_max > 1 used to enter the band path before the n*(n-1)
  // check, and a range package skipped the per-system check. One
  // system of this pair has left the cutoff. The named error is
  // raised from forceImpl, which is the call that used to miss it.
  rgpot::UmaPot pot = open_pot(package_path("UMA_BATCH_MOLECULAR_PT2"));
  const std::vector<int> z = {6, 7, 1};
  const double box[9] = {80, 0, 0, 0, 80, 0, 0, 0, 80};
  std::vector<double> compact = {12.5, 12.5, 12.5, 12.5, 12.5, 13.5, 12.5, 12.5,
                                 11.5};
  std::vector<double> stretched = compact;
  stretched.back() = 40.0;
  std::vector<double> f0(9, 0.0), f1(9, 0.0);
  rgpot::ForceInput in[2] = {
      {3, compact.data(), z.data(), box},
      {3, stretched.data(), z.data(), box},
  };
  rgpot::ForceOut out[2] = {
      {f0.data(), 0.0, 0.0},
      {f1.data(), 0.0, 0.0},
  };
  REQUIRE_THROWS_WITH(
      pot.forceImpl(in[1], &out[1]),
      Catch::Matchers::ContainsSubstring("edge_index shape is") &&
          Catch::Matchers::ContainsSubstring("[2, 6]"));
  const rgpot::ForceBatch batch{2, in, out};
  REQUIRE_THROWS_WITH(
      pot.forceBatchImpl(batch),
      Catch::Matchers::ContainsSubstring("edge_index shape is") &&
          Catch::Matchers::ContainsSubstring("[2, 2]") &&
          Catch::Matchers::ContainsSubstring("[2, 6]"));
}
