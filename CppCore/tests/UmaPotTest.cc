// MIT License — UmaPot loads an AOTI .pt2, not a metatomic checkpoint.

#include <algorithm>
#include <array>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <filesystem>
#include <fstream>
#include <map>
#include <optional>
#include <string>
#include <unordered_map>
#include <vector>

#include <catch2/catch_all.hpp>

#include "rgpot/UmaPot/UmaContract.hpp"
#include "rgpot/UmaPot/UmaPot.hpp"
#include "rgpot/UmaPot/aoti_execstack.hpp"
#include "rgpot/types/AtomMatrix.hpp"

using Catch::Matchers::WithinAbs;
using rgpot::types::AtomMatrix;
namespace fs = std::filesystem;

// ASE FAIRChemCalculator 2.22.0 / torch 2.13.0, uma-s-1p1 / omol.
// Baker 01_hcn positions are centered in double before model evaluation.
// Checkpoint SHA256:
// 07068e9c76702ca173d13155095f2117c1b327ec228557e64cd2709c777b824a
static constexpr double kHcnAseOmolEnergy = -2542.4200323230975;
static constexpr double kHcnAseOmolForces[3][3] = {
    {0.029150184243917465, 0.011640587821602821, -1.475394606590271},
    {-0.016498778015375137, -0.006508127320557833, 1.7600626945495605},
    {-0.012651405297219753, -0.005132460035383701, -0.2846679985523224},
};

static std::string resolve_uma_omol_pt2() {
  if (const char *env = std::getenv("RGPOT_UMA_OMOL_PT2"); env && *env) {
    if (fs::exists(env))
      return std::string(env);
    FAIL("RGPOT_UMA_OMOL_PT2 is set but the file is missing: "
         << env << ". Export an AOTI package with scripts/export_uma_aoti.py");
  }
  // workdir is CppCore/tests; also accept repo-root-relative paths.
  static const char *const kCandidates[] = {
      "data/uma/uma-s-1p1-omol-hcn.pt2",
      "CppCore/tests/data/uma/uma-s-1p1-omol-hcn.pt2",
      "../../bench_data/uma/uma-s-1p1-omol-hcn.pt2",
      "bench_data/uma/uma-s-1p1-omol-hcn.pt2",
  };
  for (const char *path : kCandidates) {
    if (fs::exists(path))
      return std::string(path);
  }
  // Hosts that hold the fixtures set RGPOT_UMA_REQUIRE_FIXTURES so a missing
  // package fails; elsewhere the fixture cases skip.
  static constexpr const char *kMissing =
      "UmaPot HCN fixture needs uma-s-1p1-omol-hcn.pt2. Export with "
      "scripts/export_uma_aoti.py and set RGPOT_UMA_OMOL_PT2, or place the "
      "file at CppCore/tests/data/uma/uma-s-1p1-omol-hcn.pt2 or "
      "bench_data/uma/uma-s-1p1-omol-hcn.pt2";
  if (const char *req = std::getenv("RGPOT_UMA_REQUIRE_FIXTURES");
      req && *req) {
    FAIL(kMissing);
  }
  SKIP(kMissing);
  return {};
}

static std::vector<uint8_t> elf64_gnu_stack(uint32_t flags) {
  std::vector<uint8_t> elf(120, 0);
  elf[0] = 0x7f;
  elf[1] = 'E';
  elf[2] = 'L';
  elf[3] = 'F';
  elf[4] = 2;
  elf[5] = 1;
  elf[6] = 1;
  elf[16] = 3;
  elf[18] = 62;
  elf[20] = 1;
  elf[32] = 64;
  elf[52] = 64;
  elf[54] = 56;
  elf[56] = 1;
  rgpot::aoti_execstack::wr32(elf.data() + 64, 0x6474e551u);
  rgpot::aoti_execstack::wr32(elf.data() + 68, flags);
  return elf;
}

TEST_CASE("clear_elf_gnu_stack drops PF_X", "[UmaPot][execstack]") {
  auto elf = elf64_gnu_stack(7);
  REQUIRE(
      rgpot::aoti_execstack::elf_needs_gnu_stack_clear(elf.data(), elf.size()));
  REQUIRE(rgpot::aoti_execstack::clear_elf_gnu_stack(elf.data(), elf.size()));
  REQUIRE_FALSE(
      rgpot::aoti_execstack::elf_needs_gnu_stack_clear(elf.data(), elf.size()));
  REQUIRE(rgpot::aoti_execstack::rd32(elf.data() + 68) == 6u);
}

TEST_CASE("UmaPot empty model_path throws", "[UmaPot]") {
  rgpot::UmaConfig cfg;
  REQUIRE_THROWS_WITH(rgpot::UmaPot(cfg),
                      Catch::Matchers::ContainsSubstring("model_path"));
}

// forceBatchImpl is a real AOTI batch, not Potential's one-at-a-time
// default. Callers (a NEB band) trust caps().batched for that.
TEST_CASE("UmaPot advertises native batch evaluation", "[UmaPot]") {
  rgpot::UmaConfig cfg;
  cfg.model_path = "unused.pt2";
  rgpot::UmaPot pot(cfg);
  REQUIRE(pot.caps().batched);
  REQUIRE(pot.caps().perImageInstances);
}

TEST_CASE("UmaPot rejects a metatomic checkpoint", "[UmaPot]") {
  rgpot::UmaConfig cfg;
  cfg.model_path = "data/lj38/lennard-jones.pt";
  REQUIRE_THROWS_WITH(rgpot::UmaPot(cfg),
                      Catch::Matchers::ContainsSubstring(".pt2"));
}

TEST_CASE("UmaPot paramsKey changes with charge and path", "[UmaPot]") {
  rgpot::UmaConfig cfg;
  cfg.model_path = "no-such-uma.pt2";
  rgpot::UmaPot a(cfg);
  const uint64_t k0 = a.paramsKey();
  a.setChargeSpin(-1, 1);
  REQUIRE(a.paramsKey() != k0);
  cfg.model_path = "other-uma.pt2";
  rgpot::UmaPot b(cfg);
  REQUIRE(b.paramsKey() != k0);
}

TEST_CASE("UmaPot validates spin bounds per task without rewriting spin 0",
          "[UmaPot][spin]") {
  rgpot::UmaConfig omat;
  omat.model_path = "unused.pt2";
  omat.task_name = "omat";
  omat.spin = 0;
  rgpot::UmaPot periodic(omat);
  REQUIRE(periodic.config().spin == 0);
  REQUIRE_NOTHROW(periodic.setChargeSpin(0, 0));
  REQUIRE(periodic.config().spin == 0);
  REQUIRE_THROWS_AS(periodic.setChargeSpin(0, -1), std::invalid_argument);
  REQUIRE(periodic.config().spin == 0);

  rgpot::UmaConfig omol;
  omol.model_path = "unused.pt2";
  omol.task_name = "omol";
  omol.spin = 0;
  rgpot::UmaPot mol(omol);
  REQUIRE(mol.config().spin == 0);
  REQUIRE_NOTHROW(mol.setChargeSpin(0, 0));
  REQUIRE(mol.config().spin == 0);
  REQUIRE_THROWS_AS(mol.setChargeSpin(0, -1), std::invalid_argument);
  REQUIRE_THROWS_AS(mol.setChargeSpin(0, rgpot::kUmaSpinMax + 1),
                    std::invalid_argument);
  REQUIRE(mol.config().spin == 0);
}

TEST_CASE("UmaPot constructor refuses a spin outside 0 to 100",
          "[UmaPot][spin]") {
  for (const char *task : {"omol", "omat"}) {
    for (const int bad : {-1, rgpot::kUmaSpinMax + 1}) {
      rgpot::UmaConfig cfg;
      cfg.model_path = "unused.pt2";
      cfg.task_name = task;
      cfg.spin = bad;
      REQUIRE_THROWS_AS(rgpot::UmaPot(cfg), std::invalid_argument);
      REQUIRE_THROWS_WITH(
          rgpot::UmaPot(cfg),
          Catch::Matchers::ContainsSubstring(
              "spin must be an integer from 0 to 100; got " +
              std::to_string(bad)));
    }
  }

  rgpot::UmaConfig edge;
  edge.model_path = "unused.pt2";
  edge.spin = rgpot::kUmaSpinMax;
  REQUIRE_NOTHROW(rgpot::UmaPot(edge));
  edge.spin = 0;
  rgpot::UmaPot zero(edge);
  REQUIRE(zero.config().spin == 0);
}

// Metadata as scripts/export_uma_aoti.py embeds it for Baker HCN.
static std::unordered_map<std::string, std::string> hcn_metadata() {
  return {{"task_name", "omol"},  {"charge", "0"},
          {"spin", "1"},          {"z_set", "[1, 6, 7]"},
          {"natoms", "3"},        {"counts", R"({"1": 1, "6": 1, "7": 1})"},
          {"nedges_min", "6"},    {"nedges_max", "6"},
          {"cutoff", "10.0"},     {"label", "hcn"},
          {"model", "uma-s-1p1"}, {"torch_version", "2.13.0+cpu"}};
}

static rgpot::UmaConfig hcn_config() {
  rgpot::UmaConfig cfg;
  cfg.task_name = "omol";
  cfg.charge = 0;
  cfg.spin = 1;
  return cfg;
}

template <class F> static std::string contract_field(F &&f) {
  try {
    f();
  } catch (const rgpot::UmaContractError &e) {
    return e.field();
  }
  return "";
}

TEST_CASE("UmaContract parses the exporter metadata", "[UmaPot][contract]") {
  const auto c = rgpot::UmaContract::fromMetadata(hcn_metadata());
  REQUIRE(c.task_name == std::optional<std::string>("omol"));
  REQUIRE(c.charge == std::optional<int64_t>(0));
  REQUIRE(c.spin == std::optional<int64_t>(1));
  REQUIRE(c.natoms == std::optional<int64_t>(3));
  REQUIRE(*c.z_set == std::vector<int64_t>{1, 6, 7});
  REQUIRE(*c.counts == std::map<int64_t, int64_t>{{1, 1}, {6, 1}, {7, 1}});

  auto anion = hcn_metadata();
  anion["charge"] = "-1";
  REQUIRE(rgpot::UmaContract::fromMetadata(anion).charge ==
          std::optional<int64_t>(-1));

  auto bad = hcn_metadata();
  bad["spin"] = "1, 2";
  REQUIRE_THROWS_WITH(rgpot::UmaContract::fromMetadata(bad),
                      Catch::Matchers::ContainsSubstring("spin"));
  bad = hcn_metadata();
  bad["counts"] = "[1, 1, 6]";
  REQUIRE_THROWS_WITH(rgpot::UmaContract::fromMetadata(bad),
                      Catch::Matchers::ContainsSubstring("counts"));
}

TEST_CASE("UmaContract rejects a config the package was not merged for",
          "[UmaPot][contract]") {
  const auto c = rgpot::UmaContract::fromMetadata(hcn_metadata());
  REQUIRE_NOTHROW(c.checkConfig(hcn_config()));

  auto cfg = hcn_config();
  cfg.charge = 1;
  REQUIRE(contract_field([&] { c.checkConfig(cfg); }) == "charge");
  cfg = hcn_config();
  cfg.spin = 3;
  REQUIRE(contract_field([&] { c.checkConfig(cfg); }) == "spin");
  cfg = hcn_config();
  cfg.task_name = "omat";
  REQUIRE(contract_field([&] { c.checkConfig(cfg); }) == "task_name");

  cfg = hcn_config();
  cfg.charge = -1;
  REQUIRE_THROWS_WITH(
      c.checkConfig(cfg),
      Catch::Matchers::Equals(
          "UmaPot: config charge is -1, the package was exported for 0"));
}

TEST_CASE("UmaContract rejects an input of another composition",
          "[UmaPot][contract]") {
  const auto c = rgpot::UmaContract::fromMetadata(hcn_metadata());
  const std::vector<int> hcn{6, 7, 1};
  const std::vector<int> nch{7, 6, 1};
  REQUIRE_NOTHROW(c.checkSystem(hcn.size(), hcn.data()));
  REQUIRE_NOTHROW(c.checkSystem(nch.size(), nch.data()));

  const std::vector<int> hnco{1, 7, 6, 8};
  REQUIRE(contract_field([&] { c.checkSystem(hnco.size(), hnco.data()); }) ==
          "natoms");
  const std::vector<int> ccn{6, 7, 6};
  REQUIRE(contract_field([&] { c.checkSystem(ccn.size(), ccn.data()); }) ==
          "counts");
  REQUIRE_THROWS_WITH(
      c.checkSystem(ccn.size(), ccn.data()),
      Catch::Matchers::Equals("UmaPot: input counts is {6: 2, 7: 1}, the "
                              "package was exported for {1: 1, 6: 1, 7: 1}"));

  // Ethylene and acetylene share z_set {1, 6}; only the counts tell
  // them apart.
  auto c2h2 = hcn_metadata();
  c2h2["z_set"] = "[1, 6]";
  c2h2["natoms"] = "4";
  c2h2["counts"] = R"({"1": 2, "6": 2})";
  const auto acetylene = rgpot::UmaContract::fromMetadata(c2h2);
  const std::vector<int> c2h4{6, 6, 1, 1, 1, 1};
  const std::vector<int> c3h1{6, 6, 6, 1};
  REQUIRE(contract_field([&] {
            acetylene.checkSystem(c2h4.size(), c2h4.data());
          }) == "natoms");
  REQUIRE(contract_field([&] {
            acetylene.checkSystem(c3h1.size(), c3h1.data());
          }) == "counts");
}

TEST_CASE("UmaContract refuses an edge count the graph was not exported for",
          "[UmaPot][contract]") {
  auto meta = hcn_metadata();
  meta["nedges_min"] = "6";
  meta["nedges_max"] = "6";
  const auto fixed = rgpot::UmaContract::fromMetadata(meta);
  REQUIRE_NOTHROW(fixed.checkEdges(6));
  REQUIRE(contract_field([&] { fixed.checkEdges(4); }) == "nedges");
  REQUIRE_THROWS_WITH(
      fixed.checkEdges(8),
      Catch::Matchers::Equals(
          "UmaPot: input nedges is 8, the package was exported for 6"));

  meta["nedges_min"] = "1";
  meta["nedges_max"] = "65536";
  const auto dynamic = rgpot::UmaContract::fromMetadata(meta);
  REQUIRE_NOTHROW(dynamic.checkEdges(1));
  REQUIRE_NOTHROW(dynamic.checkEdges(65536));
  REQUIRE_THROWS_WITH(
      dynamic.checkEdges(65537),
      Catch::Matchers::Equals("UmaPot: input nedges is 65537, the package "
                              "was exported for [1, 65536]"));

  auto missing = hcn_metadata();
  missing.erase("nedges_min");
  missing.erase("nedges_max");
  REQUIRE_THROWS_WITH(
      rgpot::UmaContract::fromMetadata(missing),
      Catch::Matchers::ContainsSubstring("nedges") &&
          Catch::Matchers::ContainsSubstring("absent"));

  // Grignard: 37 * 36 directed pairs, recorded only in shapes.
  auto shaped = missing;
  shaped.erase("natoms");
  shaped["shapes"] =
      "{'pos': [37, 3], 'edge_index': [2, 1332], 'cell_offsets': [1332, 3]}";
  const auto grignard = rgpot::UmaContract::fromMetadata(shaped);
  REQUIRE(grignard.nedges_min == 1332);
  REQUIRE(grignard.nedges_max == 1332);
  REQUIRE_NOTHROW(grignard.checkEdges(1332));
  REQUIRE_THROWS_WITH(
      grignard.checkCallShapes(37, 1000),
      Catch::Matchers::ContainsSubstring("edge_index shape is") &&
          Catch::Matchers::ContainsSubstring("[2, 1332]"));

  auto min_only = hcn_metadata();
  min_only.erase("nedges_max");
  REQUIRE_THROWS_WITH(
      rgpot::UmaContract::fromMetadata(min_only),
      Catch::Matchers::ContainsSubstring("nedges_max is absent"));
  auto max_only = hcn_metadata();
  max_only.erase("nedges_min");
  REQUIRE_THROWS_WITH(
      rgpot::UmaContract::fromMetadata(max_only),
      Catch::Matchers::ContainsSubstring("nedges_min is absent"));
}

TEST_CASE("UmaContract checks z_set alone for packages without counts",
          "[UmaPot][contract]") {
  auto legacy = hcn_metadata();
  legacy.erase("counts");
  legacy.erase("natoms");
  legacy["nedges_min"] = "6";
  legacy["nedges_max"] = "6";
  const auto c = rgpot::UmaContract::fromMetadata(legacy);
  REQUIRE_FALSE(c.counts.has_value());
  REQUIRE_FALSE(c.natoms.has_value());
  const std::vector<int> hcno{6, 7, 1, 8};
  REQUIRE(contract_field([&] { c.checkSystem(hcno.size(), hcno.data()); }) ==
          "z_set");
  // Same element set, other counts: only counts metadata catches it.
  const std::vector<int> hhcn{1, 6, 7, 1};
  REQUIRE_NOTHROW(c.checkSystem(hhcn.size(), hhcn.data()));

  REQUIRE_THROWS_WITH(rgpot::UmaContract::fromMetadata({}),
                      Catch::Matchers::ContainsSubstring("nedges"));
}

TEST_CASE("UmaPot refuses a charge, spin or composition the package lacks",
          "[UmaPot][omol][contract]") {
  const std::string model = resolve_uma_omol_pt2();
  const AtomMatrix positions{
      {12.49734736216627162, 12.49892801474515913, 12.54059929828148512},
      {12.50115413363106498, 12.50036504272228832, 11.38209979880783251},
      {12.50149850420264563, 12.50069809648255514, 13.61514544631068446},
  };
  const std::vector<int> atmtypes{6, 7, 1};
  const std::array<std::array<double, 3>, 3> box{
      {{25.0, 0.0, 0.0}, {0.0, 25.0, 0.0}, {0.0, 0.0, 25.0}}};

  rgpot::UmaConfig cfg = hcn_config();
  cfg.model_path = model;
  cfg.device = "cpu";

  SECTION("charge") {
    cfg.charge = 1;
    rgpot::UmaPot pot(cfg);
    REQUIRE(contract_field([&] { pot(positions, atmtypes, box); }) == "charge");
  }
  SECTION("spin set after construction") {
    rgpot::UmaPot pot(cfg);
    REQUIRE_NOTHROW(pot(positions, atmtypes, box));
    pot.setChargeSpin(0, 3);
    REQUIRE(contract_field([&] { pot(positions, atmtypes, box); }) == "spin");
  }
  SECTION("composition") {
    rgpot::UmaPot pot(cfg);
    const std::vector<int> ccn{6, 7, 6};
    REQUIRE_THROWS_AS(pot(positions, ccn, box), rgpot::UmaContractError);
  }
}

TEST_CASE("UmaPot Baker HCN matches ASE FAIRChem omol", "[UmaPot][omol]") {
  const std::string model = resolve_uma_omol_pt2();

  rgpot::UmaConfig cfg;
  cfg.model_path = model;
  cfg.device = "cpu";
  cfg.task_name = "omol";
  cfg.charge = 0;
  cfg.spin = 1;
  rgpot::UmaPot pot(cfg);

  // Baker 01_hcn reactant.con, Angstrom, 25 A cubic cell, PBC.
  const AtomMatrix positions{
      {12.49734736216627162, 12.49892801474515913, 12.54059929828148512},
      {12.50115413363106498, 12.50036504272228832, 11.38209979880783251},
      {12.50149850420264563, 12.50069809648255514, 13.61514544631068446},
  };
  const std::vector<int> atmtypes{6, 7, 1};
  const std::array<std::array<double, 3>, 3> box{
      {{25.0, 0.0, 0.0}, {0.0, 25.0, 0.0}, {0.0, 0.0, 25.0}}};

  auto [energy, forces, variance] = pot(positions, atmtypes, box);
  (void)variance;

  std::printf("backend=UmaPot\n");
  std::printf("pos_dtype=%s model_dtype=%s\n", pot.posDtypeName(),
              pot.modelDtypeName());
  std::printf("model=%s\n", model.c_str());
  std::printf("energy=%.17g\n", energy);
  double max_df = 0.0;
  REQUIRE(forces.rows() == 3);
  REQUIRE(forces.cols() == 3);
  for (size_t i = 0; i < 3; ++i) {
    std::printf("force %zu %.17g %.17g %.17g\n", i, forces(i, 0), forces(i, 1),
                forces(i, 2));
    for (size_t j = 0; j < 3; ++j) {
      max_df =
          std::max(max_df, std::abs(forces(i, j) - kHcnAseOmolForces[i][j]));
    }
  }
  std::printf("max_dF=%.17g\n", max_df);

  REQUIRE_THAT(energy, WithinAbs(kHcnAseOmolEnergy, 1e-4));
  REQUIRE_THAT(max_df, WithinAbs(0.0, 1e-4));
}

TEST_CASE("UmaPot package accepts its composition in any atom order",
          "[UmaPot][omol][contract]") {
  // The contract checks counts, not order: a package merged for HCN
  // must evaluate N, H, C to the same energy and permuted forces.
  const std::string model = resolve_uma_omol_pt2();
  rgpot::UmaConfig cfg = hcn_config();
  cfg.model_path = model;
  cfg.device = "cpu";
  rgpot::UmaPot pot(cfg);

  const AtomMatrix hcn{
      {12.49734736216627162, 12.49892801474515913, 12.54059929828148512},
      {12.50115413363106498, 12.50036504272228832, 11.38209979880783251},
      {12.50149850420264563, 12.50069809648255514, 13.61514544631068446},
  };
  const std::array<int, 3> perm{1, 2, 0}; // new row r holds old row perm[r]
  AtomMatrix nhc(3, 3);
  std::vector<int> z_nhc(3);
  const std::vector<int> z_hcn{6, 7, 1};
  for (int r = 0; r < 3; ++r) {
    z_nhc[r] = z_hcn[perm[r]];
    for (int d = 0; d < 3; ++d)
      nhc(r, d) = hcn(perm[r], d);
  }
  const std::array<std::array<double, 3>, 3> box{
      {{25.0, 0.0, 0.0}, {0.0, 25.0, 0.0}, {0.0, 0.0, 25.0}}};

  auto [e0, f0, v0] = pot(hcn, z_hcn, box);
  auto [e1, f1, v1] = pot(nhc, z_nhc, box);
  (void)v0;
  (void)v1;
  REQUIRE_THAT(e1, WithinAbs(e0, 1e-4));
  for (int r = 0; r < 3; ++r)
    for (int d = 0; d < 3; ++d)
      REQUIRE_THAT(f1(r, d), WithinAbs(f0(perm[r], d), 1e-4));
}

TEST_CASE("UmaPot band batch matches per-system evaluation",
          "[UmaPot][omol][band]") {
  // Needs a band package (exported with --batch-max) named beside the
  // single-system fixture or via RGPOT_UMA_OMOL_BAND_PT2. Skipped when
  // absent so the plain fixture suite stays runnable.
  std::string band;
  if (const char *env = std::getenv("RGPOT_UMA_OMOL_BAND_PT2"); env && *env) {
    band = env;
  }
  if (band.empty() || !fs::exists(band)) {
    SKIP("set RGPOT_UMA_OMOL_BAND_PT2 to a --batch-max export");
  }

  rgpot::UmaConfig cfg;
  cfg.model_path = band;
  cfg.device = "cpu";
  cfg.task_name = "omol";
  cfg.charge = 0;
  cfg.spin = 1;
  rgpot::UmaPot pot(cfg);

  // Three HCN geometries: the reference row plus two perturbations.
  const std::array<std::array<double, 9>, 3> geoms{{
      {12.49734736216627162, 12.49892801474515913, 12.54059929828148512,
       12.50115413363106498, 12.50036504272228832, 11.38209979880783251,
       12.50149850420264563, 12.50069809648255514, 13.61514544631068446},
      {12.53734736216627162, 12.49892801474515913, 12.54059929828148512,
       12.50115413363106498, 12.54036504272228832, 11.38209979880783251,
       12.50149850420264563, 12.50069809648255514, 13.57514544631068446},
      {12.45734736216627162, 12.52892801474515913, 12.50059929828148512,
       12.50115413363106498, 12.46036504272228832, 11.42209979880783251,
       12.54149850420264563, 12.50069809648255514, 13.65514544631068446},
  }};
  const std::vector<int> atmtypes{6, 7, 1};
  const std::array<double, 9> box{25.0, 0.0, 0.0, 0.0, 25.0,
                                  0.0,  0.0, 0.0, 25.0};
  const std::array<std::array<double, 3>, 3> box33{
      {{25.0, 0.0, 0.0}, {0.0, 25.0, 0.0}, {0.0, 0.0, 25.0}}};

  // Per-system reference through the public single-call entry (which
  // pads through the band graph itself, so this also covers padding).
  std::array<double, 3> e_single{};
  std::array<std::array<double, 9>, 3> f_single{};
  for (size_t s = 0; s < 3; ++s) {
    AtomMatrix positions(3, 3);
    for (int i = 0; i < 3; ++i)
      for (int d = 0; d < 3; ++d)
        positions(i, d) = geoms[s][static_cast<size_t>(3 * i + d)];
    auto [energy, forces, variance] = pot(positions, atmtypes, box33);
    (void)variance;
    e_single[s] = energy;
    for (int i = 0; i < 3; ++i)
      for (int d = 0; d < 3; ++d)
        f_single[s][static_cast<size_t>(3 * i + d)] = forces(i, d);
  }

  // One batched call over all three through the public batch entry.
  std::vector<rgpot::ForceInput> in;
  std::vector<rgpot::ForceOut> out;
  std::array<std::array<double, 9>, 3> f_batch{};
  for (size_t s = 0; s < 3; ++s) {
    in.push_back(
        rgpot::ForceInput{3, geoms[s].data(), atmtypes.data(), box.data()});
    out.push_back(rgpot::ForceOut{f_batch[s].data(), 0.0, 0.0});
  }
  const rgpot::ForceBatch batch{3, in.data(), out.data()};
  pot.forceBatch(batch);

  REQUIRE_THAT(out[0].energy, WithinAbs(kHcnAseOmolEnergy, 1e-4));
  for (size_t atom = 0; atom < 3; ++atom) {
    for (size_t component = 0; component < 3; ++component) {
      REQUIRE_THAT(f_batch[0][3 * atom + component],
                   WithinAbs(kHcnAseOmolForces[atom][component], 1e-4));
    }
  }

  // The batched graph is not exactly independent of a system's slot in
  // the batch: fairchem-core 2.23 eager (uma-s-1p1, omol) gives three
  // identical HCN copies energies spread over 1.8e-7 eV. The AOTI band
  // package differs from single calls by up to 1.1e-7 eV in energy and
  // by one float32 rounding in force, 1.1e-7 relative (1.7e-7 eV/A on a
  // 1.48 eV/A component, 1.4e-6 eV/A on 12.9 eV/A). Energy is bounded by
  // 5e-7 eV (2.8x the eager spread); force by 5e-7 relative with a
  // 5e-7 eV/A floor (4.3x the largest relative difference seen).
  constexpr double kBandEnergyTol = 5e-7; // eV
  constexpr double kBandForceRel = 5e-7;  // relative, floor 1 eV/A

  for (size_t s = 0; s < 3; ++s) {
    REQUIRE_THAT(out[s].energy, WithinAbs(e_single[s], kBandEnergyTol));
    for (size_t k = 0; k < 9; ++k) {
      REQUIRE_THAT(f_batch[s][k],
                   WithinAbs(f_single[s][k],
                             kBandForceRel *
                                 std::max(1.0, std::abs(f_single[s][k]))));
    }
  }
}
