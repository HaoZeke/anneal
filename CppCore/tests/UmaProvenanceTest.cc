// Copyright (c) rgpot developers
//
// The exporter records where the UMA checkpoint came from and the SHA-256
// of the file it loaded. UmaPot reads those keys back; a package written
// before them still loads and reports them absent. Meson builds the
// packages and the stand-in checkpoint, and passes their paths in.

#include <cstdlib>
#include <filesystem>
#include <stdexcept>
#include <string>
#include <unordered_map>
#include <vector>

#include <catch2/catch_all.hpp>

#include "rgpot/UmaPot/UmaPot.hpp"
#include "rgpot/UmaPot/UmaProvenance.hpp"
#include "rgpot/engine/Sha256.hpp"

namespace {

using Meta = std::unordered_map<std::string, std::string>;

const std::string kDigest =
    "b5bb9d8014a0f9b1d61e21e796d78dccdf1352f23cd32812f4850b878ae4944c";
const std::string kCommit = "0123456789abcdef0123456789abcdef01234567";

std::string env_path(const char *name) {
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

double line_energy(rgpot::UmaPot &pot) {
  std::vector<double> pos = {12.5, 12.5, 12.5, 12.5, 12.5,
                             13.5, 12.5, 12.5, 11.5};
  const std::vector<int> z = {6, 7, 1};
  const double box[9] = {25, 0, 0, 0, 25, 0, 0, 0, 25};
  std::vector<double> force(pos.size(), 0.0);
  rgpot::ForceOut slot{force.data(), 0.0, 0.0};
  const rgpot::ForceInput in{pos.size() / 3, pos.data(), z.data(), box};
  pot.forceImpl(in, &slot);
  return slot.energy;
}

} // namespace

TEST_CASE("checkpoint provenance reads a Hugging Face source",
          "[uma][provenance]") {
  const Meta meta = {{"checkpoint_source", "hf"},
                     {"checkpoint_hf_repo", "facebook/UMA"},
                     {"checkpoint_hf_filename", "checkpoints/uma-s-1p1.pt"},
                     {"checkpoint_hf_revision", kCommit},
                     {"checkpoint_sha256", kDigest},
                     {"cutoff", "6.0"}};
  const auto prov = rgpot::UmaCheckpointProvenance::fromMetadata(meta);
  REQUIRE(prov.present());
  REQUIRE(prov.source == "hf");
  REQUIRE_FALSE(prov.path);
  REQUIRE(prov.hf_repo == "facebook/UMA");
  REQUIRE(prov.hf_filename == "checkpoints/uma-s-1p1.pt");
  REQUIRE(prov.hf_revision == kCommit);
  REQUIRE(prov.sha256 == kDigest);
}

TEST_CASE("checkpoint provenance is absent when the keys are",
          "[uma][provenance]") {
  const Meta meta = {{"cutoff", "6.0"}, {"checkpoint_sha256", ""}};
  const auto prov = rgpot::UmaCheckpointProvenance::fromMetadata(meta);
  REQUIRE_FALSE(prov.present());
  REQUIRE_FALSE(prov.source);
  REQUIRE_FALSE(prov.path);
  REQUIRE_FALSE(prov.hf_repo);
  REQUIRE_FALSE(prov.hf_filename);
  REQUIRE_FALSE(prov.hf_revision);
  REQUIRE_FALSE(prov.sha256);
}

TEST_CASE("checkpoint provenance refuses a malformed digest or source",
          "[uma][provenance]") {
  REQUIRE_THROWS_WITH(rgpot::UmaCheckpointProvenance::fromMetadata(
                          {{"checkpoint_sha256", "abc"}}),
                      Catch::Matchers::ContainsSubstring("checkpoint_sha256"));
  REQUIRE_THROWS_WITH(rgpot::UmaCheckpointProvenance::fromMetadata(
                          {{"checkpoint_source", "s3"}}),
                      Catch::Matchers::ContainsSubstring("checkpoint_source"));
  REQUIRE_THROWS_WITH(rgpot::UmaCheckpointProvenance::fromMetadata(
                          {{"checkpoint_hf_revision", "main"}}),
                      Catch::Matchers::ContainsSubstring("commit"));
}

TEST_CASE("UmaPot names the digest of the checkpoint a package was built from",
          "[uma][provenance]") {
  const std::string weights = env_path("UMA_PROVENANCE_WEIGHTS");
  rgpot::UmaPot pot = open_pot(env_path("UMA_PROVENANCE_PT2"));
  const auto &prov = pot.checkpointProvenance();
  REQUIRE(prov.source == "cli");
  REQUIRE(prov.path == std::filesystem::canonical(weights).string());
  REQUIRE(prov.sha256 == rgpot::engine::Sha256::of_file(weights));
  REQUIRE_FALSE(prov.hf_repo);
  REQUIRE_FALSE(prov.hf_filename);
  REQUIRE_FALSE(prov.hf_revision);
  REQUIRE_THAT(line_energy(pot), Catch::Matchers::WithinAbs(112.5, 1e-9));
}

TEST_CASE("a package without provenance loads and reports none",
          "[uma][provenance]") {
  rgpot::UmaPot pot = open_pot(env_path("UMA_FLOAT64_PT2"));
  const auto &prov = pot.checkpointProvenance();
  REQUIRE_FALSE(prov.present());
  REQUIRE_FALSE(prov.sha256);
  REQUIRE_THAT(line_energy(pot), Catch::Matchers::WithinAbs(112.5, 1e-9));
}
