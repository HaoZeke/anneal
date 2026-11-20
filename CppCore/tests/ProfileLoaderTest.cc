// MIT License
// Copyright 2023--present rgpot developers

#include <catch2/catch_test_macros.hpp>
#include <catch2/matchers/catch_matchers_floating_point.hpp>
#include <catch2/matchers/catch_matchers_string.hpp>

#include <capnp/message.h>
#include <capnp/serialize.h>
#include <kj/array.h>

#include <cctype>
#include <cstdlib>
#include <string>
#include <vector>

#include "rgpot/abi/Compat.hpp"
#include "rgpot/abi/ProfileLoader.hpp"
#include "rgpot/rpc/Potentials.capnp.h"

using Catch::Matchers::WithinAbs;

namespace {

std::string fake_engine_path() {
  const char *path = std::getenv("RGPOT_CPMD_ENGINE");
  REQUIRE(path != nullptr);
  return path;
}

std::vector<unsigned char> flat_bytes(::capnp::MallocMessageBuilder &msg) {
  auto words = ::capnp::messageToFlatArray(msg);
  const auto bytes = words.asBytes();
  return std::vector<unsigned char>(bytes.begin(), bytes.end());
}

std::vector<unsigned char> make_config() {
  ::capnp::MallocMessageBuilder msg;
  auto config = msg.initRoot<::PotentialConfig>();
  auto cpmd = config.initCpmd();
  cpmd.setFunctional("BLYP");
  cpmd.setCutOffRy(70.0);
  return flat_bytes(msg);
}

std::vector<unsigned char> make_force_input(double cell_zz) {
  ::capnp::MallocMessageBuilder msg;
  auto input = msg.initRoot<::ForceInput>();
  auto pos = input.initPos(9);
  const double coords[9] = {0.0, 0.0, 0.0, 0.96, 0.0, 0.0, -0.24, 0.93, 0.0};
  for (unsigned int i = 0; i < 9; ++i)
    pos.set(i, coords[i]);
  auto atmnrs = input.initAtmnrs(3);
  atmnrs.set(0, 8);
  atmnrs.set(1, 1);
  atmnrs.set(2, 1);
  auto box = input.initBox(9);
  for (unsigned int i = 0; i < 9; ++i)
    box.set(i, 0.0);
  box.set(0, 10.0);
  box.set(4, 10.0);
  box.set(8, cell_zz);
  input.setLengthUnit("angstrom");
  input.setEnergyUnit("eV");
  return flat_bytes(msg);
}

} // namespace

TEST_CASE("ProfileLoader resolves the full minimum profile from one prefix",
          "[abi][profile]") {
  rgpot::abi::ProfileLoader loader;
  loader.load("cpmdc", fake_engine_path());
  REQUIRE(loader.loaded());
  REQUIRE(loader.prefix() == "cpmdc");
  REQUIRE(loader.abi_version() == 0);
  REQUIRE(loader.available() == 1);
  REQUIRE(std::string(loader.version()).find("fake") != std::string::npos);
  REQUIRE(std::string(loader.last_error()).empty());
}

TEST_CASE("ProfileLoader capabilities round-trips a Capabilities message",
          "[abi][profile]") {
  rgpot::abi::ProfileLoader loader;
  loader.load("cpmdc", fake_engine_path());

  const auto caps_bytes = loader.capabilities();
  REQUIRE(caps_bytes.size() % sizeof(::capnp::word) == 0);
  auto words = kj::arrayPtr(
      reinterpret_cast<const ::capnp::word *>(caps_bytes.data()),
      caps_bytes.size() / sizeof(::capnp::word));
  ::capnp::FlatArrayMessageReader reader(words);
  auto caps = reader.getRoot<::Capabilities>();

  REQUIRE(std::string(caps.getBackendName().cStr()) == "cpmdc");
  REQUIRE(caps.getAvailable());
  auto ops = caps.getOperations();
  REQUIRE(ops.size() == 3);
  REQUIRE(ops[0] == ::Capabilities::Operation::ENERGY);
  REQUIRE(ops[1] == ::Capabilities::Operation::FORCES);
  REQUIRE(ops[2] == ::Capabilities::Operation::GRADIENT);
  auto kinds = caps.getConfigKinds();
  REQUIRE(kinds.size() == 1);
  REQUIRE(std::string(kinds[0].cStr()) == "cpmd");
}

TEST_CASE("ProfileLoader drives a config -> session -> step evaluation",
          "[abi][profile]") {
  rgpot::abi::ProfileLoader loader;
  loader.load("cpmdc", fake_engine_path());

  const auto config = make_config();
  const double cell_zz = 12.5;
  const auto step = make_force_input(cell_zz);

  REQUIRE(loader.configure(config.data(), config.size()) == 0);

  void *session =
      loader.session_create_from_config(config.data(), config.size());
  REQUIRE(session != nullptr);
  REQUIRE(loader.session_configure(session, config.data(), config.size()) ==
          0);

  const size_t need =
      loader.potential_result_size_for_force_input(step.data(), step.size());
  REQUIRE(need > 0);
  std::vector<unsigned char> out(need);
  size_t wrote = 0;
  auto result = loader.session_calculate_result(session, step.data(),
                                                step.size(), out.data(),
                                                out.size(), &wrote);
  REQUIRE(result.ok == 1);
  REQUIRE(wrote > 0);
  loader.session_destroy(session);

  auto words = kj::arrayPtr(
      reinterpret_cast<const ::capnp::word *>(out.data()),
      wrote / sizeof(::capnp::word));
  ::capnp::FlatArrayMessageReader reader(words);
  auto decoded = reader.getRoot<::PotentialResult>();
  REQUIRE_THAT(decoded.getEnergy(), WithinAbs(0.75 + 0.001 * cell_zz, 1e-12));
  REQUIRE(decoded.getForces().size() == 9);

  std::vector<unsigned char> out_one_shot(need);
  size_t wrote_one_shot = 0;
  auto one_shot = loader.calculate_result_from_config(
      config.data(), config.size(), step.data(), step.size(),
      out_one_shot.data(), out_one_shot.size(), &wrote_one_shot);
  REQUIRE(one_shot.ok == 1);
  REQUIRE(wrote_one_shot == wrote);
  loader.finalize();
}

TEST_CASE("ProfileLoader rejects a library missing the profile symbols",
          "[abi][profile]") {
  rgpot::abi::ProfileLoader loader;
  REQUIRE_THROWS_AS(loader.load("nwchemc", fake_engine_path()),
                    std::runtime_error);
}

namespace {

// Capabilities builder with the host's own compatibility fields plus the
// required operations, edited by the caller to break exactly one of them.
template <typename Edit>
std::string verdict(Edit edit, const rgpot::abi::Expectation &want = {}) {
  ::capnp::MallocMessageBuilder msg;
  auto caps = msg.initRoot<::Capabilities>();
  rgpot::abi::fill_compatibility(caps);
  auto ops = caps.initOperations(2);
  ops.set(0, ::Capabilities::Operation::ENERGY);
  ops.set(1, ::Capabilities::Operation::FORCES);
  edit(caps);
  return rgpot::abi::check_capabilities(caps.asReader(), want);
}

} // namespace

TEST_CASE("Capabilities handshake accepts matching metadata",
          "[abi][profile][compat]") {
  REQUIRE(verdict([](::Capabilities::Builder) {}).empty());
}

TEST_CASE("Capabilities handshake does not check unstated metadata",
          "[abi][profile][compat]") {
  ::capnp::MallocMessageBuilder msg;
  auto caps = msg.initRoot<::Capabilities>();
  auto ops = caps.initOperations(2);
  ops.set(0, ::Capabilities::Operation::ENERGY);
  ops.set(1, ::Capabilities::Operation::FORCES);
  REQUIRE(rgpot::abi::check_capabilities(caps.asReader()).empty());
}

TEST_CASE("Capabilities handshake rejection matrix",
          "[abi][profile][compat]") {
  using B = ::Capabilities::Builder;
  struct Row {
    const char *name;
    std::string why;
    const char *needle;
  };
  const Row rows[] = {
      {"family", verdict([](B c) { c.setProtocolFamily("other.family"); }),
       "protocol family"},
      {"major", verdict([](B c) { c.setProtocolMajor(2); }), "protocol major"},
      {"schema", verdict([](B c) { c.setSchemaId("0xdeadbeef"); }),
       "schema id"},
      {"bridge major", verdict([](B c) { c.setBridgeAbiMajor(2); }),
       "bridge ABI major"},
      {"bridge minor", verdict([](B c) { c.setBridgeAbiMinor(9); }),
       "bridge ABI minor"},
      {"layout", verdict([](B c) { c.setBridgeLayout(7); }), "objective layout"},
      {"features", verdict([](B c) { c.setBridgeFeatures(1ULL << 40); }),
       "bridge features"},
      {"dlpack major", verdict([](B c) { c.setDlpackMajor(2); }),
       "DLPack major"},
      {"dlpack minor", verdict([](B c) { c.setDlpackMinor(9); }),
       "DLPack minor"},
      {"operation", verdict([](B c) { c.initOperations(1).set(0, ::Capabilities::Operation::ENERGY); }),
       "operation"},
  };
  for (const auto &row : rows) {
    INFO(row.name);
    REQUIRE_FALSE(row.why.empty());
    REQUIRE(row.why.find(row.needle) != std::string::npos);
  }

  rgpot::abi::Expectation strict;
  strict.protocol_minor_min = 2;
  REQUIRE(verdict([](B) {}, strict).find("protocol minor") !=
          std::string::npos);
  REQUIRE(verdict([](B c) { c.setProtocolMinor(2); }, strict).empty());
}

TEST_CASE("Capabilities handshake accepts a schema id with an optional 0x prefix",
          "[abi][profile][compat]") {
  const std::string stored = rgpot::abi::kSchemaId;
  REQUIRE(stored.rfind("0x", 0) == 0);
  const std::string bare = stored.substr(2);
  REQUIRE(bare != stored);
  REQUIRE(verdict([&](::Capabilities::Builder caps) { caps.setSchemaId(bare); })
              .empty());

  std::string upper = stored;
  for (char &ch : upper)
    ch = static_cast<char>(std::toupper(static_cast<unsigned char>(ch)));
  REQUIRE(upper != stored);
  REQUIRE(verdict([&](::Capabilities::Builder caps) { caps.setSchemaId(upper); })
              .empty());

  std::string prefixed = "0X";
  for (char ch : bare)
    prefixed.push_back(
        static_cast<char>(std::toupper(static_cast<unsigned char>(ch))));
  REQUIRE(verdict([&](::Capabilities::Builder caps) {
            caps.setSchemaId(prefixed);
          }).empty());

  const auto mismatch =
      verdict([](::Capabilities::Builder caps) { caps.setSchemaId("deadbeef"); });
  REQUIRE(mismatch.find("schema id") != std::string::npos);
}

TEST_CASE("checked_load accepts the fake engine, whose capabilities are "
          "filled by the producer helper",
          "[abi][profile][compat]") {
  rgpot::abi::ProfileLoader loader;
  rgpot::abi::checked_load(loader, "cpmdc", fake_engine_path());
  REQUIRE(loader.loaded());

  ::capnp::MallocMessageBuilder probe;
  auto bytes = loader.capabilities();
  auto words = kj::arrayPtr(
      reinterpret_cast<const ::capnp::word *>(bytes.data()),
      bytes.size() / sizeof(::capnp::word));
  ::capnp::FlatArrayMessageReader reader(words);
  auto caps = reader.getRoot<::Capabilities>();
  REQUIRE(std::string(caps.getProtocolFamily().cStr()) ==
          rgpot::abi::kProtocolFamily);
  REQUIRE(caps.getProtocolMajor() == rgpot::abi::kProtocolMajor);
  REQUIRE(std::string(caps.getSchemaId().cStr()) == rgpot::abi::kSchemaId);
#ifdef RGPOT_SCHEMA_HAS_BUILD_IDENTITY
  REQUIRE(std::string(caps.getBuildVersion().cStr()) == RGPOT_BUILD_VERSION);
  REQUIRE_FALSE(std::string(caps.getBuildVersion().cStr()).empty());
  REQUIRE(std::string(caps.getBuildRevision().cStr()) == RGPOT_BUILD_REVISION);
#endif
}

TEST_CASE("checked_load refuses an incompatible backend before dispatch",
          "[abi][profile][compat]") {
  rgpot::abi::ProfileLoader loader;
  rgpot::abi::Expectation other;
  other.family = "someone.else";
  REQUIRE_THROWS_WITH(
      rgpot::abi::checked_load(loader, "cpmdc", fake_engine_path(), other),
      Catch::Matchers::ContainsSubstring("backend refused: protocol family"));
  REQUIRE_FALSE(loader.loaded());
}

#include <capnp/ez-rpc.h>

namespace {

// A server built before Potential.getCapabilities existed: only calculate is
// overridden, so the generated default answers UNIMPLEMENTED.
class OldServer final : public Potential::Server {
public:
  kj::Promise<void> calculate(CalculateContext) override {
    return kj::READY_NOW;
  }
};

class CapabilitiesServer final : public Potential::Server {
public:
  explicit CapabilitiesServer(bool other_family = false)
      : other_family_(other_family) {}
  kj::Promise<void> calculate(CalculateContext) override {
    return kj::READY_NOW;
  }
  kj::Promise<void> getCapabilities(GetCapabilitiesContext context) override {
    auto caps = context.getResults().initCapabilities();
    auto ops = caps.initOperations(2);
    ops.set(0, ::Capabilities::Operation::ENERGY);
    ops.set(1, ::Capabilities::Operation::FORCES);
    rgpot::abi::fill_compatibility(caps);
    if (other_family_)
      caps.setProtocolFamily("other.family");
    return kj::READY_NOW;
  }

private:
  bool other_family_;
};

template <typename Server> std::string server_verdict(Server server) {
  capnp::EzRpcServer rpc_server(kj::heap<Server>(std::move(server)),
                                "127.0.0.1", 0);
  auto &wait_scope = rpc_server.getWaitScope();
  const unsigned port = rpc_server.getPort().wait(wait_scope);
  capnp::EzRpcClient rpc_client("127.0.0.1", port);
  auto cap = rpc_client.getMain<Potential>();
  return rgpot::abi::check_server(cap, rpc_client.getWaitScope());
}

} // namespace

TEST_CASE("check_server accepts a server that reports compatible capabilities",
          "[abi][compat][rpc]") {
  REQUIRE(server_verdict(CapabilitiesServer()).empty());
}

TEST_CASE("check_server refuses a server without getCapabilities and names it",
          "[abi][compat][rpc]") {
  const std::string why = server_verdict(OldServer());
  REQUIRE(why.find("Potential.getCapabilities") != std::string::npos);
  REQUIRE(why.find("upgrade the server") != std::string::npos);
}

TEST_CASE("check_server refuses a server with another protocol family",
          "[abi][compat][rpc]") {
  const std::string why = server_verdict(CapabilitiesServer(true));
  REQUIRE(why.find("protocol family") != std::string::npos);
}
