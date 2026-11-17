#pragma once
// MIT License
// Copyright 2023--present rgpot developers

/**
 * @brief Capability handshake for backends loaded through the potential
 * profile: decide from a `Capabilities` message whether this build can drive
 * the backend before any work is dispatched, and fill the same message from
 * the producing side.
 *
 * Metadata a backend leaves unset (empty family, zero bridge major) is not a
 * declaration and is not checked. The constants mirror the eindir-core
 * objective ABI stamp that rgpot-core compares against.
 */

#include "rgpot/rpc/Potentials.capnp.h"

#include <cstdint>
#include <cstring>
#include <stdexcept>
#include <string>
#include <vector>

#include <capnp/message.h>
#include <kj/async.h>
#include <capnp/serialize.h>
#include <kj/array.h>

namespace rgpot {
namespace abi {

// Build identity of this rgpot build, defined by the build system. Both are
// empty when unknown. The schema fields exist only in Potentials.capnp
// revisions that carry buildVersion/buildRevision; RGPOT_SCHEMA_HAS_BUILD_IDENTITY
// is defined by the build system when the compiled schema has them.
#ifndef RGPOT_BUILD_VERSION
#define RGPOT_BUILD_VERSION ""
#endif
#ifndef RGPOT_BUILD_REVISION
#define RGPOT_BUILD_REVISION ""
#endif

inline constexpr const char *kProtocolFamily = "rgpot.potentials";
inline constexpr uint16_t kProtocolMajor = 1;
inline constexpr uint16_t kProtocolMinor = 1;
inline constexpr const char *kSchemaId = "0xbd1f89fa17369103";

/** What the host requires of a backend. */
struct Expectation {
  std::string family = kProtocolFamily;
  uint16_t protocol_major = kProtocolMajor;
  uint16_t protocol_minor_min = 0; //!< Lowest protocol minor accepted.
  std::string schema_id = kSchemaId;
  uint16_t bridge_abi_major = 1;
  uint16_t bridge_abi_minor_max = 0; //!< Highest bridge minor consumable.
  uint32_t bridge_layout = 1;
  uint16_t dlpack_major = 1;
  uint16_t dlpack_minor_max = 0;     //!< Highest DLPack minor consumable.
  uint64_t bridge_features = 0x3;    //!< gradient | batch.
  std::vector<::Capabilities::Operation> required_operations = {
      ::Capabilities::Operation::ENERGY, ::Capabilities::Operation::FORCES};
};

/** Empty when @p caps is acceptable, else the first incompatibility. */
inline std::string check_capabilities(::Capabilities::Reader caps,
                                      const Expectation &want = {}) {
  const std::string family = caps.getProtocolFamily().cStr();
  if (!family.empty()) {
    if (family != want.family)
      return "protocol family \"" + family + "\" is not \"" + want.family + "\"";
    if (caps.getProtocolMajor() != want.protocol_major)
      return "protocol major " + std::to_string(caps.getProtocolMajor()) +
             " is not " + std::to_string(want.protocol_major);
    if (caps.getProtocolMinor() < want.protocol_minor_min)
      return "protocol minor " + std::to_string(caps.getProtocolMinor()) +
             " is below the required " +
             std::to_string(want.protocol_minor_min);
  }
  const std::string schema_id = caps.getSchemaId().cStr();
  if (!schema_id.empty() && schema_id != want.schema_id)
    return "schema id " + schema_id + " is not " + want.schema_id;
  if (caps.getBridgeAbiMajor() != 0) {
    if (caps.getBridgeAbiMajor() != want.bridge_abi_major)
      return "eindir bridge ABI major " +
             std::to_string(caps.getBridgeAbiMajor()) + " is not " +
             std::to_string(want.bridge_abi_major);
    if (caps.getBridgeAbiMinor() > want.bridge_abi_minor_max)
      return "eindir bridge ABI minor " +
             std::to_string(caps.getBridgeAbiMinor()) +
             " exceeds the supported " +
             std::to_string(want.bridge_abi_minor_max);
    if (caps.getBridgeLayout() != want.bridge_layout)
      return "eindir objective layout " +
             std::to_string(caps.getBridgeLayout()) + " is not " +
             std::to_string(want.bridge_layout);
    if ((caps.getBridgeFeatures() & ~want.bridge_features) != 0)
      return "eindir bridge features include bits outside the supported set";
  }
  if (caps.getDlpackMajor() != 0) {
    if (caps.getDlpackMajor() != want.dlpack_major)
      return "DLPack major " + std::to_string(caps.getDlpackMajor()) +
             " is not " + std::to_string(want.dlpack_major);
    if (caps.getDlpackMinor() > want.dlpack_minor_max)
      return "DLPack minor " + std::to_string(caps.getDlpackMinor()) +
             " exceeds the supported " + std::to_string(want.dlpack_minor_max);
  }
  for (const auto required : want.required_operations) {
    bool served = false;
    for (const auto op : caps.getOperations())
      served = served || op == required;
    if (!served)
      return "a required operation is not served";
  }
  return {};
}

/** Fill the compatibility fields of @p caps with what this build speaks. */
inline void fill_compatibility(::Capabilities::Builder caps) {
  const Expectation want;
  caps.setProtocolFamily(want.family);
  caps.setProtocolMajor(want.protocol_major);
  caps.setProtocolMinor(kProtocolMinor);
  caps.setSchemaId(want.schema_id);
  caps.setBridgeAbiMajor(want.bridge_abi_major);
  caps.setBridgeAbiMinor(want.bridge_abi_minor_max);
  caps.setBridgeLayout(want.bridge_layout);
  caps.setDlpackMajor(want.dlpack_major);
  caps.setDlpackMinor(want.dlpack_minor_max);
  caps.setBridgeFeatures(want.bridge_features);
#ifdef RGPOT_SCHEMA_HAS_BUILD_IDENTITY
  caps.setBuildVersion(RGPOT_BUILD_VERSION);
  caps.setBuildRevision(RGPOT_BUILD_REVISION);
#endif
}

/**
 * Ask a connected server for its `Capabilities` and return empty when it is
 * compatible, else the reason. A server built before `Potential.getCapabilities`
 * existed answers UNIMPLEMENTED; that is reported as a missing call with the
 * upgrade path, never as a successful handshake.
 */
inline std::string check_server(::Potential::Client &server,
                                kj::WaitScope &wait_scope,
                                const Expectation &want = {}) {
  try {
    auto response = server.getCapabilitiesRequest().send().wait(wait_scope);
    return check_capabilities(response.getCapabilities(), want);
  } catch (const kj::Exception &ex) {
    if (ex.getType() == kj::Exception::Type::UNIMPLEMENTED)
      return "the server does not implement Potential.getCapabilities; "
             "upgrade the server to an rgpot release whose Potentials.capnp "
             "carries it";
    return std::string("capabilities request failed: ") +
           ex.getDescription().cStr();
  }
}

} // namespace abi
} // namespace rgpot
