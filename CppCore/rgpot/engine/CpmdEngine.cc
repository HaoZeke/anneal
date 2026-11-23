// MIT License
// Copyright 2023--present rgpot developers
//
// Generic rgpot engine table for CPMD (libcpmdc through CPMDPot).
// The configuration is the CPMDParams message.

#include "rgpot/engine/EngineTable.hpp"
#include "rgpot/CPMDPot/CPMDPot.hpp"

#include <capnp/message.h>
#include <capnp/serialize.h>
#include <kj/io.h>

#include <cstdint>
#include <stdexcept>

namespace rgpot::engine {

const char *backend_name() { return "cpmd"; }

Created create(const void *config, std::size_t config_len) {
  Created out;
  try {
    const kj::ArrayPtr<const capnp::word> words(
        reinterpret_cast<const capnp::word *>(config),
        config_len / sizeof(capnp::word));
    capnp::FlatArrayMessageReader reader(words);
    auto backend = std::make_unique<Backend>();
    auto pot = std::make_unique<CPMDPot>(reader.getRoot<::CPMDParams>());
    backend->pot = std::move(pot);
    auto *raw = static_cast<CPMDPot *>(backend->pot.get());
    if (raw->keepsOrbitalsPerKey()) {
      backend->select_orbitals = [raw](long long key) {
        if (!raw->keepsOrbitalsPerKey()) {
          throw Unsupported("CPMD engine does not keep orbitals per key");
        }
        raw->selectOrbitals(static_cast<std::int64_t>(key));
      };
    }
    out.backend = std::move(backend);
  } catch (const std::exception &e) {
    out.error = e.what();
  } catch (...) {
    out.error = "rgpot_engine_create(cpmd): unknown exception";
  }
  return out;
}

} // namespace rgpot::engine
