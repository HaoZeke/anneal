// MIT License
// Copyright 2023--present rgpot developers
//
// Generic rgpot engine table for a weighted sum of potentials. The config
// root is CompositeParams: one CompositeTerm per child, each naming its
// backend with a PotentialSpec. rgpot_engine_last_components reports what
// each term returned.

#include <capnp/message.h>
#include <capnp/serialize.h>
#include <kj/io.h>

#include "rgpot/engine/EngineTable.hpp"
#include "rgpot/rpc/CompositeConfigure.hpp"

namespace rgpot::engine {

const char *backend_name() { return "composite"; }

Created create(const void *config, std::size_t config_len) {
  Created out;
  try {
    const kj::ArrayPtr<const capnp::word> words(
        reinterpret_cast<const capnp::word *>(config), config_len / sizeof(capnp::word));
    capnp::FlatArrayMessageReader reader(words);
    auto sum = rgpot::rpc::compositeFromParams(reader.getRoot<::CompositeParams>());
    const SumPot *view = sum.get();
    auto backend = std::make_unique<Backend>();
    backend->pot = std::move(sum);
    backend->components = [view]() -> const std::vector<SumComponent> & {
      return view->lastComponents();
    };
    out.backend = std::move(backend);
  } catch (const std::exception &e) {
    out.error = std::string("rgpot_engine_create(composite): ") + e.what();
  } catch (...) {
    out.error = "rgpot_engine_create(composite): unknown exception";
  }
  return out;
}

} // namespace rgpot::engine
