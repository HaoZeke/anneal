// MIT License — UMA engine behind the generic rgpot engine C ABI
// (torch linked into this .so only). The table itself is
// rgpot/engine/EngineTable.cc; this file builds the potential from UmaParams.

#include "rgpot/engine/EngineTable.hpp"
#include "rgpot/engine/TorchRuntime.hpp"
#include "rgpot/UmaPot/UmaPot.hpp"

#include <capnp/message.h>
#include <capnp/serialize.h>
#include <kj/io.h>

#include "rgpot/rpc/Potentials.capnp.h"

namespace rgpot::engine {

const char *backend_name() { return "uma"; }

Created create(const void *config, std::size_t config_len) {
  Created out;
  try {
    const kj::ArrayPtr<const capnp::word> words(
        reinterpret_cast<const capnp::word *>(config),
        config_len / sizeof(capnp::word));
    capnp::FlatArrayMessageReader reader(words);
    const auto params = reader.getRoot<::UmaParams>();

    UmaConfig c;
    c.model_path = params.getModelPath().cStr();
    if (c.model_path.empty()) {
      out.error = "rgpot_engine_create(uma): modelPath required";
      return out;
    }
    if (params.getTaskName().size() > 0)
      c.task_name = params.getTaskName().cStr();
    if (params.getDevice().size() > 0)
      c.device = params.getDevice().cStr();
    c.charge = params.getCharge();
    c.spin = params.getSpin() > 0 ? params.getSpin() : 1;
    if (params.getCutoff() > 0.0)
      c.cutoff = params.getCutoff();
    if (params.getMaxNeighbors() > 0)
      c.max_neighbors = params.getMaxNeighbors();
    if (params.hasCacheDir())
      c.cache_dir = params.getCacheDir().cStr();

    set_torch_threads(params.getIntraopThreads(), params.getInteropThreads());
    set_torch_deterministic(params.getDeterministicAlgorithms());

    auto backend = std::make_unique<Backend>();
    auto pot = std::make_unique<UmaPot>(c);
    UmaPot *raw = pot.get();
    backend->pot = std::move(pot);
    backend->model_path = c.model_path;
    backend->set_charge_spin = [raw](int charge, int spin) {
      raw->setChargeSpin(charge, spin);
    };
    backend->set_num_threads = [](int intra_op, int inter_op) {
      set_torch_threads(intra_op, inter_op);
    };
    out.backend = std::move(backend);
  } catch (const std::exception &e) {
    out.error = e.what();
  } catch (...) {
    out.error = "rgpot_engine_create(uma): unknown exception";
  }
  return out;
}

} // namespace rgpot::engine
