// MIT License
// Copyright 2023--present rgpot developers
//
// A stand-in backend for the generic engine table: E = 0.5 * sum(p^2) and
// F = -p. The first byte of the configuration picks the reentrancy the
// backend reports (0 shared, 1 per instance, 2 process serial), the second
// byte clears the optional charge/spin, thread, orbital-selection and
// orbital-copy hooks, and the third byte names the model file in
// RGPOT_ENGINE_STUB_MODEL. The library exports how many calls were inside the
// backend at once.

#include "rgpot/engine/EngineTable.hpp"
#include "rgpot/engine_c_abi.h"

#include <algorithm>
#include <atomic>
#include <chrono>
#include <cstdlib>
#include <stdexcept>
#include <thread>

namespace {
std::atomic<int> g_inside{0};
std::atomic<int> g_max{0};
std::atomic<int> g_charge{0};
std::atomic<int> g_spin{0};

class StubPot : public rgpot::PotentialBase {
public:
  explicit StubPot(rgpot::Reentrancy r)
      : PotentialBase(rgpot::PotType::UNKNOWN), reentrancy_(r) {}

  [[nodiscard]] rgpot::PotCaps caps() const noexcept override {
    return {.reentrancy = reentrancy_,
            .perImageInstances = true,
            .analyticHvp = true};
  }

  std::tuple<double, rgpot::types::AtomMatrix, double>
  operator()(const rgpot::types::AtomMatrix &positions,
             const std::vector<int> &,
             const std::array<std::array<double, 3>, 3> &) override {
    return {0.0, rgpot::types::AtomMatrix::Zero(positions.rows(), 3), 0.0};
  }

  void forceBatch(const rgpot::ForceBatch &batch) override {
    const int now = ++g_inside;
    int seen = g_max.load();
    while (now > seen && !g_max.compare_exchange_weak(seen, now)) {
    }
    std::this_thread::sleep_for(std::chrono::milliseconds(5));
    for (size_t s = 0; s < batch.nSystems; ++s) {
      const rgpot::ForceInput &in = batch.in[s];
      double e = 0.0;
      for (size_t k = 0; k < 3 * in.nAtoms; ++k) {
        e += 0.5 * in.pos[k] * in.pos[k];
        batch.out[s].F[k] = -in.pos[k];
      }
      batch.out[s].energy = e;
      batch.out[s].variance = 0.25;
    }
    --g_inside;
  }

  void hessianVectorImpl(const rgpot::ForceInput &in, const double *direction,
                         double *product) const override {
    std::copy_n(direction, 3 * in.nAtoms, product);
  }

private:
  rgpot::Reentrancy reentrancy_;
};
} // namespace

namespace rgpot::engine {

const char *backend_name() { return "stub"; }

Created create(const void *config, std::size_t) {
  const auto *bytes = static_cast<const unsigned char *>(config);
  Created out;
  auto backend = std::make_unique<Backend>();
  backend->pot = std::make_unique<StubPot>(static_cast<Reentrancy>(bytes[0]));
  if (bytes[1] == 0) {
    backend->set_charge_spin = [](int charge, int spin) {
      if (charge == 99) {
        throw rgpot::engine::Unsupported("charge 99 refused");
      }
      g_charge = charge;
      g_spin = spin;
    };
    backend->set_num_threads = [](int intra_op, int inter_op) {
      if (inter_op == 99) {
        throw std::runtime_error("inter-op threads refused");
      }
      (void)intra_op;
    };
    backend->select_orbitals = [](long long key) {
      if (key == -1)
        throw rgpot::engine::Unsupported("orbital key refused");
      if (key == -2)
        throw std::runtime_error("orbital selection failed");
    };
    backend->copy_orbitals = [](long long from_key, long long to_key) {
      if (from_key == -1)
        throw rgpot::engine::Unsupported("orbital copy refused");
      if (to_key == -2)
        throw std::runtime_error("orbital copy failed");
    };
  }
  if (bytes[2] != 0) {
    if (const char *path = std::getenv("RGPOT_ENGINE_STUB_MODEL")) {
      backend->model_path = path;
    }
  }
  out.backend = std::move(backend);
  return out;
}

} // namespace rgpot::engine

extern "C" {
RGPOT_ENGINE_API int stub_max_concurrent(void) { return g_max.load(); }
RGPOT_ENGINE_API void stub_reset(void) {
  g_max = 0;
  g_inside = 0;
  g_charge = 0;
  g_spin = 0;
}
RGPOT_ENGINE_API int stub_charge(void) { return g_charge.load(); }
RGPOT_ENGINE_API int stub_spin(void) { return g_spin.load(); }
}
