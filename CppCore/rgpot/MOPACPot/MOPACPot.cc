// MIT License
// Copyright 2023--present rgpot developers

#include "rgpot/MOPACPot/MOPACPot.hpp"

#include "rgpot/MOPACPot/mopac_c_abi.h"
#include "rgpot/abi/DynLib.hpp"
#include "rgpot/abi/EngineDlopen.hpp"
#include "rgpot/units.hpp"

#include <cstring>
#include <mutex>
#include <stdexcept>
#include <string>
#include <vector>

namespace rgpot {

using units::HARTREE_TO_EV;
using units::NEG_GRAD_TO_FORCE;

namespace {

using EnergyGradientFn = MopacCResult (*)(int, const double *, const int *,
                                          const void *, size_t, double *);
using SetParamsFn = int (*)(const void *, size_t);
using AbiVersionFn = int (*)(void);
using AvailableFn = int (*)(void);

struct EngineBundle {
  DynLib engine_lib;
  EnergyGradientFn energy_gradient = nullptr;
  SetParamsFn set_params = nullptr;
  AbiVersionFn abi_version = nullptr;
  AvailableFn available = nullptr;
  std::string load_error;
  bool loaded = false;
};

bool try_load_engine(EngineBundle &b, const std::string &engine_path) {
  b = EngineBundle{};
  const auto cands = abi::engine_candidates(
      "mopacc", engine_path, {"MOPACC_LIBRARY", "RGPOT_MOPACC_ENGINE"});
  std::string loaded_path;
  std::string eng_err;
  if (!abi::open_first_candidate(b.engine_lib, cands, &loaded_path, &eng_err)) {
    b.load_error = "libmopacc not loaded: " + eng_err;
    return false;
  }

  b.energy_gradient =
      b.engine_lib.sym_optional<EnergyGradientFn>("mopacc_energy_gradient");
  b.set_params = b.engine_lib.sym_optional<SetParamsFn>("mopacc_set_params");
  b.abi_version =
      b.engine_lib.sym_optional<AbiVersionFn>("mopacc_c_abi_version");
  b.available = b.engine_lib.sym_optional<AvailableFn>("mopacc_available");
  if (!b.energy_gradient) {
    b.load_error = "engine missing mopacc_energy_gradient";
    return false;
  }
  if (b.abi_version && b.abi_version() != RGPOT_MOPACC_C_ABI_VERSION) {
    b.load_error = "incompatible mopacc ABI";
    return false;
  }
  if (b.available && !b.available()) {
    b.load_error = "mopacc stub (libmopac not linked)";
    return false;
  }
  b.loaded = true;
  return true;
}

std::mutex g_probe_mu;
bool g_probe_done = false;
bool g_probe_ok = false;
bool g_abi_probe_done = false;
bool g_abi_probe_ok = false;

EngineBundle &probe_engine_bundle() {
  static EngineBundle retained;
  return retained;
}

} // namespace

struct MOPACPot::Impl {
  EngineBundle bundle;
  MopacCParams params{};
  std::string engine_path;
  mutable std::vector<double> grad_scratch;
};

MOPACPot::MOPACPot() : MOPACPot(Config{}) {}

MOPACPot::MOPACPot(const Config &cfg)
    : Potential(PotType::MOPAC), impl_(new Impl) {
  impl_->params.charge = cfg.charge;
  impl_->params.spin = cfg.spin;
  impl_->params.model = cfg.model;
  impl_->params.tolerance = cfg.tolerance;
  impl_->params.max_time = cfg.max_time;
  impl_->engine_path = cfg.engine_path;
  if (try_load_engine(impl_->bundle, impl_->engine_path)) {
    if (impl_->bundle.set_params)
      (void)impl_->bundle.set_params(&impl_->params, sizeof(impl_->params));
  }
}

MOPACPot::~MOPACPot() { delete impl_; }

bool MOPACPot::available() const {
  return impl_ && impl_->bundle.loaded && impl_->bundle.energy_gradient;
}

bool MOPACPot::probeAvailable() {
  return abi::probe_once(g_probe_mu, g_probe_done, g_probe_ok, [] {
    EngineBundle &tmp = probe_engine_bundle();
    return tmp.loaded || try_load_engine(tmp, "");
  });
}

bool MOPACPot::abiAvailable() {
  return abi::probe_once(g_probe_mu, g_abi_probe_done, g_abi_probe_ok, [] {
    EngineBundle &tmp = probe_engine_bundle();
    if (!tmp.loaded && !try_load_engine(tmp, "")) {
      return false;
    }
    return tmp.available != nullptr && tmp.available() != 0;
  });
}

void MOPACPot::forceImpl(const ForceInput &in, ForceOut *out) const {
  if (!available()) {
    throw std::runtime_error(std::string("MOPAC engine (libmopacc) not loaded: ") +
                             (impl_ ? impl_->bundle.load_error : "no impl"));
  }
  const int n = static_cast<int>(in.nAtoms);
  if (n <= 0)
    throw std::runtime_error("MOPACPot: nAtoms must be positive");
  if (!in.pos || !in.atmnrs || !out || !out->F)
    throw std::runtime_error("MOPACPot: null positions/atmnrs/forces buffer");

  std::vector<double> &grad = impl_->grad_scratch;
  grad.assign(static_cast<size_t>(n) * 3u, 0.0);
  MopacCResult res = impl_->bundle.energy_gradient(
      n, in.pos, in.atmnrs, &impl_->params, sizeof(impl_->params), grad.data());
  if (!res.ok) {
    throw std::runtime_error(std::string("mopacc AM1 failed: ") + res.message);
  }
  out->energy = res.energy_h * HARTREE_TO_EV;
  out->variance = 0.0;
  for (int i = 0; i < n * 3; ++i)
    out->F[static_cast<size_t>(i)] =
        grad[static_cast<size_t>(i)] * NEG_GRAD_TO_FORCE;
}

} // namespace rgpot
