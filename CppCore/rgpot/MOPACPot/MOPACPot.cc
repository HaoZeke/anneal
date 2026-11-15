// MIT License
// Copyright 2023--present rgpot developers

#include "rgpot/MOPACPot/MOPACPot.hpp"

#include "rgpot/MOPACPot/DynLib.hpp"
#include "rgpot/MOPACPot/mopac_c_abi.h"
#include "rgpot/units.hpp"

#include <cstdlib>
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

std::vector<std::string>
engine_lib_candidates(const std::string &explicit_path) {
  std::vector<std::string> out;
  if (!explicit_path.empty())
    out.emplace_back(explicit_path);
  if (const char *e = std::getenv("MOPACC_LIBRARY"))
    out.emplace_back(e);
  if (const char *e = std::getenv("RGPOT_MOPACC_ENGINE"))
    out.emplace_back(e);
  out.emplace_back("libmopacc.so");
  out.emplace_back("./libmopacc.so");
  out.emplace_back("libmopacc.dylib");
  out.emplace_back("./libmopacc.dylib");
  out.emplace_back("mopacc.dll");
  return out;
}

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
  b.load_error.clear();
  b.loaded = false;
  b.energy_gradient = nullptr;
  b.set_params = nullptr;
  b.abi_version = nullptr;
  b.available = nullptr;

  bool eng_ok = false;
  std::string eng_err;
  for (const auto &cand : engine_lib_candidates(engine_path)) {
    try {
      b.engine_lib.open(cand);
      eng_ok = true;
      break;
    } catch (const std::exception &ex) {
      eng_err = ex.what();
    }
  }
  if (!eng_ok) {
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
  std::lock_guard<std::mutex> lock(g_probe_mu);
  if (g_probe_done)
    return g_probe_ok;
  EngineBundle &tmp = probe_engine_bundle();
  if (!tmp.loaded)
    g_probe_ok = try_load_engine(tmp, "");
  else
    g_probe_ok = true;
  g_probe_done = true;
  return g_probe_ok;
}

bool MOPACPot::abiAvailable() {
  std::lock_guard<std::mutex> lock(g_probe_mu);
  if (g_abi_probe_done)
    return g_abi_probe_ok;
  EngineBundle &tmp = probe_engine_bundle();
  if (!tmp.loaded && !try_load_engine(tmp, "")) {
    g_abi_probe_ok = false;
  } else if (tmp.available) {
    g_abi_probe_ok = tmp.available() != 0;
  } else {
    g_abi_probe_ok = false;
  }
  g_abi_probe_done = true;
  return g_abi_probe_ok;
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
