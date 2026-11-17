// MIT License
// Copyright 2023--present rgpot developers

#include "rgpot/NWChemPot/NWChemPot.hpp"

#include <capnp/message.h>
#include <capnp/serialize.h>
#include <kj/array.h>

#include "rgpot/NWChemPot/DynLib.hpp"
#include "rgpot/NWChemPot/nwchem_c_abi.h"
#include "rgpot/rpc/PotentialConfigArm.hpp"
#include "rgpot/units.hpp"

#include <cstdlib>
#include <cstring>
#include <mutex>
#include <stdexcept>
#include <string>
#include <utility>
#include <vector>

namespace rgpot {

using units::HARTREE_TO_EV;
using units::NEG_GRAD_TO_FORCE;

namespace {

using EnergyGradientFn = NWChemCResult (*)(int, const double *, const int *,
                                           const void *, size_t, double *);
using SetParamsFn = int (*)(const void *, size_t);
using VersionFn = const char *(*)(void);
using AvailableFn = int (*)(void);

struct ParamsView {
  const void *data = nullptr;
  size_t size = 0;
};

std::vector<std::string>
engine_lib_candidates(const std::string &explicit_path) {
  std::vector<std::string> out;
  if (!explicit_path.empty())
    out.emplace_back(explicit_path);
  if (const char *e = std::getenv("NWCHEMC_LIBRARY"); e && *e)
    out.emplace_back(e);
  if (const char *e = std::getenv("RGPOT_NWCHEMC_ENGINE"); e && *e)
    out.emplace_back(e);
  if (const char *e = std::getenv("RGPOT_NWCHEM_ENGINE"); e && *e)
    out.emplace_back(e);
  out.emplace_back("libnwchemc.so");
  out.emplace_back("./libnwchemc.so");
  out.emplace_back("libnwchemc.dylib");
  out.emplace_back("./libnwchemc.dylib");
  out.emplace_back("nwchemc.dll");
  return out;
}

void apply_env_hints(const std::string &nwchem_root) {
  if (!nwchem_root.empty()) {
#if !defined(_WIN32)
    setenv("NWCHEM_TOP", nwchem_root.c_str(), 1);
    if (!std::getenv("NWCHEM_BASIS_LIBRARY") ||
        !std::getenv("NWCHEM_BASIS_LIBRARY")[0]) {
      std::string bas = nwchem_root + "/src/basis/libraries/";
      setenv("NWCHEM_BASIS_LIBRARY", bas.c_str(), 0);
    }
#endif
  }
}

struct EngineBundle {
  DynLib engine_lib;
  EnergyGradientFn energy_gradient = nullptr;
  SetParamsFn set_params = nullptr;
  VersionFn version = nullptr;
  AvailableFn available = nullptr;
  std::string load_error;
  bool loaded = false;
};

// NWChem/GA/MA session state and engine-owned exit handlers have process
// lifetime. Keep one owning handle for each resolved function table, so
// destroying a calculator neither finalizes a live peer nor reloads the engine.
void retain_engine(EngineBundle &bundle) {
  struct RetainedEngine {
    EnergyGradientFn energy_gradient;
    SetParamsFn set_params;
    VersionFn version;
    AvailableFn available;
    DynLib library;
  };
  static std::mutex mutex;
  // The registry has no static destructor: the engine owns finalization and
  // its callbacks require the library mapping through process teardown.
  static auto *engines = new std::vector<RetainedEngine>;
  std::lock_guard<std::mutex> lock(mutex);
  for (const auto &engine : *engines) {
    if (engine.energy_gradient == bundle.energy_gradient &&
        engine.set_params == bundle.set_params &&
        engine.version == bundle.version &&
        engine.available == bundle.available) {
      bundle.engine_lib.close();
      return;
    }
  }
  engines->push_back({bundle.energy_gradient, bundle.set_params,
                      bundle.version, bundle.available,
                      std::move(bundle.engine_lib)});
}

bool try_load_engine(EngineBundle &b, const std::string &engine_path) {
  b.load_error.clear();
  b.loaded = false;
  b.energy_gradient = nullptr;
  b.set_params = nullptr;
  b.version = nullptr;
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
    b.load_error = "libnwchemc not loaded: " + eng_err;
    return false;
  }

  b.energy_gradient =
      b.engine_lib.sym_optional<EnergyGradientFn>("nwchemc_energy_gradient");
  b.set_params = b.engine_lib.sym_optional<SetParamsFn>("nwchemc_set_params");
  b.version = b.engine_lib.sym_optional<VersionFn>("nwchemc_version");
  b.available = b.engine_lib.sym_optional<AvailableFn>("nwchemc_available");

  if (!b.energy_gradient) {
    b.load_error = "engine missing nwchemc_energy_gradient";
    return false;
  }
  if (!b.set_params) {
    b.load_error = "engine missing nwchemc_set_params";
    return false;
  }
  retain_engine(b);
  b.loaded = true;
  return true;
}

// NWChemPot is host-only dlopen of libnwchemc.so (no second "embed mode").
// Cap'n Proto fields enginePath / nwchemRoot are for *this* process to find
// and env-hint the library; the loaded ABI rejects non-empty enginePath in
// set_params / energy_gradient (not NWChem method input). Strip them from
// the blob passed across the dlopen boundary so multi-call SCF works.
std::vector<::capnp::word>
serialize_params_for_abi(const ::NWChemParams::Reader &params) {
  // Deep-copy all schema fields, then clear host-only library paths so the
  // loaded libnwchemc rejects non-empty enginePath/nwchemRoot in set_params.
  ::capnp::MallocMessageBuilder msg;
  msg.setRoot(params);
  auto out = msg.getRoot<::NWChemParams>();
  out.setEnginePath("");
  out.setNwchemRoot("");
  auto words = ::capnp::messageToFlatArray(msg);
  std::vector<::capnp::word> flat(words.size());
  std::memcpy(flat.data(), words.begin(),
              words.size() * sizeof(::capnp::word));
  return flat;
}

// Full params for getParams() / host bookkeeping (includes enginePath).
std::vector<::capnp::word>
serialize_params(const ::NWChemParams::Reader &params) {
  ::capnp::MallocMessageBuilder msg;
  msg.setRoot(params);
  auto words = ::capnp::messageToFlatArray(msg);
  std::vector<::capnp::word> out(words.size());
  std::memcpy(out.data(), words.begin(), words.size() * sizeof(::capnp::word));
  return out;
}

std::vector<::capnp::word> default_params() {
  ::capnp::MallocMessageBuilder msg;
  auto params = msg.initRoot<::NWChemParams>();
  return serialize_params(params.asReader());
}

ParamsView params_view(const std::vector<::capnp::word> &words) {
  ParamsView view;
  if (!words.empty()) {
    view.data = words.data();
    view.size = words.size() * sizeof(::capnp::word);
  }
  return view;
}

bool push_params_to_engine(EngineBundle &b,
                           const std::vector<::capnp::word> &params_words) {
  if (!b.set_params)
    return false;
  const ParamsView view = params_view(params_words);
  return b.set_params(view.data, view.size) == 0;
}

std::string params_summary(const ::NWChemParams::Reader &params) {
  std::string out = "basis=" + std::string(params.getBasis().cStr()) +
                    " theory=" + std::string(params.getTheory().cStr()) +
                    " scfType=" + std::string(params.getScfType().cStr()) +
                    " charge=" + std::to_string(params.getCharge()) +
                    " mult=" + std::to_string(params.getMultiplicity());
  if (params.getEnginePath().size() > 0)
    out += " enginePath=" + std::string(params.getEnginePath().cStr());
  if (params.getNwchemRoot().size() > 0)
    out += " nwchemRoot=" + std::string(params.getNwchemRoot().cStr());
  return out;
}

void copy_params_to_builder(const ::NWChemParams::Reader &params,
                            ::NWChemParams::Builder out) {
  out.setBasis(params.getBasis());
  out.setTheory(params.getTheory());
  out.setScfType(params.getScfType());
  out.setCharge(params.getCharge());
  out.setMultiplicity(params.getMultiplicity());
  out.setEnginePath(params.getEnginePath());
  out.setNwchemRoot(params.getNwchemRoot());
  out.setTask(params.getTask());
  out.setTitle(params.getTitle());
  out.setMemoryMb(params.getMemoryMb());
  out.setScratchDir(params.getScratchDir());
  out.setPermanentDir(params.getPermanentDir());
  const auto blocks = params.getInputBlocks();
  auto out_blocks = out.initInputBlocks(blocks.size());
  for (unsigned int i = 0; i < blocks.size(); ++i)
    out_blocks.set(i, blocks[i]);
  out.setInputStanzas(params.getInputStanzas());
}

::NWChemParams::Reader
read_params_words(const std::vector<::capnp::word> &params_words,
                  ::capnp::FlatArrayMessageReader &reader) {
  (void)params_words;
  return reader.getRoot<::NWChemParams>();
}

std::mutex g_probe_mu;
bool g_probe_done = false;
bool g_probe_ok = false;
bool g_abi_probe_done = false;
bool g_abi_probe_ok = false;

} // namespace

struct NWChemPot::Impl {
  EngineBundle bundle;
  // Full params for getParams() (may include enginePath / nwchemRoot).
  std::vector<::capnp::word> params_words;
  // ABI blob (enginePath/nwchemRoot cleared) for set_params / energy_gradient.
  std::vector<::capnp::word> abi_params_words;
  std::string engine_path;
  std::string nwchem_root;
  // Reused gradient buffer for the engine ABI; avoids a per-call heap
  // allocation in forceImpl. forceImpl is const, hence mutable.
  mutable std::vector<double> grad_scratch;

  void store(const ::NWChemParams::Reader &params) {
    params_words = serialize_params(params);
    abi_params_words = serialize_params_for_abi(params);
  }
};

NWChemPot::NWChemPot() : Potential(PotType::NWChem), impl_(new Impl) {
  {
    ::capnp::MallocMessageBuilder msg;
    auto p = msg.initRoot<::NWChemParams>();
    impl_->store(p.asReader());
  }
  apply_env_hints(impl_->nwchem_root);
  if (try_load_engine(impl_->bundle, impl_->engine_path))
    (void)push_params_to_engine(impl_->bundle, impl_->abi_params_words);
}

NWChemPot::NWChemPot(const ::NWChemParams::Reader &params)
    : Potential(PotType::NWChem), impl_(new Impl) {
  impl_->store(params);
  impl_->engine_path = params.getEnginePath().cStr();
  impl_->nwchem_root = params.getNwchemRoot().cStr();
  apply_env_hints(impl_->nwchem_root);
  if (try_load_engine(impl_->bundle, impl_->engine_path))
    (void)push_params_to_engine(impl_->bundle, impl_->abi_params_words);
}

NWChemPot::~NWChemPot() { delete impl_; }

bool NWChemPot::setParams(const ::NWChemParams::Reader &params) {
  if (!impl_)
    impl_ = new Impl;

  const std::string next_engine_path = params.getEnginePath().cStr();
  const std::string next_nwchem_root = params.getNwchemRoot().cStr();

  impl_->store(params);
  impl_->engine_path = next_engine_path;
  impl_->nwchem_root = next_nwchem_root;
  apply_env_hints(impl_->nwchem_root);

  // Do not dlclose libnwchemc once loaded — NWChem/GA/MA are process-global.
  if (!impl_->bundle.loaded) {
    if (!try_load_engine(impl_->bundle, impl_->engine_path))
      return false;
  }
  if (!impl_->bundle.loaded)
    return false;
  return push_params_to_engine(impl_->bundle, impl_->abi_params_words);
}

void NWChemPot::getParams(::NWChemParams::Builder out) const {
  if (!impl_ || impl_->params_words.empty()) {
    ::capnp::MallocMessageBuilder msg;
    auto params = msg.initRoot<::NWChemParams>();
    copy_params_to_builder(params.asReader(), out);
    return;
  }

  auto words = kj::arrayPtr<const ::capnp::word>(impl_->params_words.data(),
                                                 impl_->params_words.size());
  ::capnp::FlatArrayMessageReader reader(words);
  auto params = read_params_words(impl_->params_words, reader);
  copy_params_to_builder(params, out);
}

bool NWChemPot::setPotentialConfig(const ::PotentialConfig::Reader &cfg,
                                   std::string *message_out) {
  switch (cfg.which()) {
  case ::PotentialConfig::NONE:
    if (message_out)
      *message_out = "no-op (rgpot params: none)";
    return true;
  case ::PotentialConfig::NWCHEM: {
    const auto nw = cfg.getNwchem();
    const bool ok = setParams(nw);
    if (message_out) {
      const std::string sum = params_summary(nw);
      if (ok)
        *message_out = "rgpot params applied (nwchem arm): " + sum;
      else if (!available())
        *message_out =
            "rgpot params nwchem arm failed (engine not loaded): " + sum;
      else
        *message_out = "rgpot params nwchem arm rejected by embed: " + sum;
    }
    return ok;
  }
  default:
    if (message_out)
      *message_out = "rgpot params arm " + rgpot::rpc::potentialConfigArm(cfg) +
                     " not handled by NWChemPot (use matching backend pot)";
    return false;
  }
}

bool NWChemPot::available() const {
  return impl_ && impl_->bundle.loaded && impl_->bundle.energy_gradient &&
         impl_->bundle.set_params;
}

// Keep a process-lifetime engine handle for probes. Real libnwchemc (and NWChem
// Fortran runtime) is not safe to dlclose and reload in the same process; a
// temporary EngineBundle destructor would unload symbols and poison later calls.
EngineBundle &probe_engine_bundle() {
  static EngineBundle retained;
  return retained;
}

bool NWChemPot::probeAvailable() {
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

bool NWChemPot::abiAvailable() {
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

void NWChemPot::forceImpl(const ForceInput &in, ForceOut *out) const {
  if (!available()) {
    throw std::runtime_error(
        std::string("NWChem engine (libnwchemc) not loaded: ") +
        (impl_ ? impl_->bundle.load_error : "no impl"));
  }

  const int n = static_cast<int>(in.nAtoms);
  if (n <= 0)
    throw std::runtime_error("NWChemPot: nAtoms must be positive");
  if (!in.pos || !in.atmnrs || !out || !out->F)
    throw std::runtime_error("NWChemPot: null positions/atmnrs/forces buffer");

  std::vector<double> &grad = impl_->grad_scratch;
  grad.assign(static_cast<size_t>(n) * 3u, 0.0);
  const ParamsView params = params_view(impl_->abi_params_words.empty()
                                            ? impl_->params_words
                                            : impl_->abi_params_words);
  NWChemCResult res = impl_->bundle.energy_gradient(
      n, in.pos, in.atmnrs, params.data, params.size, grad.data());

  if (!res.ok) {
    throw std::runtime_error(std::string("NWChem engine failed: ") +
                             res.message);
  }

  out->energy = res.energy_h * HARTREE_TO_EV;
  out->variance = 0.0;
  for (int i = 0; i < n * 3; ++i)
    out->F[static_cast<size_t>(i)] =
        grad[static_cast<size_t>(i)] * NEG_GRAD_TO_FORCE;
}

} // namespace rgpot
