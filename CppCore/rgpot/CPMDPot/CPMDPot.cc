// MIT License
// Copyright 2023--present rgpot developers

#include "rgpot/CPMDPot/CPMDPot.hpp"
#include "rgpot/CPMDPot/ConfigMessage.hpp"
#include "rgpot/CalculatorGroup.hpp"

#include <capnp/message.h>
#include <capnp/serialize.h>
#include <kj/array.h>

#include "rgpot/CPMDPot/cpmd_c_abi.h"
#include "rgpot/ParamHash.hpp"
#include "rgpot/abi/CapnpParamsHelper.hpp"
#include "rgpot/abi/EngineDlopen.hpp"
#include "rgpot/abi/ProfileLoader.hpp"
#include "rgpot/rpc/PotentialConfigArm.hpp"
#include "rgpot/units.hpp"

#include <algorithm>
#include <cstddef>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <mutex>
#include <stdexcept>
#include <string>
#include <time.h>
#include <utility>
#include <vector>

namespace rgpot {

using units::HARTREE_PER_BOHR3_TO_EV_PER_ANGSTROM3;
using units::HARTREE_TO_EV;
using units::NEG_GRAD_TO_FORCE;

namespace {

using EnergyGradientFn = CPMDCResult (*)(int, const double *, const int *,
                                         const void *, size_t, double *);
using SetParamsFn = int (*)(const void *, size_t);
using SessionCreateFn = CPMDCSession *(*)(const void *, size_t);
using FeatureCountFn = size_t (*)(void);
using FeatureTableFn = const CPMDCFeatureEntry *(*)(void);
using FeatureFindFn = const CPMDCFeatureEntry *(*)(const char *);
using AdoptCommFn = int (*)(const void *, std::size_t, int);
using SelectOrbitalsFn = int (*)(CPMDCSession *, long long);
using CopyOrbitalsFn = int (*)(CPMDCSession *, long long, long long);
using LastErrorFn = const char *(*)(void);

// Layout matches cpmdc CPMDCStressTensor: int valid, then nine doubles.
struct CPMDCStressTensor {
  int valid;
  double values[9];
};
using LastStressFn = int (*)(CPMDCStressTensor *);
static_assert(offsetof(CPMDCStressTensor, values) == 8,
              "CPMDCStressTensor values follow the valid flag");

void apply_env_hints(const std::string &cpmd_root) {
  if (!cpmd_root.empty()) {
#if !defined(_WIN32)
    setenv("CPMD_ROOT", cpmd_root.c_str(), 1);
#endif
  }
}

struct EngineBundle {
  abi::ProfileLoader profile;
  EnergyGradientFn energy_gradient = nullptr;
  SetParamsFn set_params = nullptr;
  SessionCreateFn session_create = nullptr;
  FeatureCountFn feature_count = nullptr;
  FeatureTableFn feature_table = nullptr;
  FeatureFindFn feature_find = nullptr;
  LastStressFn last_stress = nullptr;
  SelectOrbitalsFn select_orbitals = nullptr;
  CopyOrbitalsFn copy_orbitals = nullptr;
  LastErrorFn last_error = nullptr;
  AdoptCommFn adopt_comm = nullptr;
  std::string load_error;
  bool loaded = false;
};

bool try_load_engine(EngineBundle &b, const std::string &engine_path) {
  b = EngineBundle{};
  const auto cands = abi::engine_candidates(
      "cpmdc", engine_path,
      {"CPMDC_LIBRARY", "RGPOT_CPMDC_ENGINE", "RGPOT_CPMD_ENGINE"});
  std::string eng_err;
  if (!b.profile.open_candidates("cpmdc", cands, &eng_err)) {
    b.load_error = "libcpmdc not loaded: " + eng_err;
    return false;
  }

  auto &lib = b.profile.lib();
  b.energy_gradient = lib.sym_optional<EnergyGradientFn>("cpmdc_energy_gradient");
  b.adopt_comm = lib.sym_optional<AdoptCommFn>("cpmdc_adopt_calculator_comm");
  b.set_params = lib.sym_optional<SetParamsFn>("cpmdc_set_params");
  b.session_create = lib.sym_optional<SessionCreateFn>("cpmdc_session_create");
  b.feature_count = lib.sym_optional<FeatureCountFn>("cpmdc_feature_count");
  b.feature_table = lib.sym_optional<FeatureTableFn>("cpmdc_feature_table");
  b.feature_find = lib.sym_optional<FeatureFindFn>("cpmdc_feature_find");
  b.last_stress = lib.sym_optional<LastStressFn>("cpmdc_last_stress");
  b.select_orbitals =
      lib.sym_optional<SelectOrbitalsFn>("cpmdc_session_select_orbitals");
  b.copy_orbitals =
      lib.sym_optional<CopyOrbitalsFn>("cpmdc_session_copy_orbitals");
  b.last_error = lib.sym_optional<LastErrorFn>("cpmdc_last_error");

  const bool has_one_shot = b.energy_gradient && b.set_params;
  const bool has_session_result =
      b.session_create && b.profile.has_session_result_profile();
  const bool has_feature_discovery =
      b.feature_count && b.feature_table && b.feature_find;
  if (!has_feature_discovery) {
    b.load_error = "engine missing cpmdc feature discovery ABI";
    return false;
  }
  if (!has_one_shot && !has_session_result) {
    b.load_error =
        "engine missing cpmdc session result ABI or one-shot gradient ABI";
    return false;
  }
  b.loaded = true;
  return true;
}

bool register_engine_adopter(const EngineBundle &b) {
  return abi::register_calculator_adopter(
      b.profile.loaded_path(), "cpmdc_adopt_calculator_comm", b.adopt_comm,
      "CPMD communicator callback changed while loading");
}

bool has_session_result_abi(const EngineBundle &b) {
  return b.session_create && b.profile.has_session_result_profile();
}

void publishForceError(const std::string &message) {
  publishCalculatorError(message);
}

bool push_params_to_engine(EngineBundle &b,
                           const std::vector<::capnp::word> &params_words) {
  if (!b.set_params)
    return false;
  const abi::BytesView view = abi::words_view(params_words);
  return b.set_params(view.data, view.size) == 0;
}

std::string read_engine_last_error(const EngineBundle &b) {
  if (!b.last_error)
    return "";
  const char *msg = b.last_error();
  if (!msg || !*msg)
    return "";
  return msg;
}


std::string params_summary(const ::CPMDParams::Reader &params) {
  std::string out = "functional=" + std::string(params.getFunctional().cStr()) +
                    " cutoffRy=" + std::to_string(params.getCutOffRy()) +
                    " charge=" + std::to_string(params.getCharge()) +
                    " mult=" + std::to_string(params.getMultiplicity());
  if (params.getEnginePath().size() > 0)
    out += " enginePath=" + std::string(params.getEnginePath().cStr());
  if (params.getCpmdRoot().size() > 0)
    out += " cpmdRoot=" + std::string(params.getCpmdRoot().cStr());
  return out;
}

void copy_params_to_builder(const ::CPMDParams::Reader &params,
                            ::CPMDParams::Builder out) {
  out.setFunctional(params.getFunctional());
  out.setCutOffRy(params.getCutOffRy());
  out.setCharge(params.getCharge());
  out.setMultiplicity(params.getMultiplicity());
  out.setTask(params.getTask());
  out.setTitle(params.getTitle());
  out.setMemoryMb(params.getMemoryMb());
  out.setScratchDir(params.getScratchDir());
  out.setPermanentDir(params.getPermanentDir());
  out.setCpmdRoot(params.getCpmdRoot());
  out.setEnginePath(params.getEnginePath());
  const auto blocks = params.getInputBlocks();
  auto out_blocks = out.initInputBlocks(blocks.size());
  for (unsigned int i = 0; i < blocks.size(); ++i)
    out_blocks.set(i, blocks[i]);
  out.setInputSections(params.getInputSections());
}

::CPMDParams::Reader
read_params_words(const std::vector<::capnp::word> &params_words,
                  ::capnp::FlatArrayMessageReader &reader) {
  (void)params_words;
  return reader.getRoot<::CPMDParams>();
}

// Every failure out of a force call goes through here. The engine runs
// one calculator across several ranks, and a host that catches the
// exception and exits leaves the peers inside a collective. Marking
// the process turns the exit handler's MPI_Finalize into MPI_Abort.
[[noreturn]] void fail_force(const std::string &message) {
  ::rgpot::abortMpiAtExit();
  throw std::runtime_error(message);
}

// capnp::word forbids copying, so the params a live session was created
// from are kept as bytes and compared bytewise.
std::vector<unsigned char> words_bytes(const std::vector<::capnp::word> &w) {
  const abi::BytesView view = abi::words_view(w);
  const auto *p = static_cast<const unsigned char *>(view.data);
  return std::vector<unsigned char>(p, p + view.size);
}

bool same_words(const std::vector<::capnp::word> &w,
                const std::vector<unsigned char> &bytes) {
  const abi::BytesView view = abi::words_view(w);
  return view.size == bytes.size() &&
         (view.size == 0 ||
          std::memcmp(view.data, bytes.data(), view.size) == 0);
}

std::mutex g_probe_mu;
bool g_probe_done = false;
bool g_probe_ok = false;
bool g_abi_probe_done = false;
bool g_abi_probe_ok = false;

} // namespace

struct CPMDPot::Impl {
  EngineBundle bundle;
  std::vector<::capnp::word> params_words;
  std::string engine_path;
  std::string cpmd_root;
  CPMDCSession *session = nullptr;
  // Params and engine path the live session was created from.
  std::vector<unsigned char> session_params;
  std::string session_engine_path;
  uint64_t params_key = 0;
  mutable std::vector<double> grad_scratch;
  // The calculation the next force belongs to, see selectOrbitals.
  bool has_orbital_key = false;
  int64_t orbital_key = 0;
  // Set when configure leaves the engine loaded but no live session.
  std::string unavailable_reason;

  void destroySession();
  bool configure();
  void recomputeParamsKey();
  void forceSession(const ForceInput &in, ForceOut *out);
  bool applySelectedOrbitalKey();
};

void CPMDPot::Impl::recomputeParamsKey() {
  // The serialized message already carries enginePath and cpmdRoot, so
  // hashing its bytes covers every field the engine sees.
  Fnv1a fp;
  fp.u64(CPMDPot::kKernelVersion);
  const abi::BytesView view = abi::words_view(params_words);
  fp.u64(view.size);
  fp.bytes(view.data, view.size);
  params_key = fp.h;
}

void CPMDPot::Impl::destroySession() {
  if (session && bundle.profile.session_destroy)
    bundle.profile.session_destroy(session);
  session = nullptr;
  session_params.clear();
  session_engine_path.clear();
}

bool CPMDPot::Impl::configure() {
  unavailable_reason.clear();
  if (!register_engine_adopter(bundle)) {
    bundle.load_error = "engine refused calculator communicator";
    bundle.loaded = false;
    destroySession();
    return false;
  }
  if (has_session_result_abi(bundle)) {
    // A session holds the engine's converged wavefunction; recreating
    // it for identical params costs a cold SCF on the next force. Keep
    // the live session when nothing the engine sees has changed.
    if (session && engine_path == session_engine_path &&
        same_words(params_words, session_params))
      return true;
    destroySession();
    const abi::BytesView view = abi::words_view(params_words);
    session = bundle.session_create(view.data, view.size);
    if (!session) {
      unavailable_reason = "CPMD engine refused the session create";
      const std::string detail = read_engine_last_error(bundle);
      if (!detail.empty())
        unavailable_reason += ": " + detail;
      return false;
    }
    // The params are the identity of this session. Commit them only
    // after the remembered orbital key is on it. A refusal drops the
    // session, so a later setParams with the same bytes tries again
    // instead of taking the identical-session return above.
    if (!applySelectedOrbitalKey()) {
      unavailable_reason = "CPMD engine refused the orbital key";
      const std::string detail = read_engine_last_error(bundle);
      if (!detail.empty())
        unavailable_reason += ": " + detail;
      destroySession();
      return false;
    }
    session_params = words_bytes(params_words);
    session_engine_path = engine_path;
    return true;
  }
  destroySession();
  return push_params_to_engine(bundle, params_words);
}

CPMDPot::CPMDPot() : Potential(PotType::CPMD), impl_(new Impl) {
  impl_->params_words = abi::default_capnp_words<::CPMDParams>();
  impl_->recomputeParamsKey();
  apply_env_hints(impl_->cpmd_root);
  if (try_load_engine(impl_->bundle, impl_->engine_path))
    (void)impl_->configure();
}

CPMDPot::CPMDPot(const ::CPMDParams::Reader &params)
    : Potential(PotType::CPMD), impl_(new Impl) {
  impl_->params_words = abi::serialize_capnp(params);
  impl_->recomputeParamsKey();
  impl_->engine_path = params.getEnginePath().cStr();
  impl_->cpmd_root = params.getCpmdRoot().cStr();
  apply_env_hints(impl_->cpmd_root);
  if (try_load_engine(impl_->bundle, impl_->engine_path))
    (void)impl_->configure();
}

void CPMDPot::selectOrbitals(int64_t key) {
  if (!impl_)
    return;
  if (impl_->session && impl_->bundle.select_orbitals &&
      impl_->bundle.select_orbitals(impl_->session,
                                    static_cast<long long>(key)) != 0) {
    throw std::runtime_error("CPMD engine refused the orbital key");
  }
  impl_->has_orbital_key = true;
  impl_->orbital_key = key;
}

bool CPMDPot::keepsOrbitalsPerKey() const {
  return impl_ && impl_->bundle.loaded && impl_->bundle.select_orbitals &&
         has_session_result_abi(impl_->bundle);
}

bool CPMDPot::copyOrbitals(int64_t from_key, int64_t to_key) {
  if (!impl_ || !impl_->bundle.loaded ||
      !has_session_result_abi(impl_->bundle) || !impl_->session ||
      !impl_->bundle.copy_orbitals)
    return false;
  return impl_->bundle.copy_orbitals(impl_->session,
                                     static_cast<long long>(from_key),
                                     static_cast<long long>(to_key)) == 0;
}

uint64_t CPMDPot::paramsKey() const noexcept {
  return impl_ ? impl_->params_key : 0;
}

CPMDPot::~CPMDPot() {
  if (impl_) {
    impl_->destroySession();
    delete impl_;
  }
}

bool CPMDPot::setParams(const ::CPMDParams::Reader &params) {
  if (!impl_)
    impl_ = new Impl;

  const std::string next_engine_path = params.getEnginePath().cStr();
  const std::string next_cpmd_root = params.getCpmdRoot().cStr();
  const bool need_reload =
      !impl_->bundle.loaded || next_engine_path != impl_->engine_path;

  impl_->params_words = abi::serialize_capnp(params);
  impl_->recomputeParamsKey();
  impl_->engine_path = next_engine_path;
  impl_->cpmd_root = next_cpmd_root;
  apply_env_hints(impl_->cpmd_root);

  if (need_reload) {
    impl_->destroySession();
    impl_->bundle = EngineBundle{};
    if (!try_load_engine(impl_->bundle, impl_->engine_path))
      return false;
  } else if (!impl_->bundle.loaded) {
    if (!try_load_engine(impl_->bundle, impl_->engine_path))
      return false;
  }
  if (!impl_->bundle.loaded)
    return false;
  return impl_->configure();
}

void CPMDPot::getParams(::CPMDParams::Builder out) const {
  if (!impl_ || impl_->params_words.empty()) {
    ::capnp::MallocMessageBuilder msg;
    auto params = msg.initRoot<::CPMDParams>();
    copy_params_to_builder(params.asReader(), out);
    return;
  }

  auto words = kj::arrayPtr<const ::capnp::word>(impl_->params_words.data(),
                                                 impl_->params_words.size());
  ::capnp::FlatArrayMessageReader reader(words);
  auto params = read_params_words(impl_->params_words, reader);
  copy_params_to_builder(params, out);
}

bool CPMDPot::setPotentialConfig(const ::PotentialConfig::Reader &cfg,
                                 std::string *message_out) {
  switch (cfg.which()) {
  case ::PotentialConfig::NONE:
    if (message_out)
      *message_out = "no-op (rgpot params: none)";
    return true;
  case ::PotentialConfig::CPMD: {
    const auto cp = cfg.getCpmd();
    const bool ok = setParams(cp);
    if (message_out) {
      *message_out = formatCpmdConfigMessage(
          params_summary(cp), ok, impl_ && impl_->bundle.loaded,
          impl_ ? impl_->unavailable_reason : std::string{});
    }
    return ok;
  }
  default:
    if (message_out)
      *message_out = "rgpot params arm " + rgpot::rpc::potentialConfigArm(cfg) +
                     " not handled by CPMDPot (use matching backend pot)";
    return false;
  }
}

bool CPMDPot::available() const {
  if (!impl_ || !impl_->bundle.loaded)
    return false;
  if (has_session_result_abi(impl_->bundle))
    return impl_->session != nullptr;
  return impl_->bundle.energy_gradient && impl_->bundle.set_params;
}

int CPMDPot::bindCalculators(int ranks_per_calc) {
  EngineBundle bundle;
  // Reject an incompatible engine before creating an MPI communicator.
  if (!try_load_engine(bundle, "") || !bundle.adopt_comm ||
      !register_engine_adopter(bundle))
    return -1;
  return ::rgpot::bindCalculators(ranks_per_calc).index;
}

bool CPMDPot::probeAvailable() {
  return abi::probe_once(g_probe_mu, g_probe_done, g_probe_ok, [] {
    EngineBundle tmp;
    return try_load_engine(tmp, "");
  });
}

bool CPMDPot::abiAvailable() {
  return abi::probe_once(g_probe_mu, g_abi_probe_done, g_abi_probe_ok, [] {
    EngineBundle tmp;
    if (!try_load_engine(tmp, "")) {
      return false;
    }
    return tmp.profile.available != nullptr && tmp.profile.available() != 0;
  });
}

void CPMDPot::forceImpl(const ForceInput &in, ForceOut *out) const {
  std::string error;
  try {
    forceImplOrThrow(in, out);
  } catch (const std::exception &ex) {
    error = ex.what();
  } catch (...) {
    error = "CPMDPot: unknown exception";
  }
  publishForceError(error);
  if (!error.empty())
    throw std::runtime_error(error);
}

void CPMDPot::forceImplOrThrow(const ForceInput &in, ForceOut *out) const {
  // A configured instance can outlive a refused calculator binding.
  if (thisCalculator().index < 0)
    fail_force("CPMDPot: calculator binding was refused");
  if (!available()) {
    if (impl_ && impl_->bundle.loaded) {
      if (!impl_->unavailable_reason.empty())
        fail_force(impl_->unavailable_reason);
      fail_force("CPMD engine session not ready");
    }
    const std::string detail =
        impl_ ? impl_->bundle.load_error : std::string("no impl");
    if (detail.empty())
      fail_force("CPMD engine (libcpmdc) not loaded");
    fail_force(std::string("CPMD engine (libcpmdc) not loaded: ") + detail);
  }

  const int n = static_cast<int>(in.nAtoms);
  if (n <= 0)
    fail_force("CPMDPot: nAtoms must be positive");
  if (!in.pos || !in.atmnrs || !in.box || !out || !out->F)
    fail_force("CPMDPot: null positions/atmnrs/box/forces buffer");

  // CPMD initializes MPI on the first calculation and never finalizes it.
  ::rgpot::finalizeMpiAtExit();

  if (has_session_result_abi(impl_->bundle) && impl_->session) {
    impl_->forceSession(in, out);
    return;
  }

  std::vector<double> &grad = impl_->grad_scratch;
  grad.assign(static_cast<size_t>(n) * 3u, 0.0);
  const abi::BytesView params = abi::words_view(impl_->params_words);
  CPMDCResult res = impl_->bundle.energy_gradient(
      n, in.pos, in.atmnrs, params.data, params.size, grad.data());

  if (!res.ok)
    fail_force(std::string("CPMD engine failed: ") + res.message);

  out->energy = res.energy_h * HARTREE_TO_EV;
  out->variance = 0.0;
  out->has_stress = 0;
  for (int i = 0; i < n * 3; ++i)
    out->F[static_cast<size_t>(i)] =
        grad[static_cast<size_t>(i)] * NEG_GRAD_TO_FORCE;
  if (impl_->bundle.last_stress) {
    CPMDCStressTensor tensor{};
    if (impl_->bundle.last_stress(&tensor) == 0 && tensor.valid) {
      for (int i = 0; i < 9; ++i)
        out->stress[i] =
            tensor.values[i] * HARTREE_PER_BOHR3_TO_EV_PER_ANGSTROM3;
      out->has_stress = 1;
    }
  }
}

void CPMDPot::Impl::forceSession(const ForceInput &in, ForceOut *out) {
  const auto force_words = abi::serialize_force_input_ev(in);
  const abi::BytesView force_view = abi::words_view(force_words);
  const size_t required = bundle.profile.potential_result_size_for_force_input(
      force_view.data, force_view.size);
  if (required == 0)
    fail_force("CPMD engine rejected ForceInput sizing");

  std::vector<::capnp::word> result_words(
      (required + sizeof(::capnp::word) - 1u) / sizeof(::capnp::word));
  size_t written = 0;
  // The engine keeps converged orbitals per key; naming the key on every
  // call keeps it right when the session was recreated since.
  if (has_orbital_key && bundle.select_orbitals &&
      bundle.select_orbitals(session, static_cast<long long>(orbital_key)) != 0)
    fail_force("CPMD engine refused the orbital key");
  abi::ProfileResult res = bundle.profile.session_calculate_result(
      session, force_view.data, force_view.size, result_words.data(),
      result_words.size() * sizeof(::capnp::word), &written);
  if (!res.ok)
    fail_force(std::string("CPMD engine failed: ") + res.message);
  if (written == 0 || written > result_words.size() * sizeof(::capnp::word) ||
      (written % sizeof(::capnp::word)) != 0)
    fail_force("CPMD engine returned invalid PotentialResult");

  auto words = kj::arrayPtr<const ::capnp::word>(
      result_words.data(), written / sizeof(::capnp::word));
  ::capnp::FlatArrayMessageReader reader(words);
  auto result = reader.getRoot<::PotentialResult>();
  const auto forces = result.getForces();
  const size_t expected_force_count = in.nAtoms * 3u;
  if (forces.size() != expected_force_count)
    fail_force("CPMD engine returned wrong force count");
  out->energy = result.getEnergy();
  out->variance = 0.0;
  for (unsigned int i = 0; i < forces.size(); ++i)
    out->F[i] = forces[i];
  out->has_stress = 0;
  const auto stress = result.getStress();
  if (stress.size() == 9) {
    for (unsigned int i = 0; i < 9; ++i)
      out->stress[i] = stress[i];
    out->has_stress = 1;
  }
}

bool CPMDPot::Impl::applySelectedOrbitalKey() {
  if (!session || !has_orbital_key || !bundle.select_orbitals)
    return true;
  return bundle.select_orbitals(session, static_cast<long long>(orbital_key)) ==
         0;
}

} // namespace rgpot
