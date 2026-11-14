// MIT License
// Copyright 2023--present rgpot developers

#include "rgpot/CP2KPot/CP2KPot.hpp"
#include "rgpot/CalculatorGroup.hpp"

#include <capnp/message.h>
#include <capnp/serialize.h>
#include <kj/array.h>

#include "rgpot/CP2KPot/CP2KDeck.hpp"
#include "rgpot/CP2KPot/cp2kc_abi.h"
#include "rgpot/NWChemPot/DynLib.hpp"
#include "rgpot/ParamHash.hpp"
#include "rgpot/rpc/PotentialConfigArm.hpp"
#include "rgpot/units.hpp"

#include <algorithm>
#include <atomic>
#include <cstdlib>
#include <cstring>
#include <filesystem>
#include <mutex>
#include <stdexcept>
#include <string>
#include <system_error>
#include <utility>
#include <vector>

#if !defined(_WIN32)
#include <unistd.h>
#endif

namespace rgpot {

using units::ANGSTROM_TO_BOHR;
using units::HARTREE_BOHR_TO_EV_ANGSTROM;
using units::HARTREE_TO_EV;

namespace {

using AbiVersionFn = int (*)(void);
using VersionFn = const char *(*)(void);
using AvailableFn = int (*)(void);
using AdoptCommFn = int (*)(const void *, std::size_t, int);
using SessionCreateFn = CP2KCSession *(*)(const char *, const char *, char *,
                                          size_t);
using SelectOrbitalsFn = int (*)(CP2KCSession *, long long);
using CalculateFn = int (*)(CP2KCSession *, int, const double *, const double *,
                            double *, double *, char *, size_t);
using SessionDestroyFn = void (*)(CP2KCSession *);

std::vector<std::string>
engine_lib_candidates(const std::string &explicit_path) {
  std::vector<std::string> out;
  if (!explicit_path.empty())
    out.emplace_back(explicit_path);
  if (const char *e = std::getenv("CP2KC_LIBRARY"))
    out.emplace_back(e);
  if (const char *e = std::getenv("RGPOT_CP2K_ENGINE"))
    out.emplace_back(e);
  out.emplace_back("libcp2kc_rgpot.so");
  out.emplace_back("./libcp2kc_rgpot.so");
  out.emplace_back("libcp2kc_rgpot.dylib");
  out.emplace_back("./libcp2kc_rgpot.dylib");
  return out;
}

struct EngineBundle {
  DynLib engine_lib;
  AbiVersionFn abi_version = nullptr;
  VersionFn version = nullptr;
  AvailableFn available = nullptr;
  AdoptCommFn adopt_comm = nullptr;
  SessionCreateFn session_create = nullptr;
  SelectOrbitalsFn select_orbitals = nullptr;
  CalculateFn calculate = nullptr;
  SessionDestroyFn session_destroy = nullptr;
  std::string loaded_path;
  std::string load_error;
  bool loaded = false;
};

bool try_load_engine(EngineBundle &b, const std::string &engine_path) {
  b = EngineBundle{};
  std::string err;
  bool opened = false;
  for (const auto &cand : engine_lib_candidates(engine_path)) {
    try {
      b.engine_lib.open(cand);
      b.loaded_path = cand;
      opened = true;
      break;
    } catch (const std::exception &ex) {
      err = ex.what();
    }
  }
  if (!opened) {
    b.load_error = "libcp2kc_rgpot not loaded: " + err;
    return false;
  }
  auto &lib = b.engine_lib;
  b.abi_version = lib.sym_optional<AbiVersionFn>("cp2kc_abi_version");
  b.version = lib.sym_optional<VersionFn>("cp2kc_version");
  b.available = lib.sym_optional<AvailableFn>("cp2kc_available");
  b.adopt_comm = lib.sym_optional<AdoptCommFn>("cp2kc_adopt_calculator_comm");
  b.session_create = lib.sym_optional<SessionCreateFn>("cp2kc_session_create");
  b.select_orbitals =
      lib.sym_optional<SelectOrbitalsFn>("cp2kc_session_select_orbitals");
  b.calculate = lib.sym_optional<CalculateFn>("cp2kc_session_calculate");
  b.session_destroy =
      lib.sym_optional<SessionDestroyFn>("cp2kc_session_destroy");
  if (!b.abi_version || b.abi_version() != CP2KC_ABI_VERSION) {
    b.load_error = b.loaded_path + " does not speak cp2kc ABI " +
                   std::to_string(CP2KC_ABI_VERSION);
    return false;
  }
  if (!b.session_create || !b.calculate || !b.session_destroy) {
    b.load_error = b.loaded_path + " lacks the cp2kc session functions";
    return false;
  }
  if (b.available && b.available() == 0) {
    b.load_error = b.loaded_path + " reports CP2K unavailable";
    return false;
  }
  b.loaded = true;
  return true;
}

bool register_engine_adopter(const EngineBundle &b) {
  if (!b.adopt_comm)
    return true;
  struct PinnedEngine {
    DynLib library;
    AdoptCommFn callback;
  };
  // The registry keeps one handle per callback until process exit, as
  // CPMDPot does: a registered callback must stay mapped.
  static auto *engines = new std::vector<PinnedEngine>;
  static std::mutex mutex;
  std::lock_guard<std::mutex> lock(mutex);
  for (const auto &engine : *engines) {
    if (engine.callback == b.adopt_comm)
      return addCalculatorCommAdopter(engine.callback);
  }
  DynLib library(b.loaded_path);
  if (library.sym_optional<AdoptCommFn>("cp2kc_adopt_calculator_comm") !=
      b.adopt_comm)
    throw std::runtime_error(
        "CP2K communicator callback changed while loading");
  engines->push_back({std::move(library), b.adopt_comm});
  return addCalculatorCommAdopter(b.adopt_comm);
}

std::vector<::capnp::word>
serialize_params(const ::CP2KParams::Reader &params) {
  ::capnp::MallocMessageBuilder msg;
  msg.setRoot(params);
  auto words = ::capnp::messageToFlatArray(msg);
  std::vector<::capnp::word> out(words.size());
  std::memcpy(out.data(), words.begin(), words.size() * sizeof(::capnp::word));
  return out;
}

std::vector<::capnp::word> default_params() {
  ::capnp::MallocMessageBuilder msg;
  auto params = msg.initRoot<::CP2KParams>();
  return serialize_params(params.asReader());
}

void copy_text_list(::capnp::List<::capnp::Text>::Reader in,
                    ::capnp::List<::capnp::Text>::Builder out) {
  for (unsigned int i = 0; i < in.size(); ++i)
    out.set(i, in[i]);
}

void copy_params(const ::CP2KParams::Reader &p, ::CP2KParams::Builder out) {
  out.setInputTemplate(p.getInputTemplate());
  out.setFunctional(p.getFunctional());
  out.setBasisSet(p.getBasisSet());
  out.setPseudopotential(p.getPseudopotential());
  out.setBasisSetFile(p.getBasisSetFile());
  out.setPotentialFile(p.getPotentialFile());
  out.setCutoffRy(p.getCutoffRy());
  out.setRelCutoffRy(p.getRelCutoffRy());
  out.setCharge(p.getCharge());
  out.setMultiplicity(p.getMultiplicity());
  out.setUks(p.getUks());
  out.setPeriodic(p.getPeriodic());
  out.setEpsScf(p.getEpsScf());
  out.setMaxScf(p.getMaxScf());
  out.setOuterMaxScf(p.getOuterMaxScf());
  out.setOt(p.getOt());
  out.setDataDir(p.getDataDir());
  out.setWorkDir(p.getWorkDir());
  out.setKeepWorkDir(p.getKeepWorkDir());
  out.setEnginePath(p.getEnginePath());
  copy_text_list(p.getExtraDftSections(),
                 out.initExtraDftSections(p.getExtraDftSections().size()));
  copy_text_list(
      p.getExtraForceEvalSections(),
      out.initExtraForceEvalSections(p.getExtraForceEvalSections().size()));
}

std::string params_summary(const ::CP2KParams::Reader &p) {
  std::string out = "functional=" + std::string(p.getFunctional().cStr()) +
                    " cutoffRy=" + std::to_string(p.getCutoffRy()) +
                    " charge=" + std::to_string(p.getCharge()) +
                    " mult=" + std::to_string(p.getMultiplicity()) +
                    " uks=" + (p.getUks() ? "true" : "false");
  if (p.getInputTemplate().size() > 0)
    out += " inputTemplate";
  if (p.getEnginePath().size() > 0)
    out += " enginePath=" + std::string(p.getEnginePath().cStr());
  return out;
}

// As in CPMDPot: a failure on one rank leaves its peers inside a
// collective, so the exit handler has to abort the world.
[[noreturn]] void fail_force(const std::string &message) {
  ::rgpot::abortMpiAtExit();
  throw std::runtime_error(message);
}

std::filesystem::path run_directory(const std::string &work_dir,
                                    std::size_t serial) {
  std::filesystem::path parent;
  if (!work_dir.empty())
    parent = work_dir;
  else if (const char *t = std::getenv("TMPDIR"); t && *t)
    parent = t;
  else
    parent = std::filesystem::temp_directory_path();
  const auto &group = thisCalculator();
#if defined(_WIN32)
  const long pid = 0;
#else
  const long pid = static_cast<long>(::getpid());
#endif
  // The instance serial keeps two potentials of one process apart.
  return parent /
         ("rgpot-cp2k-" + std::to_string(pid) + "-g" +
          std::to_string(group.index) + "-r" +
          std::to_string(group.rank_in_group) + "-i" + std::to_string(serial));
}

std::atomic<std::size_t> g_instance_serial{0};
std::mutex g_probe_mu;
bool g_probe_done = false;
bool g_probe_ok = false;

} // namespace

struct CP2KPot::Impl {
  EngineBundle bundle;
  std::vector<::capnp::word> params_words;
  std::string engine_path;
  std::string params_error;
  uint64_t params_key = 0;
  std::size_t serial = g_instance_serial.fetch_add(1);

  CP2KCSession *session = nullptr;
  std::vector<int> session_numbers;
  std::filesystem::path run_dir;
  bool keep_run_dir = false;

  bool has_orbital_key = false;
  int64_t orbital_key = 0;
  std::vector<double> pos_bohr;
  std::vector<double> forces;

  kj::ArrayPtr<const ::capnp::word> words() const {
    return kj::arrayPtr<const ::capnp::word>(params_words.data(),
                                             params_words.size());
  }

  void recomputeParamsKey() {
    Fnv1a fp;
    fp.u64(CP2KPot::kKernelVersion);
    fp.u64(params_words.size());
    fp.bytes(params_words.data(), params_words.size() * sizeof(::capnp::word));
    params_key = fp.h;
  }

  void destroySession() {
    if (session && bundle.session_destroy)
      bundle.session_destroy(session);
    session = nullptr;
    session_numbers.clear();
    if (!run_dir.empty() && !keep_run_dir) {
      std::error_code ec;
      std::filesystem::remove_all(run_dir, ec);
    }
    run_dir.clear();
  }

  void store(const ::CP2KParams::Reader &params) {
    params_words = serialize_params(params);
    recomputeParamsKey();
    engine_path = params.getEnginePath().cStr();
    params_error.clear();
    if (params.getInputTemplate().size() > 0)
      params_error = cp2k::checkTemplate(params.getInputTemplate());
  }

  bool load() {
    if (!try_load_engine(bundle, engine_path))
      return false;
    if (!register_engine_adopter(bundle)) {
      bundle.load_error = "CP2K shim refused the calculator communicator";
      bundle.loaded = false;
      return false;
    }
    return true;
  }

  void createSession(const ForceInput &in);
};

void CP2KPot::Impl::createSession(const ForceInput &in) {
  destroySession();
  ::capnp::FlatArrayMessageReader rd(words());
  const auto params = rd.getRoot<::CP2KParams>();
  std::string deck;
  try {
    deck = cp2k::renderDeck(params, in.nAtoms, in.pos, in.atmnrs, in.box);
  } catch (const std::exception &ex) {
    fail_force(std::string("CP2KPot: ") + ex.what());
  }
  if (params.getDataDir().size() > 0) {
#if !defined(_WIN32)
    setenv("CP2K_DATA_DIR", params.getDataDir().cStr(), 1);
#endif
  }
  keep_run_dir = params.getKeepWorkDir();
  const auto dir = run_directory(params.getWorkDir().cStr(), serial);
  std::error_code ec;
  std::filesystem::create_directories(dir, ec);
  if (ec)
    fail_force("CP2KPot: cannot create " + dir.string() + ": " + ec.message());
  run_dir = dir;
  char err[512] = {};
  session = bundle.session_create(deck.c_str(), dir.string().c_str(), err,
                                  sizeof(err));
  if (!session)
    fail_force(std::string("CP2K shim refused the deck: ") + err);
  session_numbers.assign(in.atmnrs, in.atmnrs + in.nAtoms);
}

CP2KPot::CP2KPot() : Potential(PotType::CP2K), impl_(new Impl) {
  impl_->params_words = default_params();
  impl_->recomputeParamsKey();
  (void)impl_->load();
}

CP2KPot::CP2KPot(const ::CP2KParams::Reader &params)
    : Potential(PotType::CP2K), impl_(new Impl) {
  impl_->store(params);
  (void)impl_->load();
}

CP2KPot::~CP2KPot() {
  if (impl_) {
    impl_->destroySession();
    delete impl_;
  }
}

uint64_t CP2KPot::paramsKey() const noexcept {
  return impl_ ? impl_->params_key : 0;
}

bool CP2KPot::setParams(const ::CP2KParams::Reader &params) {
  const auto next = serialize_params(params);
  const bool same = next.size() == impl_->params_words.size() &&
                    std::memcmp(next.data(), impl_->params_words.data(),
                                next.size() * sizeof(::capnp::word)) == 0;
  const std::string next_engine = params.getEnginePath().cStr();
  const bool reload =
      !impl_->bundle.loaded || next_engine != impl_->engine_path;
  // A live session keeps its force environments and wavefunctions when
  // nothing the deck depends on changed.
  if (!same || reload)
    impl_->destroySession();
  impl_->store(params);
  if (reload && !impl_->load())
    return false;
  return impl_->params_error.empty();
}

void CP2KPot::getParams(::CP2KParams::Builder out) const {
  ::capnp::FlatArrayMessageReader rd(impl_->words());
  copy_params(rd.getRoot<::CP2KParams>(), out);
}

bool CP2KPot::setPotentialConfig(const ::PotentialConfig::Reader &cfg,
                                 std::string *message_out) {
  switch (cfg.which()) {
  case ::PotentialConfig::NONE:
    if (message_out)
      *message_out = "no-op (rgpot params: none)";
    return true;
  case ::PotentialConfig::CP2K: {
    const auto cp = cfg.getCp2k();
    const bool ok = setParams(cp);
    if (message_out) {
      const std::string sum = params_summary(cp);
      if (ok)
        *message_out = "rgpot params applied (cp2k arm): " + sum;
      else
        *message_out =
            "rgpot params cp2k arm rejected (" + loadError() + "): " + sum;
    }
    return ok;
  }
  default:
    if (message_out)
      *message_out = "rgpot params arm " + rgpot::rpc::potentialConfigArm(cfg) +
                     " not handled by CP2KPot (use matching backend pot)";
    return false;
  }
}

void CP2KPot::selectOrbitals(int64_t key) {
  impl_->has_orbital_key = true;
  impl_->orbital_key = key;
}

bool CP2KPot::keepsOrbitalsPerKey() const {
  return impl_->bundle.loaded && impl_->bundle.select_orbitals != nullptr;
}

bool CP2KPot::available() const {
  return impl_->bundle.loaded && impl_->params_error.empty();
}

std::string CP2KPot::loadError() const {
  if (!impl_->bundle.loaded)
    return impl_->bundle.load_error;
  return impl_->params_error;
}

std::string CP2KPot::runDirectory() const { return impl_->run_dir.string(); }

bool CP2KPot::probe_available() {
  std::lock_guard<std::mutex> lock(g_probe_mu);
  if (!g_probe_done) {
    EngineBundle tmp;
    g_probe_ok = try_load_engine(tmp, "");
    g_probe_done = true;
  }
  return g_probe_ok;
}

int CP2KPot::bindCalculators(int ranks_per_calc) {
  EngineBundle bundle;
  // Reject an incompatible shim before creating an MPI communicator.
  if (!try_load_engine(bundle, "") || !bundle.adopt_comm ||
      !register_engine_adopter(bundle))
    return -1;
  return ::rgpot::bindCalculators(ranks_per_calc).index;
}

void CP2KPot::forceImpl(const ForceInput &in, ForceOut *out) const {
  std::string error;
  try {
    forceImplOrThrow(in, out);
  } catch (const CP2KScfError &) {
    // Every rank of the calculator holds the same shim code for this
    // call, so all of them leave here and none waits in the exchange.
    throw;
  } catch (const std::exception &ex) {
    error = ex.what();
  } catch (...) {
    error = "CP2KPot: unknown exception";
  }
  publishCalculatorError(error);
  if (!error.empty())
    throw std::runtime_error(error);
}

void CP2KPot::forceImplOrThrow(const ForceInput &in, ForceOut *out) const {
  if (thisCalculator().index < 0)
    fail_force("CP2KPot: calculator binding was refused");
  if (!impl_->bundle.loaded)
    fail_force("CP2K shim (libcp2kc_rgpot) not loaded: " +
               impl_->bundle.load_error);
  if (!impl_->params_error.empty())
    fail_force("CP2KPot: " + impl_->params_error);

  const int n = static_cast<int>(in.nAtoms);
  if (n <= 0)
    fail_force("CP2KPot: nAtoms must be positive");
  if (!in.pos || !in.atmnrs || !in.box || !out || !out->F)
    fail_force("CP2KPot: null positions/atmnrs/box/forces buffer");

  ::rgpot::finalizeMpiAtExit();

  // The deck fixes the element of every atom; a new sequence needs a new
  // deck and new force environments.
  const bool same_numbers = impl_->session &&
                            impl_->session_numbers.size() == in.nAtoms &&
                            std::equal(impl_->session_numbers.begin(),
                                       impl_->session_numbers.end(), in.atmnrs);
  if (!same_numbers)
    impl_->createSession(in);

  if (impl_->has_orbital_key && impl_->bundle.select_orbitals &&
      impl_->bundle.select_orbitals(
          impl_->session, static_cast<long long>(impl_->orbital_key)) != 0)
    fail_force("CP2K shim refused the orbital key");

  const std::size_t nc = static_cast<std::size_t>(n) * 3u;
  impl_->pos_bohr.resize(nc);
  for (std::size_t i = 0; i < nc; ++i)
    impl_->pos_bohr[i] = in.pos[i] * ANGSTROM_TO_BOHR;
  double cell_bohr[9];
  for (std::size_t i = 0; i < 9; ++i)
    cell_bohr[i] = in.box[i] * ANGSTROM_TO_BOHR;
  impl_->forces.assign(nc, 0.0);
  double energy_ha = 0.0;
  char err[512] = {};
  const int rc = impl_->bundle.calculate(
      impl_->session, n, impl_->pos_bohr.data(), cell_bohr, &energy_ha,
      impl_->forces.data(), err, sizeof(err));
  if (rc == CP2KC_SCF_NOT_CONVERGED)
    throw CP2KScfError(std::string("CP2K SCF did not converge: ") + err);
  if (rc != CP2KC_OK)
    fail_force("CP2K shim failed (code " + std::to_string(rc) + "): " + err);

  // libcp2k returns forces, not gradients: no sign change.
  out->energy = energy_ha * HARTREE_TO_EV;
  out->variance = 0.0;
  out->has_stress = 0;
  for (std::size_t i = 0; i < nc; ++i)
    out->F[i] = impl_->forces[i] * HARTREE_BOHR_TO_EV_ANGSTROM;
}

} // namespace rgpot
