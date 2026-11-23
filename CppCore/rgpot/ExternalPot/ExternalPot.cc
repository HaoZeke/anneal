// MIT License
// Copyright 2023--present rgpot developers

#include "rgpot/ExternalPot/ExternalPot.hpp"

#include "rgpot/ExternalPot/md_units.hpp"
#include "rgpot/LennardJones/LJPot.hpp"
#include "rgpot/engine_c_abi.h"
#include "rgpot/plugin/DynLib.hpp"

#include <array>
#include <atomic>
#include <cerrno>
#include <climits>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <map>
#include <mutex>
#include <stdexcept>
#include <utility>
#include <vector>

#if !defined(_WIN32)
#include <dlfcn.h>
#include <sys/stat.h>
#endif

namespace rgpot {
namespace {

namespace dyn = plugin::dynlib;

using EngineCreate = RgpotEnginePot *(*)(const void *, size_t, char *, size_t);
using EngineDestroy = void (*)(RgpotEnginePot *);
using EngineForce = int (*)(RgpotEnginePot *, long, const double *, const int *,
                            double *, double *, double *, const double *,
                            rgpot_engine_coord_transform, void *);
using NativeForce = int (*)(void *, long, const double *, const int *, double *,
                            double *, double *, const double *);

void require_lj(const ExternalLjConfig &config) {
  const auto usable = [](double v) { return v > 0.0 && std::isfinite(v); };
  if (!usable(config.u0) || !usable(config.cutoff) || !usable(config.psi)) {
    throw std::invalid_argument(
        "u0, cutoff and psi must be positive and finite (eV and Angstrom)");
  }
}

void add_soname(std::vector<std::string> &out, const char *stem) {
#if defined(_WIN32)
  out.emplace_back(std::string(stem) + ".dll");
  out.emplace_back(std::string("lib") + stem + ".dll");
#elif defined(__APPLE__)
  out.emplace_back(std::string("lib") + stem + ".dylib");
  out.emplace_back(std::string("lib") + stem + ".so");
#else
  out.emplace_back(std::string("lib") + stem + ".so");
#endif
}

std::string last_dl() {
  const std::string text = dyn::error();
  return text.empty() ? "dlopen failed" : text;
}

// One dlclose per successful dlopen. glibc ignores dlclose once an object is
// RTLD_NODELETE and leaves the loader reference in place, so a constructor
// that opened with that flag on every call would raise the reference forever.
// An attempt is opened without the flag and closed if it does not become the
// resident engine. The resident open is pinned RTLD_NODELETE, which keeps the
// mapping and its code loaded, and a later constructor copies that handle.
void close_library(void *handle) noexcept {
  if (handle == nullptr) {
    return;
  }
#if defined(_WIN32)
  if (FreeLibrary(reinterpret_cast<HMODULE>(handle)) == 0) {
    std::fprintf(stderr, "FreeLibrary failed: %lu\n",
                 static_cast<unsigned long>(GetLastError()));
  }
#else
  dlerror();
  if (dlclose(handle) != 0) {
    const char *msg = dlerror();
    std::fprintf(stderr, "dlclose failed: %s\n",
                 msg != nullptr && msg[0] != '\0' ? msg : "dlclose failed");
  }
#endif
}

// Copies share one loader reference. A move leaves the source empty so it
// does not close. The last owner closes, from its destructor and not by
// throwing.
class LibraryHandle {
public:
  LibraryHandle() noexcept = default;
  explicit LibraryHandle(void *raw) : ref_(nullptr) {
    if (raw == nullptr) {
      return;
    }
    try {
      auto *created = new Ref;
      created->raw = raw;
      created->count.store(1, std::memory_order_relaxed);
      ref_ = created;
    } catch (...) {
      close_library(raw);
      throw;
    }
  }
  ~LibraryHandle() { release(); }

  LibraryHandle(const LibraryHandle &other) noexcept : ref_(other.ref_) {
    retain();
  }
  LibraryHandle &operator=(const LibraryHandle &other) noexcept {
    if (this != &other) {
      release();
      ref_ = other.ref_;
      retain();
    }
    return *this;
  }
  LibraryHandle(LibraryHandle &&other) noexcept : ref_(other.ref_) {
    other.ref_ = nullptr;
  }
  LibraryHandle &operator=(LibraryHandle &&other) noexcept {
    if (this != &other) {
      release();
      ref_ = other.ref_;
      other.ref_ = nullptr;
    }
    return *this;
  }

  void *get() const noexcept { return ref_ != nullptr ? ref_->raw : nullptr; }
  explicit operator bool() const noexcept { return get() != nullptr; }

private:
  struct Ref {
    void *raw = nullptr;
    std::atomic<unsigned> count{0};
  };

  void retain() noexcept {
    if (ref_ != nullptr) {
      ref_->count.fetch_add(1, std::memory_order_relaxed);
    }
  }
  void release() noexcept {
    Ref *held = ref_;
    ref_ = nullptr;
    if (held != nullptr &&
        held->count.fetch_sub(1, std::memory_order_acq_rel) == 1) {
      close_library(held->raw);
      delete held;
    }
  }

  Ref *ref_ = nullptr;
};

dyn::Handle as_dyn(void *handle) {
  return reinterpret_cast<dyn::Handle>(handle);
}

struct Loaded {
  LibraryHandle library;
  EngineCreate create = nullptr;
  EngineDestroy destroy = nullptr;
  EngineForce force = nullptr;
  NativeForce native = nullptr;
  RgpotEnginePot *pot = nullptr;

  ~Loaded() {
    if (pot != nullptr && destroy != nullptr) {
      destroy(pot);
    }
  }

  Loaded() = default;
  Loaded(const Loaded &) = delete;
  Loaded &operator=(const Loaded &) = delete;
  Loaded(Loaded &&other) noexcept
      : library(std::move(other.library)), create(other.create),
        destroy(other.destroy), force(other.force), native(other.native),
        pot(other.pot) {
    other.create = nullptr;
    other.destroy = nullptr;
    other.force = nullptr;
    other.native = nullptr;
    other.pot = nullptr;
  }
  Loaded &operator=(Loaded &&other) noexcept {
    if (this != &other) {
      if (pot != nullptr && destroy != nullptr) {
        destroy(pot);
      }
      library = std::move(other.library);
      create = other.create;
      destroy = other.destroy;
      force = other.force;
      native = other.native;
      pot = other.pot;
      other.create = nullptr;
      other.destroy = nullptr;
      other.force = nullptr;
      other.native = nullptr;
      other.pot = nullptr;
    }
    return *this;
  }
};

struct ResidentEntry {
  LibraryHandle library;
  LibraryHandle pin;
  std::string canonical;
#if !defined(_WIN32)
  dev_t dev = 0;
  ino_t ino = 0;
#endif
};

struct ResidentTable {
  std::mutex mutex;
  std::map<std::string, ResidentEntry> by_canon;
  std::map<std::string, std::string> request_to_canon;
};

ResidentTable &residents() {
  static ResidentTable table;
  return table;
}

std::string canonical_file(const std::string &path) {
#if defined(_WIN32)
  return path;
#else
  char buf[PATH_MAX];
  if (realpath(path.c_str(), buf) == nullptr) {
    return {};
  }
  return buf;
#endif
}

struct FileIdentity {
  std::string canonical;
#if !defined(_WIN32)
  dev_t dev = 0;
  ino_t ino = 0;
#endif
};

FileIdentity identify_loaded(const void *anchor, const std::string &path) {
#if defined(_WIN32)
  (void)anchor;
  FileIdentity id;
  id.canonical = path;
  return id;
#else
  Dl_info info{};
  if (dladdr(anchor, &info) == 0 || info.dli_fname == nullptr ||
      info.dli_fname[0] == '\0') {
    throw std::runtime_error(path +
                             ": the dynamic linker did not report the file "
                             "it loaded, so the engine cannot be identified");
  }
  char canonical[PATH_MAX];
  struct stat st {};
  if (realpath(info.dli_fname, canonical) == nullptr ||
      stat(canonical, &st) != 0) {
    throw std::runtime_error(path + ": cannot identify the loaded file " +
                             info.dli_fname + ": " + std::strerror(errno));
  }
  FileIdentity id;
  id.canonical = canonical;
  id.dev = st.st_dev;
  id.ino = st.st_ino;
  return id;
#endif
}

void ensure_current(const ResidentEntry &entry, const std::string &path) {
#if defined(_WIN32)
  (void)entry;
  (void)path;
#else
  struct stat st {};
  if (stat(entry.canonical.c_str(), &st) != 0) {
    throw std::runtime_error(path + ": cannot identify the loaded file " +
                             entry.canonical + ": " + std::strerror(errno));
  }
  if (st.st_dev != entry.dev || st.st_ino != entry.ino) {
    throw std::runtime_error(
        path + ": " + entry.canonical +
        " was replaced after it was loaded, and the loaded engine stays "
        "resident; load the new file from another path or restart the process");
  }
  if (!canonical_file(path).empty()) {
    struct stat requested {};
    if (stat(path.c_str(), &requested) != 0) {
      throw std::runtime_error(path + ": cannot identify the requested file: " +
                               std::strerror(errno));
    }
    if (requested.st_dev != entry.dev || requested.st_ino != entry.ino) {
      throw std::runtime_error(
          path +
          ": the requested path was replaced or retargeted after it loaded "
          "another engine; use a new path or restart the process");
    }
  }
#endif
}

// `table.mutex` is held by the caller.
LibraryHandle find_resident(ResidentTable &table, const std::string &path) {
  std::string key = canonical_file(path);
  if (key.empty()) {
    key = path;
  }
  auto direct = table.by_canon.find(key);
  if (direct == table.by_canon.end()) {
    auto alias = table.request_to_canon.find(key);
    if (alias == table.request_to_canon.end()) {
      alias = table.request_to_canon.find(path);
    }
    if (alias == table.request_to_canon.end()) {
      return {};
    }
    direct = table.by_canon.find(alias->second);
    if (direct == table.by_canon.end()) {
      return {};
    }
  }
  ensure_current(direct->second, path);
  return direct->second.library;
}

LibraryHandle pin_resident(const std::string &path) {
#if defined(_WIN32)
  HMODULE pinned = nullptr;
  if (GetModuleHandleExA(GET_MODULE_HANDLE_EX_FLAG_PIN, path.c_str(),
                         &pinned) == 0 ||
      pinned == nullptr) {
    throw std::runtime_error(path +
                             ": the engine library could not be pinned: " +
                             dyn::error());
  }
  return LibraryHandle(reinterpret_cast<void *>(pinned));
#else
  dlerror();
  void *pin = dlopen(path.c_str(),
                     RTLD_NOW | RTLD_LOCAL | RTLD_NOLOAD | RTLD_NODELETE);
  if (pin == nullptr) {
    throw std::runtime_error(path + ": RTLD_NODELETE could not be set: " +
                             last_dl());
  }
  return LibraryHandle(pin);
#endif
}

// `table.mutex` is held by the caller.
LibraryHandle publish_resident(ResidentTable &table, const std::string &path,
                               const LibraryHandle &opened,
                               const void *anchor) {
  const FileIdentity id = identify_loaded(anchor, path);
  auto existing = table.by_canon.find(id.canonical);
  if (existing != table.by_canon.end()) {
    ensure_current(existing->second, path);
    table.request_to_canon[path] = id.canonical;
    return existing->second.library;
  }
  ResidentEntry entry;
  entry.library = opened;
  entry.pin = pin_resident(path);
  entry.canonical = id.canonical;
#if !defined(_WIN32)
  entry.dev = id.dev;
  entry.ino = id.ino;
#endif
  auto [it, inserted] = table.by_canon.emplace(id.canonical, std::move(entry));
  if (!inserted) {
    throw std::runtime_error(path + ": the engine file was registered twice");
  }
  table.request_to_canon[path] = id.canonical;
  const std::string via = canonical_file(path);
  if (!via.empty()) {
    table.request_to_canon[via] = id.canonical;
  }
  return it->second.library;
}

LibraryHandle open_attempt(const std::string &path) {
#if defined(_WIN32)
  void *raw = reinterpret_cast<void *>(dyn::open(path.c_str()));
#else
  void *raw = dlopen(path.c_str(), RTLD_NOW | RTLD_LOCAL);
#endif
  if (raw == nullptr) {
    return {};
  }
  return LibraryHandle(raw);
}

void bind_symbols(Loaded &loaded, bool native) {
  void *handle = loaded.library.get();
  loaded.create =
      dyn::loadSym<EngineCreate>(as_dyn(handle), "rgpot_engine_create");
  loaded.destroy =
      dyn::loadSym<EngineDestroy>(as_dyn(handle), "rgpot_engine_destroy");
  if (native) {
    loaded.native =
        dyn::loadSym<NativeForce>(as_dyn(handle), "rgpot_md_native_force");
  } else {
    loaded.force = dyn::loadSym<EngineForce>(as_dyn(handle), "rgpot_engine_force");
  }
}

bool have_symbols(const Loaded &loaded, bool native) {
  return loaded.create != nullptr && loaded.destroy != nullptr &&
         (native ? loaded.native != nullptr : loaded.force != nullptr);
}

Loaded open_library(const std::vector<std::string> &paths, bool native) {
  if (paths.empty()) {
    throw std::runtime_error("no engine library to open");
  }
  ResidentTable &table = residents();
  std::string tried;
  for (const std::string &path : paths) {
    Loaded loaded;
    {
      std::lock_guard<std::mutex> lock(table.mutex);
      loaded.library = find_resident(table, path);
    }
    if (loaded.library) {
      bind_symbols(loaded, native);
      if (!have_symbols(loaded, native)) {
        tried += path + ": missing " +
                 (native ? "rgpot_md_native_force" : "rgpot_engine_force") +
                 "; ";
        continue;
      }
      return loaded;
    }

    // dlopen runs library initializers. Keep them outside the registry lock:
    // an initializer may construct another external potential and consult
    // this table. Publish only after dlopen has completed and every required
    // symbol has been resolved.
    loaded.library = open_attempt(path);
    if (!loaded.library) {
      tried += path + ": " + last_dl() + "; ";
      continue;
    }
    bind_symbols(loaded, native);
    if (!have_symbols(loaded, native)) {
      tried += path + ": missing " +
               (native ? "rgpot_md_native_force" : "rgpot_engine_force") + "; ";
      continue;
    }
    {
      std::lock_guard<std::mutex> lock(table.mutex);
      loaded.library = publish_resident(
          table, path, loaded.library,
          reinterpret_cast<const void *>(loaded.create));
    }
    return loaded;
  }
  throw std::runtime_error("engine library was not opened: " + tried);
}

std::vector<std::string> lammps_paths(const std::string &explicit_path) {
  std::vector<std::string> out;
  if (!explicit_path.empty()) {
    out.push_back(explicit_path);
    return out;
  }
  if (const char *env = std::getenv("RGPOT_LAMMPS_ENGINE")) {
    if (env[0] != '\0') {
      out.emplace_back(env);
    }
  }
  if (const char *env = std::getenv("LAMMPS_ENGINE")) {
    if (env[0] != '\0') {
      out.emplace_back(env);
    }
  }
  add_soname(out, "lammps_engine");
  add_soname(out, "lj_engine");
  return out;
}

void create_engine(Loaded &loaded, const void *config, size_t bytes) {
  char err[256] = {};
  loaded.pot = loaded.create(config, bytes, err, sizeof(err));
  if (loaded.pot == nullptr) {
    throw std::runtime_error(err[0] != '\0' ? err
                                            : "rgpot_engine_create failed");
  }
}

void check_geometry(const ForceInput &in, const ForceOut *out) {
  if (out == nullptr || out->F == nullptr || in.box == nullptr ||
      (in.nAtoms > 0 && (in.pos == nullptr || in.atmnrs == nullptr))) {
    throw std::invalid_argument("potential force buffers are incomplete");
  }
}

} // namespace

struct LammpsPot::Impl {
  Loaded engine;
  mutable std::mutex mutex;
};

LammpsPot::LammpsPot(const ExternalLjConfig &config)
    : Potential(PotType::Lammps), impl_(std::make_unique<Impl>()),
      config_(config) {
  require_lj(config_);
  impl_->engine = open_library(lammps_paths(config_.engine_path), false);
  const double packed[3] = {config_.u0, config_.cutoff, config_.psi};
  create_engine(impl_->engine, packed, sizeof(packed));
}

LammpsPot::~LammpsPot() = default;
LammpsPot::LammpsPot(LammpsPot &&) noexcept = default;
LammpsPot &LammpsPot::operator=(LammpsPot &&) noexcept = default;

void LammpsPot::forceImpl(const ForceInput &in, ForceOut *out) const {
  check_geometry(in, out);
  std::lock_guard<std::mutex> lock(impl_->mutex);
  double variance = 0.0;
  const int status = impl_->engine.force(
      impl_->engine.pot, static_cast<long>(in.nAtoms), in.pos, in.atmnrs,
      out->F, &out->energy, &variance, in.box, nullptr, nullptr);
  if (status != 0) {
    throw std::runtime_error("rgpot_engine_force failed");
  }
  out->variance = variance;
  out->has_stress = 0;
}

struct NativeState {
  explicit NativeState(const ExternalLjConfig &config)
      : host(LJConfig{config.u0, config.cutoff, config.psi}) {}

  Loaded engine;
  LJPot host;
  bool use_native = false;
  mutable std::mutex mutex;
  mutable std::vector<double> positions;
  mutable std::vector<double> forces;
  mutable std::array<double, 9> box{};
};

void load_native(NativeState &state, const ExternalLjConfig &config,
                 double u0_kj, double cutoff_nm, double psi_nm) {
  if (config.engine_path.empty()) {
    return;
  }
  state.engine = open_library({config.engine_path}, true);
  const double packed[3] = {u0_kj, cutoff_nm, psi_nm};
  create_engine(state.engine, packed, sizeof(packed));
  state.use_native = true;
}

void eval_native(NativeState &state, const ForceInput &in, ForceOut *out,
                 double u0_host_unused) {
  (void)u0_host_unused;
  const size_t n3 = in.nAtoms * 3;
  state.positions.resize(n3);
  state.forces.resize(n3);
  for (size_t i = 0; i < n3; ++i) {
    state.positions[i] = in.pos[i] * md::kNmPerAngstrom;
  }
  for (int i = 0; i < 9; ++i) {
    state.box[static_cast<size_t>(i)] = in.box[i] * md::kNmPerAngstrom;
  }
  double energy_kj = 0.0;
  double variance = 0.0;
  const int status = state.engine.native(
      state.engine.pot, static_cast<long>(in.nAtoms), state.positions.data(),
      in.atmnrs, state.forces.data(), &energy_kj, &variance, state.box.data());
  if (status != 0) {
    throw std::runtime_error("rgpot_md_native_force failed");
  }
  out->energy = energy_kj * md::kEvPerKj;
  out->variance = variance * md::kEvPerKj * md::kEvPerKj;
  for (size_t i = 0; i < n3; ++i) {
    out->F[i] = state.forces[i] * md::kEvAPerKjNm;
  }
  out->has_stress = 0;
}

struct GromacsPot::Impl {
  explicit Impl(const ExternalLjConfig &config) : state(config) {}
  NativeState state;
};

GromacsPot::GromacsPot(const ExternalLjConfig &config)
    : Potential(PotType::Gromacs), impl_(std::make_unique<Impl>(config)),
      config_(config) {
  require_lj(config_);
  native_u0_ = config_.u0 * md::kKjPerEv;
  native_cutoff_ = config_.cutoff * md::kNmPerAngstrom;
  native_psi_ = config_.psi * md::kNmPerAngstrom;
  load_native(impl_->state, config_, native_u0_, native_cutoff_, native_psi_);
}

GromacsPot::~GromacsPot() = default;
GromacsPot::GromacsPot(GromacsPot &&) noexcept = default;
GromacsPot &GromacsPot::operator=(GromacsPot &&) noexcept = default;

PotCaps GromacsPot::caps() const noexcept {
  const bool native = impl_ && impl_->state.use_native;
  return {.reentrancy = Reentrancy::PerInstance,
          .periodic = true,
          .stress = !native};
}

void GromacsPot::forceImpl(const ForceInput &in, ForceOut *out) const {
  check_geometry(in, out);
  std::lock_guard<std::mutex> lock(impl_->state.mutex);
  if (impl_->state.use_native) {
    eval_native(impl_->state, in, out, config_.u0);
    return;
  }
  impl_->state.host.forceImpl(in, out);
}

struct OpenMMPot::Impl {
  explicit Impl(const ExternalLjConfig &config) : state(config) {}
  NativeState state;
};

OpenMMPot::OpenMMPot(const ExternalLjConfig &config)
    : Potential(PotType::OpenMM), impl_(std::make_unique<Impl>(config)),
      config_(config) {
  require_lj(config_);
  native_u0_ = config_.u0 * md::kKjPerEv;
  native_cutoff_ = config_.cutoff * md::kNmPerAngstrom;
  native_psi_ = config_.psi * md::kNmPerAngstrom;
  load_native(impl_->state, config_, native_u0_, native_cutoff_, native_psi_);
}

OpenMMPot::~OpenMMPot() = default;
OpenMMPot::OpenMMPot(OpenMMPot &&) noexcept = default;
OpenMMPot &OpenMMPot::operator=(OpenMMPot &&) noexcept = default;

PotCaps OpenMMPot::caps() const noexcept {
  const bool native = impl_ && impl_->state.use_native;
  return {.reentrancy = Reentrancy::PerInstance,
          .periodic = true,
          .stress = !native};
}

void OpenMMPot::forceImpl(const ForceInput &in, ForceOut *out) const {
  check_geometry(in, out);
  std::lock_guard<std::mutex> lock(impl_->state.mutex);
  if (impl_->state.use_native) {
    eval_native(impl_->state, in, out, config_.u0);
    return;
  }
  impl_->state.host.forceImpl(in, out);
}

std::unique_ptr<LammpsPot>
rgpot_lammps_potential(const ExternalLjConfig &config) {
  return std::make_unique<LammpsPot>(config);
}

std::unique_ptr<GromacsPot>
rgpot_gromacs_potential(const ExternalLjConfig &config) {
  return std::make_unique<GromacsPot>(config);
}

std::unique_ptr<OpenMMPot>
rgpot_openmm_potential(const ExternalLjConfig &config) {
  return std::make_unique<OpenMMPot>(config);
}

} // namespace rgpot

struct rgpot_external_potential {
  std::unique_ptr<rgpot::PotentialBase> pot;
};

namespace {

void write_err(char *err, size_t err_len, const char *text) {
  if (err == nullptr || err_len == 0) {
    return;
  }
  std::snprintf(err, err_len, "%s", text);
}

rgpot::ExternalLjConfig lj_config(const char *engine_path, double u0,
                                  double cutoff, double psi) {
  rgpot::ExternalLjConfig config;
  config.u0 = u0;
  config.cutoff = cutoff;
  config.psi = psi;
  if (engine_path != nullptr) {
    config.engine_path = engine_path;
  }
  return config;
}

template <typename Make>
rgpot_external_potential *adopt(Make &&make, char *err, size_t err_len) {
  try {
    std::unique_ptr<rgpot::PotentialBase> pot = make();
    return new rgpot_external_potential{std::move(pot)};
  } catch (const std::exception &ex) {
    write_err(err, err_len, ex.what());
    return nullptr;
  }
}

} // namespace

extern "C" {

double rgpot_md_kj_per_ev(void) { return rgpot::md::kKjPerEv; }

double rgpot_md_nm_per_angstrom(void) { return rgpot::md::kNmPerAngstrom; }

rgpot_external_potential *rgpot_lammps_potential(const char *engine_path,
                                                 double u0, double cutoff,
                                                 double psi, char *err,
                                                 size_t err_len) {
  return adopt(
      [&] {
        return rgpot::rgpot_lammps_potential(
            lj_config(engine_path, u0, cutoff, psi));
      },
      err, err_len);
}

rgpot_external_potential *rgpot_gromacs_potential(const char *engine_path,
                                                  double u0, double cutoff,
                                                  double psi, char *err,
                                                  size_t err_len) {
  return adopt(
      [&] {
        return rgpot::rgpot_gromacs_potential(
            lj_config(engine_path, u0, cutoff, psi));
      },
      err, err_len);
}

rgpot_external_potential *rgpot_openmm_potential(const char *engine_path,
                                                 double u0, double cutoff,
                                                 double psi, char *err,
                                                 size_t err_len) {
  return adopt(
      [&] {
        return rgpot::rgpot_openmm_potential(
            lj_config(engine_path, u0, cutoff, psi));
      },
      err, err_len);
}

void rgpot_external_potential_free(rgpot_external_potential *pot) {
  delete pot;
}

int rgpot_external_potential_energy_forces(rgpot_external_potential *pot,
                                           const double *xyz,
                                           const int *atomic_numbers, size_t n,
                                           const double *box9, double *forces,
                                           double *energy, char *err,
                                           size_t err_len) {
  if (pot == nullptr || pot->pot == nullptr || energy == nullptr ||
      box9 == nullptr ||
      (n > 0 && (xyz == nullptr || atomic_numbers == nullptr ||
                 forces == nullptr))) {
    write_err(err, err_len, "energy_forces needs a potential and a geometry");
    return 1;
  }
  try {
    rgpot::types::AtomMatrix positions(n, 3);
    if (n > 0) {
      std::memcpy(positions.data(), xyz, n * 3 * sizeof(double));
    }
    std::vector<int> types(atomic_numbers, atomic_numbers + n);
    std::array<std::array<double, 3>, 3> box{};
    std::memcpy(&box, box9, sizeof(box));
    const auto result = (*pot->pot)(positions, types, box);
    *energy = std::get<0>(result);
    if (n > 0) {
      std::memcpy(forces, std::get<1>(result).data(), n * 3 * sizeof(double));
    }
    return 0;
  } catch (const std::exception &ex) {
    write_err(err, err_len, ex.what());
    return 1;
  }
}

} // extern "C"
