// MIT License
// Copyright 2023--present rgpot developers

#include "rgpot/ExternalPot/ExternalPot.hpp"

#include "rgpot/ExternalPot/md_units.hpp"
#include "rgpot/LennardJones/LJPot.hpp"
#include "rgpot/engine_c_abi.h"
#include "rgpot/plugin/DynLib.hpp"

#include <array>
#include <cmath>
#include <cstdio>
#include <cstring>
#include <mutex>
#include <stdexcept>
#include <utility>
#include <vector>

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

struct Loaded {
  dyn::Handle library = nullptr;
  EngineCreate create = nullptr;
  EngineDestroy destroy = nullptr;
  EngineForce force = nullptr;
  NativeForce native = nullptr;
  RgpotEnginePot *pot = nullptr;

  ~Loaded() {
    if (pot != nullptr && destroy != nullptr) {
      destroy(pot);
    }
    if (library != nullptr) {
      dyn::close(library);
    }
  }

  Loaded() = default;
  Loaded(const Loaded &) = delete;
  Loaded &operator=(const Loaded &) = delete;
  Loaded(Loaded &&other) noexcept
      : library(other.library), create(other.create), destroy(other.destroy),
        force(other.force), native(other.native), pot(other.pot) {
    other.library = nullptr;
    other.pot = nullptr;
    other.destroy = nullptr;
  }
  Loaded &operator=(Loaded &&other) noexcept {
    if (this != &other) {
      if (pot != nullptr && destroy != nullptr) {
        destroy(pot);
      }
      if (library != nullptr) {
        dyn::close(library);
      }
      library = other.library;
      create = other.create;
      destroy = other.destroy;
      force = other.force;
      native = other.native;
      pot = other.pot;
      other.library = nullptr;
      other.pot = nullptr;
      other.destroy = nullptr;
    }
    return *this;
  }
};

Loaded open_library(const std::vector<std::string> &paths, bool native) {
  if (paths.empty()) {
    throw std::runtime_error("no engine library to open");
  }
  std::string tried;
  for (const std::string &path : paths) {
    dyn::Handle handle = dyn::open(path.c_str());
    if (handle == nullptr) {
      tried += path + ": " + last_dl() + "; ";
      continue;
    }
    Loaded loaded;
    loaded.library = handle;
    loaded.create = dyn::loadSym<EngineCreate>(handle, "rgpot_engine_create");
    loaded.destroy =
        dyn::loadSym<EngineDestroy>(handle, "rgpot_engine_destroy");
    if (native) {
      loaded.native =
          dyn::loadSym<NativeForce>(handle, "rgpot_md_native_force");
    } else {
      loaded.force = dyn::loadSym<EngineForce>(handle, "rgpot_engine_force");
    }
    const bool have =
        loaded.create != nullptr && loaded.destroy != nullptr &&
        (native ? loaded.native != nullptr : loaded.force != nullptr);
    if (!have) {
      tried += path + ": missing " +
               (native ? "rgpot_md_native_force" : "rgpot_engine_force") + "; ";
      dyn::close(handle);
      loaded.library = nullptr;
      continue;
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
