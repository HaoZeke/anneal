// MIT License — dlopen frontend for libxtb_engine.so
#include "rgpot/XTBPot/XTBDlopen.hpp"

#include <array>
#include <cstdlib>
#include <cstring>
#include <stdexcept>
#include <string>
#include <vector>

#ifndef _WIN32
#include <dlfcn.h>
#else
#ifndef WIN32_LEAN_AND_MEAN
#define WIN32_LEAN_AND_MEAN
#endif
#ifndef NOMINMAX
#define NOMINMAX
#endif
#include <windows.h>
#endif

namespace rgpot {

namespace {

void *open_lib(const char *path) {
#ifdef _WIN32
  return reinterpret_cast<void *>(LoadLibraryA(path));
#else
  return dlopen(path, RTLD_NOW | RTLD_LOCAL);
#endif
}

void close_lib(void *h) {
  if (!h)
    return;
#ifdef _WIN32
  FreeLibrary(reinterpret_cast<HMODULE>(h));
#else
  dlclose(h);
#endif
}

void *sym(void *h, const char *name) {
#ifdef _WIN32
  return reinterpret_cast<void *>(
      GetProcAddress(reinterpret_cast<HMODULE>(h), name));
#else
  return dlsym(h, name);
#endif
}

int method_to_c(GFNMethod m) {
  switch (m) {
  case GFNMethod::GFNFF:
    return RGPOT_XTB_METHOD_GFNFF;
  case GFNMethod::GFN0xTB:
    return RGPOT_XTB_METHOD_GFN0;
  case GFNMethod::GFN1xTB:
    return RGPOT_XTB_METHOD_GFN1;
  case GFNMethod::GFN2xTB:
  default:
    return RGPOT_XTB_METHOD_GFN2;
  }
}

// A name with no directory separator is a loader search key, not a file.
// dlopen / LoadLibraryA would walk LD_LIBRARY_PATH, DYLD_LIBRARY_PATH, PATH,
// and the ld cache, which can substitute a different libxtb_engine.
bool has_directory_separator(const std::string &path) {
  return path.find_first_of("/\\") != std::string::npos;
}

void append_loader_error(std::string &msg, bool explicit_path) {
#ifndef _WIN32
  if (const char *e = dlerror()) {
    if (*e != '\0') {
      msg += explicit_path ? "; dlerror: " : "; last dlerror: ";
      msg += e;
    }
  }
#else
  (void)explicit_path;
#endif
}

std::vector<std::string> candidate_paths(const XTBDlopenConfig &config) {
  std::vector<std::string> out;
  // An explicit path is exclusive. The constructor rejects a slash-less
  // name before this search. Falling through to the bare soname lets the
  // loader or an env/dir candidate satisfy a request for a different library.
  if (!config.engine_path.empty()) {
    out.push_back(config.engine_path);
    return out;
  }
  if (const char *e = std::getenv("RGPOT_XTB_ENGINE"))
    if (e && *e)
      out.emplace_back(e);
  if (const char *e = std::getenv("XTB_ENGINE"))
    if (e && *e)
      out.emplace_back(e);
#ifdef _WIN32
  out.emplace_back("xtb_engine.dll");
  out.emplace_back("libxtb_engine.dll");
#elif defined(__APPLE__)
  out.emplace_back("libxtb_engine.dylib");
  out.emplace_back("libxtb_engine.so");
#else
  out.emplace_back("libxtb_engine.so");
#endif
  auto append_dirs = [&](const char *env) {
    if (!env || !*env)
      return;
    std::string s(env);
    size_t start = 0;
    while (start < s.size()) {
      auto pos = s.find(':', start);
      if (pos == std::string::npos)
        pos = s.size();
      if (pos > start) {
        std::string dir = s.substr(start, pos - start);
        if (!dir.empty() && dir.back() != '/')
          dir += '/';
#ifdef __APPLE__
        out.push_back(dir + "libxtb_engine.dylib");
#endif
        out.push_back(dir + "libxtb_engine.so");
      }
      start = pos + 1;
    }
  };
  append_dirs(std::getenv("EON_POTENTIALS_PATH"));
  append_dirs(std::getenv("RGPOT_ENGINE_PATH"));
  return out;
}

} // namespace

XTBDlopen::XTBDlopen(const XTBConfig &xtb)
    : XTBDlopen(XTBDlopenConfig{xtb, {}}) {}

XTBDlopen::XTBDlopen(const XTBDlopenConfig &config) : m_config(config.xtb) {
  const bool explicit_path = !config.engine_path.empty();
  if (explicit_path && !has_directory_separator(config.engine_path)) {
    throw std::runtime_error(
        "XTBDlopen: engine_path \"" + config.engine_path +
        "\" has no directory separator; a bare name is resolved through "
        "the loader search path. Pass an absolute or relative path");
  }
  for (const auto &path : candidate_paths(config)) {
    m_lib = open_lib(path.c_str());
    if (m_lib)
      break;
  }
  if (!m_lib) {
    std::string msg;
    if (explicit_path) {
      // Discovery env vars are ignored once engine_path is set, so naming
      // them here points at a variable that was not consulted.
      msg = "XTBDlopen: libxtb_engine.so not found: " + config.engine_path;
    } else {
      msg = "XTBDlopen: libxtb_engine.so not found "
            "(set RGPOT_XTB_ENGINE / engine_path / EON_POTENTIALS_PATH)";
    }
    append_loader_error(msg, explicit_path);
    throw std::runtime_error(msg);
  }

  auto abi =
      reinterpret_cast<int (*)(void)>(sym(m_lib, "rgpot_xtb_abi_version"));
  m_create = reinterpret_cast<create_fn>(sym(m_lib, "rgpot_xtb_create"));
  m_destroy = reinterpret_cast<destroy_fn>(sym(m_lib, "rgpot_xtb_destroy"));
  m_force = reinterpret_cast<force_fn>(sym(m_lib, "rgpot_xtb_force"));
  m_reset = reinterpret_cast<reset_fn>(sym(m_lib, "rgpot_xtb_reset"));
  m_set_warm =
      reinterpret_cast<set_warm_fn>(sym(m_lib, "rgpot_xtb_set_warm_move"));
  m_warm_move =
      reinterpret_cast<warm_move_fn>(sym(m_lib, "rgpot_xtb_warm_move"));
  m_set_min_gap =
      reinterpret_cast<set_warm_fn>(sym(m_lib, "rgpot_xtb_set_warm_min_gap"));
  m_min_gap =
      reinterpret_cast<warm_move_fn>(sym(m_lib, "rgpot_xtb_warm_min_gap"));
  m_reused =
      reinterpret_cast<reused_fn>(sym(m_lib, "rgpot_xtb_reused_wavefunction"));
  if (!abi || !m_create || !m_destroy || !m_force || !m_reset || !m_set_warm ||
      !m_warm_move || !m_set_min_gap || !m_min_gap || !m_reused) {
    close_lib(m_lib);
    m_lib = nullptr;
    // The opened library is the one we selected. Do not resume the
    // candidate list: an explicit path (or the first discovery hit) that
    // lacks the ABI must not silently load a different engine.
    throw std::runtime_error(
        "XTBDlopen: missing C ABI symbols in engine library");
  }
  if (abi() != RGPOT_XTB_ABI_VERSION) {
    close_lib(m_lib);
    m_lib = nullptr;
    throw std::runtime_error("XTBDlopen: ABI version mismatch");
  }
  char err[1024]{};
  RgpotXtbConfig cfg{};
  cfg.method = method_to_c(config.xtb.method);
  cfg.accuracy = config.xtb.accuracy;
  cfg.electronic_temperature = config.xtb.electronic_temperature;
  cfg.max_iterations = config.xtb.max_iterations;
  cfg.charge = config.xtb.charge;
  cfg.uhf = config.xtb.uhf;
  m_pot = m_create(&cfg, err, sizeof err);
  if (!m_pot) {
    close_lib(m_lib);
    m_lib = nullptr;
    throw std::runtime_error(std::string("XTBDlopen: create failed: ") + err);
  }
  m_set_warm(m_pot, config.xtb.warm_move);
  m_set_min_gap(m_pot, config.xtb.warm_min_gap);
}

void XTBDlopen::reset() {
  if (!m_reset || !m_pot)
    throw std::runtime_error("XTBDlopen: engine not available");
  m_reset(m_pot);
}

void XTBDlopen::setWarmMove(double angstrom) {
  if (!m_set_warm || !m_pot)
    throw std::runtime_error("XTBDlopen: engine not available");
  m_set_warm(m_pot, angstrom);
}

double XTBDlopen::warmMove() const {
  if (!m_warm_move || !m_pot)
    throw std::runtime_error("XTBDlopen: engine not available");
  return m_warm_move(m_pot);
}

void XTBDlopen::setWarmMinGap(double ev) {
  if (!m_set_min_gap || !m_pot)
    throw std::runtime_error("XTBDlopen: engine not available");
  m_set_min_gap(m_pot, ev);
}

double XTBDlopen::warmMinGap() const {
  if (!m_min_gap || !m_pot)
    throw std::runtime_error("XTBDlopen: engine not available");
  return m_min_gap(m_pot);
}

bool XTBDlopen::reusedWavefunction() const {
  if (!m_reused || !m_pot)
    throw std::runtime_error("XTBDlopen: engine not available");
  return m_reused(m_pot) != 0;
}

XTBDlopen::~XTBDlopen() {
  if (m_pot && m_destroy)
    m_destroy(m_pot);
  m_pot = nullptr;
  if (m_lib)
    close_lib(m_lib);
  m_lib = nullptr;
}

FusedSettingsSpec XTBDlopen::fusedSettings() const {
  FusedSettingsSpec spec;
  spec.spin = m_config.uhf;
  spec.sccMaxIterations = static_cast<unsigned>(m_config.max_iterations);
  spec.charge = m_config.charge;
  spec.accuracy = m_config.accuracy;
  spec.electronicTemperature = m_config.electronic_temperature;
  spec.sccMixer = "xtb";
  spec.engineBuild = "xtb-engine-abi-" + std::to_string(RGPOT_XTB_ABI_VERSION) +
                     "-method-" + std::to_string(method_to_c(m_config.method));
  return spec;
}

void XTBDlopen::forceImpl(const ForceInput &in, ForceOut *out) const {
  if (!m_force || !m_pot)
    throw std::runtime_error("XTBDlopen: engine not available");
  double var = 0.0;
  int rc = m_force(m_pot, static_cast<long>(in.nAtoms), in.pos, in.atmnrs,
                   out->F, &out->energy, &var, in.box);
  if (rc == RGPOT_XTB_RC_SCC_NOT_CONVERGED)
    throw SccNotConverged("XTBDlopen: SCC did not converge (rc=" +
                          std::to_string(rc) + ")");
  if (rc != 0)
    throw std::runtime_error(
        "XTBDlopen: force evaluation failed (rc=" + std::to_string(rc) + ")");
  out->variance = var;
}

std::tuple<double, types::AtomMatrix, double>
XTBDlopen::operator()(const types::AtomMatrix &positions,
                      const std::vector<int> &atmtypes,
                      const std::array<std::array<double, 3>, 3> &box) const {
  const size_t nAtoms = positions.rows();
  types::AtomMatrix forces = types::AtomMatrix::Zero(nAtoms, 3);
  double flatBox[9];
  std::memcpy(flatBox, static_cast<const void *>(&box), sizeof(flatBox));
  ForceInput fi{.nAtoms = nAtoms,
                .pos = positions.data(),
                .atmnrs = atmtypes.data(),
                .box = flatBox};
  ForceOut fo{.F = forces.data(),
              .energy = 0.0,
              .variance = 0.0,
              .stress = {},
              .has_stress = 0};
  forceImpl(fi, &fo);
  return {fo.energy, std::move(forces), fo.variance};
}

} // namespace rgpot
