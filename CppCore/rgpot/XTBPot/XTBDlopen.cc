// MIT License — dlopen frontend for libxtb_engine.so
#include "rgpot/XTBPot/XTBDlopen.hpp"
#include "rgpot/abi/DynLib.hpp"
#include "rgpot/abi/EngineDlopen.hpp"

#include <array>
#include <cstring>
#include <stdexcept>
#include <string>
#include <vector>

namespace rgpot {

namespace {

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

} // namespace

XTBDlopen::XTBDlopen(const XTBConfig &xtb)
    : XTBDlopen(XTBDlopenConfig{xtb, {}}) {}

XTBDlopen::XTBDlopen(const XTBDlopenConfig &config) : m_config(config.xtb) {
  const bool explicit_path = !config.engine_path.empty();
  if (explicit_path && !abi::has_directory_separator(config.engine_path)) {
    throw std::runtime_error(
        "XTBDlopen: engine_path \"" + config.engine_path +
        "\" has no directory separator; a bare name is resolved through "
        "the loader search path. Pass an absolute or relative path");
  }
  const auto cands = abi::engine_candidates(
      "xtb_engine", config.engine_path, {"RGPOT_XTB_ENGINE", "XTB_ENGINE"},
      abi::CandidateOptions{.include_cwd_dot_slash = false,
                            .include_dll = false,
                            .include_dir_envs = true,
                            .exclusive_explicit = true,
                            .platform_only_soname = true});
  DynLib lib;
  std::string loaded_path;
  std::string open_err;
  if (!abi::open_first_candidate(lib, cands, &loaded_path, &open_err, false)) {
    std::string msg;
    if (explicit_path) {
      // Discovery env vars are ignored once engine_path is set, so naming
      // them here points at a variable that was not consulted.
      msg = "XTBDlopen: libxtb_engine.so not found: " + config.engine_path;
    } else {
      msg = "XTBDlopen: libxtb_engine.so not found "
            "(set RGPOT_XTB_ENGINE / engine_path / EON_POTENTIALS_PATH)";
    }
    if (!open_err.empty()) {
      msg += explicit_path ? "; dlerror: " : "; last dlerror: ";
      msg += open_err;
    }
    throw std::runtime_error(msg);
  }

  auto abi_fn = lib.sym_optional<int (*)(void)>("rgpot_xtb_abi_version");
  m_create = lib.sym_optional<create_fn>("rgpot_xtb_create");
  m_destroy = lib.sym_optional<destroy_fn>("rgpot_xtb_destroy");
  m_force = lib.sym_optional<force_fn>("rgpot_xtb_force");
  m_reset = lib.sym_optional<reset_fn>("rgpot_xtb_reset");
  m_set_warm = lib.sym_optional<set_warm_fn>("rgpot_xtb_set_warm_move");
  m_warm_move = lib.sym_optional<warm_move_fn>("rgpot_xtb_warm_move");
  m_set_min_gap = lib.sym_optional<set_warm_fn>("rgpot_xtb_set_warm_min_gap");
  m_min_gap = lib.sym_optional<warm_move_fn>("rgpot_xtb_warm_min_gap");
  m_reused = lib.sym_optional<reused_fn>("rgpot_xtb_reused_wavefunction");
  if (!abi_fn || !m_create || !m_destroy || !m_force || !m_reset ||
      !m_set_warm || !m_warm_move || !m_set_min_gap || !m_min_gap ||
      !m_reused) {
    // The opened library is the one we selected. Do not resume the
    // candidate list: an explicit path (or the first discovery hit) that
    // lacks the ABI must not silently load a different engine.
    throw std::runtime_error(
        "XTBDlopen: missing C ABI symbols in engine library");
  }
  if (abi_fn() != RGPOT_XTB_ABI_VERSION) {
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
    throw std::runtime_error(std::string("XTBDlopen: create failed: ") + err);
  }
  m_set_warm(m_pot, config.xtb.warm_move);
  m_set_min_gap(m_pot, config.xtb.warm_min_gap);
  m_lib = lib.native_handle();
  lib.release();
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
  DynLib::close_handle(m_lib);
  m_lib = nullptr;
}

FusedSettingsSpec XTBDlopen::fusedSettings() const {
  FusedSettingsSpec spec;
  spec.spin = m_config.uhf;
  spec.sccMaxIterations = static_cast<unsigned>(m_config.max_iterations);
  spec.charge = m_config.charge;
  spec.accuracy = m_config.accuracy;
  spec.electronicTemperature = m_config.electronic_temperature;
  spec.warmMove = warmMove();
  spec.warmMinGap = warmMinGap();
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
