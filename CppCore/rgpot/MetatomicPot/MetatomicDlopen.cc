// MIT License — dlopen frontend for libmetatomic_engine.so
#include "rgpot/MetatomicPot/MetatomicDlopen.hpp"
#include "rgpot/abi/DynLib.hpp"
#include "rgpot/abi/EngineDlopen.hpp"

#include <stdexcept>
#include <string>
#include <vector>

namespace rgpot {

namespace {

RgpotMtaConfig to_c_config(const MetatomicConfig &c) {
  RgpotMtaConfig out{};
  out.model_path = c.model_path.c_str();
  out.device = c.device.c_str();
  out.length_unit = c.length_unit.c_str();
  out.extensions_directory = c.extensions_directory.c_str();
  out.check_consistency = c.check_consistency ? 1 : 0;
  out.uncertainty_threshold = c.uncertainty_threshold;
  out.dtype_override = c.dtype_override.c_str();
  out.random_rotation = c.random_rotation ? 1 : 0;
  out.n_symmetry_rotations = c.n_symmetry_rotations;
  out.so3_probe_scatter = c.so3_probe_scatter ? 1 : 0;
  out.torch_determinism_strict =
      (c.torch_determinism == TorchDeterminismPolicy::Strict) ? 1 : 0;
  return out;
}

} // namespace

MetatomicDlopen::MetatomicDlopen(const MetatomicConfig &config) {
  const auto cands = abi::engine_candidates(
      "metatomic_engine", config.engine_path,
      {"RGPOT_METATOMIC_ENGINE", "METATOMIC_ENGINE"},
      abi::CandidateOptions{.include_cwd_dot_slash = false,
                            .include_dll = false,
                            .include_dir_envs = true,
                            .exclusive_explicit = false,
                            .platform_only_soname = true});
  DynLib lib;
  std::string loaded_path;
  std::string open_err;
  if (!abi::open_first_candidate(lib, cands, &loaded_path, &open_err, true)) {
    std::string msg =
        "MetatomicDlopen: libmetatomic_engine.so not found "
        "(set RGPOT_METATOMIC_ENGINE / engine_path / EON_POTENTIALS_PATH)";
    if (!open_err.empty()) {
      msg += "; last dlerror: ";
      msg += open_err;
    }
    throw std::runtime_error(msg);
  }

  auto abi_fn = lib.sym_optional<int (*)(void)>("rgpot_mta_abi_version");
  m_create = lib.sym_optional<create_fn>("rgpot_mta_create");
  m_destroy = lib.sym_optional<destroy_fn>("rgpot_mta_destroy");
  m_force = lib.sym_optional<force_fn>("rgpot_mta_force");
  if (!abi_fn || !m_create || !m_destroy || !m_force) {
    throw std::runtime_error(
        "MetatomicDlopen: missing C ABI symbols in engine library");
  }
  if (abi_fn() != RGPOT_MTA_ABI_VERSION) {
    throw std::runtime_error("MetatomicDlopen: ABI version mismatch");
  }
  char err[1024]{};
  auto cfg = to_c_config(config);
  m_pot = m_create(&cfg, err, sizeof err);
  if (!m_pot) {
    throw std::runtime_error(std::string("MetatomicDlopen: create failed: ") +
                             err);
  }
  m_lib = lib.native_handle();
  lib.release();
}

MetatomicDlopen::~MetatomicDlopen() {
  // Keep the engine mapped: torch/metatomic static teardown after dlclose
  // routinely SEGV at process exit.
  if (m_pot && m_destroy)
    m_destroy(m_pot);
  m_pot = nullptr;
  m_lib = nullptr;
}

void MetatomicDlopen::forceImpl(const ForceInput &in, ForceOut *out) const {
  if (!m_pot || !m_force)
    throw std::runtime_error("MetatomicDlopen: engine not available");
  const int rc =
      m_force(m_pot, static_cast<long>(in.nAtoms), in.pos, in.atmnrs, out->F,
              &out->energy, &out->variance, in.box);
  if (rc != 0)
    throw std::runtime_error("MetatomicDlopen: force failed rc=" +
                             std::to_string(rc));
}

} // namespace rgpot
