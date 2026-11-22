#pragma once
// MIT License
// Copyright 2023--present rgpot developers

/**
 * @brief Prefix-parameterized loader for the minimum potential ABI profile.
 *
 * Backends (nwchemc, cpmdc, future lammpsc/gromacsc) export the same C symbol
 * set prefix-parameterized on `<p>` (see potentials-schema PROFILE.md). This
 * loader resolves that whole set from one dlopen'd shared library, so rgpot
 * drives every conforming backend through identical function pointers with no
 * per-backend loader code. Everything stays native: flat binary Cap'n Proto
 * messages both directions, plain C calls in-process.
 */

#include "rgpot/abi/DynLib.hpp"
#include "rgpot/abi/EngineDlopen.hpp"

#include <cstddef>
#include <initializer_list>
#include <stdexcept>
#include <string>
#include <vector>

namespace rgpot {
namespace abi {

/**
 * Layout-compatible with every backend's `<P>Result` POD
 * (`NWChemCResult`, `CPMDCResult`, ...): {int ok; double energy_h;
 * char message[512];}. PROFILE.md fixes this shape, which is what makes the
 * prefix-generic function pointers below well-defined.
 */
struct ProfileResult {
  int ok;
  double energy_h;
  char message[512];
};

class ProfileLoader {
public:
  using AbiVersionFn = int (*)(void);
  using VersionFn = const char *(*)(void);
  using LastErrorFn = const char *(*)(void);
  using AvailableFn = int (*)(void);
  using FinalizeFn = void (*)(void);
  using ConfigureFn = int (*)(const void *, size_t);
  using SessionCreateFromConfigFn = void *(*)(const void *, size_t);
  using SessionConfigureFn = int (*)(void *, const void *, size_t);
  using SessionDestroyFn = void (*)(void *);
  using SessionCalculateResultFn = ProfileResult (*)(void *, const void *,
                                                     size_t, void *, size_t,
                                                     size_t *);
  using CalculateResultFromConfigFn = ProfileResult (*)(const void *, size_t,
                                                        const void *, size_t,
                                                        void *, size_t,
                                                        size_t *);
  using PotentialResultSizeFn = size_t (*)(const void *, size_t);
  using CapabilitiesResultFn = int (*)(void *, size_t, size_t *);

  ProfileLoader() = default;

  /** Candidate library paths for a prefix: explicit path, env hints,
   * platform default names. */
  static std::vector<std::string>
  candidates(const std::string &prefix, const std::string &explicit_path) {
    std::string upper;
    for (char c : prefix)
      upper.push_back(static_cast<char>(
          c >= 'a' && c <= 'z' ? c - ('a' - 'A') : c));
    const std::string env_lib = upper + "_LIBRARY";
    const std::string env_eng = "RGPOT_" + upper + "_ENGINE";
    return engine_candidates(prefix, explicit_path,
                             {env_lib.c_str(), env_eng.c_str()});
  }

  void clear() {
    lib_.close();
    prefix_.clear();
    loaded_path_.clear();
    abi_version = nullptr;
    version = nullptr;
    last_error = nullptr;
    available = nullptr;
    finalize = nullptr;
    configure = nullptr;
    session_create_from_config = nullptr;
    session_configure = nullptr;
    session_destroy = nullptr;
    session_calculate_result = nullptr;
    calculate_result_from_config = nullptr;
    potential_result_size_for_force_input = nullptr;
    capabilities_result = nullptr;
  }

  /** Open the first loadable library in @p cands and resolve any available
   * minimum-profile symbols without throwing on partial/legacy engines. */
  bool open_candidates(const std::string &prefix,
                       const std::vector<std::string> &cands,
                       std::string *open_error = nullptr) {
    clear();
    prefix_ = prefix;
    if (!open_first_candidate(lib_, cands, &loaded_path_, open_error, true)) {
      return false;
    }
    resolve_profile_symbols(nullptr);
    return true;
  }

  /**
   * dlopen the first loadable candidate and resolve the full minimum
   * profile. Throws std::runtime_error naming every missing symbol, so a
   * non-conforming build fails loudly at load time instead of at dispatch.
   */
  void load(const std::string &prefix, const std::string &explicit_path = "") {
    std::string open_error;
    if (!open_candidates(prefix, candidates(prefix, explicit_path),
                         &open_error)) {
      throw std::runtime_error("lib" + prefix + " not loaded: " + open_error);
    }

    std::vector<std::string> missing;
    resolve_profile_symbols(&missing);

    if (!missing.empty()) {
      std::string what = "lib" + prefix + " does not conform to the minimum "
                                          "potential ABI profile; missing:";
      for (const auto &sym : missing)
        what += " " + sym;
      throw std::runtime_error(what);
    }
  }

  bool loaded() const { return lib_.valid() && capabilities_result; }
  bool has_library() const { return lib_.valid(); }
  bool has_session_result_profile() const {
    return session_destroy && potential_result_size_for_force_input &&
           session_calculate_result;
  }
  const std::string &prefix() const { return prefix_; }
  const std::string &loaded_path() const { return loaded_path_; }
  DynLib &lib() { return lib_; }
  const DynLib &lib() const { return lib_; }

  /** Flat `Capabilities` message bytes (size query + fill). */
  std::vector<unsigned char> capabilities() const {
    size_t required = 0;
    (void)capabilities_result(nullptr, 0, &required);
    if (required == 0)
      throw std::runtime_error(prefix_ + "_capabilities_result sized 0");
    std::vector<unsigned char> buf(required);
    size_t written = 0;
    if (capabilities_result(buf.data(), buf.size(), &written) != 0 ||
        written == 0)
      throw std::runtime_error(prefix_ + "_capabilities_result failed");
    buf.resize(written);
    return buf;
  }

  AbiVersionFn abi_version = nullptr;
  VersionFn version = nullptr;
  LastErrorFn last_error = nullptr;
  AvailableFn available = nullptr;
  FinalizeFn finalize = nullptr;
  ConfigureFn configure = nullptr;
  SessionCreateFromConfigFn session_create_from_config = nullptr;
  SessionConfigureFn session_configure = nullptr;
  SessionDestroyFn session_destroy = nullptr;
  SessionCalculateResultFn session_calculate_result = nullptr;
  CalculateResultFromConfigFn calculate_result_from_config = nullptr;
  PotentialResultSizeFn potential_result_size_for_force_input = nullptr;
  CapabilitiesResultFn capabilities_result = nullptr;

private:
  void resolve_profile_symbols(std::vector<std::string> *missing) {
    abi_version = resolve<AbiVersionFn>("abi_version", missing);
    version = resolve<VersionFn>("version", missing);
    last_error = resolve<LastErrorFn>("last_error", missing);
    available = resolve<AvailableFn>("available", missing);
    finalize = resolve<FinalizeFn>("finalize", missing);
    configure = resolve<ConfigureFn>("configure", missing);
    session_create_from_config = resolve<SessionCreateFromConfigFn>(
        "session_create_from_config", missing);
    session_configure =
        resolve<SessionConfigureFn>("session_configure", missing);
    session_destroy = resolve<SessionDestroyFn>("session_destroy", missing);
    session_calculate_result = resolve<SessionCalculateResultFn>(
        "session_calculate_result", missing);
    calculate_result_from_config = resolve<CalculateResultFromConfigFn>(
        "calculate_result_from_config", missing);
    potential_result_size_for_force_input = resolve<PotentialResultSizeFn>(
        "potential_result_size_for_force_input", missing);
    capabilities_result =
        resolve<CapabilitiesResultFn>("capabilities_result", missing);
  }

  template <typename Fn>
  Fn resolve(const std::string &name, std::vector<std::string> *missing) {
    const std::string sym = prefix_ + "_" + name;
    Fn fn = lib_.sym_optional<Fn>(sym.c_str());
    if (!fn && missing)
      missing->push_back(sym);
    return fn;
  }

  DynLib lib_;
  std::string prefix_;
  std::string loaded_path_;
};

} // namespace abi
} // namespace rgpot
