#pragma once
// MIT License
// Copyright 2023--present rgpot developers

/**
 * @brief Shared candidate-path discovery, dlopen loop, and probe helpers for
 * runtime-loaded engine libraries.
 */

#include "rgpot/abi/DynLib.hpp"

#include <cstdlib>
#include <initializer_list>
#include <mutex>
#include <stdexcept>
#include <string>
#include <vector>

namespace rgpot {
namespace abi {

inline bool has_directory_separator(const std::string &path) {
  return path.find_first_of("/\\") != std::string::npos;
}

inline void append_env_dirs(std::vector<std::string> &out, const char *env_name,
                            const std::string &stem) {
  const char *env = std::getenv(env_name);
  if (!env || !*env) {
    return;
  }
  const std::string s(env);
  std::size_t start = 0;
  while (start < s.size()) {
    std::size_t pos = s.find(':', start);
    if (pos == std::string::npos) {
      pos = s.size();
    }
    if (pos > start) {
      std::string dir = s.substr(start, pos - start);
      if (!dir.empty() && dir.back() != '/') {
        dir += '/';
      }
#ifdef __APPLE__
      out.push_back(dir + "lib" + stem + ".dylib");
#endif
      out.push_back(dir + "lib" + stem + ".so");
    }
    start = pos + 1;
  }
}

struct CandidateOptions {
  bool include_cwd_dot_slash = true;
  bool include_dll = true;
  bool include_dir_envs = false;
  bool exclusive_explicit = false;
  bool platform_only_soname = false;
  /// When an environment variable names a library, do not also try the
  /// default file names. An explicit path still follows exclusive_explicit.
  bool exclusive_env = false;
};

inline std::vector<std::string>
engine_candidates(const std::string &stem, const std::string &explicit_path,
                  std::initializer_list<const char *> env_vars,
                  const CandidateOptions &opts = {}) {
  std::vector<std::string> out;
  if (!explicit_path.empty()) {
    out.emplace_back(explicit_path);
    if (opts.exclusive_explicit) {
      return out;
    }
  }
  const std::size_t before_env = out.size();
  for (const char *var : env_vars) {
    if (const char *e = std::getenv(var); e && *e) {
      out.emplace_back(e);
    }
  }
  if (opts.exclusive_env && out.size() > before_env) {
    return out;
  }
  if (opts.platform_only_soname) {
#ifdef _WIN32
    out.emplace_back(stem + ".dll");
    out.emplace_back("lib" + stem + ".dll");
#elif defined(__APPLE__)
    out.emplace_back("lib" + stem + ".dylib");
    out.emplace_back("lib" + stem + ".so");
#else
    out.emplace_back("lib" + stem + ".so");
#endif
  } else {
    out.emplace_back("lib" + stem + ".so");
    if (opts.include_cwd_dot_slash) {
      out.emplace_back("./lib" + stem + ".so");
    }
    out.emplace_back("lib" + stem + ".dylib");
    if (opts.include_cwd_dot_slash) {
      out.emplace_back("./lib" + stem + ".dylib");
    }
    if (opts.include_dll) {
      out.emplace_back(stem + ".dll");
    }
  }
  if (opts.include_dir_envs) {
    append_env_dirs(out, "EON_POTENTIALS_PATH", stem);
    append_env_dirs(out, "RGPOT_ENGINE_PATH", stem);
  }
  return out;
}

inline bool open_first_candidate(DynLib &lib,
                                 const std::vector<std::string> &candidates,
                                 std::string *loaded_path,
                                 std::string *open_error,
                                 bool global = true) {
  if (loaded_path) {
    loaded_path->clear();
  }
  if (open_error) {
    open_error->clear();
  }
  std::string err;
  for (const auto &cand : candidates) {
    try {
      lib.open(cand, global);
      if (loaded_path) {
        *loaded_path = cand;
      }
      return true;
    } catch (const std::exception &ex) {
      err = ex.what();
    }
  }
  if (open_error) {
    *open_error = err;
  }
  return false;
}

template <typename Fn>
inline bool probe_once(std::mutex &mu, bool &done, bool &ok, Fn &&fn) {
  std::lock_guard<std::mutex> lock(mu);
  if (!done) {
    ok = fn();
    done = true;
  }
  return ok;
}

} // namespace abi
} // namespace rgpot
