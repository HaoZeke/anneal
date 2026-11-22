#pragma once
// MIT License
// Copyright 2023--present rgpot developers

/**
 * @brief Portable dynamic library loader (dlopen / LoadLibrary family).
 *
 * Shared by ProfileLoader and the dlopen potential frontends (NWChemPot,
 * CPMDPot, CP2KPot, MOPACPot, XTBDlopen, MetatomicDlopen).
 */

#include <stdexcept>
#include <string>

#if defined(_WIN32) || defined(_WIN64)
// Cap'n Proto headers must not see full windows.h: commdlg defines
// IPrintDialogServices, which then collides with capnp's Kind templates
// (MSVC C2838 in generated-header-support.h / capability.h).
#ifndef WIN32_LEAN_AND_MEAN
#define WIN32_LEAN_AND_MEAN
#endif
#ifndef NOMINMAX
#define NOMINMAX
#endif
#include <windows.h>
#else
#include <dlfcn.h>
#endif

namespace rgpot {

class DynLib {
public:
  DynLib() = default;

  explicit DynLib(const std::string &path, bool global = true) {
    open(path, global);
  }

  DynLib(const DynLib &) = delete;
  DynLib &operator=(const DynLib &) = delete;

  DynLib(DynLib &&other) noexcept : handle_(other.handle_) {
    other.handle_ = nullptr;
  }

  DynLib &operator=(DynLib &&other) noexcept {
    if (this != &other) {
      close();
      handle_ = other.handle_;
      other.handle_ = nullptr;
    }
    return *this;
  }

  ~DynLib() { close(); }

  bool valid() const { return handle_ != nullptr; }
  void *native_handle() const noexcept { return handle_; }

  static void close_handle(void *h) {
    if (!h)
      return;
#if defined(_WIN32) || defined(_WIN64)
    FreeLibrary(static_cast<HMODULE>(h));
#else
    dlclose(h);
#endif
  }

  static std::string last_error() {
#if defined(_WIN32) || defined(_WIN64)
    return {};
#else
    const char *err = dlerror();
    return err ? std::string(err) : std::string{};
#endif
  }

  void open(const std::string &path, bool global = true) {
    close();
#if defined(_WIN32) || defined(_WIN64)
    (void)global;
    handle_ = LoadLibraryA(path.c_str());
    if (!handle_) {
      throw std::runtime_error("LoadLibrary failed: " + path);
    }
#else
    const int flags = RTLD_NOW | (global ? RTLD_GLOBAL : RTLD_LOCAL);
    handle_ = dlopen(path.c_str(), flags);
    if (!handle_) {
      const char *err = dlerror();
      throw std::runtime_error(std::string("dlopen failed: ") + path + ": " +
                               (err ? err : "unknown"));
    }
#endif
  }

  void close() {
    close_handle(handle_);
    handle_ = nullptr;
  }

  /// Relinquish ownership of the underlying handle without closing it (e.g.
  /// when keeping a library mapped through process exit).
  void release() noexcept { handle_ = nullptr; }

  template <typename T> T sym(const char *name) const {
    void *p = sym_raw(name);
    if (!p) {
      throw std::runtime_error(std::string("symbol not found: ") + name);
    }
    return reinterpret_cast<T>(p);
  }

  template <typename T> T sym_optional(const char *name) const {
    void *p = sym_raw(name);
    return p ? reinterpret_cast<T>(p) : nullptr;
  }

private:
  void *sym_raw(const char *name) const {
    if (!handle_)
      return nullptr;
#if defined(_WIN32) || defined(_WIN64)
    return reinterpret_cast<void *>(
        GetProcAddress(static_cast<HMODULE>(handle_), name));
#else
    dlerror(); // clear
    return dlsym(handle_, name);
#endif
  }

  void *handle_ = nullptr;
};

} // namespace rgpot
