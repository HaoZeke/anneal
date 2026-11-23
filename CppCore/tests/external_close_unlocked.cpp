// MIT License
// Copyright 2023--present rgpot developers
//
// The external-potential constructor must not call dlclose while it holds the
// resident table mutex. dlclose takes the dynamic loader's lock, and a plugin
// initializer that runs under that lock may construct another external
// potential, which waits for the table mutex.
//
// This executable interposes dlopen and dlclose. The first dlopen of the
// engine constructs the same engine again from inside the call, so the table
// already holds the file when the outer constructor publishes it. The outer
// constructor then closes its own, redundant, handle. Each dlclose runs a
// probe on another thread that constructs a potential from the resident
// engine. The probe needs the table mutex, so it can only finish while the
// closing thread does not hold it.

#include "rgpot/ExternalPot/external_potentials.h"

#include <dlfcn.h>

#include <atomic>
#include <chrono>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <future>

namespace {

const char *engine_path = nullptr;
bool armed = false;
std::atomic<bool> nested{false};
std::atomic<bool> probing{false};
std::atomic<bool> probes_on{false};
std::atomic<int> closes{0};
std::atomic<int> blocked{0};
std::future<bool> probe_result;

rgpot_external_potential *make(char *err, std::size_t err_len) {
  err[0] = '\0';
  return rgpot_lammps_potential(engine_path, 1.0, 8.0, 1.0, err, err_len);
}

bool construct_and_free() {
  char err[512] = {};
  rgpot_external_potential *pot = make(err, sizeof(err));
  if (pot == nullptr) {
    std::fprintf(stderr, "engine constructor: %s\n", err);
    return false;
  }
  rgpot_external_potential_free(pot);
  return true;
}

} // namespace

extern "C" void *dlopen(const char *file, int flags) {
  using Open = void *(*)(const char *, int);
  static const Open real = reinterpret_cast<Open>(dlsym(RTLD_NEXT, "dlopen"));
  void *handle = real(file, flags);
  if (handle != nullptr && armed && !nested.load() && file != nullptr &&
      (flags & RTLD_NOLOAD) == 0 && std::strcmp(file, engine_path) == 0) {
    armed = false;
    nested.store(true);
    const bool built = construct_and_free();
    nested.store(false);
    if (!built) {
      std::fputs("the nested constructor failed\n", stderr);
      std::_Exit(1);
    }
  }
  return handle;
}

extern "C" int dlclose(void *handle) {
  using Close = int (*)(void *);
  static const Close real =
      reinterpret_cast<Close>(dlsym(RTLD_NEXT, "dlclose"));
  if (!probes_on.load() || probing.exchange(true)) {
    return real(handle);
  }
  closes.fetch_add(1);
  probe_result =
      std::async(std::launch::async, [] { return construct_and_free(); });
  if (probe_result.wait_for(std::chrono::seconds(2)) !=
      std::future_status::ready) {
    blocked.fetch_add(1);
  }
  const int status = real(handle);
  probing.store(false);
  return status;
}

int main(int argc, char **argv) {
  if (argc != 2) {
    std::fprintf(stderr, "usage: external_close_unlocked engine.so\n");
    return 2;
  }
  engine_path = argv[1];
  armed = true;
  probes_on.store(true);
  const bool constructed = construct_and_free();
  probes_on.store(false);
  if (!constructed) {
    return 1;
  }
  if (probe_result.valid() && !probe_result.get()) {
    std::fputs("the probe constructor failed\n", stderr);
    return 1;
  }
  if (closes.load() == 0) {
    std::fputs("the redundant open was never closed, so the probe did not "
               "run\n",
               stderr);
    return 1;
  }
  if (blocked.load() != 0) {
    std::fputs("dlclose ran while the resident table mutex was held\n",
               stderr);
    return 1;
  }
  std::puts("dlclose runs outside the resident table mutex");
  return 0;
}
