// MIT License
// Copyright 2023--present rgpot developers
//
// The LJ engine plugin speaks rgpot_engine_force, the same entry a
// metatomic or UMA engine exports. A host dlopens it and does not link
// the potential.

#include "lj_eval.hpp"

#include <cstdio>
#include <cstring>
#include <dlfcn.h>

namespace {

using CreateFn = void *(*)(const void *, size_t, char *, size_t);
using ForceFn = int (*)(void *, long, const double *, const int *, double *, double *,
                        double *, const double *, void *, void *);
using LastErrorFn = size_t (*)(const void *, char *, size_t);
using DestroyFn = void (*)(void *);

// A failed force call leaves a message. rgpot_engine_last_error
// NUL-terminates every buffer it is given, including one of a single byte.
int check_last_error(const char *library) {
  void *lib = dlopen(library, RTLD_NOW | RTLD_LOCAL);
  if (lib == nullptr) {
    std::fprintf(stderr, "dlopen: %s\n", dlerror());
    return 1;
  }
  auto create = reinterpret_cast<CreateFn>(dlsym(lib, "rgpot_engine_create"));
  auto force = reinterpret_cast<ForceFn>(dlsym(lib, "rgpot_engine_force"));
  auto last_error = reinterpret_cast<LastErrorFn>(dlsym(lib, "rgpot_engine_last_error"));
  auto destroy = reinterpret_cast<DestroyFn>(dlsym(lib, "rgpot_engine_destroy"));
  if (create == nullptr || force == nullptr || last_error == nullptr || destroy == nullptr) {
    std::fprintf(stderr, "missing rgpot_engine_* symbol\n");
    return 1;
  }
  const double bad[3] = {1.0, -1.0, 1.0};
  char err[256] = {};
  void *pot = create(bad, sizeof(bad), err, sizeof(err));
  if (pot == nullptr) {
    std::fprintf(stderr, "create: %s\n", err);
    return 1;
  }
  const double xyz[6] = {0, 0, 0, 1.5, 0, 0};
  const int z[2] = {1, 1};
  const double box[9] = {20, 0, 0, 0, 20, 0, 0, 0, 20};
  double f[6];
  double e = 0;
  double var = 0;
  int status = 1;
  if (force(pot, 2, xyz, z, f, &e, &var, box, nullptr, nullptr) == 0) {
    std::fprintf(stderr, "negative cutoff was accepted\n");
  } else {
    char one = 'x';
    const size_t full = last_error(pot, &one, 1);
    char wide[256];
    std::memset(wide, 'x', sizeof(wide));
    const size_t again = last_error(pot, wide, sizeof(wide));
    if (full == 0 || one != '\0') {
      std::fprintf(stderr, "last_error(len 1) left the buffer unterminated\n");
    } else if (again != full || std::strlen(wide) != full) {
      std::fprintf(stderr, "last_error length %zu, message %zu\n", full, std::strlen(wide));
    } else {
      status = 0;
    }
  }
  destroy(pot);
  return status;
}

} // namespace

int main(int argc, char **argv) {
  if (argc != 2) {
    std::fprintf(stderr, "usage: test_engine liblj_engine.so\n");
    return 2;
  }
  const double xyz[6] = {0, 0, 0, 1.5, 0, 0};
  const int z[2] = {1, 1};
  const double box[9] = {20, 0, 0, 0, 20, 0, 0, 0, 20};
  const double config[3] = {1.0, 8.0, 1.0};
  double direct_f[6];
  double direct_e = 0;
  int has_stress = 0;
  double stress[9];
  if (rgpot_lj_eval(xyz, z, 2, box, config[0], config[1], config[2], direct_f, &direct_e,
                    &has_stress, stress) != 0) {
    return 1;
  }
  double plugin_f[6];
  double plugin_e = 0;
  char err[256] = {};
  if (rgpot_engine_eval(argv[1], config, sizeof(config), xyz, z, 2, box, plugin_f, &plugin_e, err,
                        sizeof(err)) != 0) {
    std::fprintf(stderr, "%s\n", err);
    return 1;
  }
  if (plugin_e != direct_e || std::memcmp(plugin_f, direct_f, sizeof(direct_f)) != 0) {
    std::fprintf(stderr, "engine %g direct %g\n", plugin_e, direct_e);
    return 1;
  }
  double again = 0;
  if (rgpot_engine_eval(argv[1], nullptr, 0, xyz, z, 2, box, plugin_f, &again, err,
                        sizeof(err)) != 0 || again != direct_e) {
    std::fprintf(stderr, "reopened engine %g\n", again);
    return 1;
  }
  if (check_last_error(argv[1]) != 0) {
    return 1;
  }
  std::printf("engine energy %.10f\n", plugin_e);
  return 0;
}
