// MIT License
// Copyright 2023--present rgpot developers

#include "lj_eval.hpp"

#include "rgpot/KeptPotential.hpp"
#include "rgpot/LennardJones/LJPot.hpp"
#include "rgpot/engine_c_abi.h"

#include <dlfcn.h>

#include <cstdio>
#include <cstring>
#include <memory>

namespace {

void fail(char *err, size_t err_len, const char *sentence) {
  if (err == nullptr || err_len == 0) {
    return;
  }
  std::snprintf(err, err_len, "%s", sentence);
}

} // namespace

int rgpot_lj_eval(const double *xyz, const int *atomic_numbers, size_t n,
                  const double *box9, double u0, double cutoff, double psi,
                  double *forces, double *energy, int *has_stress,
                  double stress[9]) {
  if (energy == nullptr || has_stress == nullptr || stress == nullptr) {
    return 1;
  }
  if (n > 0 && (xyz == nullptr || atomic_numbers == nullptr || box9 == nullptr ||
                forces == nullptr)) {
    return 1;
  }
  if (!(cutoff > 0.0) || !(psi > 0.0)) {
    return 1;
  }
  // The pair list carries over to the next call with the same parameters,
  // atom count and atomic numbers.
  rgpot::KeptKey key;
  key.f64(u0).f64(cutoff).f64(psi).atoms(n, n > 0 ? atomic_numbers : nullptr);
  rgpot::Kept<rgpot::LJPot>::with(
      key,
      [&] {
        rgpot::LJConfig config;
        config.u0 = u0;
        config.cutoff = cutoff;
        config.psi = psi;
        return std::make_unique<rgpot::LJPot>(config);
      },
      [&](const rgpot::LJPot &pot) {
        rgpot::ForceOut out{forces, 0.0, 0.0, {}, 0};
        const rgpot::ForceInput in{n, xyz, atomic_numbers, box9};
        pot.forceImpl(in, &out);
        *energy = out.energy;
        *has_stress = out.has_stress;
        std::memcpy(stress, out.stress, 9 * sizeof(double));
      });
  return 0;
}

namespace {

struct Engine {
  void *library = nullptr;
  RgpotEnginePot *pot = nullptr;
  int (*force)(RgpotEnginePot *, long, const double *, const int *, double *,
               double *, double *, const double *, rgpot_engine_coord_transform,
               void *) = nullptr;
};

Engine &cached() {
  static Engine engine;
  return engine;
}

} // namespace

int rgpot_engine_eval(const char *library, const void *config, size_t config_len,
                      const double *xyz, const int *atomic_numbers, size_t n,
                      const double *box9, double *forces, double *energy,
                      char *err, size_t err_len) {
  if (library == nullptr || energy == nullptr ||
      (n > 0 && (xyz == nullptr || atomic_numbers == nullptr || box9 == nullptr ||
                 forces == nullptr))) {
    fail(err, err_len, "engine evaluation needs a library and a geometry");
    return 1;
  }
  Engine &engine = cached();
  if (engine.library == nullptr) {
    engine.library = dlopen(library, RTLD_NOW | RTLD_LOCAL);
    if (engine.library == nullptr) {
      fail(err, err_len, dlerror() != nullptr ? dlerror() : "dlopen failed");
      return 1;
    }
    auto *create = reinterpret_cast<RgpotEnginePot *(*)(const void *, size_t, char *, size_t)>(
        dlsym(engine.library, "rgpot_engine_create"));
    engine.force = reinterpret_cast<decltype(engine.force)>(
        dlsym(engine.library, "rgpot_engine_force"));
    if (create == nullptr || engine.force == nullptr) {
      fail(err, err_len, "engine library has no rgpot_engine_create/force");
      return 1;
    }
    char why[256] = {};
    engine.pot = create(config, config_len, why, sizeof(why));
    if (engine.pot == nullptr) {
      fail(err, err_len, why[0] != '\0' ? why : "rgpot_engine_create returned null");
      return 1;
    }
  }
  double variance = 0.0;
  const int status = engine.force(engine.pot, static_cast<long>(n), xyz, atomic_numbers,
                                  forces, energy, &variance, box9, nullptr, nullptr);
  if (status != 0) {
    fail(err, err_len, "rgpot_engine_force failed");
    return 1;
  }
  return 0;
}
