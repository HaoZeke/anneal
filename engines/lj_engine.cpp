// MIT License
// Copyright 2023--present rgpot developers
//
// In-process engine plugin. Metatomic and UMA export the same
// rgpot_engine_* names from their own libraries. A null config is the
// default shifted Lennard-Jones potential (u0 = 1 eV, cutoff = 15 A,
// psi = 1 A). A 24-byte config is three little-endian float64 values:
// u0, cutoff, psi.

#include "rgpot/engine_c_abi.h"
#include "lj_eval.hpp"

#include <cstdio>
#include <cstring>
#include <new>
#include <string>
#include <vector>

namespace {

struct LjEngine {
  double u0 = 1.0;
  double cutoff = 15.0;
  double psi = 1.0;
  std::string last_error;
};

void set_error(char *errbuf, size_t errlen, const char *sentence) {
  if (errbuf == nullptr || errlen == 0) {
    return;
  }
  std::snprintf(errbuf, errlen, "%s", sentence);
}

} // namespace

extern "C" {

int rgpot_engine_abi_version(void) { return RGPOT_ENGINE_ABI_VERSION; }

int rgpot_engine_available(void) { return 1; }

RgpotEnginePot *rgpot_engine_create(const void *config, size_t config_len,
                                    char *errbuf, size_t errlen) {
  auto *pot = new (std::nothrow) LjEngine();
  if (pot == nullptr) {
    set_error(errbuf, errlen, "out of memory");
    return nullptr;
  }
  if (config == nullptr || config_len == 0) {
    return reinterpret_cast<RgpotEnginePot *>(pot);
  }
  if (config_len != 3 * sizeof(double)) {
    delete pot;
    set_error(errbuf, errlen, "LJ engine config is three float64 values");
    return nullptr;
  }
  std::memcpy(&pot->u0, config, sizeof(double));
  std::memcpy(&pot->cutoff, static_cast<const char *>(config) + sizeof(double),
              sizeof(double));
  std::memcpy(&pot->psi, static_cast<const char *>(config) + 2 * sizeof(double),
              sizeof(double));
  return reinterpret_cast<RgpotEnginePot *>(pot);
}

void rgpot_engine_destroy(RgpotEnginePot *pot) {
  delete reinterpret_cast<LjEngine *>(pot);
}

int rgpot_engine_force(RgpotEnginePot *pot, long nAtoms, const double *positions,
                       const int *atomicNrs, double *forces, double *energy,
                       double *variance, const double *box,
                       rgpot_engine_coord_transform transform, void *transform_user) {
  if (pot == nullptr || energy == nullptr || nAtoms < 0) {
    return 1;
  }
  auto *lj = reinterpret_cast<LjEngine *>(pot);
  std::vector<double> scratch_pos;
  std::vector<double> scratch_box(9);
  const double *pos = positions;
  const double *cell = box;
  if (transform != nullptr) {
    scratch_pos.assign(positions, positions + static_cast<size_t>(nAtoms) * 3);
    if (box != nullptr) {
      std::memcpy(scratch_box.data(), box, 9 * sizeof(double));
    }
    const int code = transform(transform_user, nAtoms, scratch_pos.data(), scratch_box.data());
    if (code != 0) {
      return code;
    }
    pos = scratch_pos.data();
    cell = scratch_box.data();
  }
  int has_stress = 0;
  double stress[9];
  const int code = rgpot_lj_eval(pos, atomicNrs, static_cast<size_t>(nAtoms), cell, lj->u0,
                                 lj->cutoff, lj->psi, forces, energy, &has_stress, stress);
  if (variance != nullptr) {
    *variance = 0.0;
  }
  if (code != 0) {
    lj->last_error = "Lennard-Jones evaluation rejected the geometry";
    return 2;
  }
  lj->last_error.clear();
  return 0;
}

size_t rgpot_engine_last_error(const RgpotEnginePot *pot, char *buf, size_t len) {
  if (pot == nullptr || buf == nullptr || len == 0) {
    return 0;
  }
  const std::string &message = reinterpret_cast<const LjEngine *>(pot)->last_error;
  std::snprintf(buf, len, "%s", message.c_str());
  return message.size();
}

const char *rgpot_engine_name(void) { return "lj"; }

} // extern "C"
