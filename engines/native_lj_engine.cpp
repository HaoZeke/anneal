// MIT License
// Copyright 2023--present rgpot developers
//
// Native-unit backend for the GROMACS and OpenMM constructors.
// Positions are nanometres, energy is kJ/mol, forces are kJ/mol/nm.
// The body is the in-tree shifted Lennard-Jones potential: lengths
// are scaled by 10 to Angstrom and the energy by the stored
// kJ/mol-per-eV factor. The constructor applies the inverse once.

#include "lj_eval.hpp"
#include "rgpot/ExternalPot/md_units.hpp"
#include "rgpot/engine_c_abi.h"

#include <cstdio>
#include <cstring>
#include <new>
#include <string>
#include <vector>

namespace {

struct NativeLj {
  double u0_kj = 0.0;
  double cutoff_nm = 0.0;
  double psi_nm = 0.0;
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
  if (config == nullptr || config_len != 3 * sizeof(double)) {
    set_error(errbuf, errlen, "native LJ config is three float64 values");
    return nullptr;
  }
  auto *pot = new (std::nothrow) NativeLj();
  if (pot == nullptr) {
    set_error(errbuf, errlen, "out of memory");
    return nullptr;
  }
  std::memcpy(&pot->u0_kj, config, sizeof(double));
  std::memcpy(&pot->cutoff_nm,
              static_cast<const char *>(config) + sizeof(double),
              sizeof(double));
  std::memcpy(&pot->psi_nm,
              static_cast<const char *>(config) + 2 * sizeof(double),
              sizeof(double));
  return reinterpret_cast<RgpotEnginePot *>(pot);
}

void rgpot_engine_destroy(RgpotEnginePot *pot) {
  delete reinterpret_cast<NativeLj *>(pot);
}

int rgpot_md_native_force(void *pot, long nAtoms, const double *positions,
                          const int *atomicNrs, double *forces, double *energy,
                          double *variance, const double *box) {
  if (pot == nullptr || energy == nullptr || nAtoms < 0 ||
      (nAtoms > 0 && (positions == nullptr || atomicNrs == nullptr ||
                      forces == nullptr || box == nullptr))) {
    return 1;
  }
  auto *lj = reinterpret_cast<NativeLj *>(pot);
  const auto n = static_cast<size_t>(nAtoms);
  std::vector<double> host_pos(n * 3);
  double host_box[9];
  for (size_t i = 0; i < n * 3; ++i) {
    host_pos[i] = positions[i] * rgpot::md::kAngstromPerNm;
  }
  for (int i = 0; i < 9; ++i) {
    host_box[i] = box[i] * rgpot::md::kAngstromPerNm;
  }
  std::vector<double> host_forces(n * 3);
  double host_energy = 0.0;
  int has_stress = 0;
  double stress[9];
  const double u0 = lj->u0_kj * rgpot::md::kEvPerKj;
  const double cutoff = lj->cutoff_nm * rgpot::md::kAngstromPerNm;
  const double psi = lj->psi_nm * rgpot::md::kAngstromPerNm;
  const int code =
      rgpot_lj_eval(host_pos.data(), atomicNrs, n, host_box, u0, cutoff, psi,
                    host_forces.data(), &host_energy, &has_stress, stress);
  if (code != 0) {
    lj->last_error = "native Lennard-Jones rejected the geometry";
    return 2;
  }
  *energy = host_energy * rgpot::md::kKjPerEv;
  for (size_t i = 0; i < n * 3; ++i) {
    forces[i] =
        host_forces[i] * rgpot::md::kKjPerEv * rgpot::md::kAngstromPerNm;
  }
  if (variance != nullptr) {
    *variance = 0.0;
  }
  lj->last_error.clear();
  return 0;
}

const char *rgpot_engine_name(void) { return "native-lj"; }

} // extern "C"
