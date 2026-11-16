#pragma once
// MIT License
// Copyright 2023--present rgpot developers
//
// One Lennard-Jones evaluation shared by the engine drivers. Positions
// and the cell are Angstrom, energy is eV, forces are eV/Angstrom.
// box9 is nine doubles, lattice vectors as rows.

#include <stddef.h>

#ifdef __cplusplus
extern "C" {
#endif

/* forces holds 3 * n doubles. stress holds 9 doubles when has_stress
 * is set. Returns 0, or 1 when an argument is unusable. Each thread keeps
 * the potential of its last call, whose pair list serves the next call
 * with the same parameters, atom count and atomic numbers. */
int rgpot_lj_eval(const double *xyz, const int *atomic_numbers, size_t n,
                  const double *box9, double u0, double cutoff, double psi,
                  double *forces, double *energy, int *has_stress,
                  double stress[9]);

/* Load an rgpot_engine_* plugin and evaluate one frame. config may be
 * null. The library is kept open for the process. Returns 0, or 1 and
 * writes a sentence into err (err_len bytes, including the NUL). */
int rgpot_engine_eval(const char *library, const void *config, size_t config_len,
                      const double *xyz, const int *atomic_numbers, size_t n,
                      const double *box9, double *forces, double *energy,
                      char *err, size_t err_len);

#ifdef __cplusplus
}
#endif
