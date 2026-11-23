/*
 * Public constructors for the LAMMPS, GROMACS and OpenMM potentials.
 *
 * SPDX-License-Identifier: MIT
 * Copyright 2023--present rgpot developers
 *
 * Positions are Angstrom, energies are eV, forces are eV/Angstrom.
 * A box is nine doubles, lattice vectors as rows. With an engine path,
 * GROMACS and OpenMM convert u0, cutoff and psi to kJ/mol and nanometres
 * once, in the constructor, and the geometry, energy and forces on every
 * call.
 */

#ifndef RGPOT_EXTERNAL_POTENTIALS_H
#define RGPOT_EXTERNAL_POTENTIALS_H

#include <stddef.h>

#ifdef __cplusplus
extern "C" {
#endif

typedef struct rgpot_external_potential rgpot_external_potential;

/* kJ/mol in one eV, and nanometres in one Angstrom. */
double rgpot_md_kj_per_ev(void);
double rgpot_md_nm_per_angstrom(void);

/*
 * Shifted 12-6 Lennard-Jones. u0 is eV, cutoff and psi are Angstrom.
 * engine_path NULL or "" searches RGPOT_LAMMPS_ENGINE, then
 * LAMMPS_ENGINE when they are set and not empty, then
 * liblammps_engine.so and liblj_engine.so. The plugin stays the backend: the constructor
 * dlopens it and calls rgpot_engine_force. Returns NULL and writes
 * err on failure.
 *
 * The first successful open of an engine file is kept and marked
 * RTLD_NODELETE, so the library stays mapped after
 * rgpot_external_potential_free and a later constructor of that file reuses
 * the open. An open that does not become the resident engine is closed.
 * A file replaced at a path that was already loaded is refused.
 */
rgpot_external_potential *rgpot_lammps_potential(const char *engine_path,
                                                double u0, double cutoff,
                                                double psi, char *err,
                                                size_t err_len);

/*
 * The same potential. The constructor stores u0 in kJ/mol and the
 * lengths in nanometres. engine_path NULL evaluates that potential
 * in process. A path is a library that exports rgpot_engine_create,
 * rgpot_engine_destroy and rgpot_md_native_force in those units.
 */
rgpot_external_potential *rgpot_gromacs_potential(const char *engine_path,
                                                 double u0, double cutoff,
                                                 double psi, char *err,
                                                 size_t err_len);

rgpot_external_potential *rgpot_openmm_potential(const char *engine_path,
                                                double u0, double cutoff,
                                                double psi, char *err,
                                                size_t err_len);

void rgpot_external_potential_free(rgpot_external_potential *pot);

/* Returns 0, or 1 and writes err. forces holds 3 * n doubles. */
int rgpot_external_potential_energy_forces(
    rgpot_external_potential *pot, const double *xyz, const int *atomic_numbers,
    size_t n, const double *box9, double *forces, double *energy, char *err,
    size_t err_len);

#ifdef __cplusplus
}
#endif

#endif
