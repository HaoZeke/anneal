#pragma once
// MIT License
// Copyright 2023--present rgpot developers

#include <stddef.h>

#ifdef __cplusplus
extern "C" {
#endif

#if defined(RGPOT_CP2KC_BUILD) && (defined(_WIN32) || defined(_WIN64))
#define RGPOT_CP2KC_API __declspec(dllexport)
#else
#define RGPOT_CP2KC_API
#endif

/**
 * @file cp2kc_abi.h
 * @brief C ABI between CP2KPot and the libcp2k shim (libcp2kc_rgpot).
 *
 * The shim owns the libcp2k state: initialisation, the MPI communicator,
 * one force_env per orbital key, and the SCF convergence check. It speaks
 * CP2K's atomic units and takes no Cap'n Proto: positions and cell vectors
 * arrive in Bohr, the energy leaves in Hartree and the forces in
 * Hartree/Bohr. CP2KPot converts to and from Angstrom and eV.
 *
 * Every rank of a calculator calls the session functions together, in the
 * same order and with the same arguments.
 */

/** Version of this header. CP2KPot refuses a shim that reports another. */
#define CP2KC_ABI_VERSION 1

/** Return codes of cp2kc_session_calculate. */
enum {
  /** Energy and forces are valid. */
  CP2KC_OK = 0,
  /** An argument was null, or the atom count differs from the deck. */
  CP2KC_BAD_ARGUMENT = 1,
  /**
   * The SCF did not converge. Every rank of the calculator returns this
   * code for the same call, so the caller can reject the point without
   * aborting the world. The energy and forces are not valid.
   */
  CP2KC_SCF_NOT_CONVERGED = 2,
  /** Negative values are fatal. */
  CP2KC_FATAL = -1
};

typedef struct CP2KCSession CP2KCSession;

/** @brief CP2KC_ABI_VERSION of the shim. */
RGPOT_CP2KC_API int cp2kc_abi_version(void);

/** @brief Shim and CP2K version text. The pointer stays valid. */
RGPOT_CP2KC_API const char *cp2kc_version(void);

/** @brief Non-zero when the shim can run CP2K. */
RGPOT_CP2KC_API int cp2kc_available(void);

/**
 * @brief Borrow the calculator communicator from rgpot.
 *
 * @p comm points at an MPI_Comm of @p comm_bytes bytes. The shim keeps the
 * handle, converts it to the Fortran handle libcp2k takes, and never frees
 * it. Returns the calculator index, or -1 to refuse the communicator.
 * Without this call the shim runs every force_env on MPI_COMM_WORLD.
 */
RGPOT_CP2KC_API int cp2kc_adopt_calculator_comm(const void *comm,
                                                size_t comm_bytes,
                                                int ranks_per_calc);

/**
 * @brief Create a session from a complete CP2K input deck.
 *
 * @p deck is the input text. @p dir is an existing directory: the shim
 * writes @p deck to `dir/input.inp` and CP2K writes its output to
 * `dir/cp2k.out` (the first force_env) or `dir/cp2k-<n>.out`. The first
 * session initialises libcp2k: cp2k_init_without_mpi when MPI is already
 * initialised, else cp2k_init, in which case the shim finalises CP2K and
 * MPI at exit. Returns null with a message in @p err on failure.
 */
RGPOT_CP2KC_API CP2KCSession *cp2kc_session_create(const char *deck,
                                                   const char *dir, char *err,
                                                   size_t err_len);

/**
 * @brief Name the calculation the following evaluations belong to.
 *
 * The shim keeps one force_env per key, created on the key's first
 * evaluation, so each band image or ring-polymer bead keeps its own
 * wavefunction and extrapolation history. Before the first call the key
 * is 0. Returns 0.
 */
RGPOT_CP2KC_API int cp2kc_session_select_orbitals(CP2KCSession *session,
                                                  long long key);

/**
 * @brief Evaluate the energy and forces of one geometry.
 *
 * @param n_atoms Atom count; must match the deck.
 * @param pos_bohr 3*n_atoms Cartesian positions in Bohr.
 * @param cell_bohr9 Cell vectors a, b, c as the rows of a row-major 3x3
 *        matrix, in Bohr. The shim calls cp2k_set_cell only when the cell
 *        differs from the one the key's force_env last saw.
 * @param e_ha Potential energy in Hartree.
 * @param f_ha_bohr 3*n_atoms forces (not gradients) in Hartree/Bohr.
 * @return One of the CP2KC_* codes.
 */
RGPOT_CP2KC_API int cp2kc_session_calculate(CP2KCSession *session, int n_atoms,
                                            const double *pos_bohr,
                                            const double *cell_bohr9,
                                            double *e_ha, double *f_ha_bohr,
                                            char *err, size_t err_len);

/** @brief Destroy every force_env of the session and free it. */
RGPOT_CP2KC_API void cp2kc_session_destroy(CP2KCSession *session);

#ifdef __cplusplus
}
#endif
