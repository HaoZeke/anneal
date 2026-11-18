#pragma once

#include <stddef.h>

#ifdef __cplusplus
extern "C" {
#endif

#if defined(RGPOT_CPMDC_BUILD) && (defined(_WIN32) || defined(_WIN64))
#define RGPOT_CPMDC_API __declspec(dllexport)
#else
#define RGPOT_CPMDC_API
#endif

/**
 * @file cpmd_c_abi.h
 * @brief Stable C ABI for configuring and evaluating embedded OpenCPMD.
 *
 * The parameter buffer accepted by this API is an unpacked flat Cap'n Proto
 * message whose root type is `CPMDParams` from `schema/Potentials.capnp`.
 *
 * Direct-call "socket" style: create one `CPMDCSession` from `CPMDParams`,
 * then pass a serialized `ForceInput` for each geometry step and receive an
 * unpacked flat `PotentialResult` (same carrier as nwchemc / rgpot).
 */

/**
 * @brief Result returned by energy / gradient / forces entry points.
 */
typedef struct CPMDCResult {
  /** Non-zero when the calculation succeeds. */
  int ok;
  /** Total energy in Hartree (CPMD native a.u. energy). */
  double energy_h;
  /** Null-terminated status or error message. */
  char message[512];
} CPMDCResult;

/** Opaque handle for repeated evaluations with one Cap'n Proto parameter set.
 */
typedef struct CPMDCSession CPMDCSession;

typedef enum CPMDCFeatureKind {
  CPMDC_FEATURE_SECTION = 1,
  CPMDC_FEATURE_PARAMS = 2,
  CPMDC_FEATURE_ABI = 3,
  CPMDC_FEATURE_KEYWORD = 4
} CPMDCFeatureKind;

typedef struct CPMDCFeatureEntry {
  const char *feature_id;
  CPMDCFeatureKind kind;
  int stub_applicable;
  int embed_applicable;
} CPMDCFeatureEntry;

/**
 * @brief Apply CPMD method parameters from a Cap'n Proto message.
 *
 * @param params_capnp Pointer to an unpacked flat `CPMDParams` message.
 * @param params_capnp_size_bytes Size of `params_capnp` in bytes.
 * @return 0 on success, -1 on parse or configuration failure.
 */
RGPOT_CPMDC_API int cpmdc_set_params(const void *params_capnp,
                                     size_t params_capnp_size_bytes);

/**
 * @brief Compute energy and nuclear gradient for an atomic configuration.
 *
 * Positions are Angstrom; gradient is Hartree/Bohr (CPMD ionic forces are
 * negated into a nuclear gradient for API symmetry with nwchemc).
 */
RGPOT_CPMDC_API CPMDCResult
cpmdc_energy_gradient(int n_atoms, const double *positions_ang,
                      const int *atomic_numbers, const void *params_capnp,
                      size_t params_capnp_size_bytes, double *grad_h_bohr);

/**
 * @brief Compute total energy only (no gradient allocation).
 */
RGPOT_CPMDC_API CPMDCResult cpmdc_energy(int n_atoms,
                                         const double *positions_ang,
                                         const int *atomic_numbers,
                                         const void *params_capnp,
                                         size_t params_capnp_size_bytes);

/**
 * @brief Compute energy and nuclear forces (negative gradient, Hartree/Bohr).
 */
RGPOT_CPMDC_API CPMDCResult cpmdc_energy_forces(int n_atoms,
                                                const double *positions_ang,
                                                const int *atomic_numbers,
                                                const void *params_capnp,
                                                size_t params_capnp_size_bytes,
                                                double *forces_h_bohr);

/**
 * @brief Create a persistent evaluation session from a Cap'n Proto message.
 *
 * The session owns a copy of the serialized message so callers may release the
 * input buffer after this call returns.
 */
RGPOT_CPMDC_API CPMDCSession *
cpmdc_session_create(const void *params_capnp, size_t params_capnp_size_bytes);

/**
 * @brief Replace Cap'n Proto parameters before the session accepts topology.
 */
RGPOT_CPMDC_API int cpmdc_session_set_params(CPMDCSession *session,
                                             const void *params_capnp,
                                             size_t params_capnp_size_bytes);

/** @brief Release a persistent evaluation session. */
RGPOT_CPMDC_API void cpmdc_session_destroy(CPMDCSession *session);

/**
 * @brief Name the calculation the next evaluations of this session belong to.
 *
 * The engine keeps the converged orbitals per key, so a calculator that
 * evaluates several images or beads in turn starts each SCF from that
 * image's or bead's own previous orbitals. Optional: an engine without the
 * symbol keeps one stored copy.
 *
 * @return 0 when `session` is non-null and the runtime accepts `key` (seeding
 *         a new key from the session's current orbitals or restoring a
 *         previously stored key's orbitals), and -1 when `session` is NULL,
 *         `key` is refused, or the library is the stub without an embedded
 *         runtime (leaving the session's active key and orbitals unchanged).
 *         Seeding means choosing the next SCF initial guess: the session's
 *         current converged orbitals for a key with no stored copy, or the
 *         orbitals last converged under that key when one exists.
 *         `cpmdc_last_error` is not declared here and `CPMDPot` does not read
 *         it. This note covers only the stub and the reference fake: the stub
 *         does not define `cpmdc_last_error`, and the reference fake stores one
 *         string that a failing select sets and a successful select clears.
 *         Other engines are not described.
 */
RGPOT_CPMDC_API int cpmdc_session_select_orbitals(CPMDCSession *session,
                                                  long long key);

/**
 * @brief Copy stored orbitals from one key to another in a live session.
 *
 * Returns 0 on success and -1 when either key is invalid or unsupported.
 */
RGPOT_CPMDC_API int cpmdc_session_copy_orbitals(CPMDCSession *session,
                                                long long from_key,
                                                long long to_key);

RGPOT_CPMDC_API CPMDCResult cpmdc_session_energy_gradient(
    CPMDCSession *session, int n_atoms, const double *positions_ang,
    const int *atomic_numbers, double *grad_h_bohr);

RGPOT_CPMDC_API CPMDCResult cpmdc_session_energy(CPMDCSession *session,
                                                 int n_atoms,
                                                 const double *positions_ang,
                                                 const int *atomic_numbers);

RGPOT_CPMDC_API CPMDCResult cpmdc_session_energy_forces(
    CPMDCSession *session, int n_atoms, const double *positions_ang,
    const int *atomic_numbers, double *forces_h_bohr);

/**
 * @brief Compute energy and forces for one Cap'n Proto `ForceInput` step.
 *
 * Session keeps persistent `CPMDParams`; each call supplies geometry. Returned
 * energy/forces use CPMD native units: Hartree and Hartree/Bohr.
 */
RGPOT_CPMDC_API CPMDCResult cpmdc_session_calculate_forces(
    CPMDCSession *session, const void *force_input_capnp,
    size_t force_input_capnp_size_bytes, double *forces_h_bohr,
    size_t forces_len);

/**
 * @brief Compute forces for one `ForceInput` step and write `PotentialResult`.
 *
 * Direct-call socket entry point: method state in the session, geometry in
 * `ForceInput`, output energy/forces converted to `ForceInput.energyUnit` and
 * `energyUnit / lengthUnit`.
 *
 * When `potential_result_capnp_capacity_bytes` is too small, returns `ok == 0`,
 * writes the required byte count to `potential_result_capnp_size_bytes`, and
 * does not evaluate CPMD.
 */
RGPOT_CPMDC_API CPMDCResult cpmdc_session_calculate_result(
    CPMDCSession *session, const void *force_input_capnp,
    size_t force_input_capnp_size_bytes, void *potential_result_capnp,
    size_t potential_result_capnp_capacity_bytes,
    size_t *potential_result_capnp_size_bytes);

/**
 * @brief One-shot Cap'n Proto entry point (params + ForceInput ->
 * PotentialResult).
 *
 * Multi-step callers should create one session and call
 * `cpmdc_session_calculate_result()` per step.
 */
RGPOT_CPMDC_API CPMDCResult cpmdc_calculate_result(
    const void *params_capnp, size_t params_capnp_size_bytes,
    const void *force_input_capnp, size_t force_input_capnp_size_bytes,
    void *potential_result_capnp, size_t potential_result_capnp_capacity_bytes,
    size_t *potential_result_capnp_size_bytes);

/**
 * @brief Byte count needed for a `PotentialResult` for the given `ForceInput`.
 *
 * Parses geometry only; does not initialize or evaluate CPMD. Returns 0 when
 * the message is invalid or too large for the C ABI.
 */
RGPOT_CPMDC_API size_t cpmdc_potential_result_size_for_force_input(
    const void *force_input_capnp, size_t force_input_capnp_size_bytes);

/** @brief Compiled library version string. */
RGPOT_CPMDC_API const char *cpmdc_version(void);

/** @brief 1 when the embedded OpenCPMD runtime is available. */
RGPOT_CPMDC_API int cpmdc_available(void);

/** @brief Finalize an owned embedded CPMD runtime. */
RGPOT_CPMDC_API void cpmdc_finalize(void);

RGPOT_CPMDC_API size_t cpmdc_feature_count(void);
RGPOT_CPMDC_API const CPMDCFeatureEntry *cpmdc_feature_table(void);
RGPOT_CPMDC_API const CPMDCFeatureEntry *
cpmdc_feature_find(const char *feature_id);

#ifdef __cplusplus
}
#endif
