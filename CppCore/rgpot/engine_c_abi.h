/**
 * @file engine_c_abi.h
 * @brief Generic C ABI for rgpot engine plugins.
 *
 * One symbol set for every heavy backend: an engine shared library
 * (libuma_engine.so, libmetatomic_engine.so, ...) exports these
 * functions, backend selection is which library the host dlopens, and
 * configuration travels as a Cap'n Proto flat-array message on the
 * shared Potentials.capnp wire (the engine names its root params
 * struct, e.g. UmaParams). Adding a backend therefore needs no new
 * host code and no new ABI header.
 *
 * Units: positions Angstrom, energy eV, forces eV/Angstrom.
 *
 * The optional coordinate transform lets the host inject a
 * pre-evaluation map over (positions, box) — for example the
 * molecular-box convention that recenters a gas-phase molecule in a
 * large fixed cell so compiled static-shape graphs see a constant
 * edge count. The engine applies it to its own scratch copy; the
 * caller's buffers are never mutated.
 *
 * Conventions:
 *
 * - Return values are three-valued: 0 success, positive recoverable
 *   (the evaluation failed at this input, the caller may back off or
 *   evaluate elsewhere), negative fatal (the instance is unusable).
 * - Codes the engines in this repository return: 1 invalid argument, 2 the
 *   evaluation or the setting failed (the reason is in
 *   rgpot_engine_last_error), 3 the backend does not support the operation,
 *   and for force the host transform's own nonzero code.
 *
 * Thread safety of a handle. Any thread may call force and force_batch on any
 * handle. What happens inside depends on the backend's capabilities
 * (rgpot_engine_caps().reentrancy): 0 shared instance, calls run
 * concurrently on one handle; 1 per instance, the engine admits one call at a
 * time per handle; 2 process serial, the engine admits one call at a time in
 * the process across all handles of the library. A host that wants
 * parallelism from a backend that reports 1 or 2 creates one handle per
 * worker (1 only). rgpot_engine_set_num_threads,
 * rgpot_engine_set_charge_spin, rgpot_engine_set_callbacks,
 * rgpot_engine_hint and rgpot_engine_destroy must not run concurrently with
 * a call on the same handle. rgpot_engine_last_error and
 * rgpot_engine_model_digest may run at any time.
 * - Capability discovery is by symbol presence: entry points marked
 *   optional are probed with dlsym. An absent symbol means the engine
 *   does not support the operation; a static (non-learning) engine
 *   simply omits the learning set.
 * - Engines with internal mutable state (learning surrogates) carry a
 *   monotone model-state counter. Any refit, hyperparameter change,
 *   or observation incorporation increments it; host-visible caches
 *   keyed on model output are valid only within one counter value.
 *   Acquisition (delegating a point to the oracle callback and
 *   returning its result verbatim) never increments the counter;
 *   incorporation happens only at declared boundaries (a batch call,
 *   a certify event, an explicit host hint), each of which may
 *   increment it and fire on_model_update.
 */
#pragma once

#include <stddef.h>

#ifdef __cplusplus
extern "C" {
#endif

#ifdef _WIN32
#ifdef RGPOT_ENGINE_BUILD
#define RGPOT_ENGINE_API __declspec(dllexport)
#else
#define RGPOT_ENGINE_API __declspec(dllimport)
#endif
#else
#define RGPOT_ENGINE_API __attribute__((visibility("default")))
#endif

typedef struct RgpotEnginePot RgpotEnginePot;

/**
 * Optional host-side transform applied by the engine to its scratch
 * copy of (positions, box) before evaluation. positions holds
 * 3*nAtoms doubles, box 9 doubles (row-major). Return 0 on success;
 * nonzero aborts the force call with that code.
 */
typedef int (*rgpot_engine_coord_transform)(void *user, long nAtoms,
                                            double *positions, double *box);

/**
 * ABI identity. `RGPOT_ENGINE_ABI_VERSION` is the major: an incompatible change
 * increments it, and a host compares it for equality. Additions that leave
 * every existing call valid (a new optional symbol, a new field at the end of
 * a size-prefixed struct) increment `RGPOT_ENGINE_ABI_MINOR`. The layout
 * revision increments when an existing struct changes size or field order.
 * Hosts that read only `rgpot_engine_abi_version()` keep working across minor
 * bumps; `rgpot_engine_abi_stamp()` is optional and an engine without it is
 * minor 0, layout 1.
 */
#define RGPOT_ENGINE_ABI_VERSION 1
#define RGPOT_ENGINE_ABI_MINOR 3
#define RGPOT_ENGINE_ABI_LAYOUT_REVISION 1

#define RGPOT_ENGINE_OK 0
#define RGPOT_ENGINE_ERR_INVALID 1
#define RGPOT_ENGINE_ERR_EVAL 2
#define RGPOT_ENGINE_ERR_UNSUPPORTED 3

RGPOT_ENGINE_API int rgpot_engine_abi_version(void);

RGPOT_ENGINE_API int rgpot_engine_available(void);

/**
 * Create a potential from a Cap'n Proto flat-array message whose root
 * struct is the engine's params arm in Potentials.capnp (UmaParams for
 * the UMA engine). config points at config_len bytes (word-aligned
 * capnp framing as produced by capnp::messageToFlatArray). Returns
 * NULL on failure with a message in errbuf.
 */
RGPOT_ENGINE_API RgpotEnginePot *rgpot_engine_create(const void *config,
                                                     size_t config_len,
                                                     char *errbuf,
                                                     size_t errlen);

/**
 * Gate settings for a learning engine, passed as the config blob.
 *
 * INTERIM. The config arm is specified as a Cap'n Proto message and
 * becomes GprParams once Potentials.capnp carries a nested
 * PotentialSpec (rgpot-b7zw). Until then a learning engine has no way
 * to be configured at all, which leaves certifyBelowForce at zero and
 * so leaves a driver's convergence claim unbacked by ground truth --
 * the one guarantee such an engine exists to provide. A fixed-layout
 * struct with a magic word closes that hole without pretending to be
 * capnp: an engine dispatches on the magic, so the capnp message can
 * replace it later without either side guessing which it received.
 */
#define RGPOT_ENGINE_GPR_GATE_MAGIC 0x47505247u /* "GRPG" */
#define RGPOT_ENGINE_GPR_GATE_VERSION 1u

typedef struct {
  unsigned magic;   /**< RGPOT_ENGINE_GPR_GATE_MAGIC */
  unsigned version; /**< RGPOT_ENGINE_GPR_GATE_VERSION */
  /** Answer from the model below this multiple of its likelihood noise. */
  double sigmaLoFactor;
  /** Above this multiple the geometry is not learned from. */
  double sigmaHiFactor;
  /** Predicted force magnitude under this (eV/Angstrom) forces a
   *  ground-truth call, so a driver's convergence claim is checked
   *  rather than trusted. Non-positive disables the certificate. */
  double certifyBelowForce;
  /** Refit cadence in training-set appends; zero refits at every
   *  incorporation boundary. */
  unsigned long long refitTrainingRevisions;
} RgpotEngineGprGate;

RGPOT_ENGINE_API void rgpot_engine_destroy(RgpotEnginePot *pot);

/**
 * Energy + forces. transform (with transform_user) may be NULL;
 * variance may be NULL. Returns 0 on success.
 */
RGPOT_ENGINE_API int rgpot_engine_force(
    RgpotEnginePot *pot, long nAtoms, const double *positions,
    const int *atomicNrs, double *forces, double *energy, double *variance,
    const double *box, rgpot_engine_coord_transform transform,
    void *transform_user);

/* ------------------------------------------------------------------ */
/* Optional symbols: probe with dlsym; absence means unsupported.     */
/* ------------------------------------------------------------------ */

/**
 * Analytic Hessian-vector product H(positions) * direction, in eV/Angstrom
 * for a direction in Angstrom. direction and hv_out hold 3*nAtoms doubles.
 * Returns RGPOT_ENGINE_OK (0) on success and
 * RGPOT_ENGINE_ERR_UNSUPPORTED (3) when the backend does not support
 * analytic HVP.
 */
RGPOT_ENGINE_API int rgpot_engine_hvp(
    RgpotEnginePot *pot, long nAtoms, const double *positions,
    const int *atomicNrs, const double *direction, double *hv_out,
    const double *box);

/**
 * Strided-uniform batched evaluation: nSystems systems sharing one
 * composition (atomicNrs, nAtoms). positions and forces hold
 * 3*nAtoms*nSystems doubles with system stride 3*nAtoms, energies
 * nSystems doubles, variances likewise (may be NULL), boxes
 * 9*nSystems doubles with system stride 9. The transform is applied
 * per system to the engine's scratch copy. For learning engines this
 * call is also an incorporation boundary. A grouped variant (batches
 * of uniform groups) is a separate symbol, never a flag here.
 */
RGPOT_ENGINE_API int rgpot_engine_force_batch(
    RgpotEnginePot *pot, long nSystems, long nAtoms, const double *positions,
    const int *atomicNrs, double *forces, double *energies, double *variances,
    const double *boxes, rgpot_engine_coord_transform transform,
    void *transform_user);

/**
 * Monotone model-state counter for engines with internal mutable
 * state. Increments on every refit / hyperparameter change /
 * observation incorporation; host caches keyed on model output are
 * valid only while the value is unchanged. Static engines omit the
 * symbol (equivalent to a counter frozen at 0).
 */
RGPOT_ENGINE_API unsigned long long rgpot_engine_model_state(
    const RgpotEnginePot *pot);

/** Why a learning engine delegated a point to the oracle. */
typedef enum {
  RGPOT_ACQUIRE_VARIANCE = 1, /**< predictive variance above sigma_lo */
  RGPOT_ACQUIRE_CERTIFY = 2,  /**< verifying a surrogate convergence claim */
  RGPOT_ACQUIRE_BATCH = 3,    /**< batched acquisition sweep */
  RGPOT_ACQUIRE_REJECT = 4    /**< geometry above sigma_hi, evaluated but
                                   not incorporated */
} rgpot_engine_acquire_reason;

/**
 * Host callbacks for learning engines. oracle evaluates the ground
 * truth (same conventions and three-valued return as
 * rgpot_engine_force; variance is absent because the oracle is
 * exact). on_acquire fires when the engine delegates a point;
 * on_model_update fires after each incorporation with the new
 * model-state value and marks the boundary at which history-based
 * host state (quasi-Newton memory, cached factorizations) must be
 * reset. Either notification pointer may be NULL.
 */
typedef struct {
  void *ctx;
  int (*oracle)(void *ctx, long nAtoms, const double *positions,
                const int *atomicNrs, double *forces, double *energy,
                const double *box);
  void (*on_acquire)(void *ctx, rgpot_engine_acquire_reason reason);
  void (*on_model_update)(void *ctx, unsigned long long model_state);
} RgpotEngineCallbacks;

/**
 * Register host callbacks. The struct is copied; ctx must outlive the
 * potential or a subsequent call replacing it. Learning engines
 * require an oracle before the first evaluation and fail create-time
 * validation without one.
 */
RGPOT_ENGINE_API int rgpot_engine_set_callbacks(
    RgpotEnginePot *pot, const RgpotEngineCallbacks *callbacks);

/**
 * Runtime hint/tunable message: a Cap'n Proto flat-array whose root is
 * the engine's hint arm on the shared Potentials.capnp wire (never a
 * string-keyed channel). Also the host-initiated incorporation
 * boundary for learning engines. The engine validates the resulting
 * configuration as a whole before it takes effect; on failure the
 * previous configuration stays active and errbuf holds the reason.
 */
RGPOT_ENGINE_API int rgpot_engine_hint(RgpotEnginePot *pot, const void *msg,
                                       size_t msg_len, char *errbuf,
                                       size_t errlen);

/** Compatibility identity of the loaded engine (see RGPOT_ENGINE_ABI_MINOR). */
typedef struct {
  unsigned short abi_major;
  unsigned short abi_minor;
  unsigned short layout_revision;
} RgpotEngineAbiStamp;

/** The loaded engine's identity. */
RGPOT_ENGINE_API RgpotEngineAbiStamp rgpot_engine_abi_stamp(void);

/**
 * Nonzero when the loaded engine serves a host built against `stamp`: equal
 * major, equal layout revision, engine minor at least the host's. NULL
 * returns 0.
 */
RGPOT_ENGINE_API int rgpot_engine_abi_compatible(
    const RgpotEngineAbiStamp *stamp);

/**
 * Concurrency and evaluation capabilities, the engine-side image of rgpot's
 * PotCaps. `size` is sizeof(RgpotEngineCaps) on the host; the engine fills
 * the fields that fit and sets `size` to the size it wrote.
 * reentrancy: 0 one instance may be called from many threads, 1 one instance
 * per thread, 2 every evaluation in the process is serialized.
 */
typedef struct {
  unsigned size;
  int reentrancy;
  int per_image_instances; /**< clone one instance per NEB image */
  int batched;             /**< force_batch is served natively */
  int periodic;            /**< periodic boundary conditions supported */
  int stress;              /**< cell stress is available */
  int group_collective;    /**< every rank of a group enters each call */
  int orbital_selection;   /**< select_orbitals is supported */
  int orbital_copy;        /**< copy_orbitals is supported */
  int analytic_hvp;        /**< rgpot_engine_hvp is exact for this potential */
} RgpotEngineCaps;

RGPOT_ENGINE_API int rgpot_engine_caps(const RgpotEnginePot *pot,
                                       RgpotEngineCaps *out);

/**
 * Message of the last failed call on this instance (create errors go to its
 * errbuf). Copies at most `len - 1` bytes and NUL-terminates; returns the
 * full message length, 0 when there is none. Reading does not clear it; the
 * next successful call does.
 */
RGPOT_ENGINE_API size_t rgpot_engine_last_error(const RgpotEnginePot *pot,
                                                char *buf, size_t len);

/** Short name of the backend the library serves, for example "uma". */
RGPOT_ENGINE_API const char *rgpot_engine_name(void);

/**
 * Version of the engine build: "<name>/<rgpot version>". The pointer refers to
 * static storage.
 */
RGPOT_ENGINE_API const char *rgpot_engine_version(void);

/**
 * SHA-256 of the model file or archive the instance loaded, as 64 lowercase
 * hexadecimal digits. Copies at most `len - 1` bytes and NUL-terminates;
 * returns the digest length (64), or 0 when the backend has no model file or
 * the file cannot be read. The digest is computed on first request and kept.
 */
RGPOT_ENGINE_API size_t rgpot_engine_model_digest(const RgpotEnginePot *pot,
                                                  char *buf, size_t len);

/**
 * Backend thread settings: intra-op and inter-op worker threads of the
 * engine's tensor runtime. A value below 1 leaves that setting alone. Process
 * wide in effect, and the inter-op count can be set once before the first
 * evaluation; a refused setting returns a positive code with the reason in
 * last_error. Returns 0 on success and 3 for a backend without a tensor
 * runtime. UmaParams and MetatomicParams carry the same two counts and apply
 * them at create.
 */
RGPOT_ENGINE_API int rgpot_engine_set_num_threads(RgpotEnginePot *pot,
                                                  int intra_op, int inter_op);

/**
 * Change the total charge and spin multiplicity the next evaluations assume.
 * Returns 0 when the values reach the backend, 1 for a multiplicity below 1,
 * and 3 for a backend without a charge or spin or a value it cannot take; a
 * refused call changes nothing and leaves the reason in last_error. A UMA
 * package was exported for one charge and spin; a mismatch is reported by the
 * next force call. A Metatomic model reads those it declares in
 * requested_inputs, and one it does not read must stay at charge 0 or
 * multiplicity 1.
 */
RGPOT_ENGINE_API int rgpot_engine_set_charge_spin(RgpotEnginePot *pot,
                                                  int charge, int spin);

/**
 * Select the active orbital key for subsequent evaluations.
 * On CPMD backends, accepting a key seeds the next SCF from the session's
 * current converged orbitals or from the orbitals stored for that key.
 * Returns 0 on success (clearing last_error), 1 when pot is NULL (invalid
 * argument, leaving last_error unchanged), 2 when the backend's
 * orbital-selection hook fails with an exception, and 3 when the backend has
 * no orbital selection or refuses key through Unsupported; codes 2 and 3
 * record the reason in last_error.
 */
RGPOT_ENGINE_API int rgpot_engine_select_orbitals(RgpotEnginePot *pot,
                                                  long long key);

/**
 * Copy stored orbitals from one key to another.
 * Returns 0 on success (clearing last_error), 1 when pot is NULL (invalid
 * argument, leaving last_error unchanged), 2 when the backend's orbital-copy
 * hook fails with an exception, and 3 when the backend has no orbital-copy
 * primitive or refuses the copy through Unsupported; codes 2 and 3 record
 * the reason in last_error.
 */
RGPOT_ENGINE_API int rgpot_engine_copy_orbitals(RgpotEnginePot *pot,
                                                long long from_key,
                                                long long to_key);

/**
 * Neighbor-list declaration. An engine exporting this symbol expects
 * the host to provision every declared list before evaluation; an
 * engine omitting it self-provisions (the UMA molecular-box engine
 * builds its own complete graph). strict zero permits over-inclusion.
 */
typedef struct {
  double cutoff;  /**< Angstrom */
  int full_list;  /**< 1 full, 0 half */
  int strict;     /**< 0: over-inclusion tolerated */
} RgpotEngineNeighborOptions;

/**
 * Writes up to capacity declarations into opts and returns the total
 * count (call with capacity 0 to size). Declarations may change after
 * on_model_update; hosts re-query on that boundary.
 */
RGPOT_ENGINE_API long rgpot_engine_neighbor_options(
    const RgpotEnginePot *pot, RgpotEngineNeighborOptions *opts,
    long capacity);

#ifdef __cplusplus
}
#endif
