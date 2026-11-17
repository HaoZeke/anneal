/*
 * rgpot plugin ABI -- version 1
 *
 * SPDX-License-Identifier: MIT
 * Copyright 2023--present rgpot developers
 *
 * This header defines the stable C ABI that any potential energy surface
 * implementation can conform to. A plugin is a shared library (.so/.dylib/.dll)
 * that exports a single symbol: rgpot_plugin_init(), returning a pointer to a
 * static descriptor struct.
 *
 * A host that calls rgpot::plugin::PluginRegistry::discover() scans the
 * directories listed in RGPOT_PLUGIN_PATH, dlopens each library, calls
 * rgpot_plugin_init(), validates the API version, and registers the plugin.
 * librgpot itself never calls discover().
 *
 * Plugins can be implemented in any language that can produce a C-compatible
 * shared library: C, C++, Rust, Fortran (bind(C)), Python (ctypes/cffi), etc.
 *
 * This header is STANDALONE -- it requires only <stddef.h> and <stdint.h>.
 * It has no dependency on rgpot internals, DLPack, or any C++ headers.
 */

#ifndef RGPOT_PLUGIN_H
#define RGPOT_PLUGIN_H

#include <stddef.h>
#include <stdint.h>

#ifdef __cplusplus
extern "C" {
#endif

/* ---- Version ---- */

#define RGPOT_PLUGIN_API_VERSION 1

/* ---- Status codes ---- */

typedef enum rgpot_plugin_status_t {
  RGPOT_PLUGIN_OK = 0,
  RGPOT_PLUGIN_INVALID_PARAM = 1,
  RGPOT_PLUGIN_ERROR = 2,
  RGPOT_PLUGIN_NOT_SUPPORTED = 3,
} rgpot_plugin_status_t;

/* ---- Capability flags (bitmask) ---- */

typedef enum rgpot_plugin_caps_t {
  RGPOT_CAP_ENERGY = (1 << 0),
  RGPOT_CAP_FORCES = (1 << 1),
  RGPOT_CAP_VARIANCE = (1 << 2),
  RGPOT_CAP_STRESS = (1 << 3), /* reserved */
  /* Independent instances may calculate concurrently. Absent means the
   * host serializes the call behind one process-wide lock shared by every
   * plugin without this flag. */
  RGPOT_CAP_INSTANCE_SAFE = (1 << 4),
  RGPOT_CAP_PERIODIC = (1 << 5),
} rgpot_plugin_caps_t;

/* ---- Data types (flat C arrays, no external deps) ---- */

typedef struct rgpot_plugin_input_t {
  size_t n_atoms;
  const double *positions;      /* [n_atoms * 3], row-major xyz */
  const int *atomic_numbers;    /* [n_atoms] */
  const double *cell;           /* [9], row-major 3x3 */
} rgpot_plugin_input_t;

typedef struct rgpot_plugin_output_t {
  double energy;
  double variance;
  double *forces;               /* [n_atoms * 3], caller-allocated, plugin fills */
} rgpot_plugin_output_t;

/* ---- Opaque plugin instance ---- */

typedef struct rgpot_plugin_instance rgpot_plugin_instance;

/* ---- Function pointer types ---- */

typedef rgpot_plugin_status_t (*rgpot_plugin_create_fn)(
    const char *config_json, rgpot_plugin_instance **out);

typedef rgpot_plugin_status_t (*rgpot_plugin_calculate_fn)(
    rgpot_plugin_instance *instance, const rgpot_plugin_input_t *input,
    rgpot_plugin_output_t *output);

typedef void (*rgpot_plugin_destroy_fn)(rgpot_plugin_instance *instance);

typedef uint32_t (*rgpot_plugin_capabilities_fn)(
    const rgpot_plugin_instance *instance);

typedef const char *(*rgpot_plugin_last_error_fn)(
    const rgpot_plugin_instance *instance);

/* ---- Plugin descriptor ---- */

/*
 * Forward compatibility: descriptor_size allows adding optional fields at the
 * end of the struct in future versions without bumping api_version. The host
 * checks that descriptor_size includes the entire field before accessing it.
 *
 * Breaking changes (removing fields, changing signatures) require bumping
 * RGPOT_PLUGIN_API_VERSION.
 */

typedef struct rgpot_plugin_descriptor_t {
  /* Must be sizeof(rgpot_plugin_descriptor_t) as seen by the plugin */
  size_t descriptor_size;

  /* Must match RGPOT_PLUGIN_API_VERSION */
  uint32_t api_version;

  /* Identity */
  const char *name;    /* e.g. "xtb-gfn2", "mace-mp-0" */
  const char *version; /* e.g. "6.7.1" */

  /*
   * Native units of this plugin. The host converts automatically using
   * rgpot's unit expression parser (supports "eV", "hartree", "bohr",
   * "angstrom", "kJ/mol", compound expressions like "eV/A^2", etc.).
   *
   * NULL or "" means rgpot native units (angstrom / eV).
   */
  const char *length_unit;
  const char *energy_unit;

  /* Required lifecycle functions */
  rgpot_plugin_create_fn create;
  rgpot_plugin_destroy_fn destroy;
  rgpot_plugin_calculate_fn calculate;

  /* Optional (may be NULL) */
  rgpot_plugin_capabilities_fn capabilities;
  rgpot_plugin_last_error_fn last_error;
} rgpot_plugin_descriptor_t;

/* ---- Entry point ---- */

/*
 * Every plugin shared library must export this symbol. It returns a pointer
 * to a static descriptor. The pointer must remain valid for the lifetime of
 * the loaded library (i.e., until dlclose).
 */
typedef const rgpot_plugin_descriptor_t *(*rgpot_plugin_init_fn)(void);

/* ---- Convenience macros for plugin authors ---- */

#ifdef _WIN32
#define RGPOT_PLUGIN_EXPORT __declspec(dllexport)
#else
#define RGPOT_PLUGIN_EXPORT __attribute__((visibility("default")))
#endif

/*
 * Usage in a plugin:
 *
 *   RGPOT_PLUGIN_EXPORT const rgpot_plugin_descriptor_t *rgpot_plugin_init(void) {
 *       static const rgpot_plugin_descriptor_t desc = { ... };
 *       return &desc;
 *   }
 */

#ifdef __cplusplus
}
#endif

#endif /* RGPOT_PLUGIN_H */
