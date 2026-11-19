/* A harmonic engine behind the generic engine C ABI, for tests/engine.rs.
 * E = k/2 sum x^2 over all coordinates, F = -k x. The create message is one
 * native-endian double, k. An atomic number of zero fails the evaluation with
 * a recoverable code, a negative one with a fatal code. Compile definitions
 * drop symbols or change the ABI major. */
#include "rgpot/engine_c_abi.h"

#include <math.h>
#include <stddef.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#ifndef FIXTURE_ABI
#define FIXTURE_ABI 1
#endif

#ifndef FIXTURE_STAMP_MAJOR
#define FIXTURE_STAMP_MAJOR FIXTURE_ABI
#endif

#ifndef FIXTURE_ABI_MINOR
#define FIXTURE_ABI_MINOR RGPOT_ENGINE_ABI_MINOR
#endif

#ifndef FIXTURE_ABI_LAYOUT
#define FIXTURE_ABI_LAYOUT RGPOT_ENGINE_ABI_LAYOUT_REVISION
#endif

struct RgpotEnginePot {
  double k;
  double last_energy;
  int evaluated;
  unsigned long long model_state;
  RgpotEngineCallbacks callbacks;
  char error[128];
};

static int destroyed = 0;
static int native_calls = 0;
static int batch_calls = 0;
#ifdef FATAL_ONCE
static int fatal_once_fired = 0;
#endif

int fixture_destroy_count(void) { return destroyed; }
int fixture_native_call_count(void) { return native_calls; }
int fixture_batch_call_count(void) { return batch_calls; }
int fixture_unsupported_code(void) { return RGPOT_ENGINE_ERR_UNSUPPORTED; }

int rgpot_engine_abi_version(void) { return FIXTURE_ABI; }

#ifndef NO_STAMP
RgpotEngineAbiStamp rgpot_engine_abi_stamp(void) {
  RgpotEngineAbiStamp stamp;
  stamp.abi_major = (unsigned short)FIXTURE_STAMP_MAJOR;
  stamp.abi_minor = (unsigned short)FIXTURE_ABI_MINOR;
  stamp.layout_revision = (unsigned short)FIXTURE_ABI_LAYOUT;
  return stamp;
}
#endif

RgpotEnginePot *rgpot_engine_create(const void *config, size_t config_len,
                                    char *errbuf, size_t errlen) {
  RgpotEnginePot *pot;
  if (config == NULL || config_len != sizeof(double)) {
    snprintf(errbuf, errlen, "harmonic engine: config is %zu bytes, need %zu",
             config_len, sizeof(double));
    return NULL;
  }
  if (((uintptr_t)config % 8) != 0) {
    snprintf(errbuf, errlen, "harmonic engine: config pointer %p is not 8-byte aligned",
             config);
    return NULL;
  }
  pot = calloc(1, sizeof *pot);
  memcpy(&pot->k, config, sizeof(double));
  return pot;
}

void rgpot_engine_destroy(RgpotEnginePot *pot) {
  destroyed += 1;
  free(pot);
}

#ifndef NO_FORCE
int rgpot_engine_force(RgpotEnginePot *pot, long n, const double *positions,
                       const int *atomic_nrs, double *forces, double *energy,
                       double *variance, const double *box,
                       rgpot_engine_coord_transform transform, void *user) {
  long i;
  double e = 0.0;
  double *scratch = NULL;
  double scratch_box[9];
  native_calls += 1;
  (void)variance;
  if (transform != NULL) {
    int rc;
    scratch = malloc((size_t)(3 * n) * sizeof(double));
    memcpy(scratch, positions, (size_t)(3 * n) * sizeof(double));
    memcpy(scratch_box, box, sizeof scratch_box);
    rc = transform(user, n, scratch, scratch_box);
    if (rc != 0) {
      free(scratch);
      return rc;
    }
    positions = scratch;
    box = scratch_box;
  }
#ifdef FATAL_ONCE
  if (!fatal_once_fired) {
    fatal_once_fired = 1;
    snprintf(pot->error, sizeof pot->error, "fatal once");
    free(scratch);
    return -7;
  }
#endif
  for (i = 0; i < n; i++) {
    if (atomic_nrs[i] == 0) {
      snprintf(pot->error, sizeof pot->error, "atom %ld has atomic number 0", i);
      free(scratch);
      return 1;
    }
    if (atomic_nrs[i] < 0) {
      snprintf(pot->error, sizeof pot->error, "atom %ld has a negative atomic number", i);
      free(scratch);
      return -1;
    }
  }
  if (pot->callbacks.oracle != NULL) {
    int rc;
    if (pot->callbacks.on_acquire != NULL)
      pot->callbacks.on_acquire(pot->callbacks.ctx, RGPOT_ACQUIRE_VARIANCE);
    rc = pot->callbacks.oracle(pot->callbacks.ctx, n, positions, atomic_nrs,
                               forces, energy, box);
    if (rc == 0) {
      pot->model_state += 1;
      if (pot->callbacks.on_model_update != NULL)
        pot->callbacks.on_model_update(pot->callbacks.ctx, pot->model_state);
    }
    free(scratch);
    return rc;
  }
#ifdef REJECT_COINCIDENT
  {
    long j;
    for (i = 0; i < n; i++) {
      for (j = i + 1; j < n; j++) {
        double dx = positions[3 * i] - positions[3 * j];
        double dy = positions[3 * i + 1] - positions[3 * j + 1];
        double dz = positions[3 * i + 2] - positions[3 * j + 2];
        if (!(dx * dx + dy * dy + dz * dz > 0.0)) {
          snprintf(pot->error, sizeof pot->error, "coincident atoms");
          free(scratch);
          return RGPOT_ENGINE_ERR_EVAL;
        }
      }
    }
  }
#endif
  for (i = 0; i < 3 * n; i++) {
    e += 0.5 * pot->k * positions[i] * positions[i];
    forces[i] = -pot->k * positions[i];
  }
  *energy = e;
  pot->last_energy = e;
  pot->evaluated = 1;
  pot->error[0] = '\0';
  free(scratch);
  return 0;
}
#endif

#ifndef NO_BATCH
int rgpot_engine_force_batch(RgpotEnginePot *pot, long n_systems, long n,
                             const double *positions, const int *atomic_nrs,
                             double *forces, double *energies,
                             double *variances, const double *boxes,
                             rgpot_engine_coord_transform transform,
                             void *user) {
  long s;
  native_calls += 1;
  batch_calls += 1;
  (void)variances;
  for (s = 0; s < n_systems; s++) {
    int rc = rgpot_engine_force(pot, n, positions + 3 * n * s, atomic_nrs,
                                forces + 3 * n * s, energies + s, NULL,
                                boxes + 9 * s, transform, user);
    if (rc != 0) return rc;
  }
  return 0;
}
#endif

unsigned long long rgpot_engine_model_state(const RgpotEnginePot *pot) {
  return pot->model_state;
}

int rgpot_engine_set_callbacks(RgpotEnginePot *pot,
                               const RgpotEngineCallbacks *callbacks) {
  if (pot == NULL) return RGPOT_ENGINE_ERR_INVALID;
  memset(&pot->callbacks, 0, sizeof pot->callbacks);
  if (callbacks != NULL) pot->callbacks = *callbacks;
  return RGPOT_ENGINE_OK;
}

#ifndef NO_LAST_ERROR
size_t rgpot_engine_last_error(const RgpotEnginePot *pot, char *buf, size_t len) {
  size_t full = strlen(pot->error);
  native_calls += 1;
  if (len > 0) {
    snprintf(buf, len, "%s", pot->error);
  }
  return full;
}
#endif

#ifndef NO_CAPS
int rgpot_engine_caps(const RgpotEnginePot *pot, RgpotEngineCaps *out) {
  RgpotEngineCaps filled;
  unsigned room;
  native_calls += 1;
  (void)pot;
  if (!out || out->size < sizeof(unsigned)) return 1;
  memset(&filled, 0, sizeof filled);
  filled.size = sizeof filled;
  filled.periodic = 1;
#if !defined(NO_HVP) && !defined(NO_HVP_CAP)
  filled.analytic_hvp = 1;
#endif
  room = out->size < sizeof filled ? out->size : sizeof filled;
  memcpy(out, &filled, room);
  out->size = room;
  return 0;
}
#endif

#ifndef NO_HVP
int rgpot_engine_hvp(RgpotEnginePot *pot, long n, const double *positions,
                     const int *atomic_nrs, const double *direction,
                     double *hv_out, const double *box) {
  native_calls += 1;
  (void)positions;
  (void)box;
#if defined(NO_HVP_CAP) || defined(HVP_UNSUPPORTED)
  (void)n;
  (void)atomic_nrs;
  (void)direction;
  (void)hv_out;
  snprintf(pot->error, sizeof pot->error, "harmonic engine built without analytic HVP");
  return RGPOT_ENGINE_ERR_UNSUPPORTED;
#else
  long i;
  uintptr_t db;
  uintptr_t hb;
  uintptr_t bytes;
#ifdef FATAL_ONCE
  if (!fatal_once_fired) {
    fatal_once_fired = 1;
    snprintf(pot->error, sizeof pot->error, "fatal once");
    return -7;
  }
#endif
  if (!pot || !positions || !atomic_nrs || !direction || !hv_out || !box || n <= 0) {
    if (pot) {
      snprintf(pot->error, sizeof pot->error, "invalid HVP argument");
    }
    return RGPOT_ENGINE_ERR_INVALID;
  }
  db = (uintptr_t)direction;
  hb = (uintptr_t)hv_out;
  bytes = (uintptr_t)n * 3u * sizeof(double);
  if (db < hb + bytes && hb < db + bytes) {
    snprintf(pot->error, sizeof pot->error, "direction and hv_out overlap");
    return RGPOT_ENGINE_ERR_INVALID;
  }
  for (i = 0; i < n; i++) {
    if (atomic_nrs[i] == 0) {
      snprintf(pot->error, sizeof pot->error, "atom %ld has atomic number 0", i);
      return 1;
    }
    if (atomic_nrs[i] < 0) {
      snprintf(pot->error, sizeof pot->error, "atom %ld has a negative atomic number", i);
      return -1;
    }
  }
#ifdef REJECT_COINCIDENT
  {
    long j;
    for (i = 0; i < n; i++) {
      for (j = i + 1; j < n; j++) {
        double dx = positions[3 * i] - positions[3 * j];
        double dy = positions[3 * i + 1] - positions[3 * j + 1];
        double dz = positions[3 * i + 2] - positions[3 * j + 2];
        if (!(dx * dx + dy * dy + dz * dz > 0.0)) {
          snprintf(pot->error, sizeof pot->error, "coincident atoms");
          return RGPOT_ENGINE_ERR_EVAL;
        }
      }
    }
  }
#endif
  for (i = 0; i < 3 * n; i++) {
#ifdef HVP_NONFINITE
    hv_out[i] = i == 1 ? NAN : pot->k * direction[i];
#else
    hv_out[i] = pot->k * direction[i];
#endif
  }
  pot->error[0] = '\0';
  return 0;
#endif
}
#endif

#if !defined(NO_COMPONENTS) && (FIXTURE_ABI_MINOR >= 5)
long rgpot_engine_last_components(const RgpotEnginePot *pot,
                                  RgpotEngineComponent *out, double *forces,
                                  long capacity) {
  native_calls += 1;
  (void)forces;
  if (!pot) return -1;
  if (!out || capacity <= 0) return 1;
  memset(&out[0], 0, sizeof out[0]);
  out[0].weight = 1.0;
  out[0].energy = pot->last_energy;
  out[0].enabled = 1;
  out[0].evaluated = pot->evaluated;
  out[0].has_stress = 0;
  return 1;
}

size_t rgpot_engine_component_name(const RgpotEnginePot *pot, long index,
                                   char *buf, size_t len) {
  static const char name[] = "harmonic";
  size_t full = sizeof(name) - 1;
  native_calls += 1;
  if (!pot || index != 0) {
    if (buf && len > 0) buf[0] = '\0';
    return 0;
  }
  if (buf && len > 0) {
    snprintf(buf, len, "%s", name);
  }
  return full;
}
#endif
