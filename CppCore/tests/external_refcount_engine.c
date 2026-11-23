/* MIT License
 * Copyright 2023--present rgpot developers
 *
 * Tiny engine plugin for the external-potential loader refcount test.
 */

#include "rgpot/engine_c_abi.h"

#include <stdio.h>
#include <stdlib.h>

static void record_lifecycle(const char *event) {
  const char *path = getenv("RGPOT_EXTERNAL_LIFECYCLE_LOG");
  if (path == NULL || path[0] == '\0') {
    return;
  }
  FILE *log = fopen(path, "a");
  if (log != NULL) {
    fprintf(log, "%s\n", event);
    fclose(log);
  }
}

__attribute__((constructor)) static void engine_load(void) {
  record_lifecycle("engine load");
}

__attribute__((destructor)) static void engine_unload(void) {
  record_lifecycle("engine unload");
}

RgpotEnginePot *rgpot_engine_create(const void *config, size_t config_len,
                                    char *errbuf, size_t errlen) {
  (void)config;
  (void)config_len;
  (void)errbuf;
  (void)errlen;
  return (RgpotEnginePot *)calloc(1, sizeof(int));
}

void rgpot_engine_destroy(RgpotEnginePot *pot) { free(pot); }

int rgpot_engine_force(RgpotEnginePot *pot, long nAtoms, const double *positions,
                       const int *atomicNrs, double *forces, double *energy,
                       double *variance, const double *box,
                       rgpot_engine_coord_transform transform,
                       void *transform_user) {
  (void)pot;
  (void)nAtoms;
  (void)positions;
  (void)atomicNrs;
  (void)forces;
  (void)box;
  (void)transform;
  (void)transform_user;
  if (energy != NULL) {
    *energy = 0.0;
  }
  if (variance != NULL) {
    *variance = 0.0;
  }
  return 0;
}
