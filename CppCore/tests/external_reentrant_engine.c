/* MIT License
 * Copyright 2023--present rgpot developers
 *
 * Engine whose initializer re-enters the external-potential constructor
 * through a callback exported by the test executable.
 */

#ifndef _GNU_SOURCE
#define _GNU_SOURCE 1
#endif

#include "rgpot/engine_c_abi.h"

#include <dlfcn.h>
#include <stdio.h>
#include <stdlib.h>

static int initialized;

__attribute__((constructor)) static void initialize_engine(void) {
  void (*reenter)(void) = NULL;
  *(void **)(&reenter) = dlsym(RTLD_DEFAULT, "rgpot_test_reenter");
  if (reenter == NULL) {
    initialized = -1;
    return;
  }
  reenter();
  initialized = 1;
}

RgpotEnginePot *rgpot_engine_create(const void *config, size_t config_len,
                                    char *errbuf, size_t errlen) {
  (void)config;
  (void)config_len;
  if (initialized != 1) {
    if (errbuf != NULL && errlen > 0) {
      snprintf(errbuf, errlen,
               "engine initializer could not resolve its reentry callback");
    }
    return NULL;
  }
  return (RgpotEnginePot *)calloc(1, sizeof(int));
}

void rgpot_engine_destroy(RgpotEnginePot *pot) { free(pot); }

int rgpot_engine_force(RgpotEnginePot *pot, long nAtoms,
                       const double *positions, const int *atomicNrs,
                       double *forces, double *energy, double *variance,
                       const double *box,
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
