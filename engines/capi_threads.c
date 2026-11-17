/* MIT License
 * Copyright 2023--present rgpot developers
 *
 * Const accessors of the neighbour C API, called together from several
 * threads, and two builds of different geometries at the same time.
 */
#include "rgpot/nlist/rgpot_nlist.h"

#include <pthread.h>
#include <stdio.h>
#include <string.h>

static void *read_accessors(void *arg) {
  int *bad = arg;
  for (int i = 0; i < 200; ++i) {
    if (rgpot_nlist_abi_version() != RGPOT_NLIST_ABI) {
      *bad = 1;
    }
    const char *text = rgpot_nlist_strerror(RGPOT_NLIST_OK);
    if (text == NULL || text[0] == '\0') {
      *bad = 1;
    }
    rgpot_nlist_request request;
    rgpot_nlist_request_init(&request);
    if (request.abi != (uint32_t)RGPOT_NLIST_ABI || request.cutoff != 0.0) {
      *bad = 1;
    }
  }
  return NULL;
}

typedef struct {
  const double *xyz;
  size_t n;
  const double *box;
  double cutoff;
  size_t expect;
  int bad;
} Build;

static void *build_one(void *arg) {
  Build *job = arg;
  rgpot_nlist_request request;
  rgpot_nlist_request_init(&request);
  request.cutoff = job->cutoff;
  size_t count = 0;
  size_t needed = 0;
  const int code = rgpot_nlist_build(job->xyz, job->n, job->box, &request, NULL, &count, &needed);
  if (code != RGPOT_NLIST_OK || needed != job->expect) {
    job->bad = 1;
  }
  return NULL;
}

int main(void) {
  int bad[4] = {0, 0, 0, 0};
  pthread_t threads[4];
  for (int i = 0; i < 4; ++i) {
    if (pthread_create(&threads[i], NULL, read_accessors, &bad[i]) != 0) {
      fprintf(stderr, "pthread_create failed\n");
      return 1;
    }
  }
  for (int i = 0; i < 4; ++i) {
    pthread_join(threads[i], NULL);
    if (bad[i]) {
      fprintf(stderr, "const accessor changed under threads\n");
      return 1;
    }
  }
  const double box[9] = {10, 0, 0, 0, 10, 0, 0, 0, 10};
  const double pair_a[6] = {0, 0, 0, 1, 0, 0};
  const double pair_b[9] = {0, 0, 0, 1, 0, 0, 0, 1, 0};
  Build jobs[2] = {
      {pair_a, 2, box, 2.0, 1, 0},
      {pair_b, 3, box, 2.0, 3, 0},
  };
  pthread_t builders[2];
  for (int i = 0; i < 2; ++i) {
    if (pthread_create(&builders[i], NULL, build_one, &jobs[i]) != 0) {
      return 1;
    }
  }
  for (int i = 0; i < 2; ++i) {
    pthread_join(builders[i], NULL);
    if (jobs[i].bad) {
      fprintf(stderr, "concurrent build disagreed with the serial count\n");
      return 1;
    }
  }
  return 0;
}
