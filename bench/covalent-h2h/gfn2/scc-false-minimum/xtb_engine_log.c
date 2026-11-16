/* Copyright (c) anneal developers */
/*
 * Stand-in libxtb_engine.so that forwards to the engine named by
 * XTB_ENGINE_LOG_REAL and writes every force call (index, return code,
 * whether the wavefunction was reused, energy in eV, positions in Å) and
 * every reset to XTB_ENGINE_LOG_OUT.
 *
 *   gcc -O2 -shared -fPIC -o libxtb_engine_log.so xtb_engine_log.c -ldl
 *   RGPOT_XTB_ENGINE=$PWD/libxtb_engine_log.so ...
 */
#define _GNU_SOURCE
#include <dlfcn.h>
#include <stddef.h>
#include <stdio.h>
#include <stdlib.h>

typedef struct RgpotXtbPot RgpotXtbPot;

static void *real;
static FILE *out;
static long calls;

static void *sym(const char *name) {
  if (!real) {
    real = dlopen(getenv("XTB_ENGINE_LOG_REAL"), RTLD_NOW | RTLD_LOCAL);
    if (!real) {
      fprintf(stderr, "xtb_engine_log: %s\n", dlerror());
      abort();
    }
    out = fopen(getenv("XTB_ENGINE_LOG_OUT"), "w");
  }
  return dlsym(real, name);
}

int rgpot_xtb_abi_version(void) {
  return ((int (*)(void))sym("rgpot_xtb_abi_version"))();
}
int rgpot_xtb_available(void) {
  return ((int (*)(void))sym("rgpot_xtb_available"))();
}
RgpotXtbPot *rgpot_xtb_create(const void *cfg, char *err, size_t len) {
  return ((RgpotXtbPot * (*)(const void *, char *, size_t))
              sym("rgpot_xtb_create"))(cfg, err, len);
}
void rgpot_xtb_destroy(RgpotXtbPot *p) {
  ((void (*)(RgpotXtbPot *))sym("rgpot_xtb_destroy"))(p);
}
void rgpot_xtb_reset(RgpotXtbPot *p) {
  fprintf(out, "reset\n");
  ((void (*)(RgpotXtbPot *))sym("rgpot_xtb_reset"))(p);
}
void rgpot_xtb_set_warm_move(RgpotXtbPot *p, double a) {
  ((void (*)(RgpotXtbPot *, double))sym("rgpot_xtb_set_warm_move"))(p, a);
}
double rgpot_xtb_warm_move(const RgpotXtbPot *p) {
  return ((double (*)(const RgpotXtbPot *))sym("rgpot_xtb_warm_move"))(p);
}
int rgpot_xtb_reused_wavefunction(const RgpotXtbPot *p) {
  return ((int (*)(const RgpotXtbPot *))sym("rgpot_xtb_reused_wavefunction"))(
      p);
}
int rgpot_xtb_force(RgpotXtbPot *p, long n, const double *pos, const int *z,
                    double *f, double *e, double *var, const double *box) {
  int rc = ((int (*)(RgpotXtbPot *, long, const double *, const int *,
                     double *, double *, double *, const double *))
                sym("rgpot_xtb_force"))(p, n, pos, z, f, e, var, box);
  fprintf(out, "call %ld rc %d reused %d energy %.10f\n", ++calls, rc,
          rc == 0 ? rgpot_xtb_reused_wavefunction(p) : -1, rc == 0 ? *e : 0.0);
  for (long i = 0; i < n; ++i)
    fprintf(out, "%d %.17e %.17e %.17e\n", z[i], pos[3 * i], pos[3 * i + 1],
            pos[3 * i + 2]);
  fflush(out);
  return rc;
}
