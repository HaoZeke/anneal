/* Copyright (c) anneal developers
 *
 * LD_PRELOAD counter for RDKit force-field calls. It wraps the four
 * ForceFields::ForceField energy and gradient entry points, counts the
 * outermost call on each thread, and sums the time spent inside them.
 * Calls between those entry points go through the GOT of
 * libRDKitForceField.so, so a nested call (calcGrad(grad) calling
 * calcGrad(pos, grad)) is counted once.
 *
 * Counts are kept per phase. Phase 0 is the search potential. A driver sets
 * phase 1 around ETKDG embedding, whose distance-geometry minimisation uses
 * the same ForceField class.
 *
 *   rdff_reset()             zero every counter
 *   rdff_set_phase(int)      0 or 1
 *   rdff_read(uint64 out[6]) energy calls, gradient calls, nanoseconds,
 *                            for phase 0 then phase 1
 */
#define _GNU_SOURCE
#include <dlfcn.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <time.h>

#define PHASES 2

static uint64_t energy_calls[PHASES];
static uint64_t grad_calls[PHASES];
static uint64_t inside_ns[PHASES];
static int phase;
static __thread int depth;

typedef double (*energy_pos_fn)(void *, double *);
typedef void (*grad_pos_fn)(void *, double *, double *);
typedef double (*energy_contribs_fn)(const void *, void *);
typedef void (*grad_here_fn)(const void *, double *);

static energy_pos_fn real_energy_pos;
static grad_pos_fn real_grad_pos;
static energy_contribs_fn real_energy_contribs;
static grad_here_fn real_grad_here;

/* Python loads the RDKit extension modules RTLD_LOCAL, so RTLD_NEXT does
 * not see libRDKitForceField. The library is already mapped by the time
 * any of these calls is made. */
static void *next_symbol(const char *name) {
  void *lib = dlopen("libRDKitForceField.so.1", RTLD_LAZY | RTLD_NOLOAD);
  void *sym = lib != NULL ? dlsym(lib, name) : dlsym(RTLD_NEXT, name);
  if (sym == NULL) {
    fprintf(stderr, "rdff_count: %s not found\n", name);
    abort();
  }
  return sym;
}

static uint64_t now_ns(void) {
  struct timespec ts;
  clock_gettime(CLOCK_MONOTONIC, &ts);
  return (uint64_t)ts.tv_sec * 1000000000u + (uint64_t)ts.tv_nsec;
}

static void enter(uint64_t *t0) {
  if (depth++ == 0) {
    *t0 = now_ns();
  }
}

static void leave(uint64_t t0, uint64_t *counter) {
  if (--depth == 0) {
    uint64_t dt = now_ns() - t0;
    __atomic_fetch_add(&counter[phase], 1, __ATOMIC_RELAXED);
    __atomic_fetch_add(&inside_ns[phase], dt, __ATOMIC_RELAXED);
  }
}

double _ZN11ForceFields10ForceField10calcEnergyEPd(void *self, double *pos) {
  if (real_energy_pos == NULL) {
    real_energy_pos = (energy_pos_fn)next_symbol("_ZN11ForceFields10ForceField10calcEnergyEPd");
  }
  uint64_t t0 = 0;
  enter(&t0);
  double e = real_energy_pos(self, pos);
  leave(t0, energy_calls);
  return e;
}

void _ZN11ForceFields10ForceField8calcGradEPdS1_(void *self, double *pos, double *grad) {
  if (real_grad_pos == NULL) {
    real_grad_pos = (grad_pos_fn)next_symbol("_ZN11ForceFields10ForceField8calcGradEPdS1_");
  }
  uint64_t t0 = 0;
  enter(&t0);
  real_grad_pos(self, pos, grad);
  leave(t0, grad_calls);
}

double _ZNK11ForceFields10ForceField10calcEnergyEPSt6vectorIdSaIdEE(const void *self, void *contribs) {
  if (real_energy_contribs == NULL) {
    real_energy_contribs =
        (energy_contribs_fn)next_symbol("_ZNK11ForceFields10ForceField10calcEnergyEPSt6vectorIdSaIdEE");
  }
  uint64_t t0 = 0;
  enter(&t0);
  double e = real_energy_contribs(self, contribs);
  leave(t0, energy_calls);
  return e;
}

void _ZNK11ForceFields10ForceField8calcGradEPd(const void *self, double *grad) {
  if (real_grad_here == NULL) {
    real_grad_here = (grad_here_fn)next_symbol("_ZNK11ForceFields10ForceField8calcGradEPd");
  }
  uint64_t t0 = 0;
  enter(&t0);
  real_grad_here(self, grad);
  leave(t0, grad_calls);
}

void rdff_reset(void) {
  for (int i = 0; i < PHASES; ++i) {
    energy_calls[i] = 0;
    grad_calls[i] = 0;
    inside_ns[i] = 0;
  }
  phase = 0;
}

void rdff_set_phase(int value) { phase = (value > 0 && value < PHASES) ? value : 0; }

void rdff_read(uint64_t *out) {
  for (int i = 0; i < PHASES; ++i) {
    out[3 * i] = energy_calls[i];
    out[3 * i + 1] = grad_calls[i];
    out[3 * i + 2] = inside_ns[i];
  }
}
