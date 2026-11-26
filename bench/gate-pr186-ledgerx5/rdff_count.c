/* Copyright (c) anneal developers
 *
 * LD_PRELOAD counter for force-field calls. It wraps the four
 * ForceField energy and gradient entry points, counts the outermost
 * call on each thread, and sums the time spent inside them.
 *
 * The library is already mapped when a call arrives. A wheel may
 * suffix the file name, so the mapped path is used when the plain
 * soname is not loaded.
 */
#define _GNU_SOURCE
#include <dlfcn.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
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

static void *mapped_forcefield(void) {
    FILE *file = fopen("/proc/self/maps", "r");
    if (file == NULL) {
        return NULL;
    }
    char line[4096];
    void *lib = NULL;
    while (fgets(line, sizeof line, file) != NULL) {
        char *path = strstr(line, "/");
        if (path == NULL || strstr(path, "libRDKitForceField") == NULL) {
            continue;
        }
        char *end = strpbrk(path, "\n");
        if (end != NULL) {
            *end = '\0';
        }
        lib = dlopen(path, RTLD_LAZY | RTLD_NOLOAD);
        if (lib != NULL) {
            break;
        }
    }
    fclose(file);
    return lib;
}

static void *next_symbol(const char *name) {
    void *lib = dlopen("libRDKitForceField.so.1", RTLD_LAZY | RTLD_NOLOAD);
    if (lib == NULL) {
        lib = mapped_forcefield();
    }
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
        real_energy_contribs = (energy_contribs_fn)next_symbol(
            "_ZNK11ForceFields10ForceField10calcEnergyEPSt6vectorIdSaIdEE");
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
