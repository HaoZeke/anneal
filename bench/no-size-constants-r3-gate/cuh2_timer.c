/* Copyright (c) anneal developers
 * Times rgpot_cuh2_force and writes the total nanoseconds at exit.
 * Build: gcc -O2 -shared -fPIC -o librgpot_cuh2_timed.so cuh2_timer.c -ldl
 * The real library path and the output path come from the environment
 * names the gate runner already passes. */
#define _GNU_SOURCE
#include <dlfcn.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <time.h>

typedef int (*force_fn)(int32_t, const double *, const int32_t *, const double *, double *, double *);

static force_fn real_force;
static uint64_t total_ns;

static void init_real(void) {
    const char *path;
    void *lib;
    if (real_force) {
        return;
    }
    path = getenv("GATE_REAL_CUH2_LIB");
    if (!path || !path[0]) {
        fprintf(stderr, "cuh2 timer: real library is unset\n");
        abort();
    }
    lib = dlopen(path, RTLD_NOW | RTLD_LOCAL);
    if (!lib) {
        fprintf(stderr, "cuh2 timer: dlopen failed\n");
        abort();
    }
    real_force = (force_fn)dlsym(lib, "rgpot_cuh2_force");
    if (!real_force) {
        fprintf(stderr, "cuh2 timer: rgpot_cuh2_force missing\n");
        abort();
    }
}

__attribute__((visibility("default"))) int
rgpot_cuh2_force(int32_t n, const double *pos, const int32_t *z, const double *box, double *forces,
                 double *energy) {
    struct timespec a, b;
    int rc;
    uint64_t ns;
    init_real();
    clock_gettime(CLOCK_MONOTONIC, &a);
    rc = real_force(n, pos, z, box, forces, energy);
    clock_gettime(CLOCK_MONOTONIC, &b);
    ns = (uint64_t)(b.tv_sec - a.tv_sec) * 1000000000ull + (uint64_t)(b.tv_nsec - a.tv_nsec);
    total_ns += ns;
    return rc;
}

__attribute__((destructor)) static void dump_timer(void) {
    const char *path;
    FILE *out;
    path = getenv("GATE_POT_TIMER_OUT");
    if (!path || !path[0]) {
        return;
    }
    out = fopen(path, "w");
    if (!out) {
        return;
    }
    fprintf(out, "%llu\n", (unsigned long long)total_ns);
    fclose(out);
}
