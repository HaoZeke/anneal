#define _GNU_SOURCE

#include <dlfcn.h>
#include <stdio.h>
#include <stdlib.h>

static long calls = 0;
static int (*real_force)(int, const double *, const int *, const double *,
                         double *, double *);

static void resolve(void) {
    const char *path;
    void *library;

    if (real_force != NULL) {
        return;
    }
    path = getenv("CUH2_REAL");
    if (path == NULL) {
        fprintf(stderr, "cuh2 counter has no real library\n");
        abort();
    }
    library = dlopen(path, RTLD_NOW | RTLD_LOCAL);
    if (library == NULL) {
        fprintf(stderr, "cuh2 counter could not open the real library\n");
        abort();
    }
    real_force = dlsym(library, "rgpot_cuh2_force");
    if (real_force == NULL) {
        fprintf(stderr, "cuh2 counter could not resolve the force symbol\n");
        abort();
    }
}

int rgpot_cuh2_force(int n, const double *positions, const int *numbers,
                     const double *box, double *force, double *energy) {
    resolve();
    calls += 1;
    return real_force(n, positions, numbers, box, force, energy);
}

__attribute__((destructor)) static void report_calls(void) {
    fprintf(stderr, "cuh2_engine_calls %ld\n", calls);
}
