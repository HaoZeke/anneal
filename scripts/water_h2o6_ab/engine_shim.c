/*
 * Timing and sign shim around the rgpot GFN2 engine.
 *
 * Both example drivers and the CREST call server load this library in place
 * of the engine. It forwards every call to the real engine and adds the time
 * spent inside rgpot_xtb_force to counters kept in a shared file mapping, so
 * a forked worker and its parent update the same totals.
 *
 * The real engine returns the force, minus dE/dx, in eV per angstrom. One of
 * the two code versions publishes that buffer unchanged as a force and the
 * other first negates it as if it were dE/dx. The shim hands each of them the
 * buffer it is written against.
 *
 * Configuration comes from three variables the harness sets for the child:
 * XTB_SHIM_REAL (path of the real engine), XTB_SHIM_STATS (counter file) and
 * XTB_SHIM_BUFFER ("force" or "gradient").
 */
#define _GNU_SOURCE
#include <dlfcn.h>
#include <fcntl.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <sys/mman.h>
#include <time.h>
#include <unistd.h>

typedef void *(*create_fn)(const void *, uint8_t *, size_t);
typedef void (*destroy_fn)(void *);
typedef int (*force_fn)(void *, long, const double *, const int *, double *,
                        double *, double *, const double *);
typedef void (*reset_fn)(void *);
typedef void (*set_warm_fn)(void *, double);
typedef double (*warm_fn)(const void *);
typedef int (*reused_fn)(const void *);

struct counters {
  uint64_t calls;
  uint64_t failed;
  uint64_t inside_ns;
};

static create_fn real_create;
static destroy_fn real_destroy;
static force_fn real_force;
static reset_fn real_reset;
static set_warm_fn real_set_warm;
static warm_fn real_warm;
static reused_fn real_reused;
static struct counters *shared;
static struct counters local_totals;
static int negate_buffer;

static void die(const char *what) {
  fprintf(stderr, "engine shim: %s\n", what);
  _exit(97);
}

__attribute__((constructor)) static void shim_init(void) {
  const char *real = getenv("XTB_SHIM_REAL");
  const char *stats = getenv("XTB_SHIM_STATS");
  const char *buffer = getenv("XTB_SHIM_BUFFER");
  if (!real || !*real) die("XTB_SHIM_REAL is not set");
  if (!buffer || (strcmp(buffer, "force") && strcmp(buffer, "gradient")))
    die("XTB_SHIM_BUFFER must be force or gradient");
  negate_buffer = strcmp(buffer, "gradient") == 0;
  void *lib = dlopen(real, RTLD_NOW | RTLD_LOCAL);
  if (!lib) die(dlerror());
  real_create = (create_fn)dlsym(lib, "rgpot_xtb_create");
  real_destroy = (destroy_fn)dlsym(lib, "rgpot_xtb_destroy");
  real_force = (force_fn)dlsym(lib, "rgpot_xtb_force");
  real_reset = (reset_fn)dlsym(lib, "rgpot_xtb_reset");
  real_set_warm = (set_warm_fn)dlsym(lib, "rgpot_xtb_set_warm_move");
  real_warm = (warm_fn)dlsym(lib, "rgpot_xtb_warm_move");
  real_reused = (reused_fn)dlsym(lib, "rgpot_xtb_reused_wavefunction");
  if (!real_create || !real_destroy || !real_force || !real_reset)
    die("the engine lacks a required symbol");
  shared = &local_totals;
  if (stats && *stats) {
    int fd = open(stats, O_RDWR | O_CREAT, 0600);
    if (fd < 0) die("cannot open the counter file");
    if (ftruncate(fd, 4096) != 0) die("cannot size the counter file");
    void *map = mmap(NULL, 4096, PROT_READ | PROT_WRITE, MAP_SHARED, fd, 0);
    if (map == MAP_FAILED) die("cannot map the counter file");
    close(fd);
    shared = (struct counters *)map;
  }
}

void *rgpot_xtb_create(const void *config, uint8_t *error, size_t length) {
  return real_create(config, error, length);
}

void rgpot_xtb_destroy(void *pot) { real_destroy(pot); }

void rgpot_xtb_reset(void *pot) { real_reset(pot); }

void rgpot_xtb_set_warm_move(void *pot, double angstrom) {
  if (real_set_warm) real_set_warm(pot, angstrom);
}

double rgpot_xtb_warm_move(const void *pot) {
  return real_warm ? real_warm(pot) : 0.0;
}

int rgpot_xtb_reused_wavefunction(const void *pot) {
  return real_reused ? real_reused(pot) : 0;
}

int rgpot_xtb_force(void *pot, long n_atoms, const double *positions,
                    const int *numbers, double *buffer, double *energy,
                    double *variance, const double *box) {
  struct timespec begin, end;
  clock_gettime(CLOCK_MONOTONIC, &begin);
  int status =
      real_force(pot, n_atoms, positions, numbers, buffer, energy, variance, box);
  clock_gettime(CLOCK_MONOTONIC, &end);
  uint64_t elapsed = (uint64_t)(end.tv_sec - begin.tv_sec) * 1000000000ull +
                     (uint64_t)(end.tv_nsec - begin.tv_nsec);
  __atomic_fetch_add(&shared->calls, 1, __ATOMIC_RELAXED);
  __atomic_fetch_add(&shared->inside_ns, elapsed, __ATOMIC_RELAXED);
  if (status != 0) {
    __atomic_fetch_add(&shared->failed, 1, __ATOMIC_RELAXED);
  } else if (negate_buffer) {
    for (long i = 0; i < 3 * n_atoms; ++i) buffer[i] = -buffer[i];
  }
  return status;
}
