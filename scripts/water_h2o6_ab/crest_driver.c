/*
 * Energy and gradient driver for the generic calculator of CREST, backed by
 * the rgpot GFN2 engine.
 *
 *   crest_driver serve <rundir> <budget> <engine>   long-lived call server
 *   crest_driver stop  <rundir>                     end the server
 *   crest_driver <geometry.xyz>                     one call, run by CREST
 *
 * CREST runs the call program in its calculator directory as
 * "<program> genericinp.xyz" and reads the result from an ORCA style engrad
 * file. The run directory of the server comes from CREST_DRIVER_RUNDIR in the
 * environment of CREST. The server keeps one engine handle for the whole CREST
 * run, so successive calls reuse the wavefunction the way every other consumer
 * of the engine does.
 *
 * The engine buffer is the force, minus dE/dx, in eV per angstrom. The engrad
 * file holds dE/dx in hartree per bohr.
 *
 * The first budget calls are answered. Every later call is refused and the
 * program exits 1 without writing a result, which ends the CREST step that
 * made it. The server writes <rundir>/driver.txt when it ends, with the number
 * of calls, the time spent inside the engine, and the monotonic clock readings
 * of the first call and of the call that used up the budget.
 */
#define _GNU_SOURCE
#include <ctype.h>
#include <dlfcn.h>
#include <errno.h>
#include <math.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <sys/socket.h>
#include <sys/stat.h>
#include <sys/un.h>
#include <time.h>
#include <unistd.h>

#define AUTOEV 27.21138505
#define AUTOAA 0.52917726
#define MAX_ATOMS 256
#define STOP_REQUEST UINT64_MAX
#define STATUS_OK 0
#define STATUS_REFUSED 1

typedef struct {
  int method;
  double accuracy;
  double electronic_temperature;
  int max_iterations;
  double charge;
  int uhf;
} xtb_config;

typedef void *(*create_fn)(const xtb_config *, uint8_t *, size_t);
typedef void (*destroy_fn)(void *);
typedef int (*force_fn)(void *, long, const double *, const int *, double *,
                        double *, double *, const double *);

static void fail(const char *what) {
  fprintf(stderr, "crest_driver: %s\n", what);
  exit(2);
}

static int read_full(int fd, void *buffer, size_t length) {
  char *position = buffer;
  while (length > 0) {
    ssize_t got = read(fd, position, length);
    if (got < 0 && errno == EINTR) continue;
    if (got <= 0) return 0;
    position += got;
    length -= (size_t)got;
  }
  return 1;
}

static int write_full(int fd, const void *buffer, size_t length) {
  const char *position = buffer;
  while (length > 0) {
    ssize_t put = write(fd, position, length);
    if (put < 0 && errno == EINTR) continue;
    if (put <= 0) return 0;
    position += put;
    length -= (size_t)put;
  }
  return 1;
}

static int64_t monotonic_ns(void) {
  struct timespec now;
  clock_gettime(CLOCK_MONOTONIC, &now);
  return (int64_t)now.tv_sec * 1000000000LL + (int64_t)now.tv_nsec;
}

static void socket_path(struct sockaddr_un *address, const char *rundir) {
  memset(address, 0, sizeof *address);
  address->sun_family = AF_UNIX;
  if (snprintf(address->sun_path, sizeof address->sun_path, "%s/sock", rundir) >=
      (int)sizeof address->sun_path)
    fail("socket path is too long");
}

static int atomic_number(const char *symbol) {
  char first = (char)toupper((unsigned char)symbol[0]);
  if (first == 'O') return 8;
  if (first == 'H') return 1;
  fail("only H and O are supported");
  return 0;
}

static int read_xyz(const char *path, int *numbers, double *coords) {
  FILE *file = fopen(path, "r");
  if (!file) fail("cannot read the geometry");
  int count = 0;
  char line[512];
  if (!fgets(line, sizeof line, file) || sscanf(line, "%d", &count) != 1 ||
      count <= 0 || count > MAX_ATOMS)
    fail("bad xyz header");
  if (!fgets(line, sizeof line, file)) fail("short xyz");
  for (int i = 0; i < count; ++i) {
    char symbol[16];
    if (!fgets(line, sizeof line, file) ||
        sscanf(line, "%15s %lf %lf %lf", symbol, &coords[3 * i],
               &coords[3 * i + 1], &coords[3 * i + 2]) != 4)
      fail("bad xyz atom line");
    numbers[i] = atomic_number(symbol);
  }
  fclose(file);
  return count;
}

static int serve(const char *rundir, long budget, const char *engine_path) {
  void *lib = dlopen(engine_path, RTLD_NOW | RTLD_LOCAL);
  if (!lib) fail(dlerror());
  create_fn create = (create_fn)dlsym(lib, "rgpot_xtb_create");
  destroy_fn destroy = (destroy_fn)dlsym(lib, "rgpot_xtb_destroy");
  force_fn force = (force_fn)dlsym(lib, "rgpot_xtb_force");
  if (!create || !destroy || !force) fail("the engine lacks a required symbol");
  xtb_config config = {3, 0.01, 300.0, 500, 0.0, 0};
  uint8_t error[256] = {0};
  void *pot = create(&config, error, sizeof error);
  if (!pot) fail((const char *)error);
  const double box[9] = {60, 0, 0, 0, 60, 0, 0, 0, 60};

  struct sockaddr_un address;
  socket_path(&address, rundir);
  unlink(address.sun_path);
  int listener = socket(AF_UNIX, SOCK_STREAM, 0);
  if (listener < 0 || bind(listener, (struct sockaddr *)&address, sizeof address) ||
      listen(listener, 4))
    fail("cannot listen");
  char ready[1024];
  snprintf(ready, sizeof ready, "%s/ready", rundir);
  FILE *flag = fopen(ready, "w");
  if (!flag) fail("cannot write the ready file");
  fputs("ready\n", flag);
  fclose(flag);

  long served = 0, failed = 0, refused = 0;
  double inside_s = 0.0;
  int64_t first_ns = 0, cap_ns = 0;
  for (;;) {
    int stream = accept(listener, NULL, NULL);
    if (stream < 0) {
      if (errno == EINTR) continue;
      break;
    }
    uint64_t n = 0;
    if (!read_full(stream, &n, sizeof n)) {
      close(stream);
      continue;
    }
    if (n == STOP_REQUEST) {
      uint8_t status = STATUS_OK;
      write_full(stream, &status, 1);
      close(stream);
      break;
    }
    if (n == 0 || n > MAX_ATOMS) {
      close(stream);
      continue;
    }
    double coords[3 * MAX_ATOMS], gradient[3 * MAX_ATOMS], energy = 0.0, variance = 0.0;
    int numbers[MAX_ATOMS];
    if (!read_full(stream, coords, 3 * n * sizeof(double)) ||
        !read_full(stream, numbers, n * sizeof(int))) {
      close(stream);
      continue;
    }
    uint8_t status = STATUS_REFUSED;
    int used_budget = 0;
    if (served < budget) {
      int64_t begin = monotonic_ns();
      if (!first_ns) first_ns = begin;
      int rc = force(pot, (long)n, coords, numbers, gradient, &energy, &variance, box);
      inside_s += 1e-9 * (double)(monotonic_ns() - begin);
      ++served;
      int finite = isfinite(energy);
      for (uint64_t i = 0; i < 3 * n && finite; ++i) finite = isfinite(gradient[i]);
      if (rc == 0 && finite) {
        for (uint64_t i = 0; i < 3 * n; ++i) gradient[i] = -gradient[i];
        status = STATUS_OK;
      } else {
        ++failed;
      }
      used_budget = served >= budget;
    } else {
      ++refused;
    }
    write_full(stream, &status, 1);
    if (status != STATUS_REFUSED) {
      write_full(stream, &energy, sizeof energy);
      write_full(stream, gradient, 3 * n * sizeof(double));
    }
    close(stream);
    if (used_budget) cap_ns = monotonic_ns();
  }
  char summary[1024];
  snprintf(summary, sizeof summary, "%s/driver.txt", rundir);
  FILE *meta = fopen(summary, "w");
  if (meta) {
    fprintf(meta,
            "calls: %ld\nfailed: %ld\nrefused: %ld\nbudget: %ld\ninside_s: %.6f\n"
            "first_call_monotonic_ns: %lld\ncap_monotonic_ns: %lld\n",
            served, failed, refused, budget, inside_s, (long long)first_ns, (long long)cap_ns);
    fclose(meta);
  }
  destroy(pot);
  unlink(address.sun_path);
  return 0;
}

/*
 * CREST makes a calculator copy per dynamics thread. It runs the call program
 * in the calculator directory but looks for the result in the thread
 * subdirectory _1, so the engrad file is written to both places.
 */
static void write_engrad(const char *path, int count, double energy, const double *gradient,
                         const int *numbers, const double *coords) {
  FILE *engrad = fopen(path, "w");
  if (!engrad) fail("cannot write the engrad file");
  fprintf(engrad,
          "#\n# Number of atoms\n#\n %d\n#\n# The current total energy in Eh\n#\n %.12f\n#\n"
          "# The current gradient in Eh/bohr\n#\n",
          count, energy / AUTOEV);
  for (int i = 0; i < 3 * count; ++i) fprintf(engrad, " %.12E\n", gradient[i] * AUTOAA / AUTOEV);
  fputs("#\n# The atomic numbers and current coordinates in Bohr\n#\n", engrad);
  for (int i = 0; i < count; ++i)
    fprintf(engrad, " %d %.12f %.12f %.12f\n", numbers[i], coords[3 * i] / AUTOAA,
            coords[3 * i + 1] / AUTOAA, coords[3 * i + 2] / AUTOAA);
  fclose(engrad);
}

static int call(const char *rundir, const char *geometry) {
  double coords[3 * MAX_ATOMS];
  int numbers[MAX_ATOMS];
  int count = read_xyz(geometry, numbers, coords);
  unlink("genericinp.engrad");
  unlink("_1/genericinp.engrad");

  struct sockaddr_un address;
  socket_path(&address, rundir);
  int stream = socket(AF_UNIX, SOCK_STREAM, 0);
  if (stream < 0 || connect(stream, (struct sockaddr *)&address, sizeof address))
    fail("cannot reach the server");
  uint64_t n = (uint64_t)count;
  uint8_t status = STATUS_REFUSED;
  double energy = 0.0, gradient[3 * MAX_ATOMS];
  if (!write_full(stream, &n, sizeof n) ||
      !write_full(stream, coords, 3 * n * sizeof(double)) ||
      !write_full(stream, numbers, n * sizeof(int)) || !read_full(stream, &status, 1))
    fail("lost the server");
  if (status == STATUS_REFUSED) return 1;
  if (!read_full(stream, &energy, sizeof energy) ||
      !read_full(stream, gradient, 3 * n * sizeof(double)))
    fail("short reply");
  close(stream);

  write_engrad("genericinp.engrad", count, energy, gradient, numbers, coords);
  struct stat clone_dir;
  if (!stat("_1", &clone_dir) && S_ISDIR(clone_dir.st_mode))
    write_engrad("_1/genericinp.engrad", count, energy, gradient, numbers, coords);
  return 0;
}

static int stop(const char *rundir) {
  struct sockaddr_un address;
  socket_path(&address, rundir);
  int stream = socket(AF_UNIX, SOCK_STREAM, 0);
  if (stream < 0 || connect(stream, (struct sockaddr *)&address, sizeof address)) return 1;
  uint64_t request = STOP_REQUEST;
  uint8_t status;
  write_full(stream, &request, sizeof request);
  read_full(stream, &status, 1);
  close(stream);
  return 0;
}

int main(int argc, char **argv) {
  if (argc == 5 && !strcmp(argv[1], "serve")) return serve(argv[2], atol(argv[3]), argv[4]);
  if (argc == 3 && !strcmp(argv[1], "stop")) return stop(argv[2]);
  const char *rundir = getenv("CREST_DRIVER_RUNDIR");
  if (argc == 2 && rundir) return call(rundir, argv[1]);
  fprintf(stderr,
          "usage: crest_driver serve <rundir> <budget> <engine> | stop <rundir>\n"
          "       CREST_DRIVER_RUNDIR=<rundir> crest_driver <geometry.xyz>\n");
  return 2;
}
