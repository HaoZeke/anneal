/* MIT License
 * Copyright 2023--present rgpot developers
 *
 * The C ABI of rgpot_potential_new_eam_table, called as a host calls it: a
 * copper vacancy cell evaluated against LAMMPS, and every refusal of the
 * constructor.
 *
 * argv[1] is CppCore/tests/data/eam_table. cu_u3_vacancy.ref holds the
 * LAMMPS 22 Jul 2025 run (pair_style eam on Cu_u3.eam) of 107 copper atoms
 * around one vacancy, written by make_vacancy_ref.py.
 */
/* rgpot.h names eindir_objective_t; the first header defines it for C hosts. */
/* clang-format off */
#include "rgpot/eindir_abi.hpp"
#include "rgpot.h"
/* clang-format on */

#include <math.h>
#include <stdio.h>
#include <string.h>

#if RGPOT_ABI_VERSION_MAJOR != 1 || RGPOT_ABI_VERSION_MINOR < 6
#error "rgpot_potential_new_eam_table needs C ABI 1.6 or newer"
#endif

#define MAX_ATOMS 256
/* The largest gaps of the run are 4e-13 eV and 6e-15 eV/Angstrom. */
#define ENERGY_TOLERANCE 1.0e-11
#define FORCE_TOLERANCE 1.0e-12

typedef struct {
  size_t n;
  double cell[9];
  double energy;
  double positions[3 * MAX_ATOMS];
  int numbers[MAX_ATOMS];
  double forces[3 * MAX_ATOMS];
} Reference;

static int failures = 0;

#define CHECK(cond, ...)                                                       \
  do {                                                                         \
    if (!(cond)) {                                                             \
      fprintf(stderr, "%s:%d: ", __FILE__, __LINE__);                          \
      fprintf(stderr, __VA_ARGS__);                                            \
      fprintf(stderr, "\n");                                                   \
      ++failures;                                                              \
    }                                                                          \
  } while (0)

static int load_reference(const char *path, Reference *ref) {
  FILE *file = fopen(path, "r");
  char line[1024];
  size_t atoms = 0;
  if (file == NULL) {
    fprintf(stderr, "cannot open %s\n", path);
    return 1;
  }
  ref->n = 0;
  while (fgets(line, sizeof line, file) != NULL) {
    if (line[0] == '#') {
      continue;
    }
    if (strncmp(line, "box ", 4) == 0) {
      if (sscanf(line + 4, "%lf %lf %lf %lf %lf %lf %lf %lf %lf", &ref->cell[0],
                 &ref->cell[1], &ref->cell[2], &ref->cell[3], &ref->cell[4],
                 &ref->cell[5], &ref->cell[6], &ref->cell[7],
                 &ref->cell[8]) != 9) {
        fclose(file);
        return 1;
      }
    } else if (strncmp(line, "energy ", 7) == 0) {
      if (sscanf(line + 7, "%lf", &ref->energy) != 1) {
        fclose(file);
        return 1;
      }
    } else if (strncmp(line, "atoms ", 6) == 0) {
      if (sscanf(line + 6, "%zu", &atoms) != 1 || atoms > MAX_ATOMS) {
        fclose(file);
        return 1;
      }
    } else if (strncmp(line, "stress ", 7) != 0 && ref->n < MAX_ATOMS) {
      const size_t i = ref->n;
      if (sscanf(line, "%d %lf %lf %lf %lf %lf %lf", &ref->numbers[i],
                 &ref->positions[3 * i], &ref->positions[3 * i + 1],
                 &ref->positions[3 * i + 2], &ref->forces[3 * i],
                 &ref->forces[3 * i + 1], &ref->forces[3 * i + 2]) != 7) {
        fclose(file);
        return 1;
      }
      ++ref->n;
    }
  }
  fclose(file);
  return ref->n == atoms && atoms > 0 ? 0 : 1;
}

static void join(char *out, size_t size, const char *dir, const char *name) {
  snprintf(out, size, "%s/%s", dir, name);
}

static void check_vacancy(const char *table, int format, Reference *ref) {
  struct rgpot_potential_t *pot =
      rgpot_potential_new_eam_table(table, format, true);
  struct rgpot_force_input_t input;
  struct rgpot_force_out_t output = rgpot_force_out_create();
  enum rgpot_status_t status;
  double worst = 0.0;
  size_t k;
  const double *forces;
  CHECK(pot != NULL, "constructor failed for format %d: %s", format,
        rgpot_last_error());
  if (pot == NULL) {
    return;
  }
  input =
      rgpot_force_input_create(ref->n, ref->positions, ref->numbers, ref->cell);
  status = rgpot_potential_calculate(pot, &input, &output);
  CHECK(status == RGPOT_SUCCESS, "calculate returned %d: %s", (int)status,
        rgpot_last_error());
  if (status == RGPOT_SUCCESS) {
    forces = (const double *)rgpot_tensor_data(output.forces);
    for (k = 0; k < 3 * ref->n; ++k) {
      const double gap = fabs(forces[k] - ref->forces[k]);
      worst = gap > worst ? gap : worst;
    }
    CHECK(fabs(output.energy - ref->energy) < ENERGY_TOLERANCE,
          "energy %.15g, LAMMPS %.15g", output.energy, ref->energy);
    CHECK(worst < FORCE_TOLERANCE, "largest force gap %g against LAMMPS",
          worst);
    CHECK(output.variance == 0.0, "variance %g", output.variance);
    rgpot_tensor_free(output.forces);
  }
  rgpot_force_input_free(&input);
  rgpot_potential_free(pot);
}

static void check_small_cell_is_refused(const char *table, Reference *ref) {
  struct rgpot_potential_t *pot =
      rgpot_potential_new_eam_table(table, RGPOT_EAM_TABLE_FUNCFL, true);
  double cell[9] = {8.0, 0.0, 0.0, 0.0, 8.0, 0.0, 0.0, 0.0, 8.0};
  struct rgpot_force_input_t input;
  struct rgpot_force_out_t output = rgpot_force_out_create();
  enum rgpot_status_t status;
  CHECK(pot != NULL, "constructor failed: %s", rgpot_last_error());
  if (pot == NULL) {
    return;
  }
  input = rgpot_force_input_create(ref->n, ref->positions, ref->numbers, cell);
  status = rgpot_potential_calculate(pot, &input, &output);
  CHECK(status == RGPOT_INVALID_PARAMETER, "a small cell returned %d",
        (int)status);
  CHECK(output.forces == NULL, "a refused calculation left forces behind");
  CHECK(strstr(rgpot_last_error(), "half a face spacing") != NULL,
        "message was: %s", rgpot_last_error());
  rgpot_force_input_free(&input);
  rgpot_potential_free(pot);
}

static void check_refused(const char *what, const char *path, int format,
                          const char *message) {
  struct rgpot_potential_t *pot =
      rgpot_potential_new_eam_table(path, format, true);
  CHECK(pot == NULL, "%s: a handle came back", what);
  if (pot != NULL) {
    rgpot_potential_free(pot);
    return;
  }
  CHECK(strstr(rgpot_last_error(), message) != NULL, "%s: message was: %s",
        what, rgpot_last_error());
}

int main(int argc, char **argv) {
  static Reference ref;
  char table[1024];
  char malformed[1024];
  char missing[1024];
  struct rgpot_abi_stamp_t built = {RGPOT_ABI_VERSION_MAJOR,
                                    RGPOT_ABI_VERSION_MINOR,
                                    RGPOT_ABI_LAYOUT_REVISION};
  char reference_path[1024];

  if (argc != 2) {
    fprintf(stderr, "usage: %s <CppCore/tests/data/eam_table>\n", argv[0]);
    return 2;
  }
  join(table, sizeof table, argv[1], "Cu_u3.eam");
  join(malformed, sizeof malformed, argv[1], "malformed_funcfl.eam");
  join(missing, sizeof missing, argv[1], "no_such_table.eam");
  join(reference_path, sizeof reference_path, argv[1], "cu_u3_vacancy.ref");
  if (load_reference(reference_path, &ref) != 0) {
    fprintf(stderr, "cannot read %s\n", reference_path);
    return 2;
  }
  CHECK(ref.n == 107, "the vacancy cell has %zu atoms", ref.n);

  CHECK(rgpot_abi_compatible(&built) == 1,
        "the library is older than the header");
  CHECK(rgpot_abi_stamp().abi_minor >= 6, "library minor %u",
        (unsigned)rgpot_abi_stamp().abi_minor);

  check_vacancy(table, RGPOT_EAM_TABLE_FUNCFL, &ref);
  check_vacancy(table, RGPOT_EAM_TABLE_AUTO, &ref);
  check_small_cell_is_refused(table, &ref);

  check_refused("null path", NULL, RGPOT_EAM_TABLE_FUNCFL, "path is NULL");
  check_refused("empty path", "", RGPOT_EAM_TABLE_FUNCFL, "path is empty");
  check_refused("missing file", missing, RGPOT_EAM_TABLE_FUNCFL,
                "no_such_table.eam");
  check_refused("malformed file", malformed, RGPOT_EAM_TABLE_FUNCFL,
                "unreadable");
  check_refused("funcfl read as setfl", table, RGPOT_EAM_TABLE_SETFL,
                "EAM table");
  check_refused("unknown format code", table, 17, "format code 17");
  check_refused("suffix that names no format", "Cu_u3.dat",
                RGPOT_EAM_TABLE_AUTO, "cannot tell the table format");

  if (failures != 0) {
    fprintf(stderr, "%d check(s) failed\n", failures);
    return 1;
  }
  printf("eam_table_capi: ok\n");
  return 0;
}
