// MIT License
// Copyright 2023--present rgpot developers

/*
 * libcp2kc_rgpot: the cp2kc ABI over libcp2k.
 *
 * This file is the only part of the CP2K arm that links libcp2k and MPI.
 * CP2KPot dlopens it, so librgpot carries neither dependency.
 */

#define RGPOT_CP2KC_BUILD
#include "cp2kc_abi.h"

#include <libcp2k.h>
#include <mpi.h>

#include <stdarg.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#define CP2KC_PATH_MAX 4096
static const char kScfFailure[] = "SCF run NOT converged";

typedef struct CP2KCEnv {
  long long key;
  force_env_t force_env;
  int natom;
  int has_cell;
  double cell[9];
  long output_offset;
  char output[CP2KC_PATH_MAX];
} CP2KCEnv;

struct CP2KCSession {
  char dir[CP2KC_PATH_MAX];
  char input[CP2KC_PATH_MAX];
  long long key;
  CP2KCEnv *envs;
  size_t n_envs;
  size_t cap_envs;
  CP2KCSession *next;
};

static MPI_Comm g_comm = MPI_COMM_NULL;
static int g_has_comm = 0;
static MPI_Fint g_fcomm = 0;
static int g_initialized = 0;
static int g_owns_mpi = 0;
static int g_finalized = 0;
static CP2KCSession *g_sessions = NULL;
static char g_version[320] = "cp2kc_rgpot";

static void set_error(char *err, size_t err_len, const char *fmt, ...) {
  if (err == NULL || err_len == 0)
    return;
  va_list ap;
  va_start(ap, fmt);
  vsnprintf(err, err_len, fmt, ap);
  va_end(ap);
}

static MPI_Comm session_comm(void) {
  return g_has_comm ? g_comm : MPI_COMM_WORLD;
}

static void destroy_envs(CP2KCSession *session) {
  for (size_t i = 0; i < session->n_envs; ++i) {
    if (!g_finalized)
      cp2k_destroy_force_env(session->envs[i].force_env);
  }
  free(session->envs);
  session->envs = NULL;
  session->n_envs = 0;
  session->cap_envs = 0;
}

/* Destroys the remaining force_envs, then finalises CP2K. MPI_Finalize
 * runs here only when cp2k_init initialised MPI. A host that finalised
 * MPI already leaves CP2K as it is. */
static void shutdown_at_exit(void) {
  if (!g_initialized || g_finalized)
    return;
  int finalized = 0;
  MPI_Finalized(&finalized);
  if (finalized) {
    g_finalized = 1;
    return;
  }
  for (CP2KCSession *s = g_sessions; s != NULL; s = s->next)
    destroy_envs(s);
  if (g_owns_mpi)
    cp2k_finalize();
  else
    cp2k_finalize_without_mpi();
  g_finalized = 1;
}

static int ensure_initialized(char *err, size_t err_len) {
  if (g_finalized) {
    set_error(err, err_len, "cp2kc: CP2K was already finalised");
    return -1;
  }
  if (g_initialized)
    return 0;
  int initialized = 0;
  MPI_Initialized(&initialized);
  if (initialized) {
    cp2k_init_without_mpi();
    g_owns_mpi = 0;
  } else {
    cp2k_init();
    g_owns_mpi = 1;
  }
  g_initialized = 1;
  if (!g_has_comm)
    g_fcomm = MPI_Comm_c2f(MPI_COMM_WORLD);
  char cp2k_version[256];
  memset(cp2k_version, 0, sizeof(cp2k_version));
  cp2k_get_version(cp2k_version, (int)sizeof(cp2k_version) - 1);
  for (int i = (int)strlen(cp2k_version) - 1; i >= 0 && cp2k_version[i] == ' ';
       --i)
    cp2k_version[i] = '\0';
  snprintf(g_version, sizeof(g_version), "cp2kc_rgpot %d (%s)",
           CP2KC_ABI_VERSION, cp2k_version);
  if (atexit(shutdown_at_exit) != 0) {
    set_error(err, err_len, "cp2kc: cannot register the exit handler");
    return -1;
  }
  return 0;
}

int cp2kc_abi_version(void) { return CP2KC_ABI_VERSION; }

const char *cp2kc_version(void) { return g_version; }

int cp2kc_available(void) { return 1; }

int cp2kc_adopt_calculator_comm(const void *comm, size_t comm_bytes,
                                int ranks_per_calc) {
  if (comm == NULL || comm_bytes != sizeof(MPI_Comm))
    return -1;
  int initialized = 0;
  MPI_Initialized(&initialized);
  if (!initialized)
    return -1;
  MPI_Comm incoming = MPI_COMM_NULL;
  memcpy(&incoming, comm, sizeof(incoming));
  if (incoming == MPI_COMM_NULL)
    return -1;
  int rank = 0;
  int size = 1;
  MPI_Comm_rank(MPI_COMM_WORLD, &rank);
  MPI_Comm_size(MPI_COMM_WORLD, &size);
  int per = ranks_per_calc > 0 ? ranks_per_calc : size;
  if (per > size || size % per != 0)
    return -1;
  /* A force_env keeps the communicator it was created on. */
  for (CP2KCSession *s = g_sessions; s != NULL && !g_has_comm; s = s->next) {
    if (s->n_envs > 0)
      return -1;
  }
  if (!g_has_comm) {
    g_comm = incoming;
    g_fcomm = MPI_Comm_c2f(incoming);
    g_has_comm = 1;
  }
  return rank / per;
}

CP2KCSession *cp2kc_session_create(const char *deck, const char *dir, char *err,
                                   size_t err_len) {
  if (deck == NULL || dir == NULL || dir[0] == '\0') {
    set_error(err, err_len, "cp2kc: deck and directory are required");
    return NULL;
  }
  if (ensure_initialized(err, err_len) != 0)
    return NULL;
  CP2KCSession *session = (CP2KCSession *)calloc(1, sizeof(CP2KCSession));
  if (session == NULL) {
    set_error(err, err_len, "cp2kc: out of memory");
    return NULL;
  }
  int a = snprintf(session->dir, sizeof(session->dir), "%s", dir);
  int b = snprintf(session->input, sizeof(session->input), "%s/input.inp", dir);
  if (a < 0 || (size_t)a >= sizeof(session->dir) || b < 0 ||
      (size_t)b >= sizeof(session->input)) {
    free(session);
    set_error(err, err_len, "cp2kc: directory path is too long");
    return NULL;
  }
  /* Every rank writes the same deck; CP2K's ionode reads it. */
  FILE *fp = fopen(session->input, "w");
  if (fp == NULL) {
    set_error(err, err_len, "cp2kc: cannot write %s", session->input);
    free(session);
    return NULL;
  }
  const size_t len = strlen(deck);
  const int wrote = fwrite(deck, 1, len, fp) == len;
  if (fclose(fp) != 0 || !wrote) {
    set_error(err, err_len, "cp2kc: cannot write %s", session->input);
    free(session);
    return NULL;
  }
  session->next = g_sessions;
  g_sessions = session;
  return session;
}

int cp2kc_session_select_orbitals(CP2KCSession *session, long long key) {
  if (session == NULL)
    return CP2KC_BAD_ARGUMENT;
  session->key = key;
  return CP2KC_OK;
}

static CP2KCEnv *env_for_key(CP2KCSession *session, char *err, size_t err_len) {
  for (size_t i = 0; i < session->n_envs; ++i) {
    if (session->envs[i].key == session->key)
      return &session->envs[i];
  }
  if (session->n_envs == session->cap_envs) {
    const size_t cap = session->cap_envs ? 2 * session->cap_envs : 4;
    CP2KCEnv *grown =
        (CP2KCEnv *)realloc(session->envs, cap * sizeof(CP2KCEnv));
    if (grown == NULL) {
      set_error(err, err_len, "cp2kc: out of memory");
      return NULL;
    }
    session->envs = grown;
    session->cap_envs = cap;
  }
  CP2KCEnv *env = &session->envs[session->n_envs];
  memset(env, 0, sizeof(*env));
  env->key = session->key;
  int w;
  if (session->n_envs == 0)
    w = snprintf(env->output, sizeof(env->output), "%s/cp2k.out", session->dir);
  else
    w = snprintf(env->output, sizeof(env->output), "%s/cp2k-%zu.out",
                 session->dir, session->n_envs);
  if (w < 0 || (size_t)w >= sizeof(env->output)) {
    set_error(err, err_len, "cp2kc: directory path is too long");
    return NULL;
  }
  cp2k_create_force_env_comm(&env->force_env, session->input, env->output,
                             (int)g_fcomm);
  cp2k_get_natom(env->force_env, &env->natom);
  ++session->n_envs;
  return env;
}

/* The ionode reads the output written since the previous call and looks
 * for CP2K's non-convergence notice. The deck sets
 * IGNORE_CONVERGENCE_FAILURE, so CP2K warns instead of aborting. */
static int ionode_scf_failed(CP2KCEnv *env) {
  FILE *fp = fopen(env->output, "r");
  if (fp == NULL)
    return 0;
  if (fseek(fp, 0, SEEK_END) != 0) {
    fclose(fp);
    return 0;
  }
  const long end = ftell(fp);
  int failed = 0;
  if (end > env->output_offset &&
      fseek(fp, env->output_offset, SEEK_SET) == 0) {
    const size_t n = (size_t)(end - env->output_offset);
    char *text = (char *)malloc(n + 1);
    if (text != NULL) {
      const size_t got = fread(text, 1, n, fp);
      text[got] = '\0';
      failed = strstr(text, kScfFailure) != NULL;
      free(text);
    }
  }
  if (end >= 0)
    env->output_offset = end;
  fclose(fp);
  return failed;
}

int cp2kc_session_calculate(CP2KCSession *session, int n_atoms,
                            const double *pos_bohr, const double *cell_bohr9,
                            double *e_ha, double *f_ha_bohr, char *err,
                            size_t err_len) {
  if (session == NULL || n_atoms <= 0 || pos_bohr == NULL ||
      cell_bohr9 == NULL || e_ha == NULL || f_ha_bohr == NULL) {
    set_error(err, err_len, "cp2kc: null argument or empty system");
    return CP2KC_BAD_ARGUMENT;
  }
  if (g_finalized) {
    set_error(err, err_len, "cp2kc: CP2K was already finalised");
    return CP2KC_FATAL;
  }
  CP2KCEnv *env = env_for_key(session, err, err_len);
  if (env == NULL)
    return CP2KC_FATAL;
  if (env->natom != n_atoms) {
    set_error(err, err_len, "cp2kc: %d atoms given, the deck has %d", n_atoms,
              env->natom);
    return CP2KC_BAD_ARGUMENT;
  }
  cp2k_set_positions(env->force_env, pos_bohr, 3 * n_atoms);
  if (!env->has_cell || memcmp(env->cell, cell_bohr9, sizeof(env->cell)) != 0) {
    cp2k_set_cell(env->force_env, cell_bohr9);
    memcpy(env->cell, cell_bohr9, sizeof(env->cell));
    env->has_cell = 1;
  }
  cp2k_calc_energy_force(env->force_env);
  cp2k_get_potential_energy(env->force_env, e_ha);
  cp2k_get_forces(env->force_env, f_ha_bohr, 3 * n_atoms);

  MPI_Comm comm = session_comm();
  int rank = 0;
  MPI_Comm_rank(comm, &rank);
  int failed = rank == 0 ? ionode_scf_failed(env) : 0;
  MPI_Bcast(&failed, 1, MPI_INT, 0, comm);
  if (failed) {
    set_error(err, err_len, "cp2kc: %s (see %s)", kScfFailure, env->output);
    return CP2KC_SCF_NOT_CONVERGED;
  }
  return CP2KC_OK;
}

void cp2kc_session_destroy(CP2KCSession *session) {
  if (session == NULL)
    return;
  for (CP2KCSession **p = &g_sessions; *p != NULL; p = &(*p)->next) {
    if (*p == session) {
      *p = session->next;
      break;
    }
  }
  destroy_envs(session);
  free(session);
}
