// MIT License
// Copyright 2023--present rgpot developers

// cp2kc shim that only records the calculator communicator rgpot hands it.

#if defined(_WIN32) || defined(_WIN64)
#define RGPOT_CP2KC_BUILD
#endif
#include "rgpot/CP2KPot/cp2kc_abi.h"

#include <cstdio>
#include <cstring>

#include <mpi.h>

namespace {

MPI_Comm g_comm = MPI_COMM_NULL;
bool g_set = false;
int g_adopt_calls = 0;

void write_error(char *err, size_t err_len, const char *text) {
  if (err && err_len > 0)
    std::snprintf(err, err_len, "%s", text);
}

} // namespace

extern "C" {

int cp2kc_abi_version(void) { return CP2KC_ABI_VERSION; }
const char *cp2kc_version(void) { return "cp2kc-adopt-fixture"; }
int cp2kc_available(void) { return 1; }

CP2KCSession *cp2kc_session_create(const char *, const char *, char *err,
                                   size_t err_len) {
  write_error(err, err_len, "adopt fixture has no CP2K");
  return nullptr;
}

int cp2kc_session_select_orbitals(CP2KCSession *, long long) { return 0; }

int cp2kc_session_calculate(CP2KCSession *, int, const double *, const double *,
                            double *, double *, char *err, size_t err_len) {
  std::fprintf(stderr, "cp2kc_adopt_engine: force call\n");
  write_error(err, err_len, "adopt fixture has no CP2K");
  return CP2KC_FATAL;
}

void cp2kc_session_destroy(CP2KCSession *) {}

// Stores the communicator from rgpot. Does not call MPI_Comm_split.
int cp2kc_adopt_calculator_comm(const void *comm, size_t comm_bytes,
                                int ranks_per_calc) {
  ++g_adopt_calls;
  if (comm == nullptr || comm_bytes != sizeof(MPI_Comm))
    return -1;
  MPI_Comm incoming = MPI_COMM_NULL;
  std::memcpy(&incoming, comm, sizeof(incoming));
  if (incoming == MPI_COMM_NULL)
    return -1;
  int inited = 0;
  MPI_Initialized(&inited);
  if (!inited)
    return -1;
  int rank = 0;
  int size = 1;
  MPI_Comm_rank(MPI_COMM_WORLD, &rank);
  MPI_Comm_size(MPI_COMM_WORLD, &size);
  const int per = ranks_per_calc > 0 ? ranks_per_calc : size;
  if (per > size || size % per != 0)
    return -1;
  if (!g_set) {
    g_comm = incoming;
    g_set = true;
  }
  return rank / per;
}

int cp2kc_adopt_call_count(void) { return g_adopt_calls; }

int cp2kc_adopted_comm(void *out, size_t nbytes) {
  if (!g_set || out == nullptr || nbytes != sizeof(MPI_Comm))
    return -1;
  std::memcpy(out, &g_comm, sizeof(g_comm));
  return 0;
}

} // extern "C"
