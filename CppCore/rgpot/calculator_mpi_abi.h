/* MIT License
 * Copyright 2023--present rgpot developers
 *
 * C ABI between librgpot and librgpot_mpi, the separately built library that
 * carries every MPI call of the calculator groups.
 *
 * librgpot never links MPI. The calculator-group entry points in
 * CalculatorGroup.hpp that need MPI (bindCalculators, calculatorsUseMpi)
 * load librgpot_mpi on first use with dlopen and resolve one symbol,
 * RGPOT_MPI_API_SYMBOL, which returns the table below. A process that never
 * asks for calculator groups therefore never maps libmpi or its transport
 * layers (UCX installs memory hooks at load time).
 *
 * Search order of the loader (rgpot::loadCalculatorMpi):
 *   1. an explicit path passed to loadCalculatorMpi;
 *   2. the RGPOT_MPI_LIBRARY environment variable;
 *   3. RGPOT_MPI_LIBRARY_NAME in the directory holding librgpot itself;
 *   4. RGPOT_MPI_LIBRARY_NAME through the dynamic linker's search path.
 *
 * Versioning: abi_version changes when a member changes meaning;
 * struct_size grows when members are appended. A loader accepts a table
 * whose abi_version equals RGPOT_MPI_ABI_VERSION and whose struct_size is
 * at least the size it was compiled against.
 */
#ifndef RGPOT_CALCULATOR_MPI_ABI_H
#define RGPOT_CALCULATOR_MPI_ABI_H

#include <stddef.h>
#include <stdint.h>

#ifdef __cplusplus
extern "C" {
#endif

#if !defined(_WIN32)
#define RGPOT_CALC_API __attribute__((visibility("default")))
#else
#define RGPOT_CALC_API
#endif

#define RGPOT_MPI_ABI_VERSION 1u
#define RGPOT_MPI_API_SYMBOL "rgpot_mpi_api_v1"
/* SONAME of librgpot_mpi. Independent of librgpot's kCxxSoversion:
 * this C ABI is still version 1, so the filename stays .3 / .so.3. */
#if defined(__APPLE__)
#define RGPOT_MPI_LIBRARY_NAME "librgpot_mpi.3.dylib"
#else
#define RGPOT_MPI_LIBRARY_NAME "librgpot_mpi.so.3"
#endif

/* Mirror of rgpot::CalculatorGroup. */
typedef struct rgpot_mpi_group_t {
  int index;
  int ranks;
  int rank_in_group;
  int world_size;
} rgpot_mpi_group_t;

typedef struct rgpot_mpi_api_t {
  uint32_t abi_version; /* RGPOT_MPI_ABI_VERSION */
  uint32_t struct_size; /* sizeof(rgpot_mpi_api_t) of the library */

  /* Collective on MPI_COMM_WORLD. Calls MPI_Init when MPI is not up yet.
   * Splits the world into calculators of ranks_per_calculator ranks
   * (<= 0: one calculator) and fills *out; out->index is -1 when the
   * world cannot be divided that way. */
  void (*bind)(int ranks_per_calculator, rgpot_mpi_group_t *out);
  /* Copies the calculator's MPI_Comm into comm_out when comm_bytes is
   * sizeof(MPI_Comm) and a split exists. Returns 1 then, else 0. */
  int (*comm)(void *comm_out, size_t comm_bytes);
  /* 1 when MPI_Initialized reports MPI up. */
  int (*initialized)(void);
  /* shareFromCalculator: broadcast from the first rank of calculator
   * owner to MPI_COMM_WORLD. Same return contract. */
  int (*share)(int owner, void *data, size_t bytes);
  /* finalizeMpiAtExit: registers the exit handler once. The handler calls
   * MPI_Abort when abort_requested() returns non-zero. */
  void (*finalize_at_exit)(int (*abort_requested)(void));
  /* calculatorAgree: logical AND of flags over the calculator. */
  int (*agree)(unsigned char *flags, size_t n);
  /* Error exchange of a groupCollective force call: every rank of the
   * calculator passes its message (len 0 for none). When any rank has
   * one, every message prints on every rank of the calculator and the
   * world aborts; otherwise the call returns. */
  void (*publish_error)(const char *message, size_t len);
  /* Invokes an engine callback with the calculator's MPI_Comm and its
   * byte size, then the ranks per calculator. The handle is borrowed:
   * the engine must not free it. Returns -1 without a valid split or
   * callback, otherwise the callback's result. */
  int (*adopt)(int (*callback)(const void *, size_t, int));
} rgpot_mpi_api_t;

typedef const rgpot_mpi_api_t *(*rgpot_mpi_api_fn)(void);

/* Exported by librgpot_mpi. */
const rgpot_mpi_api_t *rgpot_mpi_api_v1(void);

/* Exported by librgpot (stable C ABI for calculator-group hosts). */
RGPOT_CALC_API uint32_t rgpot_mpi_host_abi_version(void);
RGPOT_CALC_API int rgpot_mpi_bind_calculators(int ranks_per_calculator,
                                              rgpot_mpi_group_t *out);
RGPOT_CALC_API int rgpot_mpi_share_bytes(int owner, void *data, size_t bytes);

#ifdef __cplusplus
}
#endif

#endif /* RGPOT_CALCULATOR_MPI_ABI_H */
