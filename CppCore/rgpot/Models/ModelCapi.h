#pragma once
// MIT License
// Copyright 2023--present rgpot developers

#include <stddef.h>

#if defined(_WIN32)
#if defined(RGPOT_MODEL_BUILD)
#define RGPOT_MODEL_API __declspec(dllexport)
#else
#define RGPOT_MODEL_API __declspec(dllimport)
#endif
#else
#define RGPOT_MODEL_API __attribute__((visibility("default")))
#endif

#ifdef __cplusplus
extern "C" {
#endif

typedef struct rgpot_builtin_model rgpot_builtin_model;

typedef enum rgpot_builtin_kind {
  RGPOT_BUILTIN_ECKART = 1,
  RGPOT_BUILTIN_MULLER_BROWN = 2,
  RGPOT_BUILTIN_MODIFIED_MULLER_BROWN = 3,
  RGPOT_BUILTIN_LEPS_HH2 = 4,
  RGPOT_BUILTIN_LEPS_HCL2 = 5,
  RGPOT_BUILTIN_LEPS_FH2 = 6,
  RGPOT_BUILTIN_CBE = 7
} rgpot_builtin_kind;

/*
 * Construct one fixed-parameter analytic model by enum tag.
 * Returns NULL on failure; call rgpot_builtin_model_last_error() for details.
 */
RGPOT_MODEL_API rgpot_builtin_model *
rgpot_builtin_model_new(rgpot_builtin_kind kind);

/*
 * Construct a DoubleWell model with caller-provided parameters.
 * The model is analytic and supports both calculate and hvp.
 * Returns NULL on failure; call rgpot_builtin_model_last_error() for details.
 */
RGPOT_MODEL_API rgpot_builtin_model *rgpot_builtin_double_well_new(
    size_t dimensions, double v0, double x0, double omega, double coupling);

/*
 * Construct a CoupledProtonWell model with caller-provided parameters.
 * Returns NULL on failure; call rgpot_builtin_model_last_error() for details.
 */
RGPOT_MODEL_API rgpot_builtin_model *rgpot_builtin_coupled_proton_well_new(
    double v0, double x0, double spring);

RGPOT_MODEL_API void
rgpot_builtin_model_free(rgpot_builtin_model *model);

RGPOT_MODEL_API size_t
rgpot_builtin_model_dimensions(const rgpot_builtin_model *model);

/*
 * Evaluate row-major batches in rgpot native units (eV and angstrom).
 * `forces` is -dE/dx and has count*dimensions entries.
 */
RGPOT_MODEL_API int rgpot_builtin_model_calculate(
    rgpot_builtin_model *model, size_t count, const double *coordinates,
    double *energies, double *forces);

/*
 * Apply the model's existing analytic Hessian-vector hook to each row.
 * No finite-difference scheme or dense Hessian is used.
 */
RGPOT_MODEL_API int rgpot_builtin_model_hvp(
    rgpot_builtin_model *model, size_t count, const double *coordinates,
    const double *directions, double *products);

RGPOT_MODEL_API const char *rgpot_builtin_model_last_error(void);

#ifdef __cplusplus
}
#endif
