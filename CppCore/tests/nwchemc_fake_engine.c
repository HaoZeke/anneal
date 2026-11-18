#if defined(_WIN32) || defined(_WIN64)
#define RGPOT_NWCHEMC_BUILD
#endif
#include "nwchem_c_abi.h"

#include <stddef.h>
#include <stdio.h>

static int reject_params = 0;

static int has_flat_message(const void *params_capnp,
                            size_t params_capnp_size_bytes) {
  return params_capnp != NULL && params_capnp_size_bytes >= 8 &&
         (params_capnp_size_bytes % 8) == 0;
}

int nwchemc_set_params(const void *params_capnp,
                       size_t params_capnp_size_bytes) {
  if (reject_params)
    return -1;
  return has_flat_message(params_capnp, params_capnp_size_bytes) ? 0 : -1;
}

RGPOT_NWCHEMC_API void nwchemc_fake_reject_params(int reject) {
  reject_params = reject;
}

NWChemCResult nwchemc_energy_gradient(
    int n_atoms, const double *positions_ang, const int *atomic_numbers,
    const void *params_capnp, size_t params_capnp_size_bytes,
    double *grad_h_bohr) {
  NWChemCResult r;
  r.ok = 0;
  r.energy_h = 0.0;
  r.message[0] = '\0';

  if (n_atoms <= 0 || positions_ang == NULL || atomic_numbers == NULL ||
      grad_h_bohr == NULL ||
      !has_flat_message(params_capnp, params_capnp_size_bytes)) {
    snprintf(r.message, sizeof(r.message), "invalid fake-engine arguments");
    return r;
  }

  for (int i = 0; i < n_atoms * 3; ++i) {
    grad_h_bohr[i] = 0.001 * (double)(i + 1);
  }
  r.ok = 1;
  r.energy_h = 0.25;
  snprintf(r.message, sizeof(r.message), "ok");
  return r;
}

const char *nwchemc_version(void) { return "nwchemc-fake/0.1"; }

int nwchemc_available(void) { return 1; }
