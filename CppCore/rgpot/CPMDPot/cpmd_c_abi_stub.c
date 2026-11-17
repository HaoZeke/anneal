#include "cpmd_c_abi.h"

#include <stdio.h>
#include <string.h>

static const CPMDCFeatureEntry g_features[] = {
#include "cpmd_feature_table.inc"
};

static CPMDCResult stub_fail(void) {
  CPMDCResult r;
  r.ok = 0;
  r.energy_h = 0.0;
  snprintf(r.message, sizeof(r.message),
           "CPMD embed not available in cpmdc stub");
  return r;
}

int cpmdc_set_params(const void *params_capnp, size_t params_capnp_size_bytes) {
  (void)params_capnp;
  (void)params_capnp_size_bytes;
  return -1;
}

CPMDCResult cpmdc_energy_gradient(int n_atoms, const double *positions_ang,
                                  const int *atomic_numbers,
                                  const void *params_capnp,
                                  size_t params_capnp_size_bytes,
                                  double *grad_h_bohr) {
  (void)n_atoms;
  (void)positions_ang;
  (void)atomic_numbers;
  (void)params_capnp;
  (void)params_capnp_size_bytes;
  (void)grad_h_bohr;
  return stub_fail();
}

CPMDCResult cpmdc_energy(int n_atoms, const double *positions_ang,
                         const int *atomic_numbers, const void *params_capnp,
                         size_t params_capnp_size_bytes) {
  (void)n_atoms;
  (void)positions_ang;
  (void)atomic_numbers;
  (void)params_capnp;
  (void)params_capnp_size_bytes;
  return stub_fail();
}

CPMDCResult cpmdc_energy_forces(int n_atoms, const double *positions_ang,
                                const int *atomic_numbers,
                                const void *params_capnp,
                                size_t params_capnp_size_bytes,
                                double *forces_h_bohr) {
  (void)n_atoms;
  (void)positions_ang;
  (void)atomic_numbers;
  (void)params_capnp;
  (void)params_capnp_size_bytes;
  (void)forces_h_bohr;
  return stub_fail();
}

const char *cpmdc_version(void) { return "cpmdc-stub/0.1.0"; }

int cpmdc_available(void) { return 0; }

void cpmdc_finalize(void) {}

CPMDCSession *cpmdc_session_create(const void *params_capnp,
                                   size_t params_capnp_size_bytes) {
  (void)params_capnp;
  (void)params_capnp_size_bytes;
  return NULL;
}

int cpmdc_session_set_params(CPMDCSession *session, const void *params_capnp,
                             size_t params_capnp_size_bytes) {
  (void)session;
  (void)params_capnp;
  (void)params_capnp_size_bytes;
  return -1;
}

void cpmdc_session_destroy(CPMDCSession *session) { (void)session; }

int cpmdc_session_select_orbitals(CPMDCSession *session, long long key) {
  (void)session;
  (void)key;
  return -1;
}

int cpmdc_session_copy_orbitals(CPMDCSession *session, long long from_key,
                                long long to_key) {
  (void)session;
  (void)from_key;
  (void)to_key;
  return -1;
}

CPMDCResult cpmdc_session_energy(CPMDCSession *session, int n_atoms,
                                 const double *positions_ang,
                                 const int *atomic_numbers) {
  (void)session;
  (void)n_atoms;
  (void)positions_ang;
  (void)atomic_numbers;
  return stub_fail();
}

CPMDCResult cpmdc_session_energy_gradient(CPMDCSession *session, int n_atoms,
                                          const double *positions_ang,
                                          const int *atomic_numbers,
                                          double *grad_h_bohr) {
  (void)session;
  (void)n_atoms;
  (void)positions_ang;
  (void)atomic_numbers;
  (void)grad_h_bohr;
  return stub_fail();
}

CPMDCResult cpmdc_session_energy_forces(CPMDCSession *session, int n_atoms,
                                        const double *positions_ang,
                                        const int *atomic_numbers,
                                        double *forces_h_bohr) {
  (void)session;
  (void)n_atoms;
  (void)positions_ang;
  (void)atomic_numbers;
  (void)forces_h_bohr;
  return stub_fail();
}

CPMDCResult cpmdc_session_calculate_forces(
    CPMDCSession *session, const void *force_input_capnp,
    size_t force_input_capnp_size_bytes, double *forces_h_bohr,
    size_t forces_len) {
  (void)session;
  (void)force_input_capnp;
  (void)force_input_capnp_size_bytes;
  (void)forces_h_bohr;
  (void)forces_len;
  return stub_fail();
}

CPMDCResult cpmdc_session_calculate_result(
    CPMDCSession *session, const void *force_input_capnp,
    size_t force_input_capnp_size_bytes, void *potential_result_capnp,
    size_t potential_result_capnp_capacity_bytes,
    size_t *potential_result_capnp_size_bytes) {
  (void)session;
  (void)force_input_capnp;
  (void)force_input_capnp_size_bytes;
  (void)potential_result_capnp;
  (void)potential_result_capnp_capacity_bytes;
  (void)potential_result_capnp_size_bytes;
  return stub_fail();
}

CPMDCResult cpmdc_calculate_result(const void *params_capnp,
                                   size_t params_capnp_size_bytes,
                                   const void *force_input_capnp,
                                   size_t force_input_capnp_size_bytes,
                                   void *potential_result_capnp,
                                   size_t potential_result_capnp_capacity_bytes,
                                   size_t *potential_result_capnp_size_bytes) {
  (void)params_capnp;
  (void)params_capnp_size_bytes;
  (void)force_input_capnp;
  (void)force_input_capnp_size_bytes;
  (void)potential_result_capnp;
  (void)potential_result_capnp_capacity_bytes;
  (void)potential_result_capnp_size_bytes;
  return stub_fail();
}

size_t cpmdc_potential_result_size_for_force_input(
    const void *force_input_capnp, size_t force_input_capnp_size_bytes) {
  (void)force_input_capnp;
  (void)force_input_capnp_size_bytes;
  return 0;
}

size_t cpmdc_feature_count(void) {
  return sizeof(g_features) / sizeof(g_features[0]);
}

const CPMDCFeatureEntry *cpmdc_feature_table(void) { return g_features; }

const CPMDCFeatureEntry *cpmdc_feature_find(const char *feature_id) {
  if (feature_id == NULL)
    return NULL;
  for (size_t i = 0; i < cpmdc_feature_count(); ++i) {
    if (strcmp(g_features[i].feature_id, feature_id) == 0)
      return &g_features[i];
  }
  return NULL;
}
