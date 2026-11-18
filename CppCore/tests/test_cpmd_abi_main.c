/*
 * @file test_cpmd_abi_main.c
 * @brief Link against cpmd_abi_stub; verify stub reports unavailable.
 */
#include "cpmd_c_abi.h"

#include <stdio.h>
#include <string.h>

int main(void) {
  if (cpmdc_available() != 0) {
    fprintf(stderr, "cpmdc stub should report unavailable\n");
    return 1;
  }
  const char *ver = cpmdc_version();
  if (!ver || strstr(ver, "stub") == NULL) {
    fprintf(stderr, "cpmdc stub version missing stub marker\n");
    return 1;
  }
  CPMDCResult result = cpmdc_energy_gradient(0, NULL, NULL, NULL, 0, NULL);
  if (result.ok != 0 || strstr(result.message, "stub") == NULL) {
    fprintf(stderr, "cpmdc stub gradient did not fail as stub\n");
    return 1;
  }
  if (cpmdc_set_params(NULL, 0) == 0) {
    fprintf(stderr, "cpmdc stub accepted null params\n");
    return 1;
  }
  if (cpmdc_session_create(NULL, 0) != NULL) {
    fprintf(stderr, "cpmdc stub created a session\n");
    return 1;
  }
  if (cpmdc_session_select_orbitals(NULL, 0) != -1) {
    fprintf(stderr, "cpmdc stub accepted orbital selection\n");
    return 1;
  }
  if (cpmdc_session_copy_orbitals(NULL, 0, 1) != -1) {
    fprintf(stderr, "cpmdc stub accepted orbital copy\n");
    return 1;
  }
  if (cpmdc_potential_result_size_for_force_input(NULL, 0) != 0) {
    fprintf(stderr, "cpmdc stub reported a result size\n");
    return 1;
  }
  if (cpmdc_feature_count() == 0 ||
      cpmdc_feature_table() == NULL ||
      cpmdc_feature_find("abi.cpmdc_feature_find") == NULL) {
    fprintf(stderr, "cpmdc stub feature discovery is unavailable\n");
    return 1;
  }
  size_t result_size = 123;
  result = cpmdc_session_calculate_result(NULL, NULL, 0, NULL, 0, &result_size);
  if (result.ok != 0 || strstr(result.message, "stub") == NULL) {
    fprintf(stderr, "cpmdc stub session result did not fail as stub\n");
    return 1;
  }
  cpmdc_session_destroy(NULL);
  return 0;
}
