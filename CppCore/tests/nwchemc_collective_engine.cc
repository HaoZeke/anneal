// MIT License
#if defined(_WIN32) || defined(_WIN64)
#define RGPOT_NWCHEMC_BUILD
#define RGPOT_CPMDC_BUILD
#define RGPOT_CP2KC_BUILD
#endif
#include "rgpot/NWChemPot/nwchem_c_abi.h"
#include "rgpot/CPMDPot/cpmd_c_abi.h"
#include "rgpot/CP2KPot/cp2kc_abi.h"
#include "rgpot/rpc/Potentials.capnp.h"
#include <capnp/serialize.h>
#include <mpi.h>

#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <string>
#include <vector>

namespace {
constexpr CPMDCFeatureEntry kFeatures[] = {
    {"abi.cpmdc_set_params", CPMDC_FEATURE_ABI, 1, 0},
    {"abi.cpmdc_energy_gradient", CPMDC_FEATURE_ABI, 1, 0},
    {"abi.cpmdc_version", CPMDC_FEATURE_ABI, 1, 0},
    {"abi.cpmdc_available", CPMDC_FEATURE_ABI, 1, 0},
    {"abi.cpmdc_feature_count", CPMDC_FEATURE_ABI, 1, 0},
    {"abi.cpmdc_feature_table", CPMDC_FEATURE_ABI, 1, 0},
    {"abi.cpmdc_feature_find", CPMDC_FEATURE_ABI, 1, 0},
};
int charge = 0;
bool registered = false;
int calls = 0;
bool uses_mpi = false;
int rank = 0;
int size = 1;
void finish() {
  int finalized = 0;
  MPI_Finalized(&finalized);
  std::fprintf(stderr, "fixture finalize rank=%d mpi_live=%d calls=%d\n",
                rank, uses_mpi && !finalized, calls);
  if (uses_mpi && finalized)
    std::abort();
}
void setup() {
  int initialized = 0;
  MPI_Initialized(&initialized);
  uses_mpi = initialized != 0;
  if (uses_mpi) {
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &size);
  }
  if (!registered) {
    if (std::atexit(finish) != 0)
      std::abort();
    registered = true;
  }
}
}

extern "C" int nwchemc_set_params(const void *data, size_t bytes) {
  setup();
  try {
    capnp::FlatArrayMessageReader reader(kj::arrayPtr(
        static_cast<const capnp::word *>(data), bytes / sizeof(capnp::word)));
    auto parameters = reader.getRoot<NWChemParams>();
    charge = parameters.getCharge();
    if (parameters.getTitle() == "reject-worker" && rank == 1)
      return 1;
    return 0;
  } catch (...) {
    return 1;
  }
}
extern "C" NWChemCResult nwchemc_energy_gradient(
    int n, const double *positions, const int *numbers, const void *, size_t,
    double *gradient) {
  setup();
  ++calls;
  NWChemCResult result{};
  double checksum = charge;
  for (int i = 0; i < 3 * n; ++i)
    checksum += positions[i] * (i + 1);
  for (int i = 0; i < n; ++i)
    checksum += numbers[i];
  if (uses_mpi) {
    double minimum = 0;
    double maximum = 0;
    MPI_Allreduce(&checksum, &minimum, 1, MPI_DOUBLE, MPI_MIN, MPI_COMM_WORLD);
    MPI_Allreduce(&checksum, &maximum, 1, MPI_DOUBLE, MPI_MAX, MPI_COMM_WORLD);
    if (minimum != maximum) {
      std::snprintf(result.message, sizeof(result.message), "collective inputs differ");
      return result;
    }
  }
  if (rank == 1 && positions[0] == 13.0) {
    std::snprintf(result.message, sizeof(result.message), "fixture worker force failure");
    return result;
  }
  result.ok = 1;
  result.energy_h = charge;
  for (int i = 0; i < n; ++i)
    result.energy_h += 0.01 * numbers[i];
  for (int i = 0; i < 3 * n; ++i) {
    result.energy_h += 0.5 * positions[i] * positions[i];
    gradient[i] = 0.001 * (i + 1) + 0.01 * positions[i];
  }
  return result;
}
extern "C" const char *nwchemc_version() { return "nwchem-collective-fixture"; }
extern "C" int nwchemc_available() { return 1; }

extern "C" int cpmdc_set_params(const void *data, size_t bytes) {
  setup();
  try {
    capnp::FlatArrayMessageReader reader(kj::arrayPtr(
        static_cast<const capnp::word *>(data), bytes / sizeof(capnp::word)));
    auto parameters = reader.getRoot<CPMDParams>();
    charge = parameters.getCharge();
    return parameters.getTitle() == "reject-worker" && rank == 1 ? 1 : 0;
  } catch (...) {
    return 1;
  }
}
extern "C" CPMDCResult cpmdc_energy_gradient(
    int n, const double *positions, const int *numbers, const void *params,
    size_t bytes, double *gradient) {
  const auto source = nwchemc_energy_gradient(n, positions, numbers, params,
                                              bytes, gradient);
  CPMDCResult result{};
  result.ok = source.ok;
  result.energy_h = source.energy_h;
  std::snprintf(result.message, sizeof(result.message), "%s", source.message);
  return result;
}
extern "C" const char *cpmdc_version() { return "cpmd-collective-fixture"; }
extern "C" int cpmdc_available() { return 1; }

extern "C" size_t cpmdc_feature_count() {
  return sizeof(kFeatures) / sizeof(kFeatures[0]);
}
extern "C" const CPMDCFeatureEntry *cpmdc_feature_table() { return kFeatures; }
extern "C" const CPMDCFeatureEntry *cpmdc_feature_find(const char *feature_id) {
  if (feature_id == nullptr)
    return nullptr;
  for (const auto &feature : kFeatures) {
    if (std::strcmp(feature.feature_id, feature_id) == 0)
      return &feature;
  }
  return nullptr;
}

// cp2kc: the charge and the elements come from the deck CP2KPot renders.
// Positions are turned back into the caller's Angstrom values so the one
// oracle in MpiRpcContractClient covers every backend.
struct CP2KCSession {
  int charge = 0;
  std::vector<int> numbers;
};

extern "C" int cp2kc_abi_version() { return CP2KC_ABI_VERSION; }
extern "C" const char *cp2kc_version() { return "cp2k-collective-fixture"; }
extern "C" int cp2kc_available() { return 1; }
extern "C" CP2KCSession *cp2kc_session_create(const char *deck, const char *,
                                              char *, size_t) {
  setup();
  auto *session = new CP2KCSession();
  const std::string text = deck;
  const auto at = text.find("CHARGE ");
  if (at != std::string::npos)
    session->charge = std::atoi(text.c_str() + at + 7);
  const auto begin = text.find("&COORD\n");
  const auto end = text.find("&END COORD");
  for (auto line = begin; line != std::string::npos && line < end;) {
    line = text.find('\n', line);
    if (line == std::string::npos || ++line >= end)
      break;
    const auto first = text.find_first_not_of(' ', line);
    if (first < end && text.compare(first, 2, "H ") == 0)
      session->numbers.push_back(1);
    else if (first < end && text.compare(first, 2, "O ") == 0)
      session->numbers.push_back(8);
  }
  return session;
}
extern "C" int cp2kc_session_select_orbitals(CP2KCSession *, long long) {
  return 0;
}
extern "C" int cp2kc_session_calculate(CP2KCSession *session, int n,
                                       const double *pos_bohr, const double *,
                                       double *e_ha, double *f_ha_bohr,
                                       char *err, size_t err_len) {
  if (static_cast<int>(session->numbers.size()) != n) {
    std::snprintf(err, err_len, "deck has %zu atoms", session->numbers.size());
    return CP2KC_BAD_ARGUMENT;
  }
  std::vector<double> positions(3 * static_cast<std::size_t>(n));
  for (std::size_t i = 0; i < positions.size(); ++i) {
    positions[i] = pos_bohr[i] * 0.529177210903;
    const double rounded = std::round(positions[i] * 1e6) / 1e6;
    if (std::abs(positions[i] - rounded) < 1e-9)
      positions[i] = rounded;
  }
  charge = session->charge;
  std::vector<double> gradient(positions.size());
  const auto source =
      nwchemc_energy_gradient(n, positions.data(), session->numbers.data(),
                              nullptr, 0, gradient.data());
  if (!source.ok) {
    std::snprintf(err, err_len, "%s", source.message);
    return CP2KC_FATAL;
  }
  *e_ha = source.energy_h;
  for (std::size_t i = 0; i < gradient.size(); ++i)
    f_ha_bohr[i] = -gradient[i];
  return CP2KC_OK;
}
extern "C" void cp2kc_session_destroy(CP2KCSession *session) { delete session; }
