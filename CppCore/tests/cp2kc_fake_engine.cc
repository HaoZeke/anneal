// MIT License
// Copyright 2023--present rgpot developers

// Fake libcp2kc_rgpot: the cp2kc ABI over a harmonic surface in atomic
// units, E = 1/2 k sum_i |r_i - r0_i|^2 with r0_i = (0.1 i, -0.2 i, 0.3 i)
// Bohr, so CP2KPot's unit handling can be checked against a closed form.
// It keeps one "force environment" per orbital key, as the real shim does,
// and counts what CP2KPot made it do.

#if defined(_WIN32) || defined(_WIN64)
#define RGPOT_CP2KC_BUILD
#define RGPOT_FAKE_ONLY __declspec(dllexport)
#else
#define RGPOT_FAKE_ONLY
#endif
#include "rgpot/CP2KPot/cp2kc_abi.h"

#include <array>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <fstream>
#include <map>
#include <string>

namespace {

constexpr double kSpring = 0.5; // Hartree/Bohr^2

struct FakeEnv {
  int natom = 0;
  bool has_cell = false;
  std::array<double, 9> cell{};
};

int g_session_creates = 0;
int g_env_creates = 0;
int g_set_cell_calls = 0;
int g_live_sessions = 0;
long long g_last_key = -1000000;
std::array<double, 9> g_last_cell{};
std::string g_last_deck;
std::string g_last_dir;

// Non-blank lines between &COORD and &END COORD.
int count_coord_rows(const std::string &deck) {
  const auto begin = deck.find("&COORD");
  const auto end = deck.find("&END COORD");
  if (begin == std::string::npos || end == std::string::npos || end < begin)
    return 0;
  const std::string body = deck.substr(begin, end - begin);
  int rows = 0;
  std::size_t at = body.find('\n');
  while (at != std::string::npos) {
    const auto next = body.find('\n', at + 1);
    const std::string line = body.substr(
        at + 1, next == std::string::npos ? std::string::npos : next - at - 1);
    if (line.find_first_not_of(" \t") != std::string::npos)
      ++rows;
    at = next;
  }
  return rows;
}

void write_error(char *err, size_t err_len, const char *text) {
  if (err && err_len > 0)
    std::snprintf(err, err_len, "%s", text);
}

} // namespace

struct CP2KCSession {
  std::string deck;
  int natom = 0;
  long long key = 0;
  bool named_key = false;
  std::map<long long, FakeEnv> envs;
};

extern "C" {

RGPOT_FAKE_ONLY int cp2kc_abi_version(void) { return CP2KC_ABI_VERSION; }
RGPOT_FAKE_ONLY const char *cp2kc_version(void) { return "cp2kc-fake"; }
RGPOT_FAKE_ONLY int cp2kc_available(void) { return 1; }

RGPOT_FAKE_ONLY CP2KCSession *cp2kc_session_create(const char *deck,
                                                   const char *dir, char *err,
                                                   size_t err_len) {
  if (!deck || !dir) {
    write_error(err, err_len, "fake: deck and dir are required");
    return nullptr;
  }
  std::ofstream out(std::string(dir) + "/input.inp");
  out << deck;
  out.close();
  if (!out) {
    write_error(err, err_len, "fake: cannot write input.inp");
    return nullptr;
  }
  auto *session = new CP2KCSession();
  session->deck = deck;
  session->natom = count_coord_rows(session->deck);
  g_last_deck = deck;
  g_last_dir = dir;
  ++g_session_creates;
  ++g_live_sessions;
  return session;
}

RGPOT_FAKE_ONLY int cp2kc_session_select_orbitals(CP2KCSession *session,
                                                  long long key) {
  if (!session)
    return CP2KC_BAD_ARGUMENT;
  session->key = key;
  session->named_key = true;
  return CP2KC_OK;
}

RGPOT_FAKE_ONLY int cp2kc_session_calculate(CP2KCSession *session, int n_atoms,
                                            const double *pos_bohr,
                                            const double *cell_bohr9,
                                            double *e_ha, double *f_ha_bohr,
                                            char *err, size_t err_len) {
  if (!session || n_atoms <= 0 || !pos_bohr || !cell_bohr9 || !e_ha ||
      !f_ha_bohr) {
    write_error(err, err_len, "fake: null argument");
    return CP2KC_BAD_ARGUMENT;
  }
  auto found = session->envs.find(session->key);
  if (found == session->envs.end()) {
    found = session->envs.emplace(session->key, FakeEnv{}).first;
    found->second.natom = session->natom;
    ++g_env_creates;
  }
  FakeEnv &env = found->second;
  if (env.natom != n_atoms) {
    write_error(err, err_len, "fake: atom count differs from the deck");
    return CP2KC_BAD_ARGUMENT;
  }
  g_last_key = session->named_key ? session->key : -1000000;
  if (!env.has_cell ||
      std::memcmp(env.cell.data(), cell_bohr9, sizeof(double) * 9) != 0) {
    std::memcpy(env.cell.data(), cell_bohr9, sizeof(double) * 9);
    env.has_cell = true;
    ++g_set_cell_calls;
  }
  std::memcpy(g_last_cell.data(), cell_bohr9, sizeof(double) * 9);
  if (const char *flag = std::getenv("RGPOT_CP2K_FAKE_SCF_FAIL");
      flag && std::strcmp(flag, "1") == 0) {
    write_error(err, err_len, "fake: SCF run NOT converged");
    return CP2KC_SCF_NOT_CONVERGED;
  }
  double energy = 0.0;
  for (int i = 0; i < n_atoms; ++i) {
    const double r0[3] = {0.1 * i, -0.2 * i, 0.3 * i};
    for (int k = 0; k < 3; ++k) {
      const double d = pos_bohr[3 * i + k] - r0[k];
      energy += 0.5 * kSpring * d * d;
      f_ha_bohr[3 * i + k] = -kSpring * d;
    }
  }
  *e_ha = energy;
  return CP2KC_OK;
}

RGPOT_FAKE_ONLY void cp2kc_session_destroy(CP2KCSession *session) {
  if (session)
    --g_live_sessions;
  delete session;
}

// Test-only counters and records.
RGPOT_FAKE_ONLY int cp2kc_fake_session_creates(void) {
  return g_session_creates;
}
RGPOT_FAKE_ONLY int cp2kc_fake_env_creates(void) { return g_env_creates; }
RGPOT_FAKE_ONLY int cp2kc_fake_set_cell_calls(void) { return g_set_cell_calls; }
RGPOT_FAKE_ONLY int cp2kc_fake_live_sessions(void) { return g_live_sessions; }
RGPOT_FAKE_ONLY long long cp2kc_fake_last_key(void) { return g_last_key; }
RGPOT_FAKE_ONLY void cp2kc_fake_last_cell(double *out9) {
  std::memcpy(out9, g_last_cell.data(), sizeof(double) * 9);
}
RGPOT_FAKE_ONLY const char *cp2kc_fake_last_deck(void) {
  return g_last_deck.c_str();
}
RGPOT_FAKE_ONLY const char *cp2kc_fake_last_dir(void) {
  return g_last_dir.c_str();
}

} // extern "C"
