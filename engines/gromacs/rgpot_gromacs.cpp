// MIT License
// Copyright 2023--present rgpot developers
//
// librgpot_gromacs: any rgpot potential behind the libcp2k C API of the
// GROMACS QM/MM interface.
//
// GROMACS built with -DGMX_CP2K=ON (checked: 2024.6, 2025.5, 2026.4) calls
// libcp2k once per MD step from QMMMForceProvider
// (src/gromacs/applied_forces/qmmm/qmmmforceprovider.cpp): set the
// positions of every QM and MM atom and the box, evaluate, read the energy
// and the forces. This library answers with one rgpot evaluation of the QM
// atoms. GROMACS adds the energy to "Quantum En." (F_EQM), which enters
// the potential and the conserved energy, and the forces to its
// ForceWithVirial buffer, which carries no virial.
//
// The input is the file of grompp -qmi with qmmm-cp2k-qmmethod = INPUT.
// grompp checks it for CHARGE, MULTIPLICITY, RUN_TYPE ENERGY_FORCE,
// COORD_FILE_FORMAT PDB, CHARGE_EXTENDED TRUE and COORD_FILE_NAME, and
// mdrun writes it out with the name of the PDB file it generates. That
// PDB gives each atom's element (the atomic number of its GROMACS atom
// type), whether it is QM or MM, and each MM atom's charge. The rgpot
// settings sit in an &RGPOT section:
//
//   POTENTIAL Name         a potential rgpot::spec::build knows (LJ, Morse,
//                          SuttonChen, EAMTable, TersoffSi, ...)
//   OPTIONS key value ...  its options, Angstrom and eV; may repeat
//   ENGINE path            or an rgpot_engine_* library (UMA, Metatomic,
//                          XTB, classical, ...)
//   CONFIG path            the engine's Cap'n Proto params message
//   CELL PERIODIC|NONE     the GROMACS box as the cell, or none
//   EMBEDDING NONE         accept MM charges that do not act on QM atoms
//
// Errors cannot be returned through this API: the library prints
// "rgpot-gromacs: ..." to stderr and to the output file GROMACS names, and
// exits with status 1.

#include "libcp2k.h"
#include "pot_spec.hpp"
#include "qmmm_units.hpp"

#include "rgpot/engine_c_abi.h"

#include <dlfcn.h>

#include <algorithm>
#include <array>
#include <cctype>
#include <cmath>
#include <cstdarg>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <exception>
#include <fstream>
#include <memory>
#include <sstream>
#include <string>
#include <vector>

#ifndef RGPOT_GROMACS_VERSION
#define RGPOT_GROMACS_VERSION "unknown"
#endif

#define RGPOT_GMX_API extern "C" __attribute__((visibility("default")))

namespace {

namespace units = rgpot::gromacs;

constexpr const char *kProg = "rgpot-gromacs";
constexpr int kMaxEnvs = 16;

enum class Cell { Default, Periodic, None };

using EngineCreate = RgpotEnginePot *(*)(const void *, size_t, char *, size_t);
using EngineDestroy = void (*)(RgpotEnginePot *);
using EngineForce = int (*)(RgpotEnginePot *, long, const double *, const int *, double *,
                            double *, double *, const double *, rgpot_engine_coord_transform,
                            void *);
using EngineCaps = int (*)(const RgpotEnginePot *, RgpotEngineCaps *);
using EngineLastError = size_t (*)(const RgpotEnginePot *, char *, size_t);
using EngineChargeSpin = int (*)(RgpotEnginePot *, int, int);

struct Engine {
  void *library = nullptr;
  RgpotEnginePot *pot = nullptr;
  EngineDestroy destroy = nullptr;
  EngineForce force = nullptr;
  EngineLastError last_error = nullptr;
};

struct Env {
  FILE *log = nullptr;
  std::string input;

  // &RGPOT
  std::string potential;
  rgpot::spec::Options options;
  std::string engine_path;
  std::string config_path;
  Cell cell = Cell::Default;
  bool embedding_none = false;
  int charge = 0;
  int multiplicity = 1;

  rgpot::spec::Built built;
  Engine engine;
  std::string backend;
  bool periodic = false;
  double cutoff = 0.0;

  // From the PDB file.
  int natoms = 0;
  std::vector<int> qm_index;
  std::vector<int> qm_z;
  int nmm = 0;
  int charged_mm = 0;

  // GROMACS' last call, Bohr.
  std::vector<double> pos;
  double rows[3][3] = {};
  bool have_positions = false;
  bool have_cell = false;

  // QM atoms in Angstrom, the previous step's unwrapped positions and the
  // result in CP2K units for every atom.
  std::vector<double> xyz;
  std::vector<double> prev;
  bool have_prev = false;
  std::vector<double> qm_forces;
  std::vector<double> forces;
  double energy = 0.0;
  long evaluated = 0;
};

std::array<std::unique_ptr<Env>, kMaxEnvs> g_envs;

// --- errors -----------------------------------------------------------

[[noreturn]] void die(Env *env, const char *fmt, ...) __attribute__((format(printf, 2, 3)));

[[noreturn]] void die(Env *env, const char *fmt, ...) {
  char text[4096];
  va_list ap;
  va_start(ap, fmt);
  std::vsnprintf(text, sizeof(text), fmt, ap);
  va_end(ap);
  std::fprintf(stderr, "%s: %s\n", kProg, text);
  std::fflush(stderr);
  if (env != nullptr && env->log != nullptr) {
    std::fprintf(env->log, "%s: error: %s\n", kProg, text);
    std::fflush(env->log);
  }
  std::exit(EXIT_FAILURE);
}

Env *lookup(force_env_t id, const char *call) {
  if (id < 1 || id > kMaxEnvs || !g_envs[id - 1]) {
    die(nullptr, "%s: no force environment %d", call, id);
  }
  return g_envs[id - 1].get();
}

// --- input ------------------------------------------------------------

std::string upper(std::string s) {
  std::transform(s.begin(), s.end(), s.begin(),
                 [](unsigned char c) { return static_cast<char>(std::toupper(c)); });
  return s;
}

std::string readFile(Env *env, const std::string &path, const char *what) {
  std::ifstream in(path, std::ios::binary);
  if (!in) {
    die(env, "cannot open %s %s", what, path.c_str());
  }
  std::ostringstream text;
  text << in.rdbuf();
  return text.str();
}

/// Words of one line; # and ! start a comment, as in CP2K input.
std::vector<std::string> words(const std::string &line) {
  std::string body = line.substr(0, line.find_first_of("#!"));
  std::istringstream in(body);
  std::vector<std::string> out;
  for (std::string w; in >> w;) {
    out.push_back(w);
  }
  return out;
}

int integer(Env *env, const std::string &text, const char *key) {
  std::size_t end = 0;
  int v = 0;
  try {
    v = std::stoi(text, &end);
  } catch (const std::exception &) {
    end = 0;
  }
  if (end == 0 || end != text.size()) {
    die(env, "%s %s is not an integer", key, text.c_str());
  }
  return v;
}

void parseInput(Env *env, std::string *pdb_path) {
  const std::string text = readFile(env, env->input, "the QM/MM input");
  std::istringstream lines(text);
  bool in_rgpot = false;
  bool saw_rgpot = false;
  bool saw_charge = false;
  bool saw_multiplicity = false;
  std::vector<std::string> seen;
  for (std::string line; std::getline(lines, line);) {
    const std::vector<std::string> w = words(line);
    if (w.empty()) {
      continue;
    }
    const std::string key = upper(w[0]);
    if (key == "&LINK") {
      die(env,
          "%s has a &LINK section: QM-MM covalent bonds need link atoms, which "
          "rgpot-gromacs does not provide; make the QM group whole molecules",
          env->input.c_str());
    }
    if (!in_rgpot) {
      if (key == "&RGPOT") {
        if (saw_rgpot) {
          die(env, "%s has two &RGPOT sections", env->input.c_str());
        }
        in_rgpot = true;
        saw_rgpot = true;
      } else if (key == "COORD_FILE_NAME" && w.size() >= 2) {
        *pdb_path = w[1];
      } else if (key == "CHARGE" && w.size() >= 2 && !saw_charge) {
        // grompp reads the first CHARGE and MULTIPLICITY of the file.
        env->charge = integer(env, w[1], "CHARGE");
        saw_charge = true;
      } else if (key == "MULTIPLICITY" && w.size() >= 2 && !saw_multiplicity) {
        env->multiplicity = integer(env, w[1], "MULTIPLICITY");
        saw_multiplicity = true;
      }
      continue;
    }
    if (key == "&END") {
      in_rgpot = false;
      continue;
    }
    if (key[0] == '&') {
      die(env, "&RGPOT takes no subsections; found %s", w[0].c_str());
    }
    if (key != "OPTIONS") {
      if (std::find(seen.begin(), seen.end(), key) != seen.end()) {
        die(env, "&RGPOT %s is given twice", key.c_str());
      }
      seen.push_back(key);
    }
    if (key == "OPTIONS") {
      if (w.size() < 3 || (w.size() - 1) % 2 != 0) {
        die(env, "&RGPOT OPTIONS takes key value pairs, got: %s", line.c_str());
      }
      for (std::size_t k = 1; k + 1 < w.size(); k += 2) {
        env->options.emplace_back(w[k], w[k + 1]);
      }
      continue;
    }
    if (w.size() != 2) {
      die(env, "&RGPOT %s takes one value", key.c_str());
    }
    const std::string &value = w[1];
    if (key == "POTENTIAL") {
      env->potential = value;
    } else if (key == "ENGINE") {
      env->engine_path = value;
    } else if (key == "CONFIG") {
      env->config_path = value;
    } else if (key == "CELL") {
      const std::string v = upper(value);
      if (v == "PERIODIC") {
        env->cell = Cell::Periodic;
      } else if (v == "NONE") {
        env->cell = Cell::None;
      } else {
        die(env, "&RGPOT CELL is PERIODIC or NONE, not %s", value.c_str());
      }
    } else if (key == "EMBEDDING") {
      if (upper(value) != "NONE") {
        die(env,
            "&RGPOT EMBEDDING %s is not implemented: rgpot potentials take no "
            "external charges; EMBEDDING NONE is the only value",
            value.c_str());
      }
      env->embedding_none = true;
    } else {
      die(env,
          "unknown &RGPOT keyword %s (POTENTIAL, OPTIONS, ENGINE, CONFIG, CELL, "
          "EMBEDDING)",
          w[0].c_str());
    }
  }
  if (!saw_rgpot) {
    die(env,
        "%s has no &RGPOT section. Set qmmm-cp2k-qmmethod = INPUT and pass the "
        "rgpot input to grompp with -qmi",
        env->input.c_str());
  }
  if (in_rgpot) {
    die(env, "the &RGPOT section of %s has no &END", env->input.c_str());
  }
  if (env->potential.empty() == env->engine_path.empty()) {
    die(env, "&RGPOT needs exactly one of POTENTIAL and ENGINE");
  }
  if (!env->config_path.empty() && env->engine_path.empty()) {
    die(env, "&RGPOT CONFIG belongs to ENGINE; POTENTIAL takes OPTIONS");
  }
  if (!env->options.empty() && env->potential.empty()) {
    die(env, "&RGPOT OPTIONS belong to POTENTIAL; an ENGINE takes a CONFIG message");
  }
  if (pdb_path->empty()) {
    die(env, "%s names no COORD_FILE_NAME", env->input.c_str());
  }
}

int elementNumber(const std::string &symbol) {
  if (symbol.empty()) {
    return 0;
  }
  std::string s = symbol;
  std::transform(s.begin(), s.end(), s.begin(),
                 [](unsigned char c) { return static_cast<char>(std::tolower(c)); });
  s[0] = static_cast<char>(std::toupper(static_cast<unsigned char>(s[0])));
  return rgpot::spec::atomicNumber(s);
}

void parsePdb(Env *env, const std::string &path) {
  const std::string text = readFile(env, path, "the QM/MM PDB file");
  std::istringstream lines(text);
  int n = 0;
  for (std::string line; std::getline(lines, line);) {
    if (line.rfind("ATOM", 0) != 0 && line.rfind("HETATM", 0) != 0) {
      continue;
    }
    const std::vector<std::string> w = words(line);
    if (w.size() < 6) {
      die(env, "%s: cannot read atom line %d", path.c_str(), n + 1);
    }
    const bool qm = w[3] == "QM";
    if (!qm && w[3] != "MM") {
      die(env, "%s: atom %d is neither QM nor MM", path.c_str(), n + 1);
    }
    const std::string &symbol = w[w.size() - 2];
    char *end = nullptr;
    const double q = std::strtod(w.back().c_str(), &end);
    if (end == w.back().c_str() || *end != '\0') {
      die(env, "%s: atom %d has no charge", path.c_str(), n + 1);
    }
    if (qm) {
      const int z = elementNumber(symbol);
      if (z == 0) {
        die(env,
            "QM atom %d is %s, not an element: give its GROMACS atom type an "
            "atomic number and keep virtual sites out of the QM group",
            n + 1, symbol.c_str());
      }
      env->qm_index.push_back(n);
      env->qm_z.push_back(z);
    } else {
      ++env->nmm;
      env->charged_mm += q != 0.0 ? 1 : 0;
    }
    ++n;
  }
  env->natoms = n;
  if (env->qm_index.empty()) {
    die(env, "%s has no QM atoms", path.c_str());
  }
}

// --- backend ----------------------------------------------------------

void *symbol(Env *env, const char *name, bool required) {
  void *p = dlsym(env->engine.library, name);
  if (p == nullptr && required) {
    die(env, "ENGINE %s has no %s", env->engine_path.c_str(), name);
  }
  return p;
}

bool openEngine(Env *env) {
  env->engine.library = dlopen(env->engine_path.c_str(), RTLD_NOW | RTLD_LOCAL);
  if (env->engine.library == nullptr) {
    const char *why = dlerror();
    die(env, "cannot open ENGINE %s: %s", env->engine_path.c_str(),
        why != nullptr ? why : "dlopen failed");
  }
  auto version = reinterpret_cast<int (*)()>(symbol(env, "rgpot_engine_abi_version", true));
  if (version() != RGPOT_ENGINE_ABI_VERSION) {
    die(env, "ENGINE %s speaks engine ABI %d; this library needs %d",
        env->engine_path.c_str(), version(), RGPOT_ENGINE_ABI_VERSION);
  }
  auto create = reinterpret_cast<EngineCreate>(symbol(env, "rgpot_engine_create", true));
  env->engine.destroy = reinterpret_cast<EngineDestroy>(symbol(env, "rgpot_engine_destroy", true));
  env->engine.force = reinterpret_cast<EngineForce>(symbol(env, "rgpot_engine_force", true));
  env->engine.last_error =
      reinterpret_cast<EngineLastError>(symbol(env, "rgpot_engine_last_error", false));
  std::string config;
  if (!env->config_path.empty()) {
    config = readFile(env, env->config_path, "the engine CONFIG message");
    if (config.empty()) {
      die(env, "the engine CONFIG message %s is empty", env->config_path.c_str());
    }
  }
  // Cap'n Proto reads the message as 8-byte words.
  std::vector<std::uint64_t> aligned((config.size() + 7) / 8, 0);
  std::memcpy(aligned.data(), config.data(), config.size());
  char why[1024] = {};
  env->engine.pot = create(config.empty() ? nullptr : aligned.data(), config.size(), why,
                           sizeof(why));
  if (env->engine.pot == nullptr) {
    die(env, "ENGINE %s: %s", env->engine_path.c_str(),
        why[0] != '\0' ? why : "rgpot_engine_create returned null");
  }
  bool periodic = true;
  if (auto caps = reinterpret_cast<EngineCaps>(symbol(env, "rgpot_engine_caps", false))) {
    RgpotEngineCaps c{};
    c.size = sizeof(c);
    if (caps(env->engine.pot, &c) == 0) {
      periodic = c.periodic != 0;
    }
  }
  auto charge_spin =
      reinterpret_cast<EngineChargeSpin>(symbol(env, "rgpot_engine_set_charge_spin", false));
  const int status = charge_spin != nullptr
                         ? charge_spin(env->engine.pot, env->charge, env->multiplicity)
                         : 3;
  if (status == 3 && (env->charge != 0 || env->multiplicity != 1)) {
    die(env,
        "ENGINE %s takes no charge or spin, so the input needs CHARGE 0 and "
        "MULTIPLICITY 1 (it has %d and %d)",
        env->engine_path.c_str(), env->charge, env->multiplicity);
  }
  if (status != 0 && status != 3) {
    char text[1024] = {};
    if (env->engine.last_error != nullptr) {
      env->engine.last_error(env->engine.pot, text, sizeof(text));
    }
    die(env, "ENGINE %s refused CHARGE %d MULTIPLICITY %d: %s", env->engine_path.c_str(),
        env->charge, env->multiplicity, text);
  }
  const char *name = nullptr;
  if (auto get = reinterpret_cast<const char *(*)()>(symbol(env, "rgpot_engine_name", false))) {
    name = get();
  }
  env->backend = std::string("engine ") + (name != nullptr ? name : "") + " (" +
                 env->engine_path + ")";
  return periodic;
}

bool openPotential(Env *env) {
  if (env->charge != 0 || env->multiplicity != 1) {
    die(env,
        "POTENTIAL %s takes no charge or spin, so the input needs CHARGE 0 and "
        "MULTIPLICITY 1 (it has %d and %d)",
        env->potential.c_str(), env->charge, env->multiplicity);
  }
  env->built = rgpot::spec::build(env->potential, env->options, 0.0);
  if (!env->built.pot) {
    die(env, "POTENTIAL %s: %s", env->potential.c_str(), env->built.error.c_str());
  }
  env->cutoff = env->built.cutoff;
  env->backend = "potential " + env->potential;
  return env->built.pot->caps().periodic;
}

void openBackend(Env *env) {
  env->periodic = env->engine_path.empty() ? openPotential(env) : openEngine(env);
  if (env->cell == Cell::Periodic && !env->periodic) {
    die(env, "%s has free boundaries; use CELL NONE", env->backend.c_str());
  }
  if (env->charged_mm > 0 && !env->embedding_none) {
    die(env,
        "%d MM atoms carry charges, and grompp has zeroed the QM charges, so no "
        "QM-MM Coulomb term would act. rgpot potentials take no external charges. "
        "Set &RGPOT EMBEDDING NONE to run without it",
        env->charged_mm);
  }
}

// --- geometry ---------------------------------------------------------

double det3(const double m[3][3]) {
  return m[0][0] * (m[1][1] * m[2][2] - m[1][2] * m[2][1]) -
         m[0][1] * (m[1][0] * m[2][2] - m[1][2] * m[2][0]) +
         m[0][2] * (m[1][0] * m[2][1] - m[1][1] * m[2][0]);
}

/// inv with s = x inv for row vectors x and lattice vectors as rows.
bool invert3(const double m[3][3], double inv[3][3]) {
  const double d = det3(m);
  if (d == 0.0 || !std::isfinite(d)) {
    return false;
  }
  inv[0][0] = (m[1][1] * m[2][2] - m[1][2] * m[2][1]) / d;
  inv[0][1] = (m[0][2] * m[2][1] - m[0][1] * m[2][2]) / d;
  inv[0][2] = (m[0][1] * m[1][2] - m[0][2] * m[1][1]) / d;
  inv[1][0] = (m[1][2] * m[2][0] - m[1][0] * m[2][2]) / d;
  inv[1][1] = (m[0][0] * m[2][2] - m[0][2] * m[2][0]) / d;
  inv[1][2] = (m[0][2] * m[1][0] - m[0][0] * m[1][2]) / d;
  inv[2][0] = (m[1][0] * m[2][1] - m[1][1] * m[2][0]) / d;
  inv[2][1] = (m[0][1] * m[2][0] - m[0][0] * m[2][1]) / d;
  inv[2][2] = (m[0][0] * m[1][1] - m[0][1] * m[1][0]) / d;
  return true;
}

void scaled(const double inv[3][3], const double *x, double s[3]) {
  for (int a = 0; a < 3; ++a) {
    s[a] = x[0] * inv[0][a] + x[1] * inv[1][a] + x[2] * inv[2][a];
  }
}

/// Moves x by the lattice vector that brings it nearest to ref.
void nearestImage(const double rows[3][3], const double inv[3][3], const double *ref, double *x) {
  double d[3] = {x[0] - ref[0], x[1] - ref[1], x[2] - ref[2]};
  double s[3];
  scaled(inv, d, s);
  for (double &v : s) {
    v = std::nearbyint(v);
  }
  for (int k = 0; k < 3; ++k) {
    x[k] -= s[0] * rows[0][k] + s[1] * rows[1][k] + s[2] * rows[2][k];
  }
}

// --- one evaluation ---------------------------------------------------

void evaluate(Env *env) {
  const std::size_t nq = env->qm_index.size();
  double rows[3][3];
  double inv[3][3] = {};
  for (int a = 0; a < 3; ++a) {
    for (int k = 0; k < 3; ++k) {
      rows[a][k] = env->rows[a][k] * units::kAngstromPerBohr;
    }
  }
  if (env->have_cell && !invert3(rows, inv)) {
    die(env, "step %ld: the GROMACS box is singular", env->evaluated);
  }
  if (env->cell == Cell::Default) {
    env->cell = env->periodic && env->have_cell ? Cell::Periodic : Cell::None;
    if (env->log != nullptr) {
      std::fprintf(env->log, "cell %s\n", env->cell == Cell::Periodic ? "PERIODIC" : "NONE");
    }
  }
  if (env->cell == Cell::Periodic && !env->have_cell) {
    die(env, "CELL PERIODIC needs a GROMACS box; use pbc = xyz or CELL NONE");
  }
  for (std::size_t i = 0; i < nq; ++i) {
    const double *p = &env->pos[3 * static_cast<std::size_t>(env->qm_index[i])];
    for (int k = 0; k < 3; ++k) {
      env->xyz[3 * i + k] = p[k] * units::kAngstromPerBohr;
    }
  }

  double box[9] = {};
  if (env->cell == Cell::Periodic) {
    for (std::size_t i = 0; i < nq; ++i) {
      double s[3];
      scaled(inv, &env->xyz[3 * i], s);
      for (double &v : s) {
        v -= std::floor(v);
      }
      for (int k = 0; k < 3; ++k) {
        env->xyz[3 * i + k] = s[0] * rows[0][k] + s[1] * rows[1][k] + s[2] * rows[2][k];
      }
    }
    for (int a = 0; a < 3; ++a) {
      for (int k = 0; k < 3; ++k) {
        box[3 * a + k] = rows[a][k];
      }
    }
  } else {
    // GROMACS puts every atom in the box, so a QM molecule can arrive split.
    // Each atom follows the image nearest its last position; the first step
    // makes the region whole about its first atom.
    if (env->have_cell) {
      for (std::size_t i = 0; i < nq; ++i) {
        const double *ref = env->have_prev ? &env->prev[3 * i] : &env->xyz[0];
        nearestImage(rows, inv, ref, &env->xyz[3 * i]);
      }
    }
    std::copy(env->xyz.begin(), env->xyz.end(), env->prev.begin());
    env->have_prev = true;
    // Kernels apply the minimum image to the cell they are given. A cube
    // wider than twice the region plus the cutoff keeps every real pair and
    // no periodic image.
    double lo[3] = {env->xyz[0], env->xyz[1], env->xyz[2]};
    double hi[3] = {lo[0], lo[1], lo[2]};
    for (std::size_t i = 0; i < nq; ++i) {
      for (int k = 0; k < 3; ++k) {
        lo[k] = std::min(lo[k], env->xyz[3 * i + k]);
        hi[k] = std::max(hi[k], env->xyz[3 * i + k]);
      }
    }
    double extent = 0.0;
    for (int k = 0; k < 3; ++k) {
      extent = std::max(extent, hi[k] - lo[k]);
    }
    const double side = std::max(100.0, 2.0 * (extent + env->cutoff) + 10.0);
    for (std::size_t i = 0; i < nq; ++i) {
      for (int k = 0; k < 3; ++k) {
        env->xyz[3 * i + k] += 0.5 * side - 0.5 * (lo[k] + hi[k]);
      }
    }
    box[0] = box[4] = box[8] = side;
  }

  std::fill(env->qm_forces.begin(), env->qm_forces.end(), 0.0);
  double energy = 0.0;
  if (env->engine.pot != nullptr) {
    double variance = 0.0;
    const int status = env->engine.force(env->engine.pot, static_cast<long>(nq),
                                         env->xyz.data(), env->qm_z.data(),
                                         env->qm_forces.data(), &energy, &variance, box,
                                         nullptr, nullptr);
    if (status != 0) {
      char text[1024] = {};
      if (env->engine.last_error != nullptr) {
        env->engine.last_error(env->engine.pot, text, sizeof(text));
      }
      die(env, "step %ld: %s returned %d: %s", env->evaluated, env->backend.c_str(), status,
          text);
    }
  } else {
    rgpot::ForceInput in{nq, env->xyz.data(), env->qm_z.data(), box};
    rgpot::ForceOut out{env->qm_forces.data(), 0.0, 0.0, {}, 0};
    try {
      env->built.eval(in, &out);
    } catch (const std::exception &e) {
      die(env, "step %ld: %s: %s", env->evaluated, env->backend.c_str(), e.what());
    } catch (...) {
      die(env, "step %ld: %s failed with a non-standard exception", env->evaluated,
          env->backend.c_str());
    }
    energy = out.energy;
  }
  if (!std::isfinite(energy)) {
    die(env, "step %ld: %s returned a non-finite energy", env->evaluated, env->backend.c_str());
  }

  std::fill(env->forces.begin(), env->forces.end(), 0.0);
  for (std::size_t i = 0; i < nq; ++i) {
    double *f = &env->forces[3 * static_cast<std::size_t>(env->qm_index[i])];
    for (int k = 0; k < 3; ++k) {
      f[k] = env->qm_forces[3 * i + k] * units::kHartreeBohrPerEvAngstrom;
    }
  }
  env->energy = energy * units::kHartreePerEv;
  if (env->log != nullptr && env->evaluated == 0) {
    std::fprintf(env->log, "first step: energy_ev %.17g\n", energy);
    std::fflush(env->log);
  }
}

void create(force_env_t *out, const char *input, const char *output) {
  int slot = -1;
  for (int i = 0; i < kMaxEnvs && slot < 0; ++i) {
    if (!g_envs[i]) {
      slot = i;
    }
  }
  if (slot < 0) {
    die(nullptr, "more than %d force environments", kMaxEnvs);
  }
  if (input == nullptr || *input == '\0') {
    die(nullptr, "cp2k_create_force_env got no input file");
  }
  g_envs[slot] = std::make_unique<Env>();
  Env *env = g_envs[slot].get();
  env->input = input;
  if (output != nullptr && *output != '\0') {
    env->log = std::fopen(output, "w");
    if (env->log != nullptr) {
      std::fprintf(env->log, "%s %s, input %s\n", kProg, RGPOT_GROMACS_VERSION, input);
    }
  }
  std::string pdb;
  parseInput(env, &pdb);
  parsePdb(env, pdb);
  openBackend(env);
  const std::size_t nq = env->qm_index.size();
  env->pos.assign(3 * static_cast<std::size_t>(env->natoms), 0.0);
  env->forces.assign(3 * static_cast<std::size_t>(env->natoms), 0.0);
  env->xyz.assign(3 * nq, 0.0);
  env->prev.assign(3 * nq, 0.0);
  env->qm_forces.assign(3 * nq, 0.0);
  if (env->log != nullptr) {
    std::fprintf(env->log, "%s\natoms %d qm %zu mm %d charged_mm %d charge %d multiplicity %d\n",
                 env->backend.c_str(), env->natoms, nq, env->nmm, env->charged_mm, env->charge,
                 env->multiplicity);
    if (env->cell != Cell::Default) {
      std::fprintf(env->log, "cell %s\n", env->cell == Cell::Periodic ? "PERIODIC" : "NONE");
    }
    std::fflush(env->log);
  }
  *out = slot + 1;
}

void checkCount(Env *env, int n_el, const char *call) {
  if (n_el != 3 * env->natoms) {
    die(env, "%s got %d values; the PDB file has %d atoms (%d values)", call, n_el,
        env->natoms, 3 * env->natoms);
  }
}

} // namespace

// --- the libcp2k API ----------------------------------------------------

RGPOT_GMX_API void cp2k_get_version(char *version_str, int str_length) {
  if (version_str != nullptr && str_length > 0) {
    std::snprintf(version_str, static_cast<std::size_t>(str_length), "rgpot %s",
                  RGPOT_GROMACS_VERSION);
  }
}

RGPOT_GMX_API void cp2k_init(void) {}

RGPOT_GMX_API void cp2k_init_without_mpi(void) {}

RGPOT_GMX_API void cp2k_finalize(void) {}

RGPOT_GMX_API void cp2k_finalize_without_mpi(void) {}

RGPOT_GMX_API void cp2k_create_force_env(force_env_t *new_force_env, const char *input_file_path,
                                         const char *output_file_path) {
  create(new_force_env, input_file_path, output_file_path);
}

RGPOT_GMX_API void cp2k_create_force_env_comm(force_env_t *, const char *, const char *, int) {
  die(nullptr,
      "GROMACS passed an MPI communicator: run a thread-MPI GROMACS with one rank "
      "(-ntmpi 1); librgpot_gromacs evaluates on one process");
}

RGPOT_GMX_API void cp2k_destroy_force_env(force_env_t force_env) {
  Env *env = lookup(force_env, "cp2k_destroy_force_env");
  if (env->engine.pot != nullptr) {
    env->engine.destroy(env->engine.pot);
  }
  if (env->log != nullptr) {
    std::fprintf(env->log, "evaluations %ld\n", env->evaluated);
    std::fclose(env->log);
  }
  // An engine's tensor runtime keeps process-wide state; the library stays open.
  g_envs[force_env - 1].reset();
}

RGPOT_GMX_API void cp2k_set_positions(force_env_t force_env, const double *new_pos, int n_el) {
  Env *env = lookup(force_env, "cp2k_set_positions");
  checkCount(env, n_el, "cp2k_set_positions");
  std::copy(new_pos, new_pos + n_el, env->pos.begin());
  env->have_positions = true;
}

RGPOT_GMX_API void cp2k_set_cell(force_env_t force_env, const double *new_cell) {
  Env *env = lookup(force_env, "cp2k_set_cell");
  // GROMACS 2026 sends box[i][j] at 3 i + j and 2024 and 2025 send it at
  // 3 j + i. A GROMACS box is lower triangular, so the zeros tell which.
  double m[3][3];
  bool any = false;
  for (int a = 0; a < 3; ++a) {
    for (int k = 0; k < 3; ++k) {
      m[a][k] = new_cell[3 * a + k];
      any = any || m[a][k] != 0.0;
    }
  }
  const bool lower = m[0][1] == 0.0 && m[0][2] == 0.0 && m[1][2] == 0.0;
  const bool upper = m[1][0] == 0.0 && m[2][0] == 0.0 && m[2][1] == 0.0;
  if (!lower && !upper) {
    die(env,
        "cp2k_set_cell got a cell that is neither lower nor upper triangular; a "
        "GROMACS box is lower triangular");
  }
  for (int a = 0; a < 3; ++a) {
    for (int k = 0; k < 3; ++k) {
      env->rows[a][k] = lower ? m[a][k] : m[k][a];
    }
  }
  env->have_cell = any;
}

RGPOT_GMX_API void cp2k_get_natom(force_env_t force_env, int *natom) {
  *natom = lookup(force_env, "cp2k_get_natom")->natoms;
}

RGPOT_GMX_API void cp2k_get_positions(force_env_t force_env, double *pos, int n_el) {
  // GROMACS 2026 shifts its QM translation by the change between what it
  // sent and this; rgpot moves no atom, so it gets back what it sent.
  Env *env = lookup(force_env, "cp2k_get_positions");
  checkCount(env, n_el, "cp2k_get_positions");
  std::copy(env->pos.begin(), env->pos.end(), pos);
}

RGPOT_GMX_API void cp2k_calc_energy_force(force_env_t force_env) {
  Env *env = lookup(force_env, "cp2k_calc_energy_force");
  if (!env->have_positions) {
    die(env, "cp2k_calc_energy_force before cp2k_set_positions");
  }
  evaluate(env);
  env->evaluated += 1;
}

RGPOT_GMX_API void cp2k_calc_energy(force_env_t force_env) {
  cp2k_calc_energy_force(force_env);
}

RGPOT_GMX_API void cp2k_get_potential_energy(force_env_t force_env, double *e_pot) {
  Env *env = lookup(force_env, "cp2k_get_potential_energy");
  if (env->evaluated == 0) {
    die(env, "cp2k_get_potential_energy before cp2k_calc_energy_force");
  }
  *e_pot = env->energy;
}

RGPOT_GMX_API void cp2k_get_forces(force_env_t force_env, double *force, int n_el) {
  Env *env = lookup(force_env, "cp2k_get_forces");
  checkCount(env, n_el, "cp2k_get_forces");
  if (env->evaluated == 0) {
    die(env, "cp2k_get_forces before cp2k_calc_energy_force");
  }
  std::copy(env->forces.begin(), env->forces.end(), force);
}
