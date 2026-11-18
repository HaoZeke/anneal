// MIT License
// Copyright 2023--present rgpot developers

#include "rgpot/XTBPot/XTBPot.hpp"
#include "rgpot/stress.hpp"
#include "rgpot/units.hpp"

#include <cmath>
#include <limits>
#include <stdexcept>
#include <string>

namespace rgpot {

using units::ANGSTROM_TO_BOHR;
using units::HARTREE_TO_EV;
using units::NEG_GRAD_TO_FORCE;

namespace {

bool numbersDiffer(const std::vector<int> &stored, const int *atmnrs,
                   size_t n) {
  if (stored.size() != n)
    return true;
  if (n == 0)
    return false;
  if (!atmnrs)
    return true;
  for (size_t i = 0; i < n; ++i) {
    if (stored[i] != atmnrs[i])
      return true;
  }
  return false;
}

// Euclidean displacement of the atom that moved furthest, in Angstrom.
// An empty history is not a previous geometry.
double maxAtomMove(const std::vector<double> &previous, const double *next,
                   size_t n3) {
  if (previous.size() != n3 || previous.empty() || !next)
    return std::numeric_limits<double>::infinity();
  double furthest2 = 0.0;
  for (size_t i = 0; i < n3; i += 3) {
    const double dx = previous[i] - next[i];
    const double dy = previous[i + 1] - next[i + 1];
    const double dz = previous[i + 2] - next[i + 2];
    const double d2 = dx * dx + dy * dy + dz * dz;
    if (d2 > furthest2)
      furthest2 = d2;
  }
  return std::sqrt(furthest2);
}

bool sccUnconverged(const std::string &message) {
  return message.find("did not converge") != std::string::npos ||
         message.find("Self consistent") != std::string::npos;
}

// Reading the log is what clears an xtb environment.
void clearError(xtb_TEnvironment env) {
  char ignored[512];
  int ignored_len = static_cast<int>(sizeof(ignored));
  xtb_getError(env, ignored, &ignored_len);
}

} // namespace

XTBPot::XTBPot() : XTBPot(XTBConfig{}) {}

XTBPot::XTBPot(const XTBConfig &config)
    : Potential(PotType::XTB), m_config(config), m_warm_move(config.warm_move),
      m_warm_min_gap(config.warm_min_gap) {
  initHandles();
}

void XTBPot::reset() {
  freshResults();
  m_previous.clear();
  m_has_wavefunction = false;
  m_last_reused = false;
}

void XTBPot::setWarmMove(double angstrom) { m_warm_move = angstrom; }

double XTBPot::warmMove() const noexcept { return m_warm_move; }

void XTBPot::setWarmMinGap(double ev) { m_warm_min_gap = ev; }

double XTBPot::warmMinGap() const noexcept { return m_warm_min_gap; }

bool XTBPot::reusedWavefunction() const noexcept { return m_last_reused; }

void XTBPot::initHandles() const {
  m_env = xtb_newEnvironment();
  if (!m_env) {
    throw std::runtime_error("Failed to create xtb environment");
  }
  xtb_setVerbosity(m_env, XTB_VERBOSITY_MUTED);
  xtb_releaseOutput(m_env);

  m_calc = xtb_newCalculator();
  if (!m_calc) {
    xtb_delEnvironment(&m_env);
    throw std::runtime_error("Failed to create xtb calculator");
  }

  m_res = xtb_newResults();
  if (!m_res) {
    xtb_delCalculator(&m_calc);
    xtb_delEnvironment(&m_env);
    throw std::runtime_error("Failed to create xtb results");
  }
}

XTBPot::~XTBPot() {
  if (m_res)
    xtb_delResults(&m_res);
  if (m_calc)
    xtb_delCalculator(&m_calc);
  if (m_mol)
    xtb_delMolecule(&m_mol);
  if (m_env) {
    xtb_releaseOutput(m_env);
    xtb_delEnvironment(&m_env);
  }
}

void XTBPot::loadParametrisation() const {
  switch (m_config.method) {
  case GFNMethod::GFNFF:
    xtb_loadGFNFF(m_env, m_mol, m_calc, nullptr);
    break;
  case GFNMethod::GFN0xTB:
    xtb_loadGFN0xTB(m_env, m_mol, m_calc, nullptr);
    break;
  case GFNMethod::GFN1xTB:
    xtb_loadGFN1xTB(m_env, m_mol, m_calc, nullptr);
    break;
  case GFNMethod::GFN2xTB:
    xtb_loadGFN2xTB(m_env, m_mol, m_calc, nullptr);
    break;
  }
  xtb_setAccuracy(m_env, m_calc, m_config.accuracy);
  xtb_setElectronicTemp(m_env, m_calc, m_config.electronic_temperature);
  xtb_setMaxIter(m_env, m_calc, m_config.max_iterations);
}

void XTBPot::freshResults() const {
  if (m_res) {
    xtb_delResults(&m_res);
    m_res = nullptr;
  }
  m_res = xtb_newResults();
  if (!m_res)
    throw std::runtime_error("Failed to create xtb results");
  m_has_wavefunction = false;
}

double XTBPot::frontierGap() const {
  // GFN-FF has no orbitals, and asking for them logs an xtb error.
  if (m_config.method == GFNMethod::GFNFF)
    return std::numeric_limits<double>::infinity();
  int nao = 0;
  xtb_getNao(m_env, m_res, &nao);
  if (xtb_checkEnvironment(m_env) != 0 || nao <= 0) {
    clearError(m_env);
    return std::numeric_limits<double>::infinity();
  }
  m_orbital_buffer.resize(2 * static_cast<size_t>(nao));
  double *emo = m_orbital_buffer.data();
  double *occ = emo + nao;
  xtb_getOrbitalEigenvalues(m_env, m_res, emo);
  xtb_getOrbitalOccupations(m_env, m_res, occ);
  if (xtb_checkEnvironment(m_env) != 0) {
    clearError(m_env);
    return std::numeric_limits<double>::infinity();
  }
  double electrons = 0.0;
  for (int i = 0; i < nao; ++i)
    electrons += occ[i];
  // Occupations are summed over spin. The highest occupied level holds
  // the last electron, smeared or not.
  const long nel = std::lround(electrons);
  const long homo = (nel + 1) / 2 - 1;
  if (homo < 0 || homo + 1 >= nao)
    return std::numeric_limits<double>::infinity();
  return (emo[homo + 1] - emo[homo]) * HARTREE_TO_EV;
}

bool XTBPot::singlepoint(std::string *error) const {
  xtb_singlepoint(m_env, m_mol, m_calc, m_res);
  if (xtb_checkEnvironment(m_env) == 0)
    return true;
  char err_msg[512] = {};
  int err_len = static_cast<int>(sizeof(err_msg));
  xtb_getError(m_env, err_msg, &err_len);
  if (error)
    *error = std::string("xTB Error: ") + err_msg;
  return false;
}

bool XTBPot::rescueUnconvergedScc() const {
  // A dissociated or rearranged cluster can have a gap of a few tenths
  // of an eV. The charge iterator then runs out its budget at the
  // configured temperature. Smearing the levels converges that density
  // in a few iterations, and the cooled single point uses it as the
  // guess. The energy handed back is the configured-temperature one.
  //
  // A failed SCC leaves its last iterate in the results object, and on a
  // near-zero-gap cluster that iterate has diverged. Started from it, the
  // GFN2 charge iterator settles on a stationary point with charges of
  // several electrons per atom, far below the physical solution. Each hot
  // rung therefore starts from xtb's own guess.
  xtb_setMaxIter(m_env, m_calc, m_config.max_iterations);
  const double configured = m_config.electronic_temperature;
  const double ladder[] = {1000.0, 3000.0, 5000.0};
  for (double hot : ladder) {
    if (!(hot > configured))
      continue;
    freshResults();
    xtb_setElectronicTemp(m_env, m_calc, hot);
    if (!singlepoint(nullptr))
      continue;
    xtb_setElectronicTemp(m_env, m_calc, configured);
    if (singlepoint(nullptr))
      return true;
  }
  xtb_setElectronicTemp(m_env, m_calc, configured);
  return false;
}

void XTBPot::discardAfterFailure() const {
  // A failed SCC leaves the environment in an error state and the
  // wavefunction is not a guess. Replace the handle set.
  if (m_res) {
    xtb_delResults(&m_res);
    m_res = nullptr;
  }
  if (m_mol) {
    xtb_delMolecule(&m_mol);
    m_mol = nullptr;
  }
  if (m_calc) {
    xtb_delCalculator(&m_calc);
    m_calc = nullptr;
  }
  if (m_env) {
    xtb_releaseOutput(m_env);
    xtb_delEnvironment(&m_env);
    m_env = nullptr;
  }
  m_initialized = false;
  m_has_wavefunction = false;
  m_last_reused = false;
  m_previous.clear();
  m_numbers.clear();
  m_pos_bohr.clear();
  try {
    initHandles();
  } catch (...) {
    // The original xtb error is the one the caller sees.
  }
}

void XTBPot::setAccuracy(double accuracy) { m_config.accuracy = accuracy; }

void XTBPot::setElectronicTemperature(double kelvin) {
  m_config.electronic_temperature = kelvin;
}

void XTBPot::setMaxIterations(int iterations) {
  m_config.max_iterations = iterations;
}

FusedSettingsSpec XTBPot::fusedSettings() const {
  FusedSettingsSpec spec;
  spec.spin = m_config.uhf;
  spec.sccMaxIterations = static_cast<unsigned>(m_config.max_iterations);
  spec.charge = m_config.charge;
  spec.accuracy = m_config.accuracy;
  spec.electronicTemperature = m_config.electronic_temperature;
  spec.warmMove = m_warm_move;
  spec.warmMinGap = m_warm_min_gap;
  spec.sccMixer = "xtb";
  spec.engineBuild = "libxtb-api-" + std::to_string(xtb_getAPIVersion()) +
                     "-method-" + std::to_string(static_cast<int>(m_config.method));
  return spec;
}

void XTBPot::forceImpl(const ForceInput &in, ForceOut *out) const {
  if (!m_env || !m_calc || !m_res)
    throw std::runtime_error("xTB Error: calculator is not initialized");

  int intN = static_cast<int>(in.nAtoms);
  const size_t n3 = 3 * in.nAtoms;
  const bool periodicity[3] = {false, false, false};

  // Reuse preallocated buffer, resize only when atom count changes
  m_pos_bohr.resize(n3);
  for (size_t i = 0; i < n3; ++i) {
    m_pos_bohr[i] = in.pos[i] * ANGSTROM_TO_BOHR;
  }

  double box_bohr[9];
  for (int i = 0; i < 9; ++i) {
    box_bohr[i] = in.box[i] * ANGSTROM_TO_BOHR;
  }

  const bool composition_changed =
      !m_initialized || numbersDiffer(m_numbers, in.atmnrs, in.nAtoms);
  const double moved = maxAtomMove(m_previous, in.pos, n3);
  // The results object holds the previous SCC wavefunction. xtb reads it
  // as the guess. Past warm_move that guess is slower than xtb's own, so
  // the results object is replaced before the single point.
  //
  // With near-degenerate frontier orbitals the SCC can have more than one
  // self-consistent solution, and a guess carried over from another
  // geometry can settle on a different one than xtb's own guess does.
  // Below warm_min_gap the energy therefore comes from xtb's guess.
  bool reuse = !composition_changed && m_has_wavefunction &&
               m_warm_move > 0.0 && moved <= m_warm_move &&
               !(m_gap < m_warm_min_gap);

  if (composition_changed) {
    if (m_mol) {
      xtb_delMolecule(&m_mol);
      m_mol = nullptr;
    }
    if (m_has_wavefunction)
      freshResults();
    double charge = m_config.charge;
    int uhf = m_config.uhf;
    m_mol = xtb_newMolecule(m_env, &intN, in.atmnrs, m_pos_bohr.data(), &charge,
                            &uhf, box_bohr, periodicity);
    if (!m_mol)
      throw std::runtime_error("xTB Error: xtb_newMolecule failed");
    loadParametrisation();
    m_initialized = true;
  } else {
    xtb_updateMolecule(m_env, m_mol, m_pos_bohr.data(), box_bohr);
    if (!reuse && m_has_wavefunction)
      freshResults();
  }

  // A bound cluster converges in about ten iterations. One that has
  // not done so by a few times that is the near-zero-gap case, and
  // spending the rest of the iteration budget only delays the rescue.
  const int first_pass =
      m_config.max_iterations < 48 ? m_config.max_iterations : 48;
  xtb_setMaxIter(m_env, m_calc, first_pass);
  xtb_setAccuracy(m_env, m_calc, m_config.accuracy);
  xtb_setElectronicTemp(m_env, m_calc, m_config.electronic_temperature);
  std::string message;
  bool converged = singlepoint(&message);
  if (reuse && (!converged || frontierGap() < m_warm_min_gap)) {
    // What follows is a fresh call.
    reuse = false;
    freshResults();
    converged = singlepoint(&message);
  }
  if (!converged) {
    const bool scc = sccUnconverged(message);
    if (!(scc && rescueUnconvergedScc())) {
      discardAfterFailure();
      if (scc)
        throw SccNotConverged(message);
      throw std::runtime_error(message);
    }
  }
  m_gap = frontierGap();

  double energy_hartree;
  xtb_getEnergy(m_env, m_res, &energy_hartree);
  out->energy = energy_hartree * HARTREE_TO_EV;

  // Write gradient directly into output buffer, then convert in-place
  xtb_getGradient(m_env, m_res, out->F);
  bool finite = std::isfinite(energy_hartree);
  for (size_t i = 0; i < n3 && finite; ++i)
    finite = std::isfinite(out->F[i]);
  if (!finite) {
    discardAfterFailure();
    throw std::runtime_error("xTB Error: non-finite energy or gradient");
  }
  for (size_t i = 0; i < n3; ++i) {
    out->F[i] *= NEG_GRAD_TO_FORCE;
  }

  // xtb stores dE/dε in Hartree. sigma = virial / V in eV/Angstrom^3.
  double virial[9] = {};
  xtb_getVirial(m_env, m_res, virial);
  const double volume = cellVolume(in.box);
  if (volume > 0.0) {
    const double scale = units::HARTREE_TO_EV / volume;
    for (int col = 0; col < 3; ++col) {
      for (int row = 0; row < 3; ++row) {
        out->stress[row * 3 + col] = virial[row + 3 * col] * scale;
      }
    }
    out->has_stress = 1;
  } else {
    out->has_stress = 0;
  }

  out->variance = 0.0;

  m_previous.assign(in.pos, in.pos + n3);
  if (in.nAtoms == 0 || !in.atmnrs)
    m_numbers.clear();
  else
    m_numbers.assign(in.atmnrs, in.atmnrs + in.nAtoms);
  m_has_wavefunction = true;
  m_last_reused = reuse;
}

} // namespace rgpot
