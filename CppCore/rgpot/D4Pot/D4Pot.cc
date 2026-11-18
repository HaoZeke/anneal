// MIT License
// Copyright 2023--present rgpot developers

#include "rgpot/D4Pot/D4Pot.hpp"
#include "rgpot/units.hpp"

#include <algorithm>
#include <cmath>
#include <stdexcept>
#include <string>
#include <vector>

namespace rgpot {

using units::ANGSTROM_TO_BOHR;
using units::HARTREE_PER_BOHR3_TO_EV_PER_ANGSTROM3;
using units::HARTREE_TO_EV;
using units::NEG_GRAD_TO_FORCE;

namespace {

double det3(const double *h) {
  return h[0] * (h[4] * h[8] - h[5] * h[7]) -
         h[1] * (h[3] * h[8] - h[5] * h[6]) +
         h[2] * (h[3] * h[7] - h[4] * h[6]);
}

} // namespace

D4Pot::D4Pot() : D4Pot(D4Config{}) {}

D4Pot::D4Pot(const D4Config &config)
    : Potential(PotType::D4), m_config(config) {
  initHandles();
}

void D4Pot::throwIfError(const char *what) const {
  if (dftd4_check_error(m_err) == 0) {
    return;
  }
  char err_msg[512] = {};
  dftd4_get_error(m_err, err_msg, nullptr);
  throw std::runtime_error(std::string(what) + ": " + err_msg);
}

void D4Pot::initHandles() {
  m_err = dftd4_new_error();
  if (!m_err) {
    throw std::runtime_error("Failed to create dftd4 error handle");
  }
  loadParam();
}

void D4Pot::loadParam() const {
  if (m_config.functional.empty()) {
    throw std::invalid_argument("D4Config.functional must be a method key");
  }
  std::vector<char> method(m_config.functional.begin(),
                           m_config.functional.end());
  method.push_back('\0');

  m_param = dftd4_load_rational_damping(m_err, method.data(), m_config.atm);
  throwIfError("dftd4 load damping");
  if (!m_param) {
    throw std::runtime_error("dftd4 returned a null damping handle");
  }
}

D4Pot::~D4Pot() {
  if (m_param)
    dftd4_delete_param(&m_param);
  if (m_model)
    dftd4_delete_model(&m_model);
  if (m_mol)
    dftd4_delete_structure(&m_mol);
  if (m_err)
    dftd4_delete_error(&m_err);
}

void D4Pot::forceImpl(const ForceInput &in, ForceOut *out) const {
  int intN = static_cast<int>(in.nAtoms);
  const size_t n3 = 3 * in.nAtoms;
  const bool p = m_config.periodic;
  const bool periodicity[3] = {p, p, p};

  m_pos_bohr.resize(n3);
  for (size_t i = 0; i < n3; ++i) {
    m_pos_bohr[i] = in.pos[i] * ANGSTROM_TO_BOHR;
  }

  double box_bohr[9] = {};
  if (p) {
    if (in.box == nullptr) {
      throw std::invalid_argument("D4Pot: a periodic D4Config needs a box");
    }
    for (int i = 0; i < 9; ++i) {
      box_bohr[i] = in.box[i] * ANGSTROM_TO_BOHR;
    }
  }
  const double volume = std::abs(det3(box_bohr));
  if (p && !(volume > 0.0)) {
    throw std::invalid_argument(
        "D4Pot: a periodic D4Config needs a box of positive volume");
  }
  // dftd4 reads the lattice and the periodic flags together, so a molecule
  // passes neither.
  const double *lattice = p ? box_bohr : nullptr;
  const bool *periodic_flags = p ? periodicity : nullptr;

  if (!m_initialized || m_natoms != intN ||
      !std::equal(in.atmnrs, in.atmnrs + in.nAtoms, m_numbers.begin(),
                  m_numbers.end())) {
    if (m_model)
      dftd4_delete_model(&m_model);
    if (m_mol)
      dftd4_delete_structure(&m_mol);
    m_mol = dftd4_new_structure(m_err, intN, in.atmnrs, m_pos_bohr.data(),
                                &m_config.charge, lattice, periodic_flags);
    throwIfError("dftd4 new structure");
    m_model = dftd4_new_d4_model(m_err, m_mol);
    throwIfError("dftd4 new D4 model");
    m_natoms = intN;
    m_numbers.assign(in.atmnrs, in.atmnrs + in.nAtoms);
    m_initialized = true;
  } else {
    dftd4_update_structure(m_err, m_mol, m_pos_bohr.data(), lattice);
    throwIfError("dftd4 update structure");
  }

  double energy_hartree = 0.0;
  double sigma[9] = {};
  dftd4_get_dispersion(m_err, m_mol, m_model, m_param, &energy_hartree, out->F,
                       p ? sigma : nullptr);
  throwIfError("dftd4 get dispersion");

  out->energy = energy_hartree * HARTREE_TO_EV;
  for (size_t i = 0; i < n3; ++i) {
    out->F[i] *= NEG_GRAD_TO_FORCE;
  }
  out->variance = 0.0;
  if (p) {
    // dftd4's sigma is dE/dstrain, symmetric, in Hartree.
    for (int k = 0; k < 9; ++k) {
      out->stress[k] =
          sigma[k] / volume * HARTREE_PER_BOHR3_TO_EV_PER_ANGSTROM3;
    }
    out->has_stress = 1;
  }
}

} // namespace rgpot
