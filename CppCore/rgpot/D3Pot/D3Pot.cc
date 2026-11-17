// MIT License
// Copyright 2023--present rgpot developers

#include "rgpot/D3Pot/D3Pot.hpp"
#include "rgpot/units.hpp"

#include <algorithm>
#include <stdexcept>
#include <string>
#include <vector>

namespace rgpot {

using units::ANGSTROM_TO_BOHR;
using units::HARTREE_TO_EV;
using units::NEG_GRAD_TO_FORCE;

D3Pot::D3Pot() : D3Pot(D3Config{}) {}

D3Pot::D3Pot(const D3Config &config)
    : Potential(PotType::D3), m_config(config) {
  initHandles();
}

void D3Pot::throwIfError(const char *what) const {
  if (dftd3_check_error(m_err) == 0) {
    return;
  }
  char err_msg[512] = {};
  dftd3_get_error(m_err, err_msg, nullptr);
  throw std::runtime_error(std::string(what) + ": " + err_msg);
}

void D3Pot::initHandles() {
  m_err = dftd3_new_error();
  if (!m_err) {
    throw std::runtime_error("Failed to create s-dftd3 error handle");
  }
  loadParam();
}

void D3Pot::loadParam() const {
  if (m_config.functional.empty()) {
    throw std::invalid_argument("D3Config.functional must be a method key");
  }
  std::vector<char> method(m_config.functional.begin(),
                           m_config.functional.end());
  method.push_back('\0');

  switch (m_config.damping) {
  case D3Damping::BJ:
    m_param = dftd3_load_rational_damping(m_err, method.data(), m_config.atm);
    break;
  case D3Damping::Zero:
    m_param = dftd3_load_zero_damping(m_err, method.data(), m_config.atm);
    break;
  }
  throwIfError("s-dftd3 load damping");
  if (!m_param) {
    throw std::runtime_error("s-dftd3 returned a null damping handle");
  }
}

D3Pot::~D3Pot() {
  if (m_param)
    dftd3_delete_param(&m_param);
  if (m_model)
    dftd3_delete_model(&m_model);
  if (m_mol)
    dftd3_delete_structure(&m_mol);
  if (m_err)
    dftd3_delete_error(&m_err);
}

void D3Pot::forceImpl(const ForceInput &in, ForceOut *out) const {
  int intN = static_cast<int>(in.nAtoms);
  const size_t n3 = 3 * in.nAtoms;
  const bool periodicity[3] = {false, false, false};

  m_pos_bohr.resize(n3);
  for (size_t i = 0; i < n3; ++i) {
    m_pos_bohr[i] = in.pos[i] * ANGSTROM_TO_BOHR;
  }

  double box_bohr[9];
  for (int i = 0; i < 9; ++i) {
    box_bohr[i] = in.box[i] * ANGSTROM_TO_BOHR;
  }

  if (!m_initialized || m_natoms != intN ||
      !std::equal(in.atmnrs, in.atmnrs + in.nAtoms, m_numbers.begin(),
                  m_numbers.end())) {
    if (m_model)
      dftd3_delete_model(&m_model);
    if (m_mol)
      dftd3_delete_structure(&m_mol);
    m_mol = dftd3_new_structure(m_err, intN, in.atmnrs, m_pos_bohr.data(),
                                box_bohr, periodicity);
    throwIfError("s-dftd3 new structure");
    m_model = dftd3_new_d3_model(m_err, m_mol);
    throwIfError("s-dftd3 new D3 model");
    m_natoms = intN;
    m_numbers.assign(in.atmnrs, in.atmnrs + in.nAtoms);
    m_initialized = true;
  } else {
    dftd3_update_structure(m_err, m_mol, m_pos_bohr.data(), box_bohr);
    throwIfError("s-dftd3 update structure");
  }

  double energy_hartree = 0.0;
  // s-dftd3 skips the gradient unless sigma is also non-NULL.
  double sigma[9] = {};
  dftd3_get_dispersion(m_err, m_mol, m_model, m_param, &energy_hartree, out->F,
                       sigma);
  throwIfError("s-dftd3 get dispersion");

  out->energy = energy_hartree * HARTREE_TO_EV;
  for (size_t i = 0; i < n3; ++i) {
    out->F[i] *= NEG_GRAD_TO_FORCE;
  }
  out->variance = 0.0;
}

} // namespace rgpot
