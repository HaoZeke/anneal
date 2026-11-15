// MIT License
// Copyright 2023--present rgpot developers

#include "rgpot/TBLitePot/TBLitePot.hpp"
#include "rgpot/stress.hpp"
#include "rgpot/units.hpp"

#include <stdexcept>
#include <string>

namespace rgpot {

using units::ANGSTROM_TO_BOHR;
using units::HARTREE_TO_EV;
using units::KB_HARTREE;
using units::NEG_GRAD_TO_FORCE;

TBLitePot::TBLitePot() : TBLitePot(TBLiteConfig{}) {}

TBLitePot::TBLitePot(const TBLiteConfig &config)
    : Potential(PotType::TBLite), m_config(config) {
  initHandles();
}

void TBLitePot::initHandles() {
  m_ctx = tblite_new_context();
  if (!m_ctx) {
    throw std::runtime_error("Failed to create tblite context");
  }
  tblite_set_context_verbosity(m_ctx, 0);

  m_res = tblite_new_result();
  if (!m_res) {
    tblite_delete_context(&m_ctx);
    throw std::runtime_error("Failed to create tblite result");
  }
}

TBLitePot::~TBLitePot() {
  if (m_res)
    tblite_delete_result(&m_res);
  if (m_calc)
    tblite_delete_calculator(&m_calc);
  if (m_mol)
    tblite_delete_structure(&m_mol);
  if (m_ctx)
    tblite_delete_context(&m_ctx);
}

namespace {

// tblite 0.6 adds an optional xtb config. Null keeps the 0.5 defaults.
tblite_calculator newGfnCalculator(tblite_context ctx, tblite_structure mol,
                                   TBLiteMethod method) {
#ifdef RGPOT_TBLITE_XTB_CONFIG
  auto call = [](auto make, tblite_context c, tblite_structure m) {
    return make(c, m, nullptr);
  };
#else
  auto call = [](auto make, tblite_context c, tblite_structure m) {
    return make(c, m);
  };
#endif
  switch (method) {
  case TBLiteMethod::GFN1:
    return call(tblite_new_gfn1_calculator, ctx, mol);
  case TBLiteMethod::GFN2:
    return call(tblite_new_gfn2_calculator, ctx, mol);
  case TBLiteMethod::IPEA1:
    return call(tblite_new_ipea1_calculator, ctx, mol);
  }
  throw std::logic_error("unknown tblite method");
}

} // namespace

void TBLitePot::createCalculator() const {
  m_calc = newGfnCalculator(m_ctx, m_mol, m_config.method);

  if (tblite_check_context(m_ctx) != 0) {
    char err_msg[512];
    tblite_get_context_error(m_ctx, err_msg, nullptr);
    throw std::runtime_error(std::string("tblite calculator error: ") +
                             err_msg);
  }

  tblite_set_calculator_accuracy(m_ctx, m_calc, m_config.accuracy);
  tblite_set_calculator_max_iter(m_ctx, m_calc, m_config.max_iterations);
  double etemp_hartree = m_config.electronic_temperature * KB_HARTREE;
  tblite_set_calculator_temperature(m_ctx, m_calc, etemp_hartree);
}

void TBLitePot::forceImpl(const ForceInput &in, ForceOut *out) const {
  int intN = static_cast<int>(in.nAtoms);
  const size_t n3 = 3 * in.nAtoms;

  // Reuse preallocated buffer
  m_pos_bohr.resize(n3);
  for (size_t i = 0; i < n3; ++i) {
    m_pos_bohr[i] = in.pos[i] * ANGSTROM_TO_BOHR;
  }

  double box_bohr[9];
  for (int i = 0; i < 9; ++i) {
    box_bohr[i] = in.box[i] * ANGSTROM_TO_BOHR;
  }

  if (!m_initialized) {
    tblite_error err = tblite_new_error();
    double charge = m_config.charge;
    int uhf = m_config.uhf;
    m_mol = tblite_new_structure(err, intN, in.atmnrs, m_pos_bohr.data(),
                                 &charge, &uhf, box_bohr, nullptr);
    if (tblite_check_error(err) != 0) {
      char err_msg[512];
      tblite_get_error(err, err_msg, nullptr);
      tblite_delete_error(&err);
      throw std::runtime_error(std::string("tblite structure error: ") +
                               err_msg);
    }
    tblite_delete_error(&err);
    createCalculator();
    m_initialized = true;
  } else {
    tblite_error err = tblite_new_error();
    tblite_update_structure_geometry(err, m_mol, m_pos_bohr.data(), box_bohr);
    if (tblite_check_error(err) != 0) {
      char err_msg[512];
      tblite_get_error(err, err_msg, nullptr);
      tblite_delete_error(&err);
      throw std::runtime_error(std::string("tblite update error: ") + err_msg);
    }
    tblite_delete_error(&err);
  }

  tblite_get_singlepoint(m_ctx, m_mol, m_calc, m_res);

  if (tblite_check_context(m_ctx) != 0) {
    char err_msg[512];
    tblite_get_context_error(m_ctx, err_msg, nullptr);
    // The result keeps the wavefunction of the failed SCF, and tblite
    // would start the next one from it.
    tblite_delete_result(&m_res);
    m_res = tblite_new_result();
    throw std::runtime_error(std::string("tblite SCF error: ") + err_msg);
  }

  tblite_error err = tblite_new_error();

  double energy_hartree;
  tblite_get_result_energy(err, m_res, &energy_hartree);
  out->energy = energy_hartree * HARTREE_TO_EV;

  // Write gradient directly into output, convert in-place
  tblite_get_result_gradient(err, m_res, out->F);
  for (size_t i = 0; i < n3; ++i) {
    out->F[i] *= NEG_GRAD_TO_FORCE;
  }

  // tblite stores dE/dε in Hartree, column-major. sigma = virial / V.
  double virial[9] = {};
  tblite_get_result_virial(err, m_res, virial);
  const double volume = cellVolume(in.box);
  if (volume > 0.0 && tblite_check_error(err) == 0) {
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
  tblite_delete_error(&err);
}

} // namespace rgpot
