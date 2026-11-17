// MIT License
// Copyright 2023--present rgpot developers
//
// libxtb_engine.so: wraps linked XTBPot behind the C ABI for dlopen consumers.

#include "rgpot/XTBPot/xtb_c_abi.h"

#include "rgpot/XTBPot/XTBPot.hpp"
#include "rgpot/types/AtomMatrix.hpp"

#include <array>
#include <cstdio>
#include <cstring>
#include <exception>
#include <memory>
#include <vector>

struct RgpotXtbPot {
  std::unique_ptr<rgpot::XTBPot> pot;
};

static rgpot::GFNMethod method_from_c(int m) {
  switch (m) {
  case RGPOT_XTB_METHOD_GFNFF:
    return rgpot::GFNMethod::GFNFF;
  case RGPOT_XTB_METHOD_GFN0:
    return rgpot::GFNMethod::GFN0xTB;
  case RGPOT_XTB_METHOD_GFN1:
    return rgpot::GFNMethod::GFN1xTB;
  case RGPOT_XTB_METHOD_GFN2:
  default:
    return rgpot::GFNMethod::GFN2xTB;
  }
}

extern "C" {

int rgpot_xtb_abi_version(void) { return RGPOT_XTB_ABI_VERSION; }

int rgpot_xtb_available(void) { return 1; }

RgpotXtbPot *rgpot_xtb_create(const RgpotXtbConfig *cfg, char *errbuf,
                              size_t errlen) {
  try {
    rgpot::XTBConfig c{};
    if (cfg) {
      c.method = method_from_c(cfg->method);
      c.accuracy = cfg->accuracy;
      c.electronic_temperature = cfg->electronic_temperature;
      c.max_iterations = cfg->max_iterations;
      c.charge = cfg->charge;
      c.uhf = cfg->uhf;
    }
    auto *out = new RgpotXtbPot;
    out->pot = std::make_unique<rgpot::XTBPot>(c);
    return out;
  } catch (const std::exception &ex) {
    if (errbuf && errlen)
      std::snprintf(errbuf, errlen, "%s", ex.what());
    return nullptr;
  } catch (...) {
    if (errbuf && errlen)
      std::snprintf(errbuf, errlen, "unknown error creating XTB pot");
    return nullptr;
  }
}

void rgpot_xtb_destroy(RgpotXtbPot *pot) { delete pot; }

void rgpot_xtb_reset(RgpotXtbPot *pot) {
  if (pot && pot->pot)
    pot->pot->reset();
}

void rgpot_xtb_set_warm_move(RgpotXtbPot *pot, double angstrom) {
  if (pot && pot->pot)
    pot->pot->setWarmMove(angstrom);
}

double rgpot_xtb_warm_move(const RgpotXtbPot *pot) {
  if (!pot || !pot->pot)
    return 0.0;
  return pot->pot->warmMove();
}

void rgpot_xtb_set_warm_min_gap(RgpotXtbPot *pot, double ev) {
  if (pot && pot->pot)
    pot->pot->setWarmMinGap(ev);
}

double rgpot_xtb_warm_min_gap(const RgpotXtbPot *pot) {
  if (!pot || !pot->pot)
    return 0.0;
  return pot->pot->warmMinGap();
}

int rgpot_xtb_reused_wavefunction(const RgpotXtbPot *pot) {
  if (!pot || !pot->pot)
    return 0;
  return pot->pot->reusedWavefunction() ? 1 : 0;
}

int rgpot_xtb_force(RgpotXtbPot *pot, long nAtoms, const double *positions,
                    const int *atomicNrs, double *forces, double *energy,
                    double *variance, const double *box) {
  if (!pot || !pot->pot || !positions || !atomicNrs || !forces || !energy ||
      !box || nAtoms <= 0)
    return -1;
  try {
    const int n = static_cast<int>(nAtoms);
    rgpot::types::AtomMatrix pos(n, 3);
    for (int i = 0; i < n; ++i)
      for (int j = 0; j < 3; ++j)
        pos(i, j) = positions[i * 3 + j];
    std::vector<int> atm(atomicNrs, atomicNrs + n);
    std::array<std::array<double, 3>, 3> cell{};
    for (int i = 0; i < 3; ++i)
      for (int j = 0; j < 3; ++j)
        cell[static_cast<size_t>(i)][static_cast<size_t>(j)] = box[i * 3 + j];
    auto [e, f, var] = (*pot->pot)(pos, atm, cell);
    *energy = e;
    if (variance)
      *variance = var;
    for (int i = 0; i < n; ++i)
      for (int j = 0; j < 3; ++j)
        forces[i * 3 + j] = f(i, j);
    return 0;
  } catch (const std::exception &ex) {
    (void)ex;
    return -2;
  } catch (...) {
    return -3;
  }
}

} // extern "C"
