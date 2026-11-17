// MIT License
// Copyright 2023--present rgpot developers
//
// OpenMM backend. Positions are nanometres, energy is kJ/mol, forces
// are kJ/mol/nm. The expression is the shifted 12-6 potential the
// openmm_lj test evaluates. The host constructor converts once.

#include "rgpot/engine_c_abi.h"

#include <openmm/Context.h>
#include <openmm/CustomNonbondedForce.h>
#include <openmm/Platform.h>
#include <openmm/State.h>
#include <openmm/System.h>
#include <openmm/VerletIntegrator.h>

#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <memory>
#include <new>
#include <string>
#include <vector>

namespace {

const char *kExpression =
    "4*sqrt(eps1*eps2)*((sqrt(sig1*sig2)/r)^12-(sqrt(sig1*sig2)/r)^6)"
    "-4*sqrt(eps1*eps2)*((sqrt(sig1*sig2)/rc)^12-(sqrt(sig1*sig2)/rc)^6)";

struct OpenMMLj {
  double u0_kj = 0.0;
  double cutoff_nm = 0.0;
  double psi_nm = 0.0;
  long n = -1;
  std::unique_ptr<OpenMM::System> system;
  std::unique_ptr<OpenMM::VerletIntegrator> integrator;
  std::unique_ptr<OpenMM::Context> context;
  std::string last_error;
};

void set_error(char *errbuf, size_t errlen, const char *sentence) {
  if (errbuf == nullptr || errlen == 0) {
    return;
  }
  std::snprintf(errbuf, errlen, "%s", sentence);
}

void load_platforms() {
  if (const char *dir = std::getenv("OPENMM_PLUGIN_DIR")) {
    if (dir[0] != '\0') {
      OpenMM::Platform::loadPluginsFromDirectory(dir);
    }
  }
  OpenMM::Platform::loadPluginsFromDirectory(
      OpenMM::Platform::getDefaultPluginsDirectory());
}

bool ensure(OpenMMLj *pot, long n, const double *box, std::string &error) {
  try {
    if (pot->context != nullptr && pot->n == n) {
      pot->context->setPeriodicBoxVectors(OpenMM::Vec3(box[0], box[1], box[2]),
                                          OpenMM::Vec3(box[3], box[4], box[5]),
                                          OpenMM::Vec3(box[6], box[7], box[8]));
      return true;
    }
    pot->context.reset();
    pot->integrator.reset();
    pot->system = std::make_unique<OpenMM::System>();
    for (long i = 0; i < n; ++i) {
      pot->system->addParticle(1.0);
    }
    auto *force = new OpenMM::CustomNonbondedForce(kExpression);
    pot->system->addForce(force);
    force->addGlobalParameter("rc", pot->cutoff_nm);
    force->addPerParticleParameter("eps");
    force->addPerParticleParameter("sig");
    for (long i = 0; i < n; ++i) {
      force->addParticle(std::vector<double>{pot->u0_kj, pot->psi_nm});
    }
    force->setNonbondedMethod(OpenMM::CustomNonbondedForce::CutoffPeriodic);
    force->setCutoffDistance(pot->cutoff_nm);
    pot->system->setDefaultPeriodicBoxVectors(
        OpenMM::Vec3(box[0], box[1], box[2]),
        OpenMM::Vec3(box[3], box[4], box[5]),
        OpenMM::Vec3(box[6], box[7], box[8]));
    pot->integrator = std::make_unique<OpenMM::VerletIntegrator>(0.001);
    OpenMM::Platform &platform = OpenMM::Platform::getPlatformByName("CPU");
    pot->context = std::make_unique<OpenMM::Context>(
        *pot->system, *pot->integrator, platform);
    pot->n = n;
    return true;
  } catch (const std::exception &ex) {
    error = ex.what();
    return false;
  }
}

} // namespace

extern "C" {

int rgpot_engine_abi_version(void) { return RGPOT_ENGINE_ABI_VERSION; }

int rgpot_engine_available(void) { return 1; }

RgpotEnginePot *rgpot_engine_create(const void *config, size_t config_len,
                                    char *errbuf, size_t errlen) {
  if (config == nullptr || config_len != 3 * sizeof(double)) {
    set_error(errbuf, errlen, "OpenMM config is three float64 values");
    return nullptr;
  }
  try {
    load_platforms();
    (void)OpenMM::Platform::getPlatformByName("CPU");
  } catch (const std::exception &ex) {
    set_error(errbuf, errlen, ex.what());
    return nullptr;
  }
  auto *pot = new (std::nothrow) OpenMMLj();
  if (pot == nullptr) {
    set_error(errbuf, errlen, "out of memory");
    return nullptr;
  }
  std::memcpy(&pot->u0_kj, config, sizeof(double));
  std::memcpy(&pot->cutoff_nm,
              static_cast<const char *>(config) + sizeof(double),
              sizeof(double));
  std::memcpy(&pot->psi_nm,
              static_cast<const char *>(config) + 2 * sizeof(double),
              sizeof(double));
  return reinterpret_cast<RgpotEnginePot *>(pot);
}

void rgpot_engine_destroy(RgpotEnginePot *pot) {
  delete reinterpret_cast<OpenMMLj *>(pot);
}

int rgpot_md_native_force(void *pot, long nAtoms, const double *positions,
                          const int *atomicNrs, double *forces, double *energy,
                          double *variance, const double *box) {
  (void)atomicNrs;
  if (pot == nullptr || energy == nullptr || nAtoms < 1 ||
      positions == nullptr || forces == nullptr || box == nullptr) {
    return 1;
  }
  auto *lj = reinterpret_cast<OpenMMLj *>(pot);
  if (!ensure(lj, nAtoms, box, lj->last_error)) {
    return 2;
  }
  try {
    std::vector<OpenMM::Vec3> pos(static_cast<size_t>(nAtoms));
    for (long i = 0; i < nAtoms; ++i) {
      pos[static_cast<size_t>(i)] = OpenMM::Vec3(
          positions[3 * i], positions[3 * i + 1], positions[3 * i + 2]);
    }
    lj->context->setPositions(pos);
    const OpenMM::State state =
        lj->context->getState(OpenMM::State::Energy | OpenMM::State::Forces);
    *energy = state.getPotentialEnergy();
    const std::vector<OpenMM::Vec3> &got = state.getForces();
    if (static_cast<long>(got.size()) != nAtoms) {
      lj->last_error = "OpenMM returned the wrong number of forces";
      return 2;
    }
    for (long i = 0; i < nAtoms; ++i) {
      forces[3 * i] = got[static_cast<size_t>(i)][0];
      forces[3 * i + 1] = got[static_cast<size_t>(i)][1];
      forces[3 * i + 2] = got[static_cast<size_t>(i)][2];
    }
    if (variance != nullptr) {
      *variance = 0.0;
    }
    lj->last_error.clear();
    return 0;
  } catch (const std::exception &ex) {
    lj->last_error = ex.what();
    return 2;
  }
}

const char *rgpot_engine_name(void) { return "openmm"; }

} // extern "C"
