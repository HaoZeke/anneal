// MIT License
// Copyright 2023--present rgpot developers

#include "rgpot/Models/ModelCapi.h"

#include "rgpot/Models/ModelPots.hpp"
#ifdef RGPOT_HAS_CBE
#include "rgpot/CBEPot/CBEPot.hpp"
#endif

#include <algorithm>
#include <array>
#include <cmath>
#include <exception>
#include <limits>
#include <memory>
#include <stdexcept>
#include <string>
#include <vector>

struct rgpot_builtin_model {
  std::unique_ptr<rgpot::PotentialBase> potential;
  std::size_t dimensions;
};

namespace {
thread_local std::string lastError;

template <class Function>
rgpot_builtin_model *guardedNew(Function &&function) noexcept {
  try {
    lastError.clear();
    return function();
  } catch (const std::exception &error) {
    lastError = error.what();
  } catch (...) {
    lastError = "unknown rgpot built-in model error";
  }
  return nullptr;
}

template <class Function> int guardedStatus(Function &&function) noexcept {
  try {
    lastError.clear();
    function();
    return 0;
  } catch (const std::exception &error) {
    lastError = error.what();
  } catch (...) {
    lastError = "unknown rgpot built-in model error";
  }
  return 1;
}

std::unique_ptr<rgpot::PotentialBase> defaultPotential(
    rgpot_builtin_kind kind, std::size_t &dimensions) {
  switch (kind) {
  case RGPOT_BUILTIN_ECKART:
    dimensions = 1;
    return std::make_unique<rgpot::EckartPot>();
  case RGPOT_BUILTIN_MULLER_BROWN:
    dimensions = 2;
    return std::make_unique<rgpot::MullerBrownPot>();
  case RGPOT_BUILTIN_MODIFIED_MULLER_BROWN:
    dimensions = 2;
    return std::make_unique<rgpot::ModifiedMullerBrownPot>();
  case RGPOT_BUILTIN_LEPS_HH2:
    dimensions = 9;
    return std::make_unique<rgpot::LepsPot>(rgpot::LepsSystem::HH2);
  case RGPOT_BUILTIN_LEPS_HCL2:
    dimensions = 9;
    return std::make_unique<rgpot::LepsPot>(rgpot::LepsSystem::HCl2);
  case RGPOT_BUILTIN_LEPS_FH2:
    dimensions = 9;
    return std::make_unique<rgpot::LepsPot>(rgpot::LepsSystem::FH2);
  case RGPOT_BUILTIN_CBE:
#ifdef RGPOT_HAS_CBE
    dimensions = 18;
    return std::make_unique<rgpot::CBEPot>();
#else
    throw std::runtime_error("rgpot was built without the external CBE frontend");
#endif
  }
  throw std::invalid_argument("unknown rgpot built-in model kind");
}

void requireBuffers(const rgpot_builtin_model *model, std::size_t count,
                    const double *coordinates, const double *other,
                    double *output) {
  if (model == nullptr)
    throw std::invalid_argument("rgpot built-in model is null");
  if (count == 0)
    throw std::invalid_argument("rgpot built-in model batch is empty");
  if (coordinates == nullptr || output == nullptr)
    throw std::invalid_argument("rgpot built-in model buffer is null");
  if (other == nullptr)
    throw std::invalid_argument("rgpot built-in model direction is null");
  if (count > std::numeric_limits<std::size_t>::max() / model->dimensions)
    throw std::overflow_error("rgpot built-in model batch size overflow");
  const std::size_t entries = count * model->dimensions;
  for (std::size_t i = 0; i < entries; ++i) {
    if (!std::isfinite(coordinates[i]) || !std::isfinite(other[i]))
      throw std::invalid_argument("rgpot built-in model input is not finite");
  }
}

std::size_t atomCount(std::size_t dimensions) {
  if (dimensions == 0)
    throw std::invalid_argument("rgpot built-in model has no coordinates");
  if (dimensions > std::numeric_limits<std::size_t>::max() - 2)
    throw std::overflow_error("rgpot built-in model dimension overflow");
  return (dimensions + 2) / 3;
}
} // namespace

extern "C" rgpot_builtin_model *
rgpot_builtin_model_new(rgpot_builtin_kind kind) {
  return guardedNew([kind]() -> rgpot_builtin_model * {
    std::size_t dimensions = 0;
    auto potential = defaultPotential(kind, dimensions);
    return new rgpot_builtin_model{std::move(potential), dimensions};
  });
}

extern "C" rgpot_builtin_model *rgpot_builtin_double_well_new(
    std::size_t dimensions, double v0, double x0, double omega,
    double coupling) {
  return guardedNew([=]() -> rgpot_builtin_model * {
    if (dimensions == 0)
      throw std::invalid_argument("double well has no coordinates");
    auto potential = std::make_unique<rgpot::DoubleWellPot>(
        dimensions, v0, x0, omega, coupling);
    return new rgpot_builtin_model{std::move(potential), dimensions};
  });
}

extern "C" rgpot_builtin_model *rgpot_builtin_coupled_proton_well_new(
    double v0, double x0, double spring) {
  return guardedNew([=]() -> rgpot_builtin_model * {
    auto potential = std::make_unique<rgpot::CoupledProtonWellPot>(
        v0, x0, spring);
    return new rgpot_builtin_model{std::move(potential), 2};
  });
}

extern "C" void rgpot_builtin_model_free(rgpot_builtin_model *model) {
  delete model;
}

extern "C" std::size_t
rgpot_builtin_model_dimensions(const rgpot_builtin_model *model) {
  if (model == nullptr) {
    lastError = "rgpot built-in model is null";
    return 0;
  }
  lastError.clear();
  return model->dimensions;
}

extern "C" int rgpot_builtin_model_force(
    rgpot_builtin_model *model, std::size_t count, const double *coordinates,
    double *energies, double *forces) {
  return guardedStatus([=]() {
    requireBuffers(model, count, coordinates, coordinates, forces);
    if (energies == nullptr)
      throw std::invalid_argument("rgpot built-in model energy buffer is null");
    const std::size_t atoms = atomCount(model->dimensions);
    const std::size_t carrier = 3 * atoms;
    std::vector<int> atomicNumbers(atoms, 1);
    std::array<std::array<double, 3>, 3> box{};
    rgpot::types::AtomMatrix positions(atoms, 3);
    for (std::size_t row = 0; row < count; ++row) {
      for (std::size_t i = 0; i < carrier; ++i)
        positions(i / 3, i % 3) =
            i < model->dimensions
                ? coordinates[row * model->dimensions + i]
                : 0.0;
      auto [energy, force, variance] =
          (*model->potential)(positions, atomicNumbers, box);
      (void)variance;
      energies[row] = energy;
      for (std::size_t i = 0; i < model->dimensions; ++i)
        forces[row * model->dimensions + i] = force(i / 3, i % 3);
    }
  });
}

extern "C" int rgpot_builtin_model_hvp(
    rgpot_builtin_model *model, std::size_t count, const double *coordinates,
    const double *directions, double *products) {
  return guardedStatus([=]() {
    requireBuffers(model, count, coordinates, directions, products);
    if (!model->potential->caps().analyticHvp)
      throw std::runtime_error("rgpot built-in model has no analytic HVP");
    const std::size_t atoms = atomCount(model->dimensions);
    const std::size_t carrier = 3 * atoms;
    std::vector<int> atomicNumbers(atoms, 1);
    std::array<double, 9> box{};
    std::vector<double> positions(carrier);
    std::vector<double> direction(carrier);
    std::vector<double> product(carrier);
    for (std::size_t row = 0; row < count; ++row) {
      std::fill(positions.begin(), positions.end(), 0.0);
      std::fill(direction.begin(), direction.end(), 0.0);
      std::copy_n(coordinates + row * model->dimensions, model->dimensions,
                  positions.begin());
      std::copy_n(directions + row * model->dimensions, model->dimensions,
                  direction.begin());
      rgpot::ForceInput input{atoms, positions.data(), atomicNumbers.data(),
                              box.data()};
      model->potential->hessianVectorImpl(input, direction.data(),
                                          product.data());
      std::copy_n(product.begin(), model->dimensions,
                  products + row * model->dimensions);
    }
  });
}

extern "C" const char *rgpot_builtin_model_last_error() {
  return lastError.c_str();
}
