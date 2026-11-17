// MIT License
// Copyright 2023--present rgpot developers

#include "rgpot/CBEPot/CBEPot.hpp"
#include "rgpot/units.hpp"

#include <cbe.h>

#include <array>
#include <mutex>
#include <stdexcept>

namespace rgpot {
namespace {
std::once_flag initialized;
std::mutex cbeMutex;

void require(cbe_status status) {
  if (status != CBE_SUCCESS)
    throw std::runtime_error(cbe_status_string(status));
}

void initialize() {
  std::call_once(initialized, [] { require(cbe_initialize()); });
}
} // namespace

CBEPot::CBEPot() : Potential(PotType::CBE) { initialize(); }

void CBEPot::forceImpl(const ForceInput &in, ForceOut *out) const {
  if (in.nAtoms != CBE_ATOMS)
    throw std::invalid_argument("CBE requires six atoms");
  std::array<double, CBE_COORDINATES> x{}, force{};
  for (std::size_t i = 0; i < x.size(); ++i)
    x[i] = in.pos[i] / units::BOHR_TO_ANGSTROM;
  double energy = 0.0;
  std::lock_guard<std::mutex> lock(cbeMutex);
  require(cbe_evaluate(1, x.data(), &energy, force.data()));
  out->energy = energy * units::HARTREE_TO_EV;
  for (std::size_t i = 0; i < force.size(); ++i)
    out->F[i] =
        force[i] * units::HARTREE_TO_EV / units::BOHR_TO_ANGSTROM;
}

void CBEPot::hessianVectorImpl(const ForceInput &in, const double *direction,
                               double *product) const {
  if (in.nAtoms != CBE_ATOMS)
    throw std::invalid_argument("CBE requires six atoms");
  std::array<double, CBE_COORDINATES> x{}, v{}, hv{};
  for (std::size_t i = 0; i < x.size(); ++i) {
    x[i] = in.pos[i] / units::BOHR_TO_ANGSTROM;
    v[i] = direction[i] / units::BOHR_TO_ANGSTROM;
  }
  std::lock_guard<std::mutex> lock(cbeMutex);
  require(cbe_hessian_vector_product(1, x.data(), v.data(), hv.data()));
  for (std::size_t i = 0; i < hv.size(); ++i)
    product[i] = hv[i] * units::HARTREE_TO_EV / units::BOHR_TO_ANGSTROM;
}

} // namespace rgpot
