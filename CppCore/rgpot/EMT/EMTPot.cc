#include "rgpot/EMT/EMTPot.hpp"

#include "Atoms.h"
#include "EMT.h"
#include "EMTRasmussenParameterProvider.h"
#include "SuperCell.h"
#include "Vec.h"

#include <algorithm>
#include <cstring>
#include <vector>

namespace rgpot {

EMTPot::EMTPot(const EMTConfig &config)
    : Potential(PotType::EMT), m_config(config) {
  Fnv1a fp;
  fp.u64(1);
  fp.u64(config.rasmussen ? 1 : 0);
  m_paramsKey = fp.h;
}

EMTPot::~EMTPot() { reset(); }

void EMTPot::reset() const {
  delete static_cast<EMT *>(m_emt);
  m_emt = nullptr;
  delete static_cast<SuperCell *>(m_cell);
  m_cell = nullptr;
  delete static_cast<Atoms *>(m_atoms);
  m_atoms = nullptr;
  delete static_cast<EMTRasmussenParameterProvider *>(m_provider);
  m_provider = nullptr;
  m_nAtoms = 0;
  m_numbers.clear();
}

void EMTPot::forceImpl(const ForceInput &in, ForceOut *out) const {
  const long n = static_cast<long>(in.nAtoms);
  std::vector<double> pos(in.pos, in.pos + 3 * n);
  Vec basis[3] = {
      Vec(in.box[0], in.box[1], in.box[2]),
      Vec(in.box[3], in.box[4], in.box[5]),
      Vec(in.box[6], in.box[7], in.box[8]),
  };
  bool periodic[3] = {true, true, true};
  if (m_nAtoms != n || !std::equal(in.atmnrs, in.atmnrs + n,
                                   m_numbers.begin(), m_numbers.end())) {
    reset();
    m_nAtoms = n;
    m_numbers.assign(in.atmnrs, in.atmnrs + n);
    auto *cell = new SuperCell(basis, periodic);
    auto *atoms = new Atoms(reinterpret_cast<Vec *>(pos.data()), n, cell);
    std::vector<int> numbers(in.atmnrs, in.atmnrs + n);
    atoms->SetAtomicNumbers(numbers.data());
    EMT *emt = nullptr;
    EMTRasmussenParameterProvider *provider = nullptr;
    if (m_config.rasmussen) {
      provider = new EMTRasmussenParameterProvider();
      emt = new EMT(provider);
    } else {
      emt = new EMT(nullptr);
    }
    atoms->SetCalculator(emt);
    m_cell = cell;
    m_atoms = atoms;
    m_provider = provider;
    m_emt = emt;
  }
  auto *atoms = static_cast<Atoms *>(m_atoms);
  auto *emt = static_cast<EMT *>(m_emt);
  atoms->SetCartesianPositions(reinterpret_cast<Vec *>(pos.data()));
  atoms->SetUnitCell(basis, true);
  out->energy = emt->GetPotentialEnergy();
  const Vec *forces = emt->GetCartesianForces();
  std::memcpy(out->F, forces, static_cast<size_t>(n) * sizeof(Vec));
  out->variance = 0.0;
  out->has_stress = 0;
}

} // namespace rgpot
