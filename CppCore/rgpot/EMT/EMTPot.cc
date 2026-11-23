#include "rgpot/EMT/EMTPot.hpp"

#include "Atoms.h"
#include "EMT.h"
#include "EMTRasmussenParameterProvider.h"
#include "Exception.h"
#include "SuperCell.h"
#include "Vec.h"

#include <algorithm>
#include <cmath>
#include <cstring>
#include <memory>
#include <stdexcept>
#include <string>
#include <vector>

namespace rgpot {

// Members are released in reverse order: the calculator first, the
// provider it may borrow last.
struct EMTPot::State {
  std::vector<int> numbers;
  std::unique_ptr<EMTRasmussenParameterProvider> provider;
  std::unique_ptr<Atoms> atoms;
  std::unique_ptr<SuperCell> cell;
  std::unique_ptr<EMT> emt;
};

namespace {

[[noreturn]] void refuse(const std::string &what) {
  throw std::invalid_argument("EMTPot: " + what);
}

// Asap divides by the cell determinant and bins atoms by scaled
// coordinates, so a cell it cannot invert aborts inside CellList.
void checkBox(const double *box) {
  for (int k = 0; k < 9; ++k) {
    if (!std::isfinite(box[k])) {
      refuse("the cell must be finite");
    }
  }
  const Vec a(box[0], box[1], box[2]);
  const Vec b(box[3], box[4], box[5]);
  const Vec c(box[6], box[7], box[8]);
  const double volume = std::fabs(Cross(a, b) * c);
  const double scale = std::sqrt(Length2(a) * Length2(b) * Length2(c));
  if (!(volume > 1e-12 * scale)) {
    refuse("the cell must be nonsingular");
  }
}

void checkPositions(const double *pos, long n) {
  for (long k = 0; k < 3 * n; ++k) {
    if (!std::isfinite(pos[k])) {
      refuse("the positions must be finite");
    }
  }
}

} // namespace

EMTPot::EMTPot(const EMTConfig &config)
    : Potential(PotType::EMT), m_config(config) {
  Fnv1a fp;
  fp.u64(1);
  fp.u64(config.rasmussen ? 1 : 0);
  m_paramsKey = fp.h;
}

EMTPot::~EMTPot() = default;

void EMTPot::forceImpl(const ForceInput &in, ForceOut *out) const {
  const long n = static_cast<long>(in.nAtoms);
  out->variance = 0.0;
  out->has_stress = 0;
  if (n == 0) {
    out->energy = 0.0;
    return;
  }
  checkBox(in.box);
  checkPositions(in.pos, n);
  std::vector<double> pos(in.pos, in.pos + 3 * n);
  Vec basis[3] = {
      Vec(in.box[0], in.box[1], in.box[2]),
      Vec(in.box[3], in.box[4], in.box[5]),
      Vec(in.box[6], in.box[7], in.box[8]),
  };
  try {
    if (!m_state || !std::equal(in.atmnrs, in.atmnrs + n,
                                m_state->numbers.begin(),
                                m_state->numbers.end())) {
      bool periodic[3] = {true, true, true};
      auto next = std::make_unique<State>();
      next->numbers.assign(in.atmnrs, in.atmnrs + n);
      next->cell = std::make_unique<SuperCell>(basis, periodic);
      next->atoms = std::make_unique<Atoms>(reinterpret_cast<Vec *>(pos.data()),
                                            static_cast<int>(n),
                                            next->cell.get());
      next->atoms->SetAtomicNumbers(next->numbers.data());
      if (m_config.rasmussen) {
        next->provider = std::make_unique<EMTRasmussenParameterProvider>();
        next->emt = std::make_unique<EMT>(next->provider.get());
      } else {
        next->emt = std::make_unique<EMT>(nullptr);
      }
      next->atoms->SetCalculator(next->emt.get());
      m_state = std::move(next);
    }
    Atoms *atoms = m_state->atoms.get();
    EMT *emt = m_state->emt.get();
    atoms->SetCartesianPositions(reinterpret_cast<Vec *>(pos.data()));
    atoms->SetUnitCell(basis, true);
    out->energy = emt->GetPotentialEnergy();
    const Vec *forces = emt->GetCartesianForces();
    std::memcpy(out->F, forces, static_cast<size_t>(n) * sizeof(Vec));
  } catch (const Exception &error) {
    m_state.reset();
    refuse(error.GetMessage());
  } catch (...) {
    m_state.reset();
    throw;
  }
}

} // namespace rgpot
