// SPDX-License-Identifier: LGPL-2.1-or-later
// Copyright 2023--present rgpot developers
//
// PLUMED collective variable RGPOT. This file includes the PLUMED
// headers, so it is distributed under the PLUMED license. The rest of
// rgpot stays MIT.
//
//   RGPOT ATOMS=... POTENTIAL=Name [OPTIONS={key value ...}] TYPES=El[,El...] [NOPBC]
//
// POTENTIAL and OPTIONS are the name and options of pair_coeff * * Name
// (engines/pot_spec.hpp), so option values are Angstrom and eV whatever
// UNITS says. Positions, the cell, the value and its derivatives are in
// PLUMED's units and converted with its Units object. The value is the
// energy; its atomic derivatives are dE/dx and its box derivatives are
// -V sigma from the potential's stress, so BIASVALUE applies the rgpot
// forces and virial.

#include "plumed/colvar/Colvar.h"
#include "plumed/core/ActionRegister.h"
#include "plumed/core/Atoms.h"
#include "plumed/core/PlumedMain.h"
#include "plumed/tools/Pbc.h"
#include "plumed/tools/Tools.h"
#include "plumed/tools/Units.h"

#include "pot_spec.hpp"
#include "rgpot/units.hpp"

#include <algorithm>
#include <cctype>
#include <cmath>
#include <exception>
#include <string>
#include <vector>

namespace PLMD {
namespace colvar {

class Rgpot : public Colvar {
 public:
  static void registerKeywords(Keywords &keys);
  explicit Rgpot(const ActionOptions &ao);
  void calculate() override;

 private:
  /// PLUMED 2.9 reports only "An error happened while calculating" for an
  /// error raised in calculate(), so the reason also goes to the log.
  [[noreturn]] void fail(const std::string &why);

  rgpot::spec::Built built_;
  std::string name_;
  std::vector<int> z_;
  bool nopbc_ = false;
};

PLUMED_REGISTER_ACTION(Rgpot, "RGPOT")

void Rgpot::registerKeywords(Keywords &keys) {
  Colvar::registerKeywords(keys);
  keys.add("atoms", "ATOMS", "Atoms the potential sees");
  keys.add("compulsory", "POTENTIAL",
           "rgpot potential name, as in pair_coeff * * Name (LJ, Morse, SuttonChen, ...)");
  keys.add("optional", "OPTIONS",
           "Options of the potential as {key value ...}, in Angstrom and eV, as in pair_coeff");
  keys.add("compulsory", "TYPES",
           "Element symbol or atomic number of each atom in ATOMS, or one for all of them");
}

namespace {

int atomicNumber(const std::string &word) {
  if (!word.empty() && std::all_of(word.begin(), word.end(),
                                   [](unsigned char c) { return std::isdigit(c) != 0; })) {
    const int z = std::stoi(word);
    return z >= 1 && z <= 118 ? z : 0;
  }
  return rgpot::spec::atomicNumber(word);
}

} // namespace

Rgpot::Rgpot(const ActionOptions &ao) : PLUMED_COLVAR_INIT(ao) {
  std::vector<AtomNumber> atoms;
  parseAtomList("ATOMS", atoms);
  parse("POTENTIAL", name_);
  std::string option_line;
  parse("OPTIONS", option_line);
  std::vector<std::string> types;
  parseVector("TYPES", types);
  parseFlag("NOPBC", nopbc_);
  checkRead();

  if (atoms.empty()) {
    error("RGPOT needs ATOMS");
  }
  const std::vector<std::string> words = Tools::getWords(option_line);
  if (words.size() % 2 != 0) {
    error("RGPOT OPTIONS takes key value pairs, got: " + option_line);
  }
  rgpot::spec::Options options;
  for (std::size_t k = 0; k + 1 < words.size(); k += 2) {
    options.emplace_back(words[k], words[k + 1]);
  }
  if (types.size() != 1 && types.size() != atoms.size()) {
    error("RGPOT TYPES takes one entry or one per atom (" + std::to_string(atoms.size()) +
          "), got " + std::to_string(types.size()));
  }
  z_.resize(atoms.size());
  for (std::size_t i = 0; i < atoms.size(); ++i) {
    const std::string &word = types[types.size() == 1 ? 0 : i];
    z_[i] = atomicNumber(word);
    if (z_[i] == 0) {
      error("RGPOT TYPES: " + word + " is not an element symbol or atomic number");
    }
  }
  built_ = rgpot::spec::build(name_, options, 0.0);
  if (!built_.pot) {
    error("RGPOT: " + built_.error);
  }
  const rgpot::PotCaps caps = built_.pot->caps();
  if (!caps.periodic && !nopbc_) {
    error("RGPOT: " + name_ + " takes no periodic cell; add NOPBC");
  }
  log.printf("  rgpot potential %s with %zu option(s), %s\n", name_.c_str(), options.size(),
             caps.stress ? "stress available" : "no stress, so NOPBC or a cell-free run only");

  addValueWithDerivatives();
  setNotPeriodic();
  requestAtoms(atoms);
}

void Rgpot::fail(const std::string &why) {
  log.printf("  %s\n", why.c_str());
  log.flush();
  error(why);
}

void Rgpot::calculate() {
  if (plumed.getAtoms().usingNaturalUnits()) {
    fail("RGPOT cannot convert natural units to Angstrom and eV");
  }
  const Units &units = plumed.getAtoms().getUnits();
  const double to_angstrom =
      units.getLength() * rgpot::units::unit_conversion_factor("nm", "angstrom");
  const double to_ev = units.getEnergy() * rgpot::units::unit_conversion_factor("kJ/mol", "eV");
  const auto &positions = getPositions();
  const std::size_t n = positions.size();
  const bool periodic = !nopbc_ && getPbc().isSet();

  std::vector<double> xyz(3 * n);
  double box[9] = {};
  double volume = 0.0;
  if (periodic) {
    const Tensor &cell = getBox();
    for (int row = 0; row < 3; ++row) {
      for (int col = 0; col < 3; ++col) {
        box[3 * row + col] = cell(row, col) * to_angstrom;
      }
    }
    volume = std::abs(cell.determinant()) * to_angstrom * to_angstrom * to_angstrom;
    for (std::size_t i = 0; i < n; ++i) {
      Vector s = getPbc().realToScaled(positions[i]);
      for (int k = 0; k < 3; ++k) {
        s[k] -= std::floor(s[k]);
      }
      const Vector wrapped = getPbc().scaledToReal(s);
      for (int k = 0; k < 3; ++k) {
        xyz[3 * i + k] = wrapped[k] * to_angstrom;
      }
    }
  } else {
    // Kernels apply the minimum image to the cell they are given. A cube
    // wider than twice the cluster plus the cutoff keeps every real pair
    // and no periodic image.
    double lo[3] = {positions[0][0], positions[0][1], positions[0][2]};
    double hi[3] = {lo[0], lo[1], lo[2]};
    for (const Vector &p : positions) {
      for (int k = 0; k < 3; ++k) {
        lo[k] = std::min(lo[k], p[k]);
        hi[k] = std::max(hi[k], p[k]);
      }
    }
    double extent = 0.0;
    for (int k = 0; k < 3; ++k) {
      extent = std::max(extent, (hi[k] - lo[k]) * to_angstrom);
    }
    const double side = std::max(100.0, 2.0 * (extent + built_.cutoff) + 10.0);
    box[0] = box[4] = box[8] = side;
    for (std::size_t i = 0; i < n; ++i) {
      for (int k = 0; k < 3; ++k) {
        xyz[3 * i + k] = (positions[i][k] - 0.5 * (lo[k] + hi[k])) * to_angstrom + 0.5 * side;
      }
    }
  }

  std::vector<double> forces(3 * n, 0.0);
  rgpot::ForceInput in{n, xyz.data(), z_.data(), box};
  rgpot::ForceOut out{forces.data(), 0.0, 0.0, {}, 0};
  try {
    built_.eval(in, &out);
  } catch (const std::exception &e) {
    fail(std::string("RGPOT: ") + name_ + ": " + e.what());
  } catch (...) {
    fail("RGPOT: " + name_ + " failed on this geometry with a non-standard exception");
  }

  setValue(out.energy / to_ev);
  const double force_scale = to_angstrom / to_ev;
  for (std::size_t i = 0; i < n; ++i) {
    setAtomsDerivatives(static_cast<int>(i),
                        Vector(-forces[3 * i] * force_scale, -forces[3 * i + 1] * force_scale,
                               -forces[3 * i + 2] * force_scale));
  }
  if (!periodic) {
    setBoxDerivativesNoPbc();
    return;
  }
  if (out.has_stress == 0) {
    fail("RGPOT: " + name_ +
          " returns no stress, so the box derivatives of a periodic cell are unknown. "
          "Add NOPBC for an isolated cluster, or use a potential with a stress");
  }
  // PLUMED's box derivative is -sum_i x_i (x) dE/dx_i = -dE/dstrain^T.
  Tensor virial;
  for (int a = 0; a < 3; ++a) {
    for (int b = 0; b < 3; ++b) {
      virial(a, b) = -volume * out.stress[3 * b + a] / to_ev;
    }
  }
  setBoxDerivatives(virial);
}

} // namespace colvar
} // namespace PLMD
