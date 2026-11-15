#pragma once
// MIT License
// Copyright 2023--present rgpot developers
//
// Builds one rgpot Potential from a PotType name and key value options,
// for hosts that take a potential from a text line (pair_style rgpot).
// Names are the PotType enumerators, matched without case. Lengths are
// Angstrom and energies eV.

#include "rgpot/Potential.hpp"

#include <functional>
#include <memory>
#include <string>
#include <utility>
#include <vector>

namespace rgpot::spec {

using Options = std::vector<std::pair<std::string, std::string>>;

struct Built {
  std::unique_ptr<PotentialBase> pot;
  /// The kernel's own forceImpl, which writes the stress when caps().stress
  /// is set. ForceOut and its force buffer must arrive zeroed.
  std::function<void(const ForceInput &, ForceOut *)> eval;
  std::string error; ///< Reason when pot is null.
};

/// cutoff is used when the potential has a cutoff and the options do
/// not name one.
Built build(const std::string &name, const Options &options, double cutoff);

/// Names build() accepts in this build, separated by spaces.
std::string names();

/// Atomic number of an element symbol (case as printed, "Cu"), or 0.
int atomicNumber(const std::string &symbol);

} // namespace rgpot::spec
