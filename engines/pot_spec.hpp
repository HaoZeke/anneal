#pragma once
// MIT License
// Copyright 2023--present rgpot developers
//
// Builds one rgpot Potential from a PotType name and key value options,
// for hosts that take a potential from a text line (pair_style rgpot).
// Names are the PotType enumerators, matched without case. Lengths are
// Angstrom and energies eV.

#include "rgpot/Locality.hpp"
#include "rgpot/PerAtom.hpp"
#include "rgpot/Potential.hpp"
#include "rgpot/Staged.hpp"

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
  /// The kernel's forcePerAtom (PerAtom.hpp), which also fills the per-atom
  /// energies and virials. Empty when the potential cannot split them.
  std::function<void(const ForceInput &, ForceOut *, const PerAtomOut &)> perAtom;
  /// Cutoff of a potential that takes one (LJ, Morse, SuttonChen), in
  /// Angstrom. 0 for a kernel with a fixed range.
  double cutoff = 0.0;
  /// The kernel's locality() (Locality.hpp). Zero cutoff when the kernel
  /// declares no finite range.
  Locality locality;
  /// The kernel's densityStage and forceStage (Staged.hpp) on a host's
  /// neighbour list. Empty when the kernel has no staged evaluation.
  std::function<void(const HalfList &, double *)> densityStage;
  std::function<void(const HalfList &, const double *, double *, StageTally *)> forceStage;
  std::string error; ///< Reason when pot is null.
};

/// cutoff is used when the potential has a cutoff and the options do
/// not name one. A cutoff <= 0 keeps the potential's own default.
Built build(const std::string &name, const Options &options, double cutoff);

/// Names build() accepts in this build, separated by spaces.
std::string names();

/// Atomic number of an element symbol (case as printed, "Cu"), or 0.
int atomicNumber(const std::string &symbol);

} // namespace rgpot::spec
