#pragma once
// MIT License
// Copyright 2023--present rgpot developers
//
// Config POD shared by linked XTBPot and the XTBDlopen frontend.
// No libxtb headers: hosts may compile XTBDlopen without -Dwith_xtb.

#include <string>

namespace rgpot {

enum class GFNMethod { GFNFF, GFN0xTB, GFN1xTB, GFN2xTB };

struct XTBConfig {
  GFNMethod method = GFNMethod::GFN2xTB;
  double accuracy = 1.0;
  double electronic_temperature = 300.0; // Kelvin
  int max_iterations = 250;
  double charge = 0.0;
  /** Number of unpaired electrons (libxtb uhf; multiplicity 2 => uhf 1). */
  int uhf = 0;
  /**
   * Largest atom move, in Angstrom, that still reuses the previous SCC
   * wavefunction as the next guess. A larger move drops the results
   * object, and the next single point uses xtb's own guess. Zero or a
   * negative value disables reuse.
   */
  double warm_move = 1.0;
  /**
   * Smallest HOMO-LUMO gap, in eV, at which the previous SCC wavefunction
   * is still the next guess. Below it the GFN SCC can have several
   * self-consistent solutions, and a carried-over guess can land on a
   * different one than xtb's own guess. Such a geometry is solved from
   * xtb's guess, so the energy does not depend on the call history. Zero
   * or a negative value reuses the wavefunction at any gap.
   */
  double warm_min_gap = 1.0;
};

/** Optional engine path for the dlopen frontend (also RGPOT_XTB_ENGINE). */
struct XTBDlopenConfig {
  XTBConfig xtb{};
  std::string engine_path;
};

} // namespace rgpot
