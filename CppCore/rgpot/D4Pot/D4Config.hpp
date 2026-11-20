#pragma once
// MIT License
// Copyright 2023--present rgpot developers

// The parameters of rgpot::D4Pot. They live apart from the class so the C++
// layout stamp can cover them without dftd4.h.

#include <string>

namespace rgpot {

/**
 * Linked dftd4 pot (NEEDED libdftd4).
 *
 * ``atm`` is the many-body (Axilrod-Teller-Muto) term the C API names
 * ``mdb``. The literature D4-ATM default is on; the flag is an explicit
 * constructor field so tests can pin both states. ``functional`` is the
 * dftd4 method key for damping parameters (for example ``"pbe"``).
 * ``charge`` is the total molecular charge passed to
 * ``dftd4_new_structure``. Charge scaling lives inside dftd4; this
 * wrapper does not reimplement it. ``periodic`` passes the box to dftd4 as
 * a lattice periodic along all three vectors, which then returns the
 * virial; off, the box is ignored and the system is a molecule.
 */
struct D4Config {
  std::string functional = "pbe";
  double charge = 0.0;
  bool atm = true;       //!< many-body ATM (``mdb``); default on (D4-ATM).
  bool periodic = false; //!< 3D lattice from the box, with the stress.
};

} // namespace rgpot
