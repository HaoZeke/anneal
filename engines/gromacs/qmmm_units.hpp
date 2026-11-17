#pragma once
// MIT License
// Copyright 2023--present rgpot developers
//
// The constants GROMACS' QM/MM module converts with, written as in
// src/gromacs/math/units.h of GROMACS 2024 through 2026 (CODATA 2018) so
// each one has the same bits. GROMACS sends positions in nm / c_bohr2Nm
// and multiplies the returned energy by c_hartree2Kj * c_avogadro and the
// forces by c_hartreeBohr2Md. The factors below undo that, so the energy
// GROMACS reports is the rgpot energy times rgpot::md::kKjPerEv.

#include "rgpot/ExternalPot/md_units.hpp"

namespace rgpot::gromacs {

inline constexpr double kAngstrom = 1e-10;
inline constexpr double kKilo = 1e3;
inline constexpr double kNano = 1e-9;
inline constexpr double kPico = 1e-12;
inline constexpr double kNm2A = kNano / kAngstrom;
inline constexpr double kAvogadro = 6.02214076e23;
inline constexpr double kPlanck1 = 6.62607015e-34;
inline constexpr double kPlanck = (kPlanck1 * kAvogadro / (kPico * kKilo));
inline constexpr double kSpeedOfLight = 2.99792458e05;
inline constexpr double kRydberg = 1.0973731568160e-02;
inline constexpr double kHartree2Kj = ((2.0 * kRydberg * kPlanck * kSpeedOfLight) / kAvogadro);
inline constexpr double kBohr2Nm = 0.0529177210903;
inline constexpr double kHartreeBohr2Md = (kHartree2Kj * kAvogadro / kBohr2Nm);

/// Bohr as GROMACS sends it, to Angstrom.
inline constexpr double kAngstromPerBohr = kBohr2Nm * kNm2A;
/// eV to the Hartree that GROMACS turns into rgpot's kJ/mol.
inline constexpr double kHartreePerEv = md::kKjPerEv / (kHartree2Kj * kAvogadro);
/// eV/Angstrom to the Hartree/Bohr that GROMACS turns into kJ/(mol nm).
inline constexpr double kHartreeBohrPerEvAngstrom = md::kKjPerEv * kNm2A / kHartreeBohr2Md;

} // namespace rgpot::gromacs
