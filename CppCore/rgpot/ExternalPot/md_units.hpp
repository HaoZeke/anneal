#pragma once
// MIT License
// Copyright 2023--present rgpot developers
//
// GROMACS and OpenMM speak nanometres and kJ/mol. rgpot speaks
// Angstrom and eV. These factors are the SI definitions (exact
// elementary charge and Avogadro number, 2019), so a caller converts
// once and then reuses the stored values.

namespace rgpot::md {

// 1 nm = 10 Angstrom, exactly.
inline constexpr double kAngstromPerNm = 10.0;
inline constexpr double kNmPerAngstrom = 0.1;

// Faraday constant / 1000: kJ/mol in one eV.
// e = 1.602176634e-19 C, N_A = 6.02214076e23 /mol.
inline constexpr double kCoulomb = 1.602176634e-19;
inline constexpr double kAvogadro = 6.02214076e23;
inline constexpr double kKjPerEv = kCoulomb * kAvogadro / 1000.0;
inline constexpr double kEvPerKj = 1.0 / kKjPerEv;

// (kJ/mol)/nm to eV/Angstrom.
inline constexpr double kEvAPerKjNm = kEvPerKj / kAngstromPerNm;

} // namespace rgpot::md
