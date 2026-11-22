// MIT License
// Copyright 2023--present rgpot developers

//! Native rgpot (eV, Angstrom) to atomic-unit conversion.

/// One Bohr in Angstrom (CODATA 2018).
pub const BOHR_TO_ANGSTROM: f64 = 0.529_177_210_903;

/// One eV in Hartree (CODATA 2018).
pub const EV_TO_HARTREE: f64 = 1.0 / 27.211_386_245_988;

/// Convert a length in Bohr to rgpot's native Angstrom.
#[cfg(feature = "mdi")]
pub(crate) fn bohr_to_angstrom(value: f64) -> f64 {
    value * BOHR_TO_ANGSTROM
}

/// Convert rgpot's native eV to Hartree.
#[cfg(feature = "mdi")]
pub(crate) fn ev_to_hartree(value: f64) -> f64 {
    value * EV_TO_HARTREE
}

/// Convert rgpot's native eV/Angstrom to Hartree/Bohr.
#[cfg(feature = "mdi")]
pub(crate) fn ev_per_angstrom_to_hartree_per_bohr(value: f64) -> f64 {
    value * EV_TO_HARTREE * BOHR_TO_ANGSTROM
}
