//! Cambridge Cluster Database energies used as stops and scores.
//!
//! Lennard-Jones values are the global-minima columns of the N ≤ 150 table,
//! <https://www-wales.ch.cam.ac.uk/~jon/structures/LJ/tables.150.html>,
//! read on 2026-10-26. The unit is the pair well depth. The second table on
//! that page, "Lowest energy icosahedral minima", is not a stop target.
//!
//! Morse values are the bold entries of
//! <https://www-wales.ch.cam.ac.uk/~jon/structures/Morse/tables.html>
//! (Doye and Wales). Bold marks the global minimum at that range. The energy
//! columns are ρ₀ = 3, 6, 10 and 14. The key used here is `(ρ₀ * 2).round()`,
//! so ρ₀ = 6, 10, 14 are 12, 20 and 28.

/// Cs icosahedral LJ98 minimum.
///
/// Deaven, Tit, Morris and Ho, Chem. Phys. Lett. 256, 195 (1996). The
/// Cambridge page lists it under the icosahedral minima, points file `98i`.
/// It is not the global minimum.
pub const LJ98_ICOSAHEDRAL: f64 = -543.642957;

/// Td LJ98 global minimum.
///
/// R. H. Leary and J. P. K. Doye, Phys. Rev. E 60, R6320 (1999). The
/// Cambridge page lists point group Td, points file `98`, reference Leary2.
pub const LJ98_GLOBAL: f64 = -543.665361;

/// Published Lennard-Jones global minimum for `n`, when this crate scores that size.
pub fn lennard_jones(n: usize) -> Option<f64> {
    Some(match n {
        13 => -44.326801,
        38 => -173.928427,
        55 => -279.248470,
        75 => -397.492331,
        98 => LJ98_GLOBAL,
        102 => -569.363652,
        104 => -582.086642,
        150 => -893.310258,
        _ => return None,
    })
}

/// Bold Doye–Wales Morse global minimum at range `rho` for `n` points.
pub fn morse(n: usize, rho: f64) -> Option<f64> {
    let key = (n, (rho * 2.0).round() as i64);
    Some(match key {
        (38, 12) => -157.477108,
        (55, 12) => -250.286609,
        (75, 12) => -351.472365,
        (38, 20) => -145.849817,
        (55, 20) => -225.814286,
        (75, 20) => -322.643558,
        (38, 28) => -144.321054,
        (55, 28) => -220.646208,
        (75, 28) => -318.407330,
        (80, 28) => -340.811371,
        _ => return None,
    })
}

/// Refuse a stop threshold that is the LJ98 icosahedral minimum.
///
/// `limit` is the energy at which a search halts (`energy <= limit`). The
/// tetrahedral minimum plus the usual 0.001 window sits 0.021 below the
/// icosahedral shelf, so that window is kept. A threshold within `1e-3` of
/// the Deaven energy is the shelf itself.
pub fn lj98_stop_limit(n: usize, limit: f64) -> Result<f64, &'static str> {
    if n == 98 && (limit - LJ98_ICOSAHEDRAL).abs() <= 1e-3 {
        return Err(
            "LJ98 stop target is the Cs icosahedral minimum (Deaven), not the Td global minimum (Leary and Doye 1999)",
        );
    }
    Ok(limit)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::methods::cluster_hopping::Ledger;
    use ndarray::Array1;

    #[test]
    fn lj98_stop_is_the_tetrahedral_minimum() {
        assert_eq!(lennard_jones(98), Some(LJ98_GLOBAL));
        assert!((LJ98_ICOSAHEDRAL - LJ98_GLOBAL - 0.022404).abs() < 1e-12);
        let mut ledger = Ledger::new(8);
        ledger.stop_below(lj98_stop_limit(98, LJ98_GLOBAL).unwrap());
        let state = Array1::zeros(3);
        ledger.record(LJ98_ICOSAHEDRAL, state.view());
        assert!(!ledger.below_target(LJ98_ICOSAHEDRAL));
        assert!(!ledger.target_reached());
        ledger.record(LJ98_GLOBAL, state.view());
        assert!(ledger.below_target(LJ98_GLOBAL));
        assert!(ledger.target_reached());
        // The success window used by the harness sits below the icosahedral shelf.
        let window = LJ98_GLOBAL + 1e-3;
        assert!(lj98_stop_limit(98, window).is_ok());
        assert!((window - LJ98_ICOSAHEDRAL).abs() > 1e-3);
        assert!(lj98_stop_limit(98, LJ98_ICOSAHEDRAL).is_err());
        assert!(lj98_stop_limit(98, LJ98_ICOSAHEDRAL + 1e-4).is_err());
        assert!(lj98_stop_limit(75, LJ98_ICOSAHEDRAL).is_ok());
    }
}
