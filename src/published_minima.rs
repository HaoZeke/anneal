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

/// Energy above a published minimum that the harness still counts as a stop.
///
/// A halt limit equal to the minimum, or equal to the minimum plus this
/// window, is legal. A higher limit is kept, with a warning.
pub const HARNESS_WINDOW: f64 = 1e-3;

/// C1 icosahedral LJ75 minimum.
///
/// The Cambridge N ≤ 150 table lists it under "Lowest energy icosahedral
/// minima", not in the global-minima column. It is not a stop target.
pub const LJ75_ICOSAHEDRAL: f64 = -396.282249;

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

/// Icosahedral shelf that must not be installed as a stop for this size.
fn icosahedral_shelf(n: usize) -> Option<f64> {
    match n {
        75 => Some(LJ75_ICOSAHEDRAL),
        98 => Some(LJ98_ICOSAHEDRAL),
        _ => None,
    }
}

fn shelf_error(n: usize) -> &'static str {
    match n {
        75 => "LJ75 stop target is the C1 icosahedral minimum, not the D5h global minimum",
        _ => {
            "LJ98 stop target is the Cs icosahedral minimum (Deaven), not the Td global minimum (Leary and Doye 1999)"
        }
    }
}

/// Check a halt limit against the published Lennard-Jones minimum for `n`.
///
/// `limit` is the energy at which a search halts (`energy <= limit`).
/// Installing the LJ75 or LJ98 icosahedral shelf is an error. When
/// [`lennard_jones`] knows `n`, a limit above that minimum plus
/// [`HARNESS_WINDOW`] is kept and a warning is written. The minimum itself
/// and the minimum plus the window are quiet. A non-finite limit is left
/// unchanged and is not a shelf.
pub fn check_stop_limit(n: usize, limit: f64) -> Result<f64, &'static str> {
    if !limit.is_finite() {
        return Ok(limit);
    }
    if let Some(shelf) = icosahedral_shelf(n)
        && limit == shelf
    {
        return Err(shelf_error(n));
    }
    if let Some(minimum) = lennard_jones(n)
        && limit > minimum + HARNESS_WINDOW
    {
        eprintln!(
            "warning: LJ{n} halt limit {limit} sits above the published minimum {minimum} plus the harness window {HARNESS_WINDOW}"
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
        ledger.stop_below(98, LJ98_GLOBAL).unwrap();
        let state = Array1::zeros(3);
        ledger.record(LJ98_ICOSAHEDRAL, state.view());
        assert!(!ledger.below_target(LJ98_ICOSAHEDRAL));
        assert!(!ledger.target_reached());
        ledger.record(LJ98_GLOBAL, state.view());
        assert!(ledger.below_target(LJ98_GLOBAL));
        assert!(ledger.target_reached());
        let window = LJ98_GLOBAL + HARNESS_WINDOW;
        assert!(check_stop_limit(98, window).is_ok());
        assert!(window < LJ98_ICOSAHEDRAL);
        assert_eq!(check_stop_limit(98, LJ98_ICOSAHEDRAL), Err(shelf_error(98)));
        // 1e-4 above the shelf is not the shelf. It sits above the
        // tetrahedral minimum plus the harness window, so it warns.
        assert!(check_stop_limit(98, LJ98_ICOSAHEDRAL + 1e-4).is_ok());
        assert!(check_stop_limit(75, LJ98_ICOSAHEDRAL).is_ok());
    }

    #[test]
    fn a_known_shelf_is_refused_and_not_installed() {
        let mut ledger = Ledger::new(4);
        assert_eq!(
            ledger.stop_below(98, LJ98_ICOSAHEDRAL),
            Err(shelf_error(98))
        );
        assert!(ledger.target().is_none());
        assert_eq!(
            ledger.stop_below(75, LJ75_ICOSAHEDRAL),
            Err(shelf_error(75))
        );
        assert!(ledger.target().is_none());
        let minimum = lennard_jones(75).unwrap();
        assert!(LJ75_ICOSAHEDRAL > minimum + HARNESS_WINDOW);
        assert!(ledger.stop_below(75, minimum).is_ok());
        assert_eq!(ledger.target(), Some(minimum));
        assert!(ledger.stop_below(75, minimum + HARNESS_WINDOW).is_ok());
        assert_eq!(ledger.target(), Some(minimum + HARNESS_WINDOW));
    }

    #[test]
    fn a_halt_above_the_window_warns_and_is_kept() {
        let minimum = LJ98_GLOBAL;
        assert!(check_stop_limit(98, minimum).is_ok());
        assert!(check_stop_limit(98, minimum + HARNESS_WINDOW).is_ok());
        let above = minimum + HARNESS_WINDOW + 1e-9;
        assert!(check_stop_limit(98, above).is_ok());
        // -543.641 is farther than 1e-3 from the Cs shelf, so the old ball
        // accepted it. It is still above the tetrahedral minimum plus the window.
        let missed = -543.641;
        assert!(check_stop_limit(98, missed).is_ok());
        assert!(missed > minimum + HARNESS_WINDOW);
        assert!(missed != LJ98_ICOSAHEDRAL);
        let mut ledger = Ledger::new(4);
        ledger.stop_below(98, above).unwrap();
        assert_eq!(ledger.target(), Some(above));
        assert!(check_stop_limit(4, -0.5).is_ok());
        assert!(check_stop_limit(98, f64::NAN).is_ok());
        assert!(check_stop_limit(98, f64::INFINITY).is_ok());
    }
}
