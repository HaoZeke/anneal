//! Shared one-sided target-energy hit detection.

/// Default hit tolerance above a reference target energy.
pub const DEFAULT_HIT_TOL: f64 = 1.0e-3;

/// Hit window of the cluster-search solved flag and the first encounter.
///
/// Solved flags and first encounters were already scored at 1e-4.
/// The ensemble SOLVED marker, `tally.solved` and `first_target_calls` moved
/// from reference + 1e-3 to reference + 1e-4. Widening that window back to
/// [`DEFAULT_HIT_TOL`] would count a different set of runs.
pub const SEARCH_HIT_TOL: f64 = 1.0e-4;

/// Whether `energy` hits `target` under the one-sided rule `energy <= target + tol`.
///
/// Non-finite values (including `NaN` and infinities) and negative tolerances
/// never count as a hit.
#[inline]
pub fn energy_hits(energy: f64, target: f64, tol: f64) -> bool {
    energy.is_finite()
        && target.is_finite()
        && tol.is_finite()
        && tol >= 0.0
        && energy <= target + tol
}

#[cfg(test)]
mod tests {
    use super::{DEFAULT_HIT_TOL, SEARCH_HIT_TOL, energy_hits};

    #[test]
    fn exact_boundary_target_plus_tol_is_a_hit() {
        let target = -173.928427;
        let tol = DEFAULT_HIT_TOL;
        let boundary = target + tol;
        assert!(energy_hits(boundary, target, tol));
        assert!(!energy_hits(boundary.next_up(), target, tol));
        assert!(energy_hits(target, target, 0.0));
        assert!(!energy_hits(target.next_up(), target, 0.0));
    }

    #[test]
    fn energies_below_target_are_hits_regardless_of_gap() {
        let target = -397.492331;
        let tol = DEFAULT_HIT_TOL;
        assert!(energy_hits(target, target, tol));
        assert!(energy_hits(target - tol, target, tol));
        assert!(energy_hits(target - 10.0 * tol, target, tol));
        assert!(energy_hits(target - 100.0, target, tol));
    }

    #[test]
    fn nan_infinities_and_negative_tolerances_never_hit() {
        assert!(!energy_hits(f64::NAN, -10.0, DEFAULT_HIT_TOL));
        assert!(!energy_hits(-10.0, f64::NAN, DEFAULT_HIT_TOL));
        assert!(!energy_hits(-10.0, -10.0, f64::NAN));
        assert!(!energy_hits(f64::NAN, f64::NAN, f64::NAN));
        assert!(!energy_hits(f64::NEG_INFINITY, -10.0, DEFAULT_HIT_TOL));
        assert!(!energy_hits(f64::INFINITY, -10.0, DEFAULT_HIT_TOL));
        assert!(!energy_hits(-10.0, f64::INFINITY, DEFAULT_HIT_TOL));
        assert!(!energy_hits(-10.0, f64::NEG_INFINITY, DEFAULT_HIT_TOL));
        assert!(!energy_hits(-10.0, -10.0, f64::INFINITY));
        assert!(!energy_hits(-10.0, -10.0, f64::NEG_INFINITY));
        assert!(!energy_hits(-11.0, -10.0, -1.0e-12));
    }

    #[test]
    fn the_search_window_rejects_a_hit_the_default_accepts() {
        assert_eq!(SEARCH_HIT_TOL.to_bits(), 1.0e-4_f64.to_bits());
        // `black_box` keeps the comparison out of clippy's constant-assertion
        // lint. The relationship is the contract, and both operands are consts.
        assert!(std::hint::black_box(SEARCH_HIT_TOL) < std::hint::black_box(DEFAULT_HIT_TOL));
        let target = -173.928427;
        let on_search = target + SEARCH_HIT_TOL;
        let between = target + SEARCH_HIT_TOL + (DEFAULT_HIT_TOL - SEARCH_HIT_TOL) / 2.0;
        assert!(energy_hits(on_search, target, SEARCH_HIT_TOL));
        assert!(!energy_hits(on_search.next_up(), target, SEARCH_HIT_TOL));
        assert!(!energy_hits(between, target, SEARCH_HIT_TOL));
        assert!(energy_hits(between, target, DEFAULT_HIT_TOL));
        // The ensemble SOLVED marker, `tally.solved` and `first_target_calls` moved
        // from reference + 1e-3 to reference + 1e-4. They follow that level
        // with no extra slack. The default window would accept `between`.
        assert!(energy_hits(on_search, target + SEARCH_HIT_TOL, 0.0));
        assert!(!energy_hits(between, target + SEARCH_HIT_TOL, 0.0));
        assert!(energy_hits(between, target + DEFAULT_HIT_TOL, 0.0));
    }
}
