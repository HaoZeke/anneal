//! Shared one-sided target-energy hit detection.

/// Default hit tolerance above a reference target energy.
pub const DEFAULT_HIT_TOL: f64 = 1.0e-3;

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
    use super::{DEFAULT_HIT_TOL, energy_hits};

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
}
