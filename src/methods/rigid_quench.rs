//! Rigid-body record tolerance, and when a rigid walker is replaced.
//!
//! The hop quench is the same Wolfe L-BFGS as a Cartesian quench. A
//! rigid record accepts the root-mean-square force at the ranked
//! tolerance of `1e-3`. A walker that goes 1000 hops without a new
//! lowest minimum returns to that minimum when it lies within one
//! hopping temperature of the run target. Farther above the target the
//! walker is drawn again. That 1000 is the restart-file
//! interval in the source. The compiled reseed window is zero and is
//! not used.

use ndarray::ArrayView1;

/// RMS force at which a rigid quench may be recorded. Ranked `SLOPPYCONV`.
pub(crate) const SLOPPY_RMS: f64 = 1.0e-3;
/// Hops without a new lowest minimum before a rigid walker may be replaced.
///
/// GMIN's reseed predicate is `J1 - JBEST > NRELAX`. The compiled
/// `NRELAX` is 0, which would reseed on every hop that does not lower
/// the best, and the ranked TIP4P file leaves `NEWRESTART` off. The
/// only positive MC-cycle count initialised beside a restart in
/// `keywords.f90` is `DUMPINT = 1000`: `mc.f` writes a restart file
/// every 1000 cycles. The reseed waits that many hops. It is not
/// `NRELAX`. Inside one hopping temperature of the run target the
/// walker returns to the lowest minimum. A fresh draw there quenches
/// far above that minimum. Farther above the target the fresh draw
/// is the escape.
pub(crate) const STAGNATION_HOPS: usize = 1000;
/// Failed reseeds tried before the walker is left where it is.
/// `NEWRESATTEMPTS` stops after ten.
pub(crate) const RESEED_ATTEMPTS: usize = 10;

/// True when `hops` since the last new lowest minimum exceeds [`STAGNATION_HOPS`].
pub(crate) fn stagnation_restart(hops: usize, hops_at_lowest: usize) -> bool {
    hops.saturating_sub(hops_at_lowest) > STAGNATION_HOPS
}

/// Root-mean-square force, `sqrt(g · g / n)`.
pub(crate) fn rms_force(gradient: ArrayView1<f64>) -> f64 {
    let n = gradient.len();
    if n == 0 || gradient.iter().any(|value| !value.is_finite()) {
        return f64::INFINITY;
    }
    (gradient.dot(&gradient) / n as f64).sqrt()
}

/// Whether a rigid quench is tight enough to record.
pub(crate) fn rms_is_converged(
    gradient: ArrayView1<f64>,
    dimensions: usize,
    tolerance: f64,
) -> bool {
    gradient.len() == dimensions && dimensions > 0 && rms_force(gradient) <= tolerance
}

#[cfg(test)]
mod tests {
    use super::*;
    use ndarray::Array1;

    #[test]
    fn the_rms_force_is_the_root_mean_square() {
        let gradient = Array1::from_vec(vec![3.0, 4.0]);
        assert!((rms_force(gradient.view()) - (12.5_f64).sqrt()).abs() < 1e-12);
        assert!(rms_is_converged(gradient.view(), 2, 4.0));
        assert!(!rms_is_converged(gradient.view(), 2, SLOPPY_RMS));
        assert!(!rms_is_converged(gradient.view(), 1, 10.0));
    }

    #[test]
    fn stagnation_waits_for_the_dump_interval() {
        assert_eq!(STAGNATION_HOPS, 1000);
        assert_eq!(RESEED_ATTEMPTS, 10);
        assert!(!stagnation_restart(1000, 0));
        assert!(stagnation_restart(1001, 0));
        assert!(!stagnation_restart(1500, 500));
        assert!(stagnation_restart(1501, 500));
    }
}
