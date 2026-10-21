//! Rigid-body record tolerance.
//!
//! The hop quench is the same Wolfe L-BFGS as a Cartesian quench. A
//! rigid record accepts the root-mean-square force at the ranked
//! tolerance of `1e-3`.

use ndarray::ArrayView1;

/// RMS force at which a rigid quench may be recorded. Ranked `SLOPPYCONV`.
pub(crate) const SLOPPY_RMS: f64 = 1.0e-3;

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
}
