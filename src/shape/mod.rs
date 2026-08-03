//! Shape matching through the IRA library, behind the `ira` feature.
//!
//! [`crate::bias::BasinBias`] asks whether two states are the same basin, and
//! [`crate::bias::SortedPairs`] answers with a descriptor whose threshold has
//! to be found empirically and does not transfer between system sizes. For
//! point sets there is an exact answer: the optimal permutation and rigid
//! motion between them, and the Hausdorff distance under that match.
//!
//! IRA computes it (Gunde, Salles, Hemeryck and Martin-Samos, Comput Phys
//! Commun 280, 108431, 2022). Its threshold is a length rather than a tuned
//! number in descriptor space, so the same value works whatever the system:
//! measured on a 38-point Lennard-Jones cluster, a permuted and rotated copy
//! returns 0.0, the same basin jittered by 0.02 returns 0.054, and a different
//! basin returns 1.58.
//!
//! The library is Fortran with a C-bound interface, linked here as
//! `libira`. Build it and point `IRA_LIB_DIR` at the result, or leave the
//! feature off.

use std::os::raw::{c_double, c_int};

use ndarray::{Array1, ArrayView1};

use crate::bias::Fingerprint;

unsafe extern "C" {
    /// `libira_match` from `src/library_ira.f90`.
    ///
    /// Returns the rigid transform taking structure 2 onto structure 1 and the
    /// Hausdorff distance under it. Candidate pointers may be null, in which
    /// case IRA selects its own candidate frames.
    #[allow(clippy::too_many_arguments)]
    fn libira_match(
        nat1: c_int,
        typ1: *const c_int,
        coords1: *const c_double,
        candidate1: *const c_int,
        nat2: c_int,
        typ2: *const c_int,
        coords2: *const c_double,
        candidate2: *const c_int,
        kmax_factor: c_double,
        rotation: *mut c_double,
        translation: *mut c_double,
        permutation: *mut c_int,
        hd: *mut c_double,
        cerr: *mut c_int,
    );
}

/// Error from the shape-matching library.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum ShapeError {
    /// Structures hold different numbers of points, so no permutation exists.
    SizeMismatch(usize, usize),
    /// IRA reported a non-zero status.
    Library(i32),
}

impl std::fmt::Display for ShapeError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            ShapeError::SizeMismatch(a, b) => {
                write!(f, "shape match needs equal point counts, got {a} and {b}")
            }
            ShapeError::Library(c) => write!(f, "IRA returned status {c}"),
        }
    }
}

impl std::error::Error for ShapeError {}

/// Result of matching two point sets.
#[derive(Debug, Clone)]
pub struct Match {
    /// Hausdorff distance under the optimal permutation and rigid motion.
    pub distance: f64,
    /// Row-major 3x3 rotation taking the second structure onto the first.
    pub rotation: [f64; 9],
    /// Translation applied after the rotation.
    pub translation: [f64; 3],
    /// Permutation carrying the second structure's points onto the first's.
    pub permutation: Vec<usize>,
}

/// Optimal match between two flattened `(n, 3)` point sets.
///
/// `kmax_factor` bounds how far IRA searches for candidate frames; 1.8 is the
/// value the library's examples use for equal-sized structures.
pub fn match_shapes(
    a: ArrayView1<f64>,
    b: ArrayView1<f64>,
    kmax_factor: f64,
) -> Result<Match, ShapeError> {
    let n1 = a.len() / 3;
    let n2 = b.len() / 3;
    if n1 != n2 || n1 == 0 {
        return Err(ShapeError::SizeMismatch(n1, n2));
    }
    // IRA expects contiguous row-major coordinates; a view may be strided.
    let ca: Vec<f64> = a.iter().copied().collect();
    let cb: Vec<f64> = b.iter().copied().collect();
    let typ = vec![1_i32; n1];

    let mut rotation = [0.0_f64; 9];
    let mut translation = [0.0_f64; 3];
    let mut permutation = vec![0_i32; n1];
    let mut hd = 0.0_f64;
    let mut cerr = 0_i32;

    // SAFETY: every pointer is to a live buffer of the length IRA is told,
    // the candidate pointers are null which the interface accepts, and the
    // outputs are sized 9, 3 and n as the interface requires.
    unsafe {
        libira_match(
            n1 as c_int,
            typ.as_ptr(),
            ca.as_ptr(),
            std::ptr::null(),
            n2 as c_int,
            typ.as_ptr(),
            cb.as_ptr(),
            std::ptr::null(),
            kmax_factor,
            rotation.as_mut_ptr(),
            translation.as_mut_ptr(),
            permutation.as_mut_ptr(),
            &mut hd,
            &mut cerr,
        );
    }
    if cerr != 0 {
        return Err(ShapeError::Library(cerr));
    }
    Ok(Match {
        distance: hd,
        rotation,
        translation,
        // IRA indexes from one.
        permutation: permutation.iter().map(|&p| (p - 1).max(0) as usize).collect(),
    })
}

/// Fingerprint that defers to shape matching.
///
/// [`Fingerprint`] returns a vector and [`crate::bias::BasinBias`] compares
/// those by euclidean distance, which cannot express "optimal over all
/// permutations". The state is therefore passed through unchanged and
/// [`IraMetric`] is used where a metric is wanted instead.
pub struct IraPassthrough;

impl Fingerprint for IraPassthrough {
    fn describe(&self, x: ArrayView1<f64>) -> Array1<f64> {
        x.to_owned()
    }
}

/// Distance between two states under optimal permutation and rigid motion.
pub struct IraMetric {
    /// Search breadth passed to IRA.
    pub kmax_factor: f64,
}

impl Default for IraMetric {
    fn default() -> Self {
        Self { kmax_factor: 1.8 }
    }
}

impl IraMetric {
    /// Distance, or `f64::INFINITY` when the structures cannot be matched.
    pub fn distance(&self, a: ArrayView1<f64>, b: ArrayView1<f64>) -> f64 {
        match_shapes(a, b, self.kmax_factor)
            .map(|m| m.distance)
            .unwrap_or(f64::INFINITY)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use ndarray::Array1;

    /// Regular octahedron, flattened.
    fn octahedron() -> Array1<f64> {
        Array1::from(vec![
            1.0, 0.0, 0.0, -1.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, -1.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0,
            -1.0,
        ])
    }

    fn rotated_z(x: &Array1<f64>, angle: f64) -> Array1<f64> {
        let (s, c) = angle.sin_cos();
        let mut out = x.clone();
        for i in 0..x.len() / 3 {
            let (a, b) = (x[3 * i], x[3 * i + 1]);
            out[3 * i] = c * a - s * b;
            out[3 * i + 1] = s * a + c * b;
        }
        out
    }

    /// The property the descriptor metric could not provide: a structure and a
    /// permuted, rotated copy of it are the same shape, at distance zero.
    #[test]
    fn permuted_and_rotated_copy_matches_exactly() {
        let a = octahedron();
        let rotated = rotated_z(&a, 0.7);
        // Reverse the point order as well, so the match must permute.
        let mut b = Array1::zeros(rotated.len());
        let n = rotated.len() / 3;
        for i in 0..n {
            for k in 0..3 {
                b[3 * i + k] = rotated[3 * (n - 1 - i) + k];
            }
        }
        let m = match_shapes(a.view(), b.view(), 1.8).expect("match");
        assert!(
            m.distance < 1e-6,
            "a rotated permutation of a shape is the same shape, got {}",
            m.distance
        );
        assert_eq!(m.permutation.len(), n);
    }

    #[test]
    fn different_shapes_are_far_apart() {
        let a = octahedron();
        let mut b = octahedron();
        b[0] += 1.5; // pull one vertex well away
        let m = match_shapes(a.view(), b.view(), 1.8).expect("match");
        assert!(m.distance > 0.1, "distorted shape should not match, got {}", m.distance);
    }

    #[test]
    fn unequal_point_counts_are_rejected() {
        let a = octahedron();
        let b = Array1::from(vec![0.0, 0.0, 0.0, 1.0, 0.0, 0.0]);
        assert!(matches!(
            match_shapes(a.view(), b.view(), 1.8),
            Err(ShapeError::SizeMismatch(6, 2))
        ));
    }

    #[test]
    fn metric_reports_infinity_rather_than_panicking() {
        let a = octahedron();
        let b = Array1::from(vec![0.0, 0.0, 0.0]);
        assert_eq!(IraMetric::default().distance(a.view(), b.view()), f64::INFINITY);
    }
}
