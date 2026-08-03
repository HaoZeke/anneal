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

use crate::bias::{BasinMetric, Fingerprint};

unsafe extern "C" {
    /// `libira_try_mat` from `src/library_sofi.f90`.
    ///
    /// Applies one candidate symmetry operation to a structure and returns the
    /// Hausdorff distance between the structure and its own image under that
    /// operation, together with the permutation realising it.
    ///
    /// This is the SOFI entry point that matters for a search. Asking SOFI for
    /// a point group returns a discrete answer, and on this landscape that
    /// answer is C1 for every quenched minimum at every threshold from 0.3 to
    /// 2.0, so a coordinate built on it is constant and a bias deposited on it
    /// cancels. The deviation under a named operation is continuous and is
    /// defined whether or not the structure has the symmetry: it measures how
    /// far a structure is from having a fivefold axis rather than whether it
    /// has one.
    fn libira_try_mat(
        nat: c_int,
        typ: *const c_int,
        coords: *const c_double,
        rmat: *const c_double,
        dh: *mut c_double,
        // By reference, as for the output buffers of `libira_match`.
        perm: *const *mut c_int,
    );

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
        // Declared `type(c_ptr), intent(in)` without `value`, so Fortran
        // receives these by reference and dereferences once to reach the
        // buffer. Passing the buffer address directly makes it read the first
        // eight bytes of the buffer as an address, which segfaults. The
        // arguments above carry `value` and are passed directly; the
        // difference is one keyword in the Fortran and the whole calling
        // convention here.
        rotation: *const *mut c_double,
        translation: *const *mut c_double,
        permutation: *const *mut c_int,
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
    /// A coordinate buffer whose length is not a multiple of three.
    NotThreeDimensional(usize),
    /// An empty structure, which has no symmetry to measure.
    Empty,
}

impl std::fmt::Display for ShapeError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            ShapeError::SizeMismatch(a, b) => {
                write!(f, "shape match needs equal point counts, got {a} and {b}")
            }
            ShapeError::NotThreeDimensional(n) => {
                write!(f, "coordinate length {n} is not a multiple of three")
            }
            ShapeError::Empty => write!(f, "an empty structure has no symmetry"),
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

    // Length `nat` with a leading -1, the sentinel IRA's own interface uses for
    // equal atom counts to mean "choose your own frames". A null pointer is not
    // an option: the Fortran calls `c_f_pointer` on this argument with no
    // `c_associated` guard, so null becomes a Fortran pointer to address zero.
    // One array per structure, never one buffer passed twice. The Fortran
    // takes both as pointers and reaches them through c_f_pointer, so aliasing
    // them lets work on the first alter what the second is read as. It fails
    // data-dependently, as a bounds error on a candidate index of zero, which
    // is why some runs survived it.
    let mut candidate_a = vec![0_i32; n1];
    candidate_a[0] = -1;
    let mut candidate_b = vec![0_i32; n2];
    candidate_b[0] = -1;

    let mut rotation = [0.0_f64; 9];
    let mut translation = [0.0_f64; 3];
    // The identity, one-based, not zeros. The Fortran indexes its type and
    // coordinate arrays with this buffer after the call, and zero is out of
    // range there. IRA can return a zero status having left slots unwritten on
    // a degenerate structure, which a search produces routinely and a unit test
    // on clean structures never does, and the run then dies inside the library
    // with a bounds error rather than at the call.
    let mut permutation: Vec<i32> = (1..=n1 as i32).collect();
    let mut hd = 0.0_f64;
    let mut cerr = 0_i32;

    // SAFETY: every pointer is to a live buffer of the length IRA is told. The
    // candidate arrays are allocated and carry the sentinel rather than being
    // null. The three output buffers are passed by reference because Fortran
    // declares them `type(c_ptr), intent(in)` without `value`, unlike the
    // inputs above, and the outputs are sized 9, 3 and n as required.
    unsafe {
        libira_match(
            n1 as c_int,
            typ.as_ptr(),
            ca.as_ptr(),
            candidate_a.as_ptr(),
            n2 as c_int,
            typ.as_ptr(),
            cb.as_ptr(),
            candidate_b.as_ptr(),
            kmax_factor,
            &rotation.as_mut_ptr(),
            &translation.as_mut_ptr(),
            &permutation.as_mut_ptr(),
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

/// Deviation of a structure from invariance under one operation.
///
/// `matrix` is a three by three orthogonal operation in row-major order.
/// Returns the Hausdorff distance between the structure and its image, which
/// is zero when the operation is an exact symmetry and grows smoothly with the
/// departure from it.
///
/// The continuous quantity is the useful one. A point group is a discrete
/// answer that reads C1 for every structure a search on this landscape visits,
/// so a bias on it deposits on a constant; the deviation under a chosen
/// operation separates structures that no group label distinguishes.
pub fn symmetry_deviation(
    coords: ArrayView1<f64>,
    matrix: &[f64; 9],
) -> Result<f64, ShapeError> {
    if coords.len() % 3 != 0 {
        return Err(ShapeError::NotThreeDimensional(coords.len()));
    }
    let n = coords.len() / 3;
    if n == 0 {
        return Err(ShapeError::Empty);
    }
    let owned: Vec<f64> = coords.iter().copied().collect();
    let types = vec![1_i32; n];
    let mut perm = vec![0_i32; n];
    let mut dh: c_double = 0.0;
    // Safety: every pointer is to a live buffer of the length the Fortran
    // declares from `nat`, and `dh` and `perm` are the only ones written.
    unsafe {
        libira_try_mat(
            n as c_int,
            types.as_ptr(),
            owned.as_ptr(),
            matrix.as_ptr(),
            &mut dh,
            &perm.as_mut_ptr(),
        );
    }
    if !dh.is_finite() || dh < 0.0 {
        return Err(ShapeError::Library(-1));
    }
    Ok(dh)
}

/// Rotation by `angle` about a unit `axis`, row-major, for [`symmetry_deviation`].
pub fn rotation_matrix(axis: [f64; 3], angle: f64) -> [f64; 9] {
    let norm = (axis[0] * axis[0] + axis[1] * axis[1] + axis[2] * axis[2]).sqrt();
    let [x, y, z] = [axis[0] / norm, axis[1] / norm, axis[2] / norm];
    let (s, c) = angle.sin_cos();
    let t = 1.0 - c;
    [
        t * x * x + c,
        t * x * y - s * z,
        t * x * z + s * y,
        t * x * y + s * z,
        t * y * y + c,
        t * y * z - s * x,
        t * x * z - s * y,
        t * y * z + s * x,
        t * z * z + c,
    ]
}

/// Smallest deviation over the rotations of an `order`-fold axis.
///
/// A structure is scored against a symmetry it need not possess, so the result
/// is a distance from that symmetry rather than a test for it. Minimising over
/// the non-trivial powers keeps the quantity a property of the axis rather than
/// of which power happens to be tried.
pub fn axis_deviation(
    coords: ArrayView1<f64>,
    axis: [f64; 3],
    order: usize,
) -> Result<f64, ShapeError> {
    if order < 2 {
        return Err(ShapeError::Library(-2));
    }
    let mut best = f64::INFINITY;
    for k in 1..order {
        let angle = 2.0 * std::f64::consts::PI * (k as f64) / (order as f64);
        let d = symmetry_deviation(coords, &rotation_matrix(axis, angle))?;
        best = best.min(d);
    }
    Ok(best)
}

#[cfg(test)]
mod sofi_tests {
    use super::*;
    use ndarray::Array1;

    /// Twelve vertices of a regular icosahedron, which has exact fivefold axes.
    fn icosahedron() -> Array1<f64> {
        let phi = (1.0 + 5.0_f64.sqrt()) / 2.0;
        let mut v = Vec::new();
        for s1 in [-1.0_f64, 1.0] {
            for s2 in [-1.0_f64, 1.0] {
                v.extend_from_slice(&[0.0, s1, s2 * phi]);
                v.extend_from_slice(&[s1, s2 * phi, 0.0]);
                v.extend_from_slice(&[s1 * phi, 0.0, s2]);
            }
        }
        Array1::from(v)
    }

    #[test]
    fn rotation_matrix_is_orthogonal_with_unit_determinant() {
        let m = rotation_matrix([0.3, -0.5, 0.81], 1.1);
        let r = [
            [m[0], m[1], m[2]],
            [m[3], m[4], m[5]],
            [m[6], m[7], m[8]],
        ];
        for i in 0..3 {
            for j in 0..3 {
                let dot: f64 = (0..3).map(|k| r[i][k] * r[j][k]).sum();
                let want = if i == j { 1.0 } else { 0.0 };
                assert!((dot - want).abs() < 1e-12, "not orthogonal at {i},{j}");
            }
        }
        let det = r[0][0] * (r[1][1] * r[2][2] - r[1][2] * r[2][1])
            - r[0][1] * (r[1][0] * r[2][2] - r[1][2] * r[2][0])
            + r[0][2] * (r[1][0] * r[2][1] - r[1][1] * r[2][0]);
        assert!((det - 1.0).abs() < 1e-12, "determinant {det}");
    }

    /// The property the module exists for: the deviation is near zero along a
    /// symmetry the structure has, and clearly positive along one it does not.
    #[test]
    fn deviation_separates_a_real_axis_from_a_wrong_one() {
        let x = icosahedron();
        let phi = (1.0 + 5.0_f64.sqrt()) / 2.0;
        // A fivefold axis of the icosahedron passes through a vertex.
        let real = axis_deviation(x.view(), [0.0, 1.0, phi], 5).unwrap();
        // A fourfold rotation is not a symmetry of an icosahedron.
        let wrong = axis_deviation(x.view(), [0.0, 1.0, phi], 4).unwrap();
        assert!(real < 1e-6, "fivefold deviation {real} should vanish");
        assert!(wrong > 0.1, "fourfold deviation {wrong} should not");
    }
}

impl BasinMetric for IraMetric {
    /// Hausdorff distance under the optimal permutation and rigid motion.
    ///
    /// This is what makes a merge threshold a length rather than a number in
    /// descriptor space. Two states differing by relabelling or by a rotation
    /// return zero however their coordinates are written, so the threshold does
    /// not have to absorb those symmetries and does not have to be re-found at
    /// every system size.
    ///
    /// A failed match reads as infinity, which puts the pair in different
    /// basins. That is the safe direction: merging on a distance that was never
    /// computed empties the bias that separating them exists for.
    fn distance(&self, a: ArrayView1<f64>, b: ArrayView1<f64>) -> f64 {
        IraMetric::distance(self, a, b)
    }
}
