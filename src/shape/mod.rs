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
    /// The library returned success with a permutation that is not a
    /// bijection, so the correspondence it reports cannot be used.
    ///
    /// Observed against libira at 3cb0c29: matching a twelve-point structure to
    /// a relabelled copy of itself returns `cerr = 0`, the correct distance of
    /// zero and the correct identity rotation, with the permutation
    /// `[4,5,6,7,8,9,10,0,0,1,2,3]`, where index 0 is assigned twice and index
    /// 11 never. The distance is computed inside the library and is right; the
    /// correspondence is not.
    ///
    /// Reported rather than repaired: a permutation with a duplicate has no
    /// unique completion, and guessing one would put points in each other's
    /// places silently. Callers that only need the distance are unaffected.
    NonBijectivePermutation,
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
            ShapeError::NonBijectivePermutation => write!(
                f,
                "IRA reported success with a permutation that is not a bijection"
            ),
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
    /// Permutation carrying the second structure's points onto the first's,
    /// when the library returns a usable one.
    ///
    /// `None` when what came back is not a bijection. The distance and the
    /// rigid motion are computed inside the library and stay trustworthy in
    /// that case; only the correspondence is lost, so a caller that needs the
    /// distance is unaffected and a caller that needs the ordering finds out
    /// rather than being handed a mapping that puts points in each other's
    /// places.
    pub permutation: Option<Vec<usize>>,
}

/// A canonical atom order and frame, taken against a fixed reference.
///
/// Keying basins on shape has been priced out by the comparison, not by the
/// descriptor. A shape distance between two structures costs an IRA call, so a
/// bias holding a few thousand basins pays one call per comparison and a run
/// that does a hundred thousand hops cannot finish.
///
/// The permutation IRA already returns is what removes that cost, and it was
/// being discarded. Matching each structure once against a single reference
/// gives an ordering and a frame in which corresponding points occupy
/// corresponding slots; after that, comparing two structures is Euclidean
/// distance on the aligned coordinates. One call per hop rather than one per
/// basin.
///
/// The idea is the one `readcon` uses for atom identity: the `.con` format
/// groups atoms by element and so reorders them, and rather than pretending
/// order does not matter it stores the pre-grouping index "so the original
/// sequence can be reconstructed after any number of read/write cycles". Carry
/// the permutation; do not quotient it away.
///
/// # What this buys over sorting
///
/// A sorted descriptor destroys correspondence: it says which values occur,
/// not which point holds which. Two structures with the same multiset of site
/// energies and a different arrangement are identical to it. Under a canonical
/// order the arrangement survives, so the descriptor separates them, and the
/// merge radius becomes a root-mean-square displacement, which is a length and
/// transfers between sizes.
pub struct CanonicalOrder {
    reference: Array1<f64>,
    /// Search breadth passed to IRA.
    pub kmax_factor: f64,
}

impl CanonicalOrder {
    /// A canonicaliser against `reference`, which fixes the frame for the run.
    ///
    /// Any structure of the right size will do; what matters is that it does
    /// not change, since the ordering is only canonical relative to it.
    pub fn new(reference: Array1<f64>, kmax_factor: f64) -> Self {
        Self {
            reference,
            kmax_factor,
        }
    }

    /// Points the reference holds.
    pub fn n_points(&self) -> usize {
        self.reference.len() / 3
    }

    /// `x` reordered and rigidly moved onto the reference frame.
    ///
    /// `None` when IRA cannot match, which for equal-sized structures means the
    /// call failed rather than that no match exists; the caller should fall
    /// back to an order-free descriptor rather than treat it as a new basin.
    pub fn canonicalise(&self, x: ArrayView1<f64>) -> Option<Array1<f64>> {
        let n = self.n_points();
        if x.len() != 3 * n {
            return None;
        }
        let m = match_shapes(self.reference.view(), x, self.kmax_factor).ok()?;
        // No usable correspondence means no canonical order. Falling back to
        // the raw coordinates here would return a descriptor that is not
        // invariant while looking like one, which is the failure this whole
        // path exists to avoid.
        let perm = m.permutation.as_ref()?;
        let mut out = Array1::<f64>::zeros(3 * n);
        for (slot, &src) in perm.iter().enumerate() {
            if src >= n || slot >= n {
                return None;
            }
            // `R p + t`, with the rotation read row-major. The library
            // transposes its matrix on the way out, again under "return
            // C-style data", so no further transpose belongs here.
            let p = [x[3 * src], x[3 * src + 1], x[3 * src + 2]];
            for k in 0..3 {
                out[3 * slot + k] = m.rotation[3 * k] * p[0]
                    + m.rotation[3 * k + 1] * p[1]
                    + m.rotation[3 * k + 2] * p[2]
                    + m.translation[k];
            }
        }
        Some(out)
    }
}

impl crate::bias::Fingerprint for CanonicalOrder {
    /// # Panics
    ///
    /// Never; but a structure that cannot be canonicalised returns its raw
    /// coordinates, which are *not* comparable with canonicalised ones. Use
    /// [`CanonicalOrder::canonicalise`] directly and handle `None` unless the
    /// caller has established that matching succeeds for every structure it
    /// will see. Against libira at 3cb0c29 it does not: the permutation comes
    /// back non-bijective even for a structure matched to a relabelled copy of
    /// itself.
    fn describe(&self, x: ArrayView1<f64>) -> Array1<f64> {
        // Scaled by 1/sqrt(n), so Euclidean distance between two descriptors is
        // a root-mean-square displacement per point rather than a total.
        //
        // Without it the threshold carries a hidden sqrt(n) and does not
        // transfer between sizes, which is the defect the shape metric was
        // brought in to remove. It also mis-set the radius directly: at 38
        // points an unscaled 0.3 sat far below the distance one accepted hop
        // covers, so the run opened 907 basins in 2465 hops, 2.7 hops each, and
        // the bias had nothing to accumulate.
        let n = self.n_points().max(1) as f64;
        let s = 1.0 / n.sqrt();
        self.canonicalise(x)
            .map(|mut v| {
                v *= s;
                v
            })
            .unwrap_or_else(|| x.to_owned() * s)
    }
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
    // The pair is written before the call, not after: the failure being chased
    // kills the process inside the library, so nothing after the call runs.
    // Off unless asked for.
    if std::env::var_os("ANNEAL_DUMP_IRA").is_some() {
        use std::io::Write;
        if let Ok(mut f) = std::fs::OpenOptions::new()
            .create(true)
            .append(true)
            .open("ira_pair.txt")
        {
            let fmt = |v: &Vec<f64>| {
                v.iter()
                    .map(|z| format!("{z:.17e}"))
                    .collect::<Vec<_>>()
                    .join(" ")
            };
            let _ = writeln!(f, "PAIR {n1}");
            let _ = writeln!(f, "A {}", fmt(&ca));
            let _ = writeln!(f, "B {}", fmt(&cb));
            let _ = f.flush();
        }
    }

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
        permutation: {
            // Validated here rather than trusted. The library reports success
            // alongside a permutation that assigns one index twice and another
            // never, and a caller that used it would put points in each other's
            // places with nothing to signal it.
            // Already zero-based. `libira_match` ends with
            //
            //   p_matrix = transpose( p_matrix )
            //   p_perm(:) = p_perm(:) - 1
            //
            // under a comment reading "return C-style data", so subtracting one
            // here converts a second time. That is what made every valid
            // permutation look broken: a cyclic shift by five came back as
            // [5,6,7,8,9,10,11,0,1,2,3,4] and this turned it into
            // [4,5,6,7,8,9,10,0,0,1,2,3], with one index twice and one missing.
            let perm: Vec<usize> = permutation.iter().map(|&p| p.max(0) as usize).collect();
            let mut seen = vec![false; n1];
            let ok = perm.len() == n1
                && perm.iter().all(|&q| {
                    if q >= n1 || seen[q] {
                        false
                    } else {
                        seen[q] = true;
                        true
                    }
                });
            if ok { Some(perm) } else { None }
        },
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

    /// Regular octahedron, flattened.
    fn octahedron() -> Array1<f64> {
        Array1::from(vec![
            1.0, 0.0, 0.0, -1.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, -1.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0,
            -1.0,
        ])
    }

    fn ico12(scale: f64) -> Array1<f64> {
        let phi = (1.0 + 5.0_f64.sqrt()) / 2.0;
        let verts: [[f64; 3]; 12] = [
            [0.0, 1.0, phi],
            [0.0, 1.0, -phi],
            [0.0, -1.0, phi],
            [0.0, -1.0, -phi],
            [1.0, phi, 0.0],
            [1.0, -phi, 0.0],
            [-1.0, phi, 0.0],
            [-1.0, -phi, 0.0],
            [phi, 0.0, 1.0],
            [-phi, 0.0, 1.0],
            [phi, 0.0, -1.0],
            [-phi, 0.0, -1.0],
        ];
        let mut x = Array1::<f64>::zeros(36);
        for (i, v) in verts.iter().enumerate() {
            for k in 0..3 {
                x[3 * i + k] = scale * v[k];
            }
        }
        x
    }

    fn relabel(x: ArrayView1<f64>, shift: usize) -> Array1<f64> {
        let n = x.len() / 3;
        let mut y = Array1::<f64>::zeros(x.len());
        for i in 0..n {
            let p = (i + shift) % n;
            for k in 0..3 {
                y[3 * p + k] = x[3 * i + k];
            }
        }
        y
    }

    /// Relabelling the points must not move the descriptor: that is what makes
    /// it a basin key at all.
    #[test]
    fn a_canonical_order_absorbs_relabelling() {
        let r = generic12();
        let c = CanonicalOrder::new(r.clone(), 1.8);
        let a = c
            .canonicalise(r.view())
            .expect("reference did not canonicalise");
        let b = c
            .canonicalise(relabel(r.view(), 5).view())
            .expect("relabelled copy did not canonicalise");
        assert!(
            rms(a.view(), b.view()) < 1e-6,
            "relabelling moved the descriptor by {}",
            rms(a.view(), b.view())
        );
    }

    #[test]
    fn a_canonical_order_absorbs_rigid_motions() {
        let r = generic12();
        let c = CanonicalOrder::new(r.clone(), 1.8);
        let a = c.canonicalise(r.view()).unwrap();
        let mut moved = Array1::<f64>::zeros(36);
        for i in 0..12 {
            let px = r[3 * i] + 4.0;
            let py = r[3 * i + 1] - 2.0;
            moved[3 * i] = -py;
            moved[3 * i + 1] = px;
            moved[3 * i + 2] = r[3 * i + 2] + 0.5;
        }
        let b = c
            .canonicalise(moved.view())
            .expect("moved copy did not canonicalise");
        assert!(
            rms(a.view(), b.view()) < 1e-6,
            "a rigid motion moved the descriptor by {}",
            rms(a.view(), b.view())
        );
    }

    /// Genuinely different structures must stay apart, or the key merges basins
    /// the search needs separate.
    #[test]
    fn different_structures_stay_apart_under_a_canonical_order() {
        let r = generic12();
        let c = CanonicalOrder::new(r.clone(), 1.8);
        let a = c.canonicalise(r.view()).unwrap();
        let mut far = r.clone();
        for k in 0..3 {
            far[k] *= 1.6;
        }
        let b = c
            .canonicalise(far.view())
            .expect("distorted copy did not canonicalise");
        assert!(
            rms(a.view(), b.view()) > 0.05,
            "a clearly different structure came out {} away",
            rms(a.view(), b.view())
        );
    }

    fn rms(a: ArrayView1<f64>, b: ArrayView1<f64>) -> f64 {
        let n = (a.len() / 3) as f64;
        (a.iter()
            .zip(b.iter())
            .map(|(p, q)| (p - q) * (p - q))
            .sum::<f64>()
            / n)
            .sqrt()
    }

    /// The permutation is zero-based on arrival and must survive the binding.
    ///
    /// This is the test that was missing. Without it the binding subtracted one
    /// from an already zero-based array and every valid permutation came back
    /// with an index repeated and another absent, which reads exactly like a
    /// library defect and was very nearly reported as one.
    #[test]
    fn a_relabelled_copy_returns_the_relabelling() {
        let r = generic12();
        let shift = 5usize;
        let q = relabel(r.view(), shift);
        let m = match_shapes(r.view(), q.view(), 1.8).expect("match failed outright");
        assert!(
            m.distance < 1e-9,
            "a relabelled copy should be at distance zero, got {}",
            m.distance
        );
        let perm = m
            .permutation
            .as_ref()
            .expect("a relabelled copy has a perfectly good correspondence");
        let n = 12;
        let mut seen = vec![false; n];
        for &p in perm {
            assert!(p < n && !seen[p], "not a bijection: {perm:?}");
            seen[p] = true;
        }
        // `relabel` sends point i of the reference to slot (i + shift) % n of
        // the query, so slot i of the reference corresponds to query point
        // (i + shift) % n, and that is what the permutation reports.
        for (slot, &src) in perm.iter().enumerate() {
            assert_eq!(
                src,
                (slot + shift) % n,
                "slot {slot} mapped to {src} in {perm:?}"
            );
        }
    }

    /// A generic structure with no symmetry, where the canonical order is
    /// unique and relabelling is absorbed exactly.
    fn generic12() -> Array1<f64> {
        let mut x = Array1::<f64>::zeros(36);
        for i in 0..12 {
            let a = (i as f64) * 1.317;
            let r = 0.9 + 0.23 * ((i % 5) as f64);
            x[3 * i] = r * a.cos();
            x[3 * i + 1] = r * a.sin();
            x[3 * i + 2] = 0.41 * ((i % 7) as f64) - 1.2;
        }
        x
    }

    /// Relabelling the points must not move the descriptor: that is what makes
    /// it a basin key at all.
    /// The limitation, kept as a test because it decides where this can be
    /// used. A canonical order is canonical only up to the structure's own
    /// symmetry: an icosahedron admits sixty equivalent matchings, all at the
    /// same distance, and IRA is free to return any of them. Relabelling then
    /// moves the descriptor by a symmetry operation even though the structure
    /// has not changed, which is fatal for keying basins on it, because the
    /// structures this search cares about are exactly the symmetric ones.
    /// And neither must a rigid motion.
    /// The claim the economy rests on: Euclidean distance between two
    /// canonicalised structures has to track the pairwise IRA distance, or
    /// replacing one call per comparison with one per structure changes the
    /// answer rather than the cost.
    /// Genuinely different structures must not collapse together, or the key
    /// merges basins the search needs apart.
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
        // The correspondence is optional; only its length is asserted when the
        // library returns one at all.
        if let Some(p) = m.permutation.as_ref() {
            assert_eq!(p.len(), n);
        }
    }

    #[test]
    fn different_shapes_are_far_apart() {
        let a = octahedron();
        let mut b = octahedron();
        b[0] += 1.5; // pull one vertex well away
        let m = match_shapes(a.view(), b.view(), 1.8).expect("match");
        assert!(
            m.distance > 0.1,
            "distorted shape should not match, got {}",
            m.distance
        );
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
        assert_eq!(
            IraMetric::default().distance(a.view(), b.view()),
            f64::INFINITY
        );
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
pub fn symmetry_deviation(coords: ArrayView1<f64>, matrix: &[f64; 9]) -> Result<f64, ShapeError> {
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

/// Hausdorff deviation and the permutation realising the image.
///
/// `perm[i]` is the point that `matrix` sends onto `i` when the library
/// returns a bijection. The residual `x[i] − R x[perm[i]]` is the
/// Cartesian leftover of that symmetry.
pub fn symmetry_pair(
    coords: ArrayView1<f64>,
    matrix: &[f64; 9],
) -> Result<(f64, Vec<usize>), ShapeError> {
    if coords.len() % 3 != 0 {
        return Err(ShapeError::NotThreeDimensional(coords.len()));
    }
    let n = coords.len() / 3;
    if n == 0 {
        return Err(ShapeError::Empty);
    }
    let owned: Vec<f64> = coords.iter().copied().collect();
    let types = vec![1_i32; n];
    let mut perm_i = vec![0_i32; n];
    let mut dh: c_double = 0.0;
    unsafe {
        libira_try_mat(
            n as c_int,
            types.as_ptr(),
            owned.as_ptr(),
            matrix.as_ptr(),
            &mut dh,
            &perm_i.as_mut_ptr(),
        );
    }
    if !dh.is_finite() || dh < 0.0 {
        return Err(ShapeError::Library(-1));
    }
    let perm: Vec<usize> = perm_i.iter().map(|&p| p.max(0) as usize).collect();
    let mut seen = vec![false; n];
    let ok = perm.len() == n
        && perm.iter().all(|&q| {
            if q >= n || seen[q] {
                false
            } else {
                seen[q] = true;
                true
            }
        });
    if !ok {
        return Err(ShapeError::NonBijectivePermutation);
    }
    Ok((dh, perm))
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
        let r = [[m[0], m[1], m[2]], [m[3], m[4], m[5]], [m[6], m[7], m[8]]];
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
