//! Basin identity from a three-body tensor invariant.
//!
//! [`crate::bias::SortedPairs`] identifies a basin by the multiset of pairwise
//! distances. That multiset is a known-incomplete permutation invariant:
//! homometric point sets share it and are not congruent. The consequence in
//! this crate is a merge radius with no room in it, 9 seeds of 16 solved at
//! 0.7 and 0 of 18 at anything coarser, and the exact repair through IRA shape
//! matching costs 52 charged evaluations per hop against 31 and solves 4 of 23
//! against 7 of 24.
//!
//! What is missing has a name. Write `A_ij = exp(-r_ij^2 / (2 sigma^2))` for a
//! Gaussian kernel on the distances. The distance multiset determines
//! `tr A^2 = N + 2 sum_{i<j} A_ij^2` and stops there; it does not determine
//! `tr A^3 = sum_{ijk} A_ij A_jk A_ki`, the weighted triangle sum, because that
//! asks which distances share a vertex and a multiset has thrown that away.
//! Triangles are the first thing a distance spectrum cannot see.
//!
//! # The tensor
//!
//! Collect the triangles into `T_ijk = A_ij A_jk A_ik`, supersymmetric under
//! all six permutations of its indices, and the total mass of `T` is `tr A^3`.
//! Relabelling the points by `pi` sends `T` to `T x_1 P x_2 P x_3 P` with the
//! same permutation matrix `P` in every mode, so every mode-wise spectral
//! quantity of `T` is a permutation invariant. Rigid motions and reflections
//! act trivially because `T` is built from distances.
//!
//! # What is invariant, and what is only invariant up to a gauge
//!
//! Let `T_(n)` be the mode-`n` unfolding and `G = T_(1) T_(1)^T`. Supersymmetry
//! makes the three unfoldings share their singular values, so there is one
//! Gram matrix rather than three.
//!
//! 1. `G -> P G P^T` under relabelling, so `spec(G)` is exactly invariant, and
//!    with it the mode singular values `sqrt(spec(G))`. This is the part that
//!    can be made into a descriptor; which function of `T` the descriptor
//!    actually takes is settled below on cost and on measured separation.
//!
//! 2. The mode factor matrix `U` from `G = U Lambda U^T` transforms as
//!    `U -> P U Q`, where `Q` is block orthogonal with one block per distinct
//!    eigenvalue of `G`. `U` is equivariant, not invariant, and the residual
//!    gauge `Q` does not cancel.
//!
//! 3. The Tucker core `C = T x_1 U^T x_2 U^T x_3 U^T` therefore transforms as
//!    `C -> C x_1 Q^T x_2 Q^T x_3 Q^T`. It is not invariant. When `spec(G)` is
//!    simple, `Q = diag(s_a)` with `s_a = +-1` and `C_abc -> s_a s_b s_c
//!    C_abc`, so `|C_abc|` is invariant and the signs are not. When `spec(G)`
//!    is degenerate, not even `|C_abc|` survives; what survives is the
//!    Frobenius norm of each block `C_{S_t, S_u, S_v}` cut along the
//!    eigenvalue groups.
//!
//! 4. The total core norm carries nothing new: the HOSVD is norm preserving,
//!    so `||C||_F^2 = ||T||_F^2 = tr G = sum_a lambda_a`, which is already the
//!    sum of the mode spectrum.
//!
//! # The Tucker core buys nothing, for two separate reasons
//!
//! The first is numerical. Clusters have near point-group symmetry, so
//! `spec(G)` has near-degenerate groups, and the block cut in 3 above is
//! discontinuous in the structure exactly where the groups nearly touch. A
//! descriptor read by a merge radius has to be continuous or the radius
//! measures the tie-breaking and not the structure. The invariant that
//! survives degeneracy is a block norm, and the blocks are the near-degenerate
//! groups, which is to say the core reduces to quantities already coarser than
//! the spectrum that indexed them.
//!
//! The second is that the whole mode spectrum, core or no core, separates no
//! better than one contraction of it. On the homometric pair below, at the
//! shipped kernel width 2.5, the exact mode singular values of `T` separate at
//! 2.7 times their own jitter response where the contracted spectra reach
//! 10.3; at width 3 the figures are 8.6 and 21.4, and the contraction leads at
//! every width tried from 1 to 6. Unfolding a mode and keeping all of it does
//! not pay.
//!
//! It does cost. `G` has `N^2` entries and each is a sum over `N^2` terms,
//! `G_ab = sum_jk (A_aj A_bj) A_jk^2 (A_ak A_bk)`, with no factorisation, so
//! [`mode_gram`] is order `N^4`, measured at 50 times the cost from 38 points
//! to 98 against the 44 the exponent predicts: 257 us and 12.8 ms, against the
//! 91 us and 1.05 ms of the whole descriptor. It stays as a reference the
//! tests check the cheap path against.
//!
//! The way out is to contract one mode instead of unfolding it. The mode-3
//! marginal `M_ij = sum_k T_ijk = A_ij (A A)_ij` is one matrix product, order
//! `N^3`, it is exactly permutation equivariant, `M -> P M P^T`, so `spec(M)`
//! is exactly invariant, and it still carries the triangle content:
//! `tr M = tr A^2` while `sum_ij M_ij = tr A^3`. That is [`triplet_matrix`],
//! and it is what the descriptor uses.
//!
//! # The descriptor is a superset, not a replacement
//!
//! `spec(A)` does not determine the distance multiset and the distance
//! multiset does not determine `spec(A)`: the two invariants are incomparable,
//! so dropping one for the other trades unknown failures for unknown failures.
//! [`TripletSpectrum`] therefore emits the sorted distances unchanged followed
//! by the two spectra. Writing `d_new^2 = d_old^2 + w^2 d_spec^2`, any two
//! structures the new descriptor merges at radius `R` the old one also merged
//! at `R`, so at a fixed radius the new descriptor can only split what the old
//! one confused.
//!
//! That ordering is about a fixed radius, and a radius has to be set, so the
//! measurement below normalises: `z = ||f(X) - f(Y)|| / j(f)` with `j(f)` the
//! largest response of `f` to a 0.02 displacement plus a relabelling and a
//! rotation. A merge radius has to sit above `j(f)` and below the closest pair
//! of distinct minima, so the worst-case `z` over pairs is the width of the
//! band the radius has to find, and `z` below 1 means there is no band. Both
//! ends are extreme-order statistics, so the descriptors are read on one
//! shared set of perturbations rather than on independent draws.
//!
//! # What it is worth, measured
//!
//! Over distinct quenched Lennard-Jones minima, worst case and median `z` over
//! all pairs, at the shipped `sigma = 2.5` and `spectral_weight = 2.5`:
//!
//! | system | pairs | distances worst | joint worst | distances median | joint median |
//! |--------|-------|-----------------|-------------|------------------|--------------|
//! | LJ13   | 66    | 1.71            | 1.70        | 3.72             | 3.96         |
//! | LJ26   | 153   | 1.12            | 1.31        | 3.82             | 4.59         |
//! | LJ38   | 190   | 0.85            | 1.17        | 2.69             | 3.57         |
//! | LJ55   | 91    | 0.87            | 1.11        | 2.63             | 3.59         |
//!
//! At 38 and 55 points the distance descriptor's worst case is 0.85 and 0.87,
//! below 1: the closest pair of distinct minima is nearer than a jittered copy
//! of one of them, so no radius both separates that pair and holds a quench
//! together, and a coarser radius merges the two minima outright. That is the
//! knife edge, read off the descriptor rather than off the seed counts. The
//! joint descriptor lifts the worst case past 1, to 1.17 and 1.11, and the
//! median from 2.69 to 3.57 and from 2.63 to 3.59. The band opens by about a
//! third. It does not open by an order of magnitude, and at 13 points, where
//! the distance descriptor was not failing, it does not open at all: 1.70
//! against 1.71 is a wash.
//!
//! On the case sorted distances provably cannot do, Bloom's homometric pair
//! `{0, 1, 4, 10, 12, 17}` and `{0, 1, 8, 11, 13, 17}` on a line, which share
//! the multiset `{1..13, 16, 17}` and are not congruent, sorted distances
//! separate by exactly 0 and the spectra separate by 10.3 jitters.
//!
//! `experiments/tensor_id_separation.py` produces every table here.
//!
//! # What was tried and does not work
//!
//! Power traces in place of eigenvalues. Three matrix products give
//! `tr A^2, tr A^3, tr A^4` and `tr M .. tr M^6`, nine numbers instead of
//! `2N`, and on the homometric pair they do as well as the spectra, 10.5
//! jitters against 10.3. On Lennard-Jones minima they are worse than having no
//! spectral block at all: worst-case `z` of 1.29, 0.81, 0.64 and 0.62 at 13,
//! 26, 38 and 55 points against 1.71, 1.12, 0.85 and 0.87 for the distances
//! alone. The block adds jitter faster than it adds separation, and the gap
//! widens with system size, which is what a fixed nine-number summary of a
//! growing spectrum should do.
//!
//! The compression does not even pay for itself. Two extra matrix products at
//! `2N^3` stand against two tridiagonal reductions at `4N^3/3`, so the traces
//! are the same order and more arithmetic than the spectra they replace. The
//! eigenvalues are not optional and are not the expensive part.
//!
//! # Cost
//!
//! `A` is `O(N^2)`, `M` is one symmetric product at `O(N^3)`, and the two
//! spectra are Householder plus QL at `2N^3/3` each through
//! [`crate::spectral::symmetric_eigenvalues`], so the descriptor is `O(N^3)`
//! and charges no potential evaluations. Measured per call against the
//! descriptor it replaces and against one Lennard-Jones gradient, of which a
//! hop spends thirty-one:
//!
//! | points | sorted distances | this descriptor | one gradient | hops per call |
//! |--------|------------------|-----------------|--------------|---------------|
//! | 38     | 5.27 us          | 91.1 us         | 2.45 us      | 1.20          |
//! | 75     | 24.3 us          | 478 us          | 10.3 us      | 1.50          |
//! | 98     | 45.3 us          | 1.05 ms         | 17.8 us      | 1.90          |
//!
//! So a hop that identifies its basin this way costs about twice a hop that
//! does not, at 98 points, and about a fifth more at 38. IRA shape matching
//! charges 52 evaluations against 31, which is 1.7 times the evaluations plus
//! its own matching; this charges 1.0 times the evaluations and 1.9 times the
//! arithmetic, and the arithmetic is the cheap half.
//!
//! Cyclic Jacobi, which [`crate::spectral::symmetric_eigen`] runs and which is
//! the right choice on the tens-of-nodes matrices the rest of the crate feeds
//! it, puts the same two spectra at 1.40 ms, 12.2 ms and 28.5 ms over the same
//! three sizes, 17 to 28 times the tridiagonal path. That path is what makes
//! the descriptor affordable at all: through Jacobi it would cost 52 hops at
//! 98 points instead of 2.

use ndarray::{Array1, Array2, ArrayView1, ArrayView2};

use crate::bias::Fingerprint;
use crate::spectral::{symmetric_eigen, symmetric_eigenvalues};

/// Sweeps allowed to the Jacobi eigensolver on the reference paths.
///
/// [`crate::spectral::symmetric_eigenvalues`] is what the descriptor uses;
/// Jacobi appears here only in [`hosvd_core`], which needs the eigenvectors
/// and is a diagnostic.
const EIGEN_SWEEPS: usize = 30;

/// Gaussian kernel matrix `A_ij = exp(-r_ij^2 / (2 sigma^2))` of a flattened
/// `(n, 3)` point set.
///
/// Symmetric with unit diagonal, and a function of the distances alone, so it
/// is invariant to rigid motions and equivariant to relabelling.
///
/// Panics if `sigma` is not positive or `x` is not `3 * n_points` long.
pub fn kernel_matrix(x: ArrayView1<f64>, n_points: usize, sigma: f64) -> Array2<f64> {
    assert!(sigma > 0.0, "sigma must be > 0");
    assert_eq!(
        x.len(),
        3 * n_points,
        "state length must be 3 * n_points, got {} for {n_points} points",
        x.len()
    );
    let n = n_points;
    let inv = 1.0 / (2.0 * sigma * sigma);
    let mut a = Array2::<f64>::eye(n);
    for i in 0..n {
        for j in (i + 1)..n {
            let dx = x[3 * i] - x[3 * j];
            let dy = x[3 * i + 1] - x[3 * j + 1];
            let dz = x[3 * i + 2] - x[3 * j + 2];
            let v = (-(dx * dx + dy * dy + dz * dz) * inv).exp();
            a[[i, j]] = v;
            a[[j, i]] = v;
        }
    }
    a
}

/// Mode-3 contraction of the triangle tensor, `M_ij = sum_k A_ij A_jk A_ik`.
///
/// Equivalently `M = A * (A A)` with `*` elementwise, so it costs one
/// symmetric matrix product. The Hadamard factor is what keeps the three-body
/// content: an orthogonally invariant function of `A` cannot see which
/// distances share a vertex, and an elementwise product is basis dependent and
/// permutation equivariant at the same time, which is exactly the pairing the
/// descriptor needs.
///
/// Scaled by `n / tr(A^2)`, itself a permutation invariant so the equivariance
/// is untouched, which puts `spec(M)` on the scale of `spec(A)` and lets one
/// merge radius read both blocks. The unscaled marginal has `tr M = tr A^2`,
/// so the scaling fixes `tr M = n` exactly and the spectrum always sums to `n`:
/// what the block carries is the shape of that spectrum, not its total.
pub fn triplet_matrix(a: ArrayView2<f64>) -> Array2<f64> {
    let n = a.nrows();
    assert_eq!(n, a.ncols(), "triplet_matrix needs a square matrix");
    let a2 = a.dot(&a);
    // tr(A^2) = ||A||_F^2 for symmetric A, and is bounded below by n because
    // the diagonal of A is one, so the division is safe.
    let scale = n as f64 / a2.diag().sum();
    let mut m = Array2::<f64>::zeros((n, n));
    for i in 0..n {
        for j in 0..n {
            m[[i, j]] = a[[i, j]] * a2[[i, j]] * scale;
        }
    }
    m
}

/// The triangle tensor `T_ijk = A_ij A_jk A_ik`, flattened mode-1 into an
/// `(n, n * n)` matrix.
///
/// Order `N^3` in storage and only worth forming to check the cheaper paths
/// against it. [`triplet_matrix`] is the working contraction.
pub fn triangle_unfolding(a: ArrayView2<f64>) -> Array2<f64> {
    let n = a.nrows();
    assert_eq!(n, a.ncols(), "triangle_unfolding needs a square matrix");
    let mut t = Array2::<f64>::zeros((n, n * n));
    for i in 0..n {
        for j in 0..n {
            let aij = a[[i, j]];
            for k in 0..n {
                t[[i, j * n + k]] = aij * a[[j, k]] * a[[i, k]];
            }
        }
    }
    t
}

/// Mode-1 Gram matrix `G = T_(1) T_(1)^T` of the triangle tensor.
///
/// `G_ab = sum_jk (A_aj A_bj) A_jk^2 (A_ak A_bk)`, computed without forming
/// `T`, at order `N^4`. Supersymmetry makes the three mode Grams equal, so
/// this one carries the whole HOSVD mode spectrum.
///
/// Reference path. At 98 points it costs 12.8 ms against the 1.05 ms of the
/// whole descriptor and the 0.55 ms of the energies a hop spends, which is why
/// the descriptor uses [`triplet_matrix`] instead; the tests are what this
/// exists for.
pub fn mode_gram(a: ArrayView2<f64>) -> Array2<f64> {
    let n = a.nrows();
    assert_eq!(n, a.ncols(), "mode_gram needs a square matrix");
    let mut w = Array2::<f64>::zeros((n, n));
    for j in 0..n {
        for k in 0..n {
            w[[j, k]] = a[[j, k]] * a[[j, k]];
        }
    }
    let mut g = Array2::<f64>::zeros((n, n));
    let mut u = vec![0.0_f64; n];
    for p in 0..n {
        for q in p..n {
            for j in 0..n {
                u[j] = a[[p, j]] * a[[q, j]];
            }
            let mut acc = 0.0;
            for j in 0..n {
                if u[j] == 0.0 {
                    continue;
                }
                let mut inner = 0.0;
                for k in 0..n {
                    inner += w[[j, k]] * u[k];
                }
                acc += u[j] * inner;
            }
            g[[p, q]] = acc;
            g[[q, p]] = acc;
        }
    }
    g
}

/// Tucker core of the triangle tensor in the HOSVD basis, `C = T x_1 U^T x_2
/// U^T x_3 U^T`, returned with the mode eigenvalues.
///
/// Returns `(lambda, c)` with `lambda` the eigenvalues of [`mode_gram`]
/// ascending and `c` the `(n, n * n)` mode-1 unfolding of the core, so
/// `c[[a, b * n + c]] = C_abc`.
///
/// Diagnostic only. The core is invariant only up to the block-orthogonal
/// gauge described in the module documentation, so it is not a descriptor;
/// this exists so the tests can say by how much the gauge bites.
pub fn hosvd_core(a: ArrayView2<f64>) -> (Array1<f64>, Array2<f64>) {
    let n = a.nrows();
    let (lambda, u) = symmetric_eigen(mode_gram(a).view(), EIGEN_SWEEPS);
    let t1 = triangle_unfolding(a);
    // C_(1) = U^T T_(1) (U kron U).
    let left = u.t().dot(&t1); // (n, n*n)
    let mut mid = Array2::<f64>::zeros((n, n * n));
    // Apply U^T in mode 2 then mode 3 by reshaping each row of `left`.
    for a_idx in 0..n {
        let mut blk = Array2::<f64>::zeros((n, n));
        for j in 0..n {
            for k in 0..n {
                blk[[j, k]] = left[[a_idx, j * n + k]];
            }
        }
        let out = u.t().dot(&blk).dot(&u);
        for b in 0..n {
            for c in 0..n {
                mid[[a_idx, b * n + c]] = out[[b, c]];
            }
        }
    }
    (lambda, mid)
}

/// Sorted pair distances of a flattened `(n, 3)` point set, the block
/// [`crate::bias::SortedPairs`] emits on its own.
fn sorted_pair_block(x: ArrayView1<f64>, n: usize) -> Vec<f64> {
    let mut d = Vec::with_capacity(n * (n - 1) / 2);
    for i in 0..n {
        for j in (i + 1)..n {
            let dx = x[3 * i] - x[3 * j];
            let dy = x[3 * i + 1] - x[3 * j + 1];
            let dz = x[3 * i + 2] - x[3 * j + 2];
            d.push((dx * dx + dy * dy + dz * dz).sqrt());
        }
    }
    d.sort_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal));
    d
}

/// The two-body and three-body kernel spectra followed by the sorted
/// distances.
///
/// The descriptor is `[ w * spec(A) , w * spec(M) , d_sorted ]` with `A` the
/// Gaussian kernel of [`kernel_matrix`] and `M` the triangle contraction of
/// [`triplet_matrix`], both spectra ascending. Length `2n + n(n-1)/2`, and the
/// distance block is exactly what [`crate::bias::SortedPairs`] emits.
///
/// Carrying the old descriptor entire is deliberate. Under a Euclidean metric
/// the squared distances add, `d_new^2 = w^2 d_spec^2 + d_old^2`, so at a
/// fixed radius the new descriptor separates every pair the old one separated
/// and can only add to the separation of the pairs it confused.
///
/// The spectra lead for a reason of the scan rather than of the mathematics,
/// the ordering being immaterial to any distance. `EuclideanMetric` stops
/// accumulating once the partial sum passes the merge radius, and a chain that
/// has opened twenty thousand basins pays that partial sum per centre on every
/// miss. At 98 points the spectra are 196 entries against the distances' 4753,
/// and they are the entries on which two different structures differ most, so
/// a miss can be settled in the first tenth of the vector.
pub struct TripletSpectrum {
    /// Points per state; the state length must be `3 * n_points`.
    pub n_points: usize,
    /// Gaussian kernel width, in the same length units as the coordinates.
    pub sigma: f64,
    /// Weight on the two spectral blocks relative to the distance block.
    pub spectral_weight: f64,
}

impl TripletSpectrum {
    /// Descriptor over `n_points` points with the defaults measured on
    /// Lennard-Jones clusters.
    ///
    /// `sigma = 2.5` is a little over two nearest-neighbour distances, the
    /// Lennard-Jones minimum being at `2^(1/6) = 1.122`, so the kernel reaches
    /// the second and third shells and the triangles it weights are the ones
    /// that distinguish packings.
    ///
    /// It is a compromise, and the two measurements pull opposite ways. The
    /// clusters prefer narrow: worst-case `z` at 38 points runs 1.15, 1.24,
    /// 1.20, 1.17, 1.15, 1.12 over widths 1.0, 1.4, 2.0, 2.5, 3.0, 4.0, and at
    /// 55 points 1.53, 1.37, 1.13, 1.11, 1.14, 0.99. The homometric pair
    /// demands width: 0.01, 0.29, 3.73, 10.26, 21.43, 20.25 on the same scan,
    /// so anything below 2.5 leaves it inseparable. Taking 2.5 gives up about
    /// 0.1 of cluster `z` against each system's own best width and buys the
    /// pair that sorted distances cannot do at all. Wider still flattens the
    /// kernel towards the all-ones matrix, and by `sigma = 6` a 6-point
    /// spectrum has collapsed onto one eigenvalue.
    ///
    /// `spectral_weight = 2.5` puts the two blocks on one noise scale. The
    /// spectral block responds to a 0.02 displacement at 0.32, 0.38, 0.29 and
    /// 0.39 times the distance block's response over 13, 26, 38 and 55 points,
    /// a ratio steady enough to fix rather than calibrate. Its reciprocal runs
    /// 2.6 to 3.4 and 2.5 sits just under, which leaves the distance block
    /// marginally the louder of the two: the conservative side, since that is
    /// the block whose radius is already calibrated. Unweighted, a merge
    /// radius reading the concatenation would be reading the distances alone.
    pub fn new(n_points: usize) -> Self {
        Self {
            n_points,
            sigma: 2.5,
            spectral_weight: 2.5,
        }
    }

    /// Replaces the kernel width.
    pub fn with_sigma(mut self, sigma: f64) -> Self {
        assert!(sigma > 0.0, "sigma must be > 0");
        self.sigma = sigma;
        self
    }

    /// Replaces the weight on the spectral blocks.
    pub fn with_spectral_weight(mut self, w: f64) -> Self {
        assert!(w >= 0.0, "spectral_weight must be >= 0");
        self.spectral_weight = w;
        self
    }

    /// The two spectra alone, `[ spec(A) , spec(M) ]`, unweighted.
    ///
    /// Split out because the separation measurements read the spectral block
    /// on its own, and because a caller wanting the three-body invariant
    /// without the `n(n-1)/2` distance block can take it here.
    pub fn spectra(&self, x: ArrayView1<f64>) -> Array1<f64> {
        let a = kernel_matrix(x, self.n_points, self.sigma);
        let m = triplet_matrix(a.view());
        let la = symmetric_eigenvalues(a.view());
        let lm = symmetric_eigenvalues(m.view());
        let mut out = Array1::zeros(2 * self.n_points);
        for i in 0..self.n_points {
            out[i] = la[i];
            out[self.n_points + i] = lm[i];
        }
        out
    }
}

impl Fingerprint for TripletSpectrum {
    fn describe(&self, x: ArrayView1<f64>) -> Array1<f64> {
        let n = self.n_points;
        let mut out = Vec::with_capacity(2 * n + n * (n - 1) / 2);
        for v in self.spectra(x).iter() {
            out.push(self.spectral_weight * v);
        }
        out.extend(sorted_pair_block(x, n));
        Array1::from(out)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::bias::BasinIndex;
    use ndarray::Array1;

    /// Deterministic linear congruential stream, so the tests carry no
    /// dependency on the rng crate's sequence.
    struct Lcg(u64);

    impl Lcg {
        fn next_f64(&mut self) -> f64 {
            self.0 = self.0.wrapping_mul(6364136223846793005).wrapping_add(1);
            ((self.0 >> 11) as f64) / ((1u64 << 53) as f64)
        }

        fn normal(&mut self) -> f64 {
            let u1 = self.next_f64().max(1e-12);
            let u2 = self.next_f64();
            (-2.0 * u1.ln()).sqrt() * (std::f64::consts::TAU * u2).cos()
        }

        fn permutation(&mut self, n: usize) -> Vec<usize> {
            let mut p: Vec<usize> = (0..n).collect();
            for i in (1..n).rev() {
                let j = (self.next_f64() * (i + 1) as f64) as usize;
                p.swap(i, j.min(i));
            }
            p
        }

        /// A rotation from the exponential of a random skew matrix, built by
        /// Gram-Schmidt on three random vectors so it needs no matrix
        /// exponential.
        fn rotation(&mut self) -> [[f64; 3]; 3] {
            let mut m = [[0.0; 3]; 3];
            for row in m.iter_mut() {
                for v in row.iter_mut() {
                    *v = self.normal();
                }
            }
            for i in 0..3 {
                for k in 0..i {
                    let dot: f64 = (0..3).map(|c| m[i][c] * m[k][c]).sum();
                    let basis = m[k];
                    for (v, b) in m[i].iter_mut().zip(basis.iter()) {
                        *v -= dot * b;
                    }
                }
                let nrm: f64 = (0..3).map(|c| m[i][c] * m[i][c]).sum::<f64>().sqrt();
                for v in m[i].iter_mut() {
                    *v /= nrm;
                }
            }
            m
        }
    }

    fn random_points(n: usize, seed: u64) -> Array1<f64> {
        let mut r = Lcg(seed);
        Array1::from(
            (0..3 * n)
                .map(|_| 1.6 * (n as f64).cbrt() * r.normal() / 2.0)
                .collect::<Vec<_>>(),
        )
    }

    fn permute_rotate(x: &Array1<f64>, n: usize, seed: u64) -> Array1<f64> {
        let mut r = Lcg(seed);
        let perm = r.permutation(n);
        let q = r.rotation();
        let shift = [r.normal(), r.normal(), r.normal()];
        let mut y = Array1::zeros(3 * n);
        for (i, &p) in perm.iter().enumerate() {
            let v = [x[3 * p], x[3 * p + 1], x[3 * p + 2]];
            for c in 0..3 {
                y[3 * i + c] = q[c][0] * v[0] + q[c][1] * v[1] + q[c][2] * v[2] + shift[c];
            }
        }
        y
    }

    /// Points on a line, for the homometric pair.
    fn on_a_line(vals: &[f64]) -> Array1<f64> {
        let mut x = Array1::zeros(3 * vals.len());
        for (i, v) in vals.iter().enumerate() {
            x[3 * i] = *v;
        }
        x
    }

    /// Bloom's homometric pair: the same multiset of pairwise distances,
    /// `{1..13, 16, 17}` both times, from two point sets that are not
    /// congruent, their gap sequences being `1 3 6 2 5` and `1 7 3 2 4` with
    /// neither equal to the other reversed.
    const HOM_A: [f64; 6] = [0.0, 1.0, 4.0, 10.0, 12.0, 17.0];
    const HOM_B: [f64; 6] = [0.0, 1.0, 8.0, 11.0, 13.0, 17.0];

    fn l2(a: &Array1<f64>, b: &Array1<f64>) -> f64 {
        a.iter()
            .zip(b.iter())
            .map(|(p, q)| (p - q) * (p - q))
            .sum::<f64>()
            .sqrt()
    }

    #[test]
    fn contraction_matches_the_explicit_triangle_tensor() {
        let n = 9;
        let x = random_points(n, 12345);
        let a = kernel_matrix(x.view(), n, 1.4);
        let t1 = triangle_unfolding(a.view());
        let scale = n as f64 / a.dot(&a).diag().sum();
        let m = triplet_matrix(a.view());
        for i in 0..n {
            for j in 0..n {
                let direct: f64 = (0..n).map(|k| t1[[i, j * n + k]]).sum();
                assert!(
                    (m[[i, j]] - direct * scale).abs() < 1e-12,
                    "M_{i}{j} is not the mode-3 marginal"
                );
            }
        }
    }

    #[test]
    fn the_distance_multiset_fixes_tr_a2_and_not_tr_a3() {
        let n = 6;
        let xa = on_a_line(&HOM_A);
        let xb = on_a_line(&HOM_B);
        let (aa, ab) = (
            kernel_matrix(xa.view(), n, 2.0),
            kernel_matrix(xb.view(), n, 2.0),
        );
        let tr2 = |m: &Array2<f64>| m.dot(m).diag().sum();
        let tr3 = |m: &Array2<f64>| m.dot(m).dot(m).diag().sum();
        assert!(
            (tr2(&aa) - tr2(&ab)).abs() < 1e-12,
            "tr A^2 is a function of the distance multiset and must agree"
        );
        assert!(
            (tr3(&aa) - tr3(&ab)).abs() > 1e-3,
            "tr A^3 is the weighted triangle sum and must not agree"
        );
    }

    #[test]
    fn mode_spectrum_is_invariant_under_relabelling_and_rigid_motion() {
        let n = 8;
        let x = random_points(n, 909);
        let y = permute_rotate(&x, n, 4242);
        let ga = mode_gram(kernel_matrix(x.view(), n, 1.4).view());
        let gb = mode_gram(kernel_matrix(y.view(), n, 1.4).view());
        let (la, _) = symmetric_eigen(ga.view(), EIGEN_SWEEPS);
        let (lb, _) = symmetric_eigen(gb.view(), EIGEN_SWEEPS);
        let scale = la.iter().map(|v| v.abs()).fold(0.0_f64, f64::max);
        for i in 0..n {
            assert!(
                (la[i] - lb[i]).abs() <= 1e-10 * scale,
                "mode eigenvalue {i} moved under a relabelling"
            );
        }
    }

    #[test]
    fn the_core_is_invariant_only_up_to_a_sign_per_mode() {
        let n = 7;
        let x = random_points(n, 777);
        let y = permute_rotate(&x, n, 31337);
        let (_, ca) = hosvd_core(kernel_matrix(x.view(), n, 1.4).view());
        let (_, cb) = hosvd_core(kernel_matrix(y.view(), n, 1.4).view());
        let mut worst_abs = 0.0_f64;
        let mut worst_signed = 0.0_f64;
        for a in 0..n {
            for b in 0..n {
                for c in 0..n {
                    let (p, q) = (ca[[a, b * n + c]], cb[[a, b * n + c]]);
                    worst_abs = worst_abs.max((p.abs() - q.abs()).abs());
                    worst_signed = worst_signed.max((p - q).abs());
                }
            }
        }
        assert!(
            worst_abs < 1e-8,
            "|C_abc| must be invariant, worst gap {worst_abs:e}"
        );
        assert!(
            worst_signed > 1e-3,
            "the signed core must not be invariant, worst gap {worst_signed:e}"
        );
    }

    #[test]
    fn the_core_norm_repeats_the_mode_spectrum() {
        let n = 7;
        let x = random_points(n, 2024);
        let (lambda, c) = hosvd_core(kernel_matrix(x.view(), n, 1.4).view());
        let core_norm: f64 = c.iter().map(|v| v * v).sum();
        let trace: f64 = lambda.sum();
        assert!(
            (core_norm - trace).abs() <= 1e-8 * trace,
            "HOSVD is norm preserving: ||C||_F^2 = tr G, got {core_norm} and {trace}"
        );
    }

    #[test]
    fn descriptor_is_invariant_under_relabelling_and_rigid_motion() {
        let n = 30;
        let f = TripletSpectrum::new(n);
        let x = random_points(n, 5150);
        for seed in [11_u64, 222, 3333] {
            let y = permute_rotate(&x, n, seed);
            let d = l2(&f.describe(x.view()), &f.describe(y.view()));
            assert!(d < 1e-8, "descriptor moved by {d:e} under a symmetry");
        }
    }

    #[test]
    fn descriptor_separates_a_homometric_pair_that_sorted_distances_cannot() {
        let n = 6;
        let xa = on_a_line(&HOM_A);
        let xb = on_a_line(&HOM_B);

        let pairs = |x: &Array1<f64>| Array1::from(sorted_pair_block(x.view(), n));
        let d_pairs = l2(&pairs(&xa), &pairs(&xb));
        assert!(
            d_pairs < 1e-12,
            "the pair must be homometric, sorted distances differ by {d_pairs:e}"
        );

        let f = TripletSpectrum::new(n);
        let sep = l2(&f.spectra(xa.view()), &f.spectra(xb.view()));

        // What the same descriptor does to a structure moved by 0.02 per
        // coordinate, which is the scale a quench leaves behind.
        let mut r = Lcg(8080);
        let mut jitter = 0.0_f64;
        for _ in 0..8 {
            let mut y = xa.clone();
            for v in y.iter_mut() {
                *v += 0.02 * r.normal();
            }
            let y = permute_rotate(&y, n, r.0);
            jitter = jitter.max(l2(&f.spectra(xa.view()), &f.spectra(y.view())));
        }
        // Measured at 8.9 to 10.0 across jitter draws, so the bar is set at
        // half of that: the claim under test is a decisive separation, not a
        // particular realisation of the noise.
        assert!(
            sep > 5.0 * jitter,
            "separation {sep:e} must clear the jitter response {jitter:e} by a wide margin"
        );
    }

    /// The descriptor has to work through the index, not only in isolation:
    /// a merge radius set from the jitter response must hold a jittered copy
    /// in one basin and open a second for the homometric partner.
    #[test]
    fn basin_index_merges_a_jitter_and_splits_a_homometric_pair() {
        let n = 6;
        let f = TripletSpectrum::new(n);
        let xa = on_a_line(&HOM_A);
        let xb = on_a_line(&HOM_B);

        let mut r = Lcg(4711);
        let mut copies = Vec::new();
        let mut jitter = 0.0_f64;
        let base = f.describe(xa.view());
        for _ in 0..8 {
            let mut y = xa.clone();
            for v in y.iter_mut() {
                *v += 0.02 * r.normal();
            }
            let y = permute_rotate(&y, n, r.0);
            jitter = jitter.max(l2(&base, &f.describe(y.view())));
            copies.push(y);
        }

        let mut idx = BasinIndex::new(f, 2.0 * jitter);
        let home = idx.basin_of(xa.view());
        for y in &copies {
            assert_eq!(
                idx.basin_of(y.view()),
                home,
                "a 0.02 jitter must stay in its basin at twice the jitter radius"
            );
        }
        assert_ne!(
            idx.basin_of(xb.view()),
            home,
            "the homometric partner must open its own basin"
        );
        assert_eq!(idx.n_basins(), 2);
    }
}
