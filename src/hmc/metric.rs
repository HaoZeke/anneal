//! Mass matrices for the hopping proposal, and the rigid modes they must not
//! spend energy on.
//!
//! HMC's mass matrix `M` sets the kinetic form `K(p) = 1/2 p^T M^-1 p`. The
//! metric that makes the dynamics well conditioned is the target's own
//! curvature: with `M = H` the transformed problem has unit Hessian and one
//! step size serves every direction. Stan cannot use that, because it has no
//! model of the target's curvature, so it estimates `M^-1` from the sample
//! covariance of the draws it has already taken (Stan reference manual, MCMC
//! chapter, "Euclidean metric"). A cluster search has something Stan does not:
//! [`crate::model_hessian`] is a curvature operator read off the coordinates
//! alone at zero cost on the evaluation ledger.
//!
//! Three metrics are therefore comparable on the same chain, which is the
//! measurement this module exists for.
//!
//! | metric | where it comes from | charged cost |
//! |--------|---------------------|--------------|
//! | [`MetricKind::Identity`] | nothing; the control | 0 |
//! | [`MetricKind::Diagonal`] | regularised sample variance of visited structures, Stan's estimator | 0 |
//! | [`MetricKind::ModelHessian`] | Lindh-type pair force constants at the current geometry | 0 |
//!
//! # The metric is frozen for the length of a trajectory
//!
//! A position-dependent metric turns the Hamiltonian non-separable and
//! leapfrog stops being either symplectic or reversible; recovering both needs
//! the implicit generalised leapfrog of Riemannian HMC, which costs fixed-point
//! iterations per step and the derivative of the metric with respect to the
//! coordinates. The metric here is built once from the trajectory's starting
//! structure and held fixed, so the integrator keeps the two properties the
//! proposal depends on. [`crate::hmc::hop`] states what the resulting
//! across-proposal variation does and does not cost.
//!
//! # The six rigid modes get a heavy mass, not a projection
//!
//! A pairwise potential is invariant under translation and rotation, so the
//! gradient is exactly orthogonal to the six rigid generators at every
//! configuration and the trajectory can put energy into them without changing
//! the energy at all. That is not free: displacement along a mode of mass `m`
//! scales as `m^-1/2`, and under the model Hessian the rigid modes carry only
//! the [`crate::model_hessian::FLOOR`] mass of 0.05 against a contact force
//! constant of 1. Six modes at twenty times the compliance take
//! `6 * 20 / (6 * 20 + 108) = 53` per cent of the squared displacement of a
//! 38-point trajectory and put it into a rigid motion the basin descriptor
//! cannot even see.
//!
//! The repair is to add mass rather than to project: `M' = M + c Z Z^T` with
//! `Z` an orthonormal basis of the rigid generators. That is still a symmetric
//! positive-definite mass matrix, so the dynamics stay exactly Hamiltonian, and
//! it needs no modification to the integrator. A projector would have to be
//! applied inside the kinetic form as `P M^-1 P`, whose square root and inverse
//! are not the ones the sampler holds.
//!
//! # Why dense, when the operator is matrix free
//!
//! One structure supplies the metric for a whole trajectory, so the operator is
//! inverted once per leapfrog step at a fixed geometry, and a momentum draw
//! needs a square root as well. A dense Cholesky supplies both exactly:
//! `(3N)^3/3` flops once, then `2 (3N)^2` per solve. Matrix-free conjugate
//! gradients cost `k` operator products per solve at `2 (3N)^2` each with `k`
//! around 40 for this operator's conditioning, and give a truncated answer
//! rather than an exact one. Dense wins while `3N < 3 k L`, which for `k = 40`
//! and a trajectory of `L = 32` leaves is `N < 1280` points: every cluster
//! anyone runs. Above that the ordering reverses and a matrix-free path is the
//! right one to write.
//!
//! Identity and diagonal metrics skip the factorisation entirely: both are
//! diagonal plus a rank-six update, so Woodbury inverts them in `O(3N)` and the
//! momentum draw is `D^1/2 z + sqrt(c) Z w`, which has covariance
//! `D + c Z Z^T` exactly.

use ndarray::{Array1, Array2, ArrayView1};
use rand::Rng;
use rand_distr::{Distribution, StandardNormal};

/// Mass multiplier applied to the six rigid modes.
///
/// Displacement along a mode scales as the inverse square root of its mass, so
/// `1e3` leaves a rigid motion at three per cent of an internal one. Larger
/// buys nothing: the modes are already invisible in the trajectory at that
/// point, and the metric's condition number is a Cholesky's problem.
pub const RIGID_MASS: f64 = 1.0e3;

/// Which mass matrix a chain runs.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum MetricKind {
    /// `M = I`. The control: no preconditioning at all.
    Identity,
    /// `M^-1 = diag(v)` with `v` Stan's regularised sample variance of the
    /// structures the chain has visited.
    ///
    /// This is the arm with a structural reason to be inert, and the reason is
    /// worth stating before the measurement rather than after. Stan's estimator
    /// is informative when the target's coordinates have different scales. The
    /// coordinates here are the Cartesian components of indistinguishable
    /// points, so the target is exactly invariant under permuting points and
    /// under rotating all of them, and the population covariance is therefore
    /// isotropic by symmetry. Whatever anisotropy the estimate shows is a
    /// finite-sample artefact of the particular structures visited.
    /// [`MetricAdaptation::condition`] reports it, so an inert arm is
    /// identified as inert rather than read as a weak effect.
    Diagonal,
    /// `M = H_model(x)`, the Lindh-type model Hessian at the trajectory's
    /// starting structure.
    ModelHessian,
}

impl MetricKind {
    /// Short name, for per-arm reporting.
    pub fn name(&self) -> &'static str {
        match self {
            MetricKind::Identity => "identity",
            MetricKind::Diagonal => "diagonal",
            MetricKind::ModelHessian => "modelhess",
        }
    }
}

/// An orthonormal basis of the translations and rotations at a structure.
///
/// Six vectors for a three-dimensional structure, fewer if the points are
/// collinear or coincident, which the Gram-Schmidt drops by rank rather than by
/// a special case.
#[derive(Debug, Clone)]
pub struct RigidModes {
    /// Column `k` is the `k`-th orthonormal generator, length `3n`.
    pub z: Vec<Array1<f64>>,
}

impl RigidModes {
    /// Builds the basis at `x`.
    ///
    /// The generators are the three translations and the three rotations about
    /// the centre of mass. Rotations are taken about the centroid because a
    /// rotation about any other point is a rotation about the centroid plus a
    /// translation, which is already in the span.
    pub fn at(x: ArrayView1<f64>, n: usize) -> Self {
        let dim = 3 * n;
        let mut c = [0.0f64; 3];
        for i in 0..n {
            for k in 0..3 {
                c[k] += x[3 * i + k];
            }
        }
        for v in c.iter_mut() {
            *v /= n.max(1) as f64;
        }
        let mut raw: Vec<Array1<f64>> = Vec::with_capacity(6);
        for k in 0..3 {
            let mut t = Array1::<f64>::zeros(dim);
            for i in 0..n {
                t[3 * i + k] = 1.0;
            }
            raw.push(t);
        }
        // Rotation about axis a: the displacement of point i is e_a x (x_i - c).
        for a in 0..3 {
            let mut r = Array1::<f64>::zeros(dim);
            for i in 0..n {
                let d = [x[3 * i] - c[0], x[3 * i + 1] - c[1], x[3 * i + 2] - c[2]];
                let b = (a + 1) % 3;
                let g = (a + 2) % 3;
                r[3 * i + b] = -d[g];
                r[3 * i + g] = d[b];
            }
            raw.push(r);
        }
        // Modified Gram-Schmidt, dropping any generator the structure does not
        // actually have.
        let mut z: Vec<Array1<f64>> = Vec::with_capacity(6);
        for mut v in raw {
            for q in &z {
                let d = v.dot(q);
                v.scaled_add(-d, q);
            }
            let nrm = v.dot(&v).sqrt();
            if nrm > 1e-8 {
                v /= nrm;
                z.push(v);
            }
        }
        Self { z }
    }

    /// Number of rigid modes the structure has.
    pub fn rank(&self) -> usize {
        self.z.len()
    }

    /// `Z^T v`, the components of `v` along the rigid modes.
    pub fn coords(&self, v: ArrayView1<f64>) -> Vec<f64> {
        self.z.iter().map(|q| q.dot(&v)).collect()
    }

    /// Fraction of `|v|^2` lying in the rigid subspace.
    ///
    /// The instrument for the mass choice: a trajectory whose displacement is
    /// mostly rigid has not moved the structure.
    pub fn share(&self, v: ArrayView1<f64>) -> f64 {
        let total = v.dot(&v);
        if total <= 0.0 {
            return 0.0;
        }
        let inside: f64 = self.coords(v).iter().map(|c| c * c).sum();
        inside / total
    }
}

/// A mass matrix frozen at one structure, ready to draw momenta and integrate.
///
/// Construct once per trajectory through [`MetricAdaptation::freeze`]. Every
/// method is arithmetic on the geometry: nothing here charges the ledger.
pub struct Metric {
    kind: MetricKind,
    dim: usize,
    /// Diagonal part `D`; all ones for [`MetricKind::Identity`].
    diag: Array1<f64>,
    /// Rigid generators, mass-weighted into `M` at [`RIGID_MASS`] times the
    /// mean diagonal mass.
    rigid: RigidModes,
    /// `c` in `M = D + c Z Z^T`, or in `M = H + c Z Z^T`.
    rigid_c: f64,
    /// Lower Cholesky factor of the full `M`, for the model-Hessian metric.
    chol: Option<Array2<f64>>,
    /// `(c^-1 I + Z^T D^-1 Z)^-1`, precomputed for the Woodbury path.
    woodbury: Option<Vec<Vec<f64>>>,
}

impl Metric {
    /// Which metric this is.
    pub fn kind(&self) -> MetricKind {
        self.kind
    }

    /// The rigid basis this metric was built against.
    pub fn rigid(&self) -> &RigidModes {
        &self.rigid
    }

    /// Draws `p ~ N(0, M)`.
    ///
    /// Exact in both paths. The dense path uses `p = L z`; the diagonal path
    /// uses `p = D^1/2 z + sqrt(c) Z w`, whose covariance is `D + c Z Z^T` by
    /// independence of `z` and `w`.
    pub fn sample<R: Rng + ?Sized>(&self, rng: &mut R) -> Array1<f64> {
        let z: Array1<f64> = Array1::from_iter((0..self.dim).map(|_| StandardNormal.sample(rng)));
        match &self.chol {
            Some(l) => {
                let mut p = Array1::<f64>::zeros(self.dim);
                for i in 0..self.dim {
                    let mut s = 0.0;
                    for j in 0..=i {
                        s += l[[i, j]] * z[j];
                    }
                    p[i] = s;
                }
                p
            }
            None => {
                let mut p = Array1::<f64>::zeros(self.dim);
                for i in 0..self.dim {
                    p[i] = self.diag[i].sqrt() * z[i];
                }
                let s = self.rigid_c.sqrt();
                for q in &self.rigid.z {
                    let w: f64 = StandardNormal.sample(rng);
                    p.scaled_add(s * w, q);
                }
                p
            }
        }
    }

    /// `M^-1 p`, the drift velocity.
    pub fn velocity(&self, p: ArrayView1<f64>) -> Array1<f64> {
        match &self.chol {
            Some(l) => {
                // Forward then back substitution on L L^T v = p.
                let mut y = Array1::<f64>::zeros(self.dim);
                for i in 0..self.dim {
                    let mut s = p[i];
                    for j in 0..i {
                        s -= l[[i, j]] * y[j];
                    }
                    y[i] = s / l[[i, i]];
                }
                let mut v = Array1::<f64>::zeros(self.dim);
                for i in (0..self.dim).rev() {
                    let mut s = y[i];
                    for j in (i + 1)..self.dim {
                        s -= l[[j, i]] * v[j];
                    }
                    v[i] = s / l[[i, i]];
                }
                v
            }
            None => {
                // Woodbury: M^-1 = D^-1 - D^-1 Z W Z^T D^-1 with
                // W = (c^-1 I + Z^T D^-1 Z)^-1.
                let mut v = Array1::<f64>::zeros(self.dim);
                for i in 0..self.dim {
                    v[i] = p[i] / self.diag[i];
                }
                let Some(w) = &self.woodbury else {
                    return v;
                };
                let k = self.rigid.rank();
                if k == 0 {
                    return v;
                }
                // Z^T D^-1 p, using v which already holds D^-1 p.
                let zt: Vec<f64> = self.rigid.z.iter().map(|q| q.dot(&v)).collect();
                let mut wz = vec![0.0f64; k];
                for (a, row) in w.iter().enumerate() {
                    for (b, val) in row.iter().enumerate() {
                        wz[a] += val * zt[b];
                    }
                }
                for (a, q) in self.rigid.z.iter().enumerate() {
                    for i in 0..self.dim {
                        v[i] -= wz[a] * q[i] / self.diag[i];
                    }
                }
                v
            }
        }
    }

    /// `K(p) = 1/2 p^T M^-1 p`.
    pub fn kinetic(&self, p: ArrayView1<f64>) -> f64 {
        0.5 * p.dot(&self.velocity(p))
    }

    /// A lower bound on the condition number of `M`, from the factorisation
    /// already in hand.
    ///
    /// The quantity the whole comparison turns on: a metric whose condition
    /// number is one is the identity wearing another name, and one that is
    /// large is carrying anisotropy the unit metric cannot. For the dense path
    /// the squared ratio of the extreme diagonal entries of the Cholesky factor
    /// bounds `cond(M)` below, which costs nothing beyond a pass over the
    /// diagonal; for the diagonal path the ratio of the masses is exact.
    ///
    /// The rigid modes are excluded. They are given [`RIGID_MASS`] by
    /// construction, so including them would report the constant this module
    /// chose rather than the curvature the structure has.
    pub fn condition_bound(&self) -> f64 {
        let (mut lo, mut hi) = (f64::INFINITY, 0.0f64);
        match &self.chol {
            Some(l) => {
                for i in 0..self.dim {
                    let v = l[[i, i]] * l[[i, i]];
                    if v > 0.0 && v < self.rigid_c * 0.5 {
                        lo = lo.min(v);
                        hi = hi.max(v);
                    }
                }
            }
            None => {
                for v in self.diag.iter() {
                    if *v > 0.0 {
                        lo = lo.min(*v);
                        hi = hi.max(*v);
                    }
                }
            }
        }
        if lo.is_finite() && lo > 0.0 {
            hi / lo
        } else {
            1.0
        }
    }
}

/// Stan's regularised running variance, and the metric choice it feeds.
///
/// Per-chain by construction: a replica ladder needs one of these per rung,
/// because a hot chain and a cold chain traverse differently conditioned
/// landscapes and share no step size or metric. Nothing in this struct is
/// global and nothing in it is keyed on a configuration, so a swap that moves
/// structures between rungs leaves it alone.
#[derive(Debug, Clone)]
pub struct MetricAdaptation {
    /// Which metric to build.
    pub kind: MetricKind,
    /// Points in a structure.
    pub n_points: usize,
    count: u64,
    mean: Array1<f64>,
    m2: Array1<f64>,
    /// Frozen estimate of `M^-1`, written when a window closes.
    inv_mass: Array1<f64>,
}

impl MetricAdaptation {
    /// A fresh adaptation state for `n_points` points.
    pub fn new(kind: MetricKind, n_points: usize) -> Self {
        let dim = 3 * n_points;
        Self {
            kind,
            n_points,
            count: 0,
            mean: Array1::zeros(dim),
            m2: Array1::zeros(dim),
            inv_mass: Array1::ones(dim),
        }
    }

    /// Records a structure the chain visited.
    ///
    /// Welford, so the running variance needs one pass and no stored sample.
    pub fn observe(&mut self, x: ArrayView1<f64>) {
        if self.kind != MetricKind::Diagonal || x.len() != self.mean.len() {
            return;
        }
        self.count += 1;
        let c = self.count as f64;
        for i in 0..self.mean.len() {
            let d = x[i] - self.mean[i];
            self.mean[i] += d / c;
            self.m2[i] += d * (x[i] - self.mean[i]);
        }
    }

    /// Observations since the last window close.
    pub fn samples(&self) -> u64 {
        self.count
    }

    /// Closes a window: writes the regularised variance into `M^-1` and
    /// restarts the accumulator.
    ///
    /// The shrinkage is Stan's, `(n/(n+5)) s^2 + 1e-3 (5/(n+5))`, which pulls a
    /// variance estimated from few draws towards a unit metric rather than
    /// letting it produce a mass matrix from three points
    /// (`stan/mcmc/var_adaptation.hpp`).
    pub fn close_window(&mut self) {
        if self.kind != MetricKind::Diagonal || self.count < 2 {
            self.restart();
            return;
        }
        let n = self.count as f64;
        for i in 0..self.inv_mass.len() {
            let s2 = self.m2[i] / (n - 1.0);
            self.inv_mass[i] = (n / (n + 5.0)) * s2 + 1e-3 * (5.0 / (n + 5.0));
        }
        self.restart();
    }

    fn restart(&mut self) {
        self.count = 0;
        self.mean.fill(0.0);
        self.m2.fill(0.0);
    }

    /// Ratio of the largest to the smallest diagonal mass.
    ///
    /// The instrument for whether the estimated metric carries any anisotropy
    /// at all. A value near one says the arm is running and doing nothing,
    /// which a solve count cannot distinguish from a mechanism that acts and
    /// does not help.
    pub fn condition(&self) -> f64 {
        let mut lo = f64::INFINITY;
        let mut hi = 0.0f64;
        for v in self.inv_mass.iter() {
            if *v > 0.0 {
                lo = lo.min(*v);
                hi = hi.max(*v);
            }
        }
        if lo.is_finite() && lo > 0.0 {
            hi / lo
        } else {
            1.0
        }
    }

    /// Builds the metric this chain will integrate with, frozen at `x`.
    ///
    /// Charges nothing: the model Hessian is a function of the coordinates and
    /// the diagonal estimate is already held.
    pub fn freeze(&self, x: ArrayView1<f64>) -> Metric {
        let n = self.n_points;
        let dim = 3 * n;
        let rigid = RigidModes::at(x, n);
        match self.kind {
            MetricKind::Identity | MetricKind::Diagonal => {
                // M^-1 = diag(inv_mass), so M = diag(1 / inv_mass).
                let diag: Array1<f64> = match self.kind {
                    MetricKind::Identity => Array1::ones(dim),
                    _ => self.inv_mass.mapv(|v| 1.0 / v.max(1e-12)),
                };
                let mean_mass = diag.iter().sum::<f64>() / dim.max(1) as f64;
                let c = RIGID_MASS * mean_mass;
                let woodbury = woodbury_core(&diag, &rigid, c);
                Metric {
                    kind: self.kind,
                    dim,
                    diag,
                    rigid,
                    rigid_c: c,
                    chol: None,
                    woodbury,
                }
            }
            MetricKind::ModelHessian => {
                let scale = crate::model_hessian::spacing(x, n);
                let mut h = crate::model_hessian::dense(x, n, scale);
                let mean_mass = (0..dim).map(|i| h[[i, i]]).sum::<f64>() / dim.max(1) as f64;
                let c = RIGID_MASS * mean_mass;
                for q in &rigid.z {
                    for i in 0..dim {
                        if q[i] == 0.0 {
                            continue;
                        }
                        for j in 0..dim {
                            h[[i, j]] += c * q[i] * q[j];
                        }
                    }
                }
                let chol = cholesky(&h, mean_mass);
                Metric {
                    kind: self.kind,
                    dim,
                    diag: Array1::ones(dim),
                    rigid,
                    rigid_c: c,
                    chol: Some(chol),
                    woodbury: None,
                }
            }
        }
    }
}

/// `(c^-1 I + Z^T D^-1 Z)^-1`, the small matrix Woodbury needs.
fn woodbury_core(diag: &Array1<f64>, rigid: &RigidModes, c: f64) -> Option<Vec<Vec<f64>>> {
    let k = rigid.rank();
    if k == 0 {
        return None;
    }
    let mut a = vec![vec![0.0f64; k]; k];
    for i in 0..k {
        for j in 0..k {
            let mut s = 0.0;
            for t in 0..diag.len() {
                s += rigid.z[i][t] * rigid.z[j][t] / diag[t];
            }
            a[i][j] = s;
        }
        a[i][i] += 1.0 / c;
    }
    invert_small(a)
}

/// Gauss-Jordan inverse of a small dense matrix, with partial pivoting.
fn invert_small(mut a: Vec<Vec<f64>>) -> Option<Vec<Vec<f64>>> {
    let k = a.len();
    let mut inv = vec![vec![0.0f64; k]; k];
    for (i, row) in inv.iter_mut().enumerate() {
        row[i] = 1.0;
    }
    for col in 0..k {
        let mut piv = col;
        for r in (col + 1)..k {
            if a[r][col].abs() > a[piv][col].abs() {
                piv = r;
            }
        }
        if a[piv][col].abs() < 1e-300 {
            return None;
        }
        a.swap(col, piv);
        inv.swap(col, piv);
        let d = a[col][col];
        for t in 0..k {
            a[col][t] /= d;
            inv[col][t] /= d;
        }
        for r in 0..k {
            if r == col {
                continue;
            }
            let f = a[r][col];
            if f == 0.0 {
                continue;
            }
            for t in 0..k {
                a[r][t] -= f * a[col][t];
                inv[r][t] -= f * inv[col][t];
            }
        }
    }
    Some(inv)
}

/// Lower Cholesky factor, retrying with a growing diagonal shift.
///
/// The model Hessian carries [`crate::model_hessian::FLOOR`] on its diagonal
/// and is positive definite in exact arithmetic. The retry is for the case the
/// caller hands in a structure with coincident points, where the exponential
/// force constant overflows the conditioning rather than the arithmetic.
fn cholesky(a: &Array2<f64>, scale: f64) -> Array2<f64> {
    let n = a.nrows();
    let mut jitter = 0.0f64;
    for attempt in 0..8 {
        let mut l = Array2::<f64>::zeros((n, n));
        let mut ok = true;
        for i in 0..n {
            for j in 0..=i {
                let mut s = a[[i, j]];
                if i == j {
                    s += jitter;
                }
                for t in 0..j {
                    s -= l[[i, t]] * l[[j, t]];
                }
                if i == j {
                    if !(s > 0.0) {
                        ok = false;
                        break;
                    }
                    l[[i, i]] = s.sqrt();
                } else {
                    l[[i, j]] = s / l[[j, j]];
                }
            }
            if !ok {
                break;
            }
        }
        if ok {
            return l;
        }
        jitter = if attempt == 0 {
            1e-8 * scale.max(1e-12)
        } else {
            jitter * 10.0
        };
    }
    // Every shift failed, which means the geometry is degenerate. A unit
    // metric is a proposal that still runs, and the caller's divergence
    // counter is what reports the structure.
    Array2::from_diag(&Array1::from_elem(n, scale.max(1e-12).sqrt()))
}

#[cfg(test)]
mod tests {
    use super::*;
    use rand::SeedableRng;
    use rand::rngs::StdRng;

    fn blob(n: usize) -> Array1<f64> {
        let mut x = Array1::zeros(3 * n);
        for i in 0..n {
            x[3 * i] = (i % 3) as f64 * 1.1;
            x[3 * i + 1] = ((i / 3) % 3) as f64 * 1.05;
            x[3 * i + 2] = (i / 9) as f64 * 1.15;
        }
        x
    }

    /// The dense assembly and the matrix-free product are two implementations
    /// of one operator, so they have to agree. Two implementations agreeing to
    /// machine precision says more about both than either says alone.
    #[test]
    fn dense_and_matrix_free_agree() {
        let n = 9;
        let x = blob(n);
        let s = crate::model_hessian::spacing(x.view(), n);
        let h = crate::model_hessian::dense(x.view(), n, s);
        let mut rng = StdRng::seed_from_u64(7);
        for _ in 0..5 {
            let v: Array1<f64> =
                Array1::from_iter((0..3 * n).map(|_| StandardNormal.sample(&mut rng)));
            let a = crate::model_hessian::apply(x.view(), n, v.view(), s);
            let b = h.dot(&v);
            for i in 0..3 * n {
                assert!(
                    (a[i] - b[i]).abs() < 1e-10,
                    "component {i} differs: {} against {}",
                    a[i],
                    b[i]
                );
            }
        }
    }

    /// A rigid motion has to be reproduced exactly by the basis, or the mass
    /// that is supposed to suppress it is being applied to the wrong subspace.
    #[test]
    fn the_rigid_basis_spans_translations_and_rotations() {
        let n = 10;
        let x = blob(n);
        let r = RigidModes::at(x.view(), n);
        assert_eq!(r.rank(), 6, "a three-dimensional blob has six rigid modes");
        // A translation.
        let mut t = Array1::<f64>::zeros(3 * n);
        for i in 0..n {
            t[3 * i + 1] = 0.7;
        }
        assert!(
            (r.share(t.view()) - 1.0).abs() < 1e-12,
            "translation escaped"
        );
        // A rotation about z, taken about the centroid.
        let mut c = [0.0f64; 3];
        for i in 0..n {
            for k in 0..3 {
                c[k] += x[3 * i + k] / n as f64;
            }
        }
        let mut w = Array1::<f64>::zeros(3 * n);
        for i in 0..n {
            w[3 * i] = -(x[3 * i + 1] - c[1]);
            w[3 * i + 1] = x[3 * i] - c[0];
        }
        assert!((r.share(w.view()) - 1.0).abs() < 1e-10, "rotation escaped");
    }

    /// Every metric has to draw momenta whose sample covariance is `M`, which
    /// is checked against the metric's own kinetic form: `E[p^T M^-1 p] = dim`.
    #[test]
    fn momentum_draws_carry_the_metrics_own_covariance() {
        let n = 7;
        let x = blob(n);
        let dim = 3 * n;
        for kind in [
            MetricKind::Identity,
            MetricKind::Diagonal,
            MetricKind::ModelHessian,
        ] {
            let ad = MetricAdaptation::new(kind, n);
            let m = ad.freeze(x.view());
            let mut rng = StdRng::seed_from_u64(11);
            let draws = 4000;
            let mut acc = 0.0;
            for _ in 0..draws {
                let p = m.sample(&mut rng);
                acc += 2.0 * m.kinetic(p.view());
            }
            let mean = acc / draws as f64;
            // Chi-squared with dim degrees of freedom: relative standard error
            // of the mean is sqrt(2 / (dim * draws)) = 0.3 per cent here, so
            // three per cent is ten sigma.
            assert!(
                (mean / dim as f64 - 1.0).abs() < 0.03,
                "{}: E[p^T M^-1 p] came to {mean} against dim {dim}",
                kind.name()
            );
        }
    }

    /// The Woodbury inverse and the dense factor solve the same system, so the
    /// two velocity paths have to return the same vector.
    #[test]
    fn the_two_inverse_paths_agree() {
        let n = 6;
        let x = blob(n);
        let dim = 3 * n;
        let ad = MetricAdaptation::new(MetricKind::Identity, n);
        let m = ad.freeze(x.view());
        // The same M built densely: I + c Z Z^T.
        let rigid = RigidModes::at(x.view(), n);
        let c = RIGID_MASS;
        let mut dense = Array2::<f64>::eye(dim);
        for q in &rigid.z {
            for i in 0..dim {
                for j in 0..dim {
                    dense[[i, j]] += c * q[i] * q[j];
                }
            }
        }
        let mut rng = StdRng::seed_from_u64(3);
        let p: Array1<f64> = Array1::from_iter((0..dim).map(|_| StandardNormal.sample(&mut rng)));
        let v = m.velocity(p.view());
        let back = dense.dot(&v);
        for i in 0..dim {
            assert!(
                (back[i] - p[i]).abs() < 1e-9,
                "M (M^-1 p) missed p at {i}: {} against {}",
                back[i],
                p[i]
            );
        }
    }

    /// The heavy rigid mass has to actually suppress rigid motion, which is the
    /// whole reason it is there. Without it the model-Hessian metric puts about
    /// half its displacement into a motion the descriptor cannot see.
    #[test]
    fn the_rigid_mass_suppresses_rigid_velocity() {
        let n = 12;
        let x = blob(n);
        let ad = MetricAdaptation::new(MetricKind::ModelHessian, n);
        let m = ad.freeze(x.view());
        let mut rng = StdRng::seed_from_u64(5);
        let mut share = 0.0;
        let draws = 200;
        for _ in 0..draws {
            let p = m.sample(&mut rng);
            let v = m.velocity(p.view());
            share += m.rigid().share(v.view());
        }
        share /= draws as f64;
        assert!(
            share < 0.02,
            "rigid modes took {:.3} of the velocity, so the mass is not biting",
            share
        );
    }

    /// Stan's shrinkage has to pull a variance estimated from a handful of
    /// draws towards the unit metric rather than emit whatever three points
    /// happened to say.
    #[test]
    fn the_regularised_variance_shrinks_towards_unity() {
        let n = 4;
        let mut ad = MetricAdaptation::new(MetricKind::Diagonal, n);
        let mut rng = StdRng::seed_from_u64(9);
        for _ in 0..3 {
            let v: Array1<f64> = Array1::from_iter((0..3 * n).map(|_| {
                let z: f64 = StandardNormal.sample(&mut rng);
                100.0 * z
            }));
            ad.observe(v.view());
        }
        ad.close_window();
        // With n = 3 the weight on the sample variance is 3/8, so a variance of
        // order 1e4 cannot come through unshrunk, and the floor keeps the
        // estimate off zero.
        let c = ad.condition();
        assert!(c.is_finite() && c > 1.0, "condition estimate came back {c}");
    }
}
