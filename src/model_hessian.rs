//! A model Hessian, and the quench depth it predicts for free.
//!
//! The delayed-acceptance first stage needs the energy a relaxation will
//! recover from an unrelaxed structure. That quantity has a closed form to
//! leading order. Expanding about the minimum the descent will reach,
//!
//! ```text
//! E(x) - E(Q(x)) ~ 1/2 g^T H^-1 g
//! ```
//!
//! with `g` the gradient at `x` and `H` the Hessian. Regressing a scalar proxy
//! for that, which is what the surrogate did with `|g|^2 / 2 lambda`, collapses
//! an anisotropic quantity onto one curvature and was measured to leave the
//! second stage rejecting 65 per cent of what the first passed.
//!
//! The true `H` costs second derivatives. A *model* Hessian costs nothing: the
//! empirical pairwise force constants that quantum chemistry codes use to start
//! a quasi-Newton optimiser, where the constant between two points falls off
//! exponentially with their separation. Lindh and co-workers set them for
//! exactly this purpose, and the same construction appears as the exponential
//! preconditioner for atomistic optimisation. It is a function of the geometry
//! alone, so evaluating it charges nothing on the ledger.
//!
//! # The operator
//!
//! Stretch terms only, which is what a preconditioner needs and what a cluster
//! of one species has:
//!
//! ```text
//! H = sum_{i<j} k(r_ij) (u_ij u_ij^T) (x) [[1, -1], [-1, 1]]
//! ```
//!
//! with `u_ij` the unit vector along the pair and `k` the force constant. The
//! form is a weighted graph Laplacian with a directional projector on each
//! edge, so it is positive semidefinite and its nullspace is the three
//! translations, which the solver projects out.
//!
//! Applied matrix free, so a product costs `O(N^2)` arithmetic and no storage
//! beyond the structure.
//!
//! # Why this is not a preconditioner here
//!
//! It is the same object used differently. A preconditioner uses `H^-1 g` as a
//! direction; this uses `g^T H^-1 g` as a number, the depth, and hands it to a
//! decision about whether the real relaxation is worth paying for.

use ndarray::{Array1, ArrayView1};

/// Force-constant scale, in the units the structure's own spacing sets.
///
/// The magnitude does not matter for the decision the depth feeds, since the
/// surrogate regresses on it and absorbs any constant, but the *shape* does:
/// what carries information is that near pairs are stiff and far pairs are
/// soft.
pub const K0: f64 = 1.0;

/// Floor added to the diagonal, as a fraction of the contact force constant.
///
/// The stretch-only operator does not resist bending, so its nullspace holds
/// every transverse motion as well as the rigid ones, and a gradient lying in
/// it drives the solve to infinity: without this the depth of a chain pushed
/// sideways came back as 5e19. The true Hessian has no such nullspace, because
/// bending costs energy; the shift stands in for the angle terms the model
/// omits, and is what makes the operator invertible rather than merely
/// semidefinite.
pub const FLOOR: f64 = 0.05;

/// Decay of the force constant with separation, in units of the spacing.
///
/// A pair at twice the nearest-neighbour distance contributes about `e^-3` of
/// what a contacting pair does, which is the range over which a displacement
/// is actually resisted.
pub const ALPHA: f64 = 1.0;

/// The three constants the operator is built from, carried as a value.
///
/// The constants above are covalent-chemistry practice, which is the wrong
/// regime for a Lennard-Jones or Morse cluster: covalent numbers describe bonds
/// that break, and a rare-gas cluster is held by a well that is shallower and
/// longer ranged. Carrying the constants as a value rather than reading them
/// from statics is what lets [`crate::hessian_fit`] replace them with numbers
/// measured off the run's own descents.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct ModelParams {
    /// Force-constant scale, `K0`.
    pub k0: f64,
    /// Decay of the force constant with separation, `ALPHA`.
    pub alpha: f64,
    /// Diagonal shift as a fraction of `k0`, `FLOOR`.
    pub floor: f64,
}

impl Default for ModelParams {
    fn default() -> Self {
        Self {
            k0: K0,
            alpha: ALPHA,
            floor: FLOOR,
        }
    }
}

impl ModelParams {
    /// The operator's shape at unit scale.
    ///
    /// Every term carries `k0` linearly, the pair constants through
    /// `k0 exp(...)` and the diagonal through `floor k0`, so
    /// `H(k0, alpha, floor) = k0 H(1, alpha, floor)` exactly and the depth
    /// `1/2 g^T H^-1 g` goes as `1/k0`. A fit over the shape therefore never
    /// varies `k0`: it recovers it afterwards in closed form, as the one factor
    /// that lines the predicted depths up with the observed ones.
    pub fn unit_scale(alpha: f64, floor: f64) -> Self {
        Self {
            k0: 1.0,
            alpha,
            floor,
        }
    }
}

/// Mean nearest-neighbour distance, which sets the scale everything is in.
pub fn spacing(x: ArrayView1<f64>, n: usize) -> f64 {
    if n < 2 {
        return 1.0;
    }
    let mut total = 0.0;
    for i in 0..n {
        let mut best = f64::INFINITY;
        for j in 0..n {
            if i == j {
                continue;
            }
            let d: f64 = (0..3)
                .map(|k| {
                    let v = x[3 * i + k] - x[3 * j + k];
                    v * v
                })
                .sum();
            best = best.min(d);
        }
        total += best.sqrt();
    }
    total / n as f64
}

/// Applies the model Hessian to `v`.
///
/// Matrix free: each pair contributes a rank-one term along its own direction,
/// so nothing is stored and a product is one pass over the pairs.
pub fn apply(x: ArrayView1<f64>, n: usize, v: ArrayView1<f64>, scale: f64) -> Array1<f64> {
    apply_with(x, n, v, scale, ModelParams::default())
}

/// Applies the model Hessian to `v` with the constants given by `p`.
///
/// The same one pass over the pairs as [`apply`], with the three constants read
/// from a value instead of from statics, so a calibration can sweep them.
pub fn apply_with(
    x: ArrayView1<f64>,
    n: usize,
    v: ArrayView1<f64>,
    scale: f64,
    p: ModelParams,
) -> Array1<f64> {
    let mut out = Array1::zeros(3 * n);
    for i in 0..n {
        for j in (i + 1)..n {
            let mut u = [0.0f64; 3];
            let mut r2 = 0.0;
            for k in 0..3 {
                u[k] = x[3 * i + k] - x[3 * j + k];
                r2 += u[k] * u[k];
            }
            let r = r2.sqrt();
            if r < 1e-12 {
                continue;
            }
            for k in 0..3 {
                u[k] /= r;
            }
            // Exponential falloff in units of the structure's own spacing, so
            // the operator carries no length of its own.
            let k_ij = p.k0 * (-p.alpha * (r / scale - 1.0)).exp();
            // Relative displacement projected on the pair direction: only the
            // stretching part of the motion is resisted.
            let mut du = 0.0;
            for k in 0..3 {
                du += u[k] * (v[3 * i + k] - v[3 * j + k]);
            }
            let f = k_ij * du;
            for k in 0..3 {
                out[3 * i + k] += f * u[k];
                out[3 * j + k] -= f * u[k];
            }
        }
    }
    // The bending the stretch terms do not carry.
    for k in 0..(3 * n) {
        out[k] += p.floor * p.k0 * v[k];
    }
    out
}

/// Removes the translational component, which the operator annihilates.
fn deflate(v: &mut Array1<f64>, n: usize) {
    if n == 0 {
        return;
    }
    let mut mean = [0.0f64; 3];
    for i in 0..n {
        for k in 0..3 {
            mean[k] += v[3 * i + k];
        }
    }
    for m in mean.iter_mut() {
        *m /= n as f64;
    }
    for i in 0..n {
        for k in 0..3 {
            v[3 * i + k] -= mean[k];
        }
    }
}

/// Predicted energy a relaxation from `x` will recover, from the model Hessian.
///
/// Solves `H d = g` by conjugate gradients in the space orthogonal to the
/// translations and returns `1/2 g^T d`. Charges nothing: every operation is on
/// the geometry and the gradient the caller already holds.
///
/// `iters` bounds the solve. A handful is enough, because the number feeds a
/// decision rather than a step, and the surrogate that consumes it regresses
/// away any systematic bias a truncated solve leaves.
pub fn depth(x: ArrayView1<f64>, n: usize, gradient: ArrayView1<f64>, iters: usize) -> f64 {
    depth_with(x, n, gradient, iters, ModelParams::default())
}

/// Predicted depth with the constants given by `p`.
///
/// The solve [`depth`] runs, with the operator's three constants read from a
/// value. A calibration evaluates this once per candidate per sample, so the
/// cost of a refit is `grid points x samples x iters` products.
pub fn depth_with(
    x: ArrayView1<f64>,
    n: usize,
    gradient: ArrayView1<f64>,
    iters: usize,
    p: ModelParams,
) -> f64 {
    if n < 2 || gradient.len() != 3 * n {
        return 0.0;
    }
    let scale = spacing(x, n);
    let mut b = gradient.to_owned();
    deflate(&mut b, n);
    let mut d = Array1::<f64>::zeros(3 * n);
    let mut r = b.clone();
    let mut p_dir = r.clone();
    let mut rr = r.dot(&r);
    if rr <= 0.0 {
        return 0.0;
    }
    for _ in 0..iters.max(1) {
        let mut ap = apply_with(x, n, p_dir.view(), scale, p);
        deflate(&mut ap, n);
        let pap = p_dir.dot(&ap);
        // A non-positive curvature means the truncated operator has run out of
        // usable information in this direction, which for a semidefinite
        // operator means the residual has entered its nullspace.
        if !(pap > 1e-300) {
            break;
        }
        let alpha = rr / pap;
        d.scaled_add(alpha, &p_dir);
        r.scaled_add(-alpha, &ap);
        let rr_new = r.dot(&r);
        if rr_new <= 1e-24 * rr {
            break;
        }
        let beta = rr_new / rr;
        p_dir = &r + &(p_dir.mapv(|v| v * beta));
        rr = rr_new;
    }
    0.5 * b.dot(&d)
}

#[cfg(test)]
mod tests {
    use super::*;
    use ndarray::Array1;

    /// A compact three-dimensional arrangement, since a chain is degenerate
    /// for an operator that only resists stretching.
    fn blob(n: usize) -> Array1<f64> {
        let mut x = Array1::zeros(3 * n);
        for i in 0..n {
            x[3 * i] = (i % 3) as f64 * 1.1;
            x[3 * i + 1] = ((i / 3) % 3) as f64 * 1.1;
            x[3 * i + 2] = (i / 9) as f64 * 1.1;
        }
        x
    }

    /// Two points along x, displaced apart: the operator has to resist the
    /// stretch and ignore a rigid translation.
    #[test]
    fn it_resists_stretching_and_ignores_translation() {
        let x = Array1::from(vec![0.0, 0.0, 0.0, 1.0, 0.0, 0.0]);
        let stretch = Array1::from(vec![-1.0, 0.0, 0.0, 1.0, 0.0, 0.0]);
        let shift = Array1::from(vec![1.0, 0.0, 0.0, 1.0, 0.0, 0.0]);
        let hs = apply(x.view(), 2, stretch.view(), 1.0);
        let ht = apply(x.view(), 2, shift.view(), 1.0);
        // The floor acts on a translation too, so the check is that the
        // stretch terms do not.
        assert!(hs.dot(&stretch) > 0.1, "stretch was not resisted: {hs:?}");
        for (i, v) in ht.iter().enumerate() {
            assert!(
                (v - FLOOR * K0 * shift[i]).abs() < 1e-12,
                "a translation drew a stretch response: {ht:?}"
            );
        }
    }

    /// Motion across a pair is not a stretch, so the stretch-only operator has
    /// to leave it alone. This is the operator's known limitation stated as a
    /// test rather than left implicit: it is a preconditioner's Hessian, not a
    /// Hessian.
    #[test]
    fn transverse_motion_is_not_resisted() {
        let x = Array1::from(vec![0.0, 0.0, 0.0, 1.0, 0.0, 0.0]);
        let shear = Array1::from(vec![0.0, 1.0, 0.0, 0.0, -1.0, 0.0]);
        let h = apply(x.view(), 2, shear.view(), 1.0);
        // Only the floor resists it, which is the point: the stretch terms
        // contribute nothing to bending and the shift stands in for them.
        for (i, v) in h.iter().enumerate() {
            assert!(
                (v - FLOOR * K0 * shear[i]).abs() < 1e-12,
                "shear drew a stretch response: {h:?}"
            );
        }
    }

    /// The operator has to be positive semidefinite, or the conjugate-gradient
    /// solve is not solving anything.
    #[test]
    fn the_operator_is_positive_semidefinite() {
        let mut rng = 12345u64;
        let mut next = || {
            rng ^= rng << 13;
            rng ^= rng >> 7;
            rng ^= rng << 17;
            (rng >> 11) as f64 / (1u64 << 53) as f64 - 0.5
        };
        let n = 12;
        let x: Array1<f64> = Array1::from((0..3 * n).map(|_| next() * 4.0).collect::<Vec<_>>());
        let s = spacing(x.view(), n);
        for _ in 0..20 {
            let v: Array1<f64> = Array1::from((0..3 * n).map(|_| next()).collect::<Vec<_>>());
            let h = apply(x.view(), n, v.view(), s);
            assert!(v.dot(&h) >= -1e-9, "negative curvature {}", v.dot(&h));
        }
    }

    /// The depth has to be positive for a structure with a gradient, and zero
    /// for one without, since a relaxed structure has nothing left to recover.
    #[test]
    fn depth_is_positive_and_vanishes_at_a_minimum() {
        let n = 8;
        let mut x = Array1::zeros(3 * n);
        for i in 0..n {
            x[3 * i] = i as f64 * 1.1;
        }
        let zero = Array1::zeros(3 * n);
        assert_eq!(depth(x.view(), n, zero.view(), 12), 0.0);
        let mut g = Array1::zeros(3 * n);
        g[0] = 0.3;
        g[3] = -0.2;
        let d = depth(x.view(), n, g.view(), 12);
        assert!(d > 0.0, "depth {d} is not positive");
    }

    /// A larger gradient has to predict a larger recovery, or the number
    /// carries no information about how far a structure will fall.
    #[test]
    fn a_larger_gradient_predicts_a_deeper_fall() {
        let n = 10;
        let x = blob(n);
        let mut g = Array1::zeros(3 * n);
        for i in 0..n {
            g[3 * i + 1] = 0.1 * ((i % 5) as f64 - 2.0);
        }
        let small = depth(x.view(), n, g.view(), 20);
        let big = depth(x.view(), n, (&g * 3.0).view(), 20);
        assert!(
            big > 4.0 * small,
            "tripling the gradient moved the depth from {small} to {big}, which is not quadratic"
        );
    }

    /// The scale enters the depth as a pure reciprocal, which is the identity a
    /// calibration relies on when it fits the shape at `k0 = 1` and recovers the
    /// scale in closed form afterwards. If this fails, the shape and the scale
    /// are entangled and a two-parameter search is not enough.
    #[test]
    fn the_depth_goes_as_the_reciprocal_of_the_force_constant_scale() {
        let n = 10;
        let x = blob(n);
        let mut g = Array1::zeros(3 * n);
        for i in 0..n {
            g[3 * i] = 0.07 * ((i % 4) as f64 - 1.5);
            g[3 * i + 2] = 0.03 * ((i % 3) as f64 - 1.0);
        }
        let unit = ModelParams::unit_scale(1.7, 0.02);
        let scaled = ModelParams {
            k0: 8.0,
            ..unit
        };
        let a = depth_with(x.view(), n, g.view(), 40, unit);
        let b = depth_with(x.view(), n, g.view(), 40, scaled);
        assert!(
            (a / (8.0 * b) - 1.0).abs() < 1e-9,
            "eightfold stiffer gave depth {b} against {a}, not an eighth"
        );
    }

    /// The parametrised entry points have to reproduce the constants exactly,
    /// or the calibration is fitting a different operator than the one the
    /// depth consumer runs.
    #[test]
    fn the_defaults_reproduce_the_fixed_constants() {
        let n = 9;
        let x = blob(n);
        let mut g = Array1::zeros(3 * n);
        for i in 0..n {
            g[3 * i + 1] = 0.11 * ((i % 5) as f64 - 2.0);
        }
        let fixed = depth(x.view(), n, g.view(), 25);
        let parametrised = depth_with(x.view(), n, g.view(), 25, ModelParams::default());
        assert_eq!(
            fixed, parametrised,
            "the parametrised solve disagreed with the fixed one"
        );
    }

    /// And the scaling has to be the quadratic form's, since that is the claim:
    /// the depth goes as the square of the gradient at fixed geometry.
    #[test]
    fn the_depth_is_quadratic_in_the_gradient() {
        let n = 10;
        let x = blob(n);
        let mut g = Array1::zeros(3 * n);
        for i in 0..n {
            g[3 * i + 2] = 0.05 * ((i % 7) as f64 - 3.0);
        }
        let a = depth(x.view(), n, g.view(), 30);
        let b = depth(x.view(), n, (&g * 2.0).view(), 30);
        assert!(
            (b / a - 4.0).abs() < 1e-6,
            "doubling the gradient scaled the depth by {}, wanted 4",
            b / a
        );
    }
}
