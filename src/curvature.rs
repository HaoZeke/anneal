//! Spectral statistics of the curvature at a point, without forming a Hessian.
//!
//! A work ledger charges gradient evaluations, so anything computed alongside a
//! relaxation has to be cheap in those units. Building the Hessian of a
//! 75-point cluster explicitly and diagonalising it costs of order twenty times
//! the relaxation it would inform, which makes it useless as a screen however
//! informative it is.
//!
//! Lanczos needs only the action of the Hessian on a vector, and that action is
//! a central difference of the gradient: two evaluations per matrix-vector
//! product, so the lowest handful of eigenpairs costs a few tens of gradients
//! against the roughly eighty a relaxation from a deep minimum takes.
//!
//! What the spectrum is for. On a multi-funnel landscape the structures a search
//! settles into and the structures worth leaving differ in stiffness before they
//! differ in any order parameter: a relaxation started near the minimum a search
//! plateaus at costs about 79 charged evaluations, while one started from a
//! random configuration costs about 270, because a deep minimum is stiff and a
//! perturbation falls straight back into it. That stiffness is the smallest
//! eigenvalues of the curvature, and it is available before the relaxation that
//! would otherwise discover it by returning.
//!
//! Three quantities come out of the low end of the spectrum:
//!
//! The smallest eigenvalue, which is the softest direction and near zero at a
//! saddle or a flat shoulder.
//!
//! The participation ratio of its eigenvector, which distinguishes a soft mode
//! carried by a few atoms, a surface rearrangement, from one carried by the
//! whole cluster, a collective distortion. Both can have the same eigenvalue.
//!
//! The gap between the first and second, which says whether the softest
//! direction is isolated or one of a degenerate family, and a degenerate soft
//! family is what a symmetric structure has.
//!
//! Rigid motions are projected out. A cluster in free space has three
//! translational zero modes, and near a minimum three rotational ones, and
//! leaving them in makes the smallest eigenvalues numerical noise about zero
//! rather than a property of the structure.

use ndarray::{Array1, ArrayView1};

/// Low-end spectral summary of the curvature at a point.
#[derive(Debug, Clone, PartialEq)]
pub struct CurvatureFeatures {
    /// Smallest non-rigid eigenvalue.
    pub lambda_min: f64,
    /// Second smallest, for the gap.
    pub lambda_second: f64,
    /// `lambda_second - lambda_min`; small when the softest mode is degenerate.
    pub gap: f64,
    /// Participation ratio of the softest mode, in `(0, 1]`.
    ///
    /// One when every atom moves equally, of order `1/n` when one atom carries
    /// the mode. Defined as `1 / (n * sum p_i^2)` with `p_i` the fraction of
    /// the squared norm on atom `i`, so it does not depend on cluster size for
    /// a collective mode.
    pub participation: f64,
    /// The softest non-rigid direction, unit norm.
    ///
    /// Returned rather than summarised because a direction is what an escape
    /// needs. Goedecker's argument for molecular dynamics as the escape is that
    /// it follows the soft modes and so crosses low barriers, by the
    /// Bell-Evans-Polanyi correlation between barrier height and the energy
    /// change along the path. An isotropic displacement has no such preference:
    /// scaled up it crosses high barriers or none, measured on LJ38 as a
    /// controller pinned at its ceiling with a discovery rate of 67 in 1871.
    pub mode: Array1<f64>,
    /// Gradient evaluations spent.
    pub evaluations: usize,
}

/// Orthonormal basis of the rigid motions of the structure at `x`.
///
/// Three translations and three infinitesimal rotations about the centre of
/// mass, orthonormalised against each other by modified Gram-Schmidt.
///
/// The orthonormalisation is the part that matters. Projecting against the six
/// generators one at a time only removes the rigid subspace when they happen to
/// be mutually orthogonal, and the rotation generators of a general geometry
/// are not. The residue survives as a near-zero eigenvalue of the projected
/// operator, so the reported softest curvature is a leftover rotation rather
/// than a property of the structure.
pub fn rigid_basis(x: ArrayView1<f64>) -> Vec<Array1<f64>> {
    let dim = x.len();
    let n = dim / 3;
    if n == 0 {
        return Vec::new();
    }
    let mut centre = [0.0_f64; 3];
    for i in 0..n {
        for k in 0..3 {
            centre[k] += x[3 * i + k];
        }
    }
    for c in centre.iter_mut() {
        *c /= n as f64;
    }

    let mut raw: Vec<Array1<f64>> = Vec::with_capacity(6);
    for k in 0..3 {
        let mut t = Array1::<f64>::zeros(dim);
        for i in 0..n {
            t[3 * i + k] = 1.0;
        }
        raw.push(t);
    }
    for axis in 0..3 {
        let mut r = Array1::<f64>::zeros(dim);
        for i in 0..n {
            let d = [
                x[3 * i] - centre[0],
                x[3 * i + 1] - centre[1],
                x[3 * i + 2] - centre[2],
            ];
            let (a, b) = ((axis + 1) % 3, (axis + 2) % 3);
            r[3 * i + a] = -d[b];
            r[3 * i + b] = d[a];
        }
        raw.push(r);
    }

    // Scale for the degeneracy test below, so the threshold is relative to the
    // size of the structure rather than absolute: a small cluster and a
    // near-degenerate generator are otherwise indistinguishable.
    let scale = raw
        .iter()
        .map(|v| v.iter().map(|z| z * z).sum::<f64>().sqrt())
        .fold(0.0_f64, f64::max)
        .max(1e-30);

    let mut basis: Vec<Array1<f64>> = Vec::with_capacity(6);
    for mut v in raw {
        // Two passes, because one pass of Gram-Schmidt loses orthogonality when
        // a vector is nearly dependent on those before it, as for a collinear
        // or highly symmetric arrangement.
        for _ in 0..2 {
            for b in &basis {
                let dot: f64 = v.iter().zip(b.iter()).map(|(a, c)| a * c).sum();
                for i in 0..dim {
                    v[i] -= dot * b[i];
                }
            }
        }
        let norm: f64 = v.iter().map(|z| z * z).sum::<f64>().sqrt();
        // A degenerate generator is dropped rather than normalised into noise.
        // A collinear arrangement has five rigid modes rather than six, because
        // rotation about its own axis moves nothing, and normalising that
        // generator would amplify rounding into a spurious basis vector that
        // then removes a real direction from the operator.
        if norm > 1e-7 * scale {
            basis.push(v / norm);
        }
    }
    basis
}

/// Removes the rigid motions in `basis` from `v`, in place.
pub fn project_rigid_with(v: &mut Array1<f64>, basis: &[Array1<f64>]) {
    for b in basis {
        let dot: f64 = v.iter().zip(b.iter()).map(|(a, c)| a * c).sum();
        for i in 0..v.len() {
            v[i] -= dot * b[i];
        }
    }
}

/// Removes translations and rotations of `x` from `v`, in place.
fn project_rigid(v: &mut Array1<f64>, x: ArrayView1<f64>) {
    let basis = rigid_basis(x);
    project_rigid_with(v, &basis);
}

/// Lowest curvature eigenvalues at `x`, by Lanczos on gradient differences.
///
/// `grad` returns the gradient at a point and counts against the caller's
/// budget. `steps` is the Krylov dimension; twenty is enough for the low end of
/// a cluster spectrum and costs forty gradients.
///
/// `epsilon` is the central-difference step. Too small and the difference is
/// dominated by rounding in the gradient, too large and it stops being a
/// directional derivative; the default is set for gradients accurate to about
/// machine precision.
///
/// Returns `None` when the Krylov space collapses before two eigenvalues are
/// available, which happens when the projected start vector is already an
/// eigenvector, rather than reporting a gap computed from one number.
/// Eigenvalues and eigenvectors of a Lanczos tridiagonal, ascending.
fn ritz(alphas: &[f64], betas: &[f64]) -> Option<(Vec<f64>, ndarray::Array2<f64>)> {
    let m = alphas.len();
    if m < 2 {
        return None;
    }
    let mut t = ndarray::Array2::<f64>::zeros((m, m));
    for i in 0..m {
        t[[i, i]] = alphas[i];
        if i + 1 < m {
            t[[i, i + 1]] = betas[i];
            t[[i + 1, i]] = betas[i];
        }
    }
    let (vals, vecs) = crate::spectral::symmetric_eigen(t.view(), 128);
    Some((vals.to_vec(), vecs))
}

/// Lanczos with full reorthogonalisation on an arbitrary symmetric operator.
///
/// Returns the diagonal, the off-diagonal and the basis. Split out because the
/// low end of a spectrum is reached by running this twice, once on the operator
/// and once on a shift of it.
fn lanczos_tridiag<F, P>(
    start: &Array1<f64>,
    steps: usize,
    op: &mut F,
    project: &P,
) -> Option<(Vec<f64>, Vec<f64>, Vec<Array1<f64>>)>
where
    F: FnMut(&Array1<f64>) -> Option<Array1<f64>>,
    P: Fn(&mut Array1<f64>),
{
    let dim = start.len();
    let mut q = start.clone();
    let mut alphas: Vec<f64> = Vec::new();
    let mut betas: Vec<f64> = Vec::new();
    let mut q_prev: Option<Array1<f64>> = None;
    let mut basis: Vec<Array1<f64>> = Vec::new();

    for _ in 0..steps {
        basis.push(q.clone());
        let mut w = op(&q)?;
        let alpha: f64 = w.iter().zip(q.iter()).map(|(a, b)| a * b).sum();
        alphas.push(alpha);
        for i in 0..dim {
            w[i] -= alpha * q[i];
        }
        if let (Some(prev), Some(beta)) = (q_prev.as_ref(), betas.last()) {
            for i in 0..dim {
                w[i] -= beta * prev[i];
            }
        }
        // Full reorthogonalisation. Lanczos loses orthogonality in floating
        // point and the lost directions reappear as spurious repeats of the
        // extreme eigenvalues, which is exactly what this is asked for.
        for b in &basis {
            let dot: f64 = w.iter().zip(b.iter()).map(|(a, c)| a * c).sum();
            for i in 0..dim {
                w[i] -= dot * b[i];
            }
        }
        // Re-projected explicitly, not only through the operator. The
        // constrained subspace is an invariant subspace in exact arithmetic
        // and not in floating point, and the directions the caller removed are
        // a strong attractor: under the spectral shift below they carry the
        // largest eigenvalue of the shifted operator, so a Krylov space that
        // drifts into them returns the removed subspace as the answer.
        project(&mut w);
        let beta: f64 = w.iter().map(|z| z * z).sum::<f64>().sqrt();
        if beta <= 1e-10 {
            break;
        }
        betas.push(beta);
        q_prev = Some(q.clone());
        q = w / beta;
    }
    if alphas.len() < 2 {
        return None;
    }
    Some((alphas, betas, basis))
}

pub fn curvature_features<G>(
    x: ArrayView1<f64>,
    mut grad: G,
    steps: usize,
    epsilon: f64,
) -> Option<CurvatureFeatures>
where
    G: FnMut(ArrayView1<f64>) -> Option<Array1<f64>>,
{
    let dim = x.len();
    if dim < 6 || steps < 2 {
        return None;
    }
    let mut evaluations = 0usize;
    // Built once: it depends only on `x`, and rebuilding it inside every
    // matrix-vector product would dominate the cheap part of this routine.
    let rigid = rigid_basis(x);

    // Hessian-vector product by central difference of the gradient, with the
    // rigid directions projected out on the way in and on the way out so the
    // operator acts on the non-rigid subspace only.
    let mut hv = |v: &Array1<f64>, evaluations: &mut usize| -> Option<Array1<f64>> {
        let mut d = v.clone();
        project_rigid_with(&mut d, &rigid);
        let mut xp = x.to_owned();
        let mut xm = x.to_owned();
        for i in 0..dim {
            xp[i] += epsilon * d[i];
            xm[i] -= epsilon * d[i];
        }
        let gp = grad(xp.view())?;
        let gm = grad(xm.view())?;
        *evaluations += 2;
        let mut out = Array1::zeros(dim);
        for i in 0..dim {
            out[i] = (gp[i] - gm[i]) / (2.0 * epsilon);
        }
        project_rigid_with(&mut out, &rigid);
        Some(out)
    };

    // Deterministic start, spread over all coordinates so it is unlikely to be
    // orthogonal to the softest mode.
    let mut q0 = Array1::from_shape_fn(dim, |i| ((i % 7) as f64 - 3.0) + 0.5);
    project_rigid_with(&mut q0, &rigid);
    let n0: f64 = q0.iter().map(|z| z * z).sum::<f64>().sqrt();
    if n0 <= 1e-12 {
        return None;
    }
    q0 /= n0;

    // Two passes, because Lanczos converges from the ends of the spectrum
    // outward and far faster at the top than at the bottom. A short run
    // returns the largest eigenvalue to several digits and the smallest barely
    // at all, and its lowest Ritz vector is dominated by stiff directions.
    //
    // That is not a small inaccuracy when the vector is used as a direction.
    // Displacing an LJ38 minimum along the eight-step "softest" mode raised the
    // energy so reliably that 6494 of 6550 trials failed the energy screen, and
    // the reported smallest eigenvalue sat at 6.84 across every seed.
    //
    // The fix is the standard spectral transformation. Run once to bound the
    // spectrum, then run again on `sigma I - H`, whose largest eigenvalue is
    // the smallest of `H` and which Lanczos therefore resolves quickly. The
    // cost is two passes rather than one.
    let project = |w: &mut Array1<f64>| project_rigid_with(w, &rigid);
    let (alphas, betas, _) =
        lanczos_tridiag(&q0, steps, &mut |v| hv(v, &mut evaluations), &project)?;
    let (top, _) = ritz(&alphas, &betas)?;
    let sigma = top[top.len() - 1] * 1.05 + 1.0;

    let (alphas, betas, basis) = lanczos_tridiag(
        &q0,
        steps,
        &mut |v| {
            // Both terms are formed on the projected vector, so the shifted
            // operator maps the constrained subspace into itself and the
            // removed directions never acquire the eigenvalue `sigma`.
            let mut vp = v.clone();
            project_rigid_with(&mut vp, &rigid);
            let h = hv(&vp, &mut evaluations)?;
            let mut out = Array1::zeros(dim);
            for i in 0..dim {
                out[i] = sigma * vp[i] - h[i];
            }
            project_rigid_with(&mut out, &rigid);
            Some(out)
        },
        &project,
    )?;
    let (mu, vecs) = ritz(&alphas, &betas)?;
    let m = mu.len();
    // Ascending in `mu` is descending in the eigenvalue of `H`.
    let lambda_min = sigma - mu[m - 1];
    let lambda_second = sigma - mu[m - 2];

    // The softest Ritz vector, lifted back to the full space.
    let mut mode = Array1::<f64>::zeros(dim);
    for (j, b) in basis.iter().enumerate().take(m) {
        let c = vecs[[j, m - 1]];
        for i in 0..dim {
            mode[i] += c * b[i];
        }
    }
    project_rigid_with(&mut mode, &rigid);
    let norm: f64 = mode.iter().map(|z| z * z).sum::<f64>().sqrt();
    if norm <= 1e-12 {
        return None;
    }
    mode /= norm;

    let n_atoms = dim / 3;
    let mut p2 = 0.0;
    for i in 0..n_atoms {
        let pi = mode[3 * i] * mode[3 * i]
            + mode[3 * i + 1] * mode[3 * i + 1]
            + mode[3 * i + 2] * mode[3 * i + 2];
        p2 += pi * pi;
    }
    let participation = if p2 > 0.0 {
        1.0 / (n_atoms as f64 * p2)
    } else {
        0.0
    };

    Some(CurvatureFeatures {
        lambda_min,
        lambda_second,
        gap: lambda_second - lambda_min,
        participation,
        mode,
        evaluations,
    })
}

/// The `k` softest non-rigid eigenpairs at a point.
///
/// The same two-pass shifted Lanczos as [`curvature_features`], returning the
/// subspace rather than the softest vector alone. A single soft mode is a poor
/// displacement -- measured here, 6494 of 6550 single-mode trials failed the
/// energy screen -- because one parabola climbed at fixed amplitude is all
/// rise. A draw over the soft *subspace* with per-mode thermal amplitudes is a
/// different object: it is the local Gaussian `N(0, T H^{-1})` truncated to
/// the modes that carry displacement at temperature `T`, the correct
/// preconditioned perturbation in the Hessian metric.
pub fn soft_subspace<G>(
    x: ArrayView1<f64>,
    mut grad: G,
    steps: usize,
    epsilon: f64,
    k: usize,
) -> Option<(Vec<f64>, Vec<Array1<f64>>, usize)>
where
    G: FnMut(ArrayView1<f64>) -> Option<Array1<f64>>,
{
    let dim = x.len();
    if dim < 6 || steps < 2 || k == 0 {
        return None;
    }
    let mut evaluations = 0usize;
    let rigid = rigid_basis(x);
    let mut hv = |v: &Array1<f64>, evaluations: &mut usize| -> Option<Array1<f64>> {
        let mut d = v.clone();
        project_rigid_with(&mut d, &rigid);
        let mut xp = x.to_owned();
        let mut xm = x.to_owned();
        for i in 0..dim {
            xp[i] += epsilon * d[i];
            xm[i] -= epsilon * d[i];
        }
        let gp = grad(xp.view())?;
        let gm = grad(xm.view())?;
        *evaluations += 2;
        let mut out = Array1::zeros(dim);
        for i in 0..dim {
            out[i] = (gp[i] - gm[i]) / (2.0 * epsilon);
        }
        project_rigid_with(&mut out, &rigid);
        Some(out)
    };
    let mut q0 = Array1::from_shape_fn(dim, |i| ((i % 7) as f64 - 3.0) + 0.5);
    project_rigid_with(&mut q0, &rigid);
    let n0: f64 = q0.iter().map(|z| z * z).sum::<f64>().sqrt();
    if n0 <= 1e-12 {
        return None;
    }
    q0 /= n0;
    let project = |w: &mut Array1<f64>| project_rigid_with(w, &rigid);
    let (alphas, betas, _) =
        lanczos_tridiag(&q0, steps, &mut |v| hv(v, &mut evaluations), &project)?;
    let (top, _) = ritz(&alphas, &betas)?;
    let sigma = top[top.len() - 1] * 1.05 + 1.0;
    let (alphas, betas, basis) = lanczos_tridiag(
        &q0,
        steps,
        &mut |v| {
            let mut vp = v.clone();
            project_rigid_with(&mut vp, &rigid);
            let h = hv(&vp, &mut evaluations)?;
            let mut out = Array1::zeros(dim);
            for i in 0..dim {
                out[i] = sigma * vp[i] - h[i];
            }
            project_rigid_with(&mut out, &rigid);
            Some(out)
        },
        &project,
    )?;
    let (mu, vecs) = ritz(&alphas, &betas)?;
    let m = mu.len();
    let take = k.min(m.saturating_sub(1));
    if take == 0 {
        return None;
    }
    let mut lambdas = Vec::with_capacity(take);
    let mut modes = Vec::with_capacity(take);
    // Ascending in mu is descending in the eigenvalue of H, so the softest
    // eigenpairs of H sit at the top of mu.
    for j in 0..take {
        let col = m - 1 - j;
        lambdas.push(sigma - mu[col]);
        let mut mode = Array1::<f64>::zeros(dim);
        for (row, b) in basis.iter().enumerate().take(m) {
            let c = vecs[[row, col]];
            for i in 0..dim {
                mode[i] += c * b[i];
            }
        }
        project_rigid_with(&mut mode, &rigid);
        let norm: f64 = mode.iter().map(|z| z * z).sum::<f64>().sqrt();
        if norm <= 1e-12 {
            continue;
        }
        mode /= norm;
        modes.push(mode);
    }
    if modes.is_empty() {
        return None;
    }
    lambdas.truncate(modes.len());
    Some((lambdas, modes, evaluations))
}

#[cfg(test)]
mod tests {
    use super::*;

    /// A three-dimensional arrangement of `n` points, not collinear.
    ///
    /// Geometry matters here: a collinear set has five rigid modes rather than
    /// six, so a degenerate generator is dropped and any test that assumes six
    /// is testing the wrong structure.
    fn scattered(n: usize) -> Array1<f64> {
        let mut v = Vec::with_capacity(3 * n);
        for i in 0..n {
            let t = i as f64;
            v.push(1.7 * (0.9 * t).sin() + 0.3 * t);
            v.push(1.3 * (1.7 * t).cos() - 0.2 * t);
            v.push(0.9 * (2.3 * t).sin() + 0.11 * t * t.sqrt());
        }
        Array1::from(v)
    }

    /// A separable quadratic whose curvature is known exactly, embedded so the
    /// rigid projection has something to remove.
    fn quadratic_grad(scales: &[f64]) -> impl Fn(ArrayView1<f64>) -> Option<Array1<f64>> + '_ {
        move |x: ArrayView1<f64>| {
            let mut g = Array1::zeros(x.len());
            for i in 0..x.len() {
                g[i] = scales[i % scales.len()] * x[i];
            }
            Some(g)
        }
    }

    #[test]
    fn recovers_the_low_end_of_a_known_spectrum() {
        // Curvatures 2, 20, 200 repeating; after rigid projection the smallest
        // genuine curvature is still 2.
        let scales = [2.0, 20.0, 200.0];
        let x = scattered(10);
        let f = curvature_features(x.view(), quadratic_grad(&scales), 24, 1e-4).unwrap();
        assert!(
            (f.lambda_min - 2.0).abs() < 1e-3,
            "lambda_min {} should be 2",
            f.lambda_min
        );
        assert!(f.lambda_second >= f.lambda_min);
        assert!(f.evaluations <= 96, "spent {} gradients", f.evaluations);
    }

    /// The projected operator, formed column by column, so a test can compare
    /// against a dense decomposition rather than against a constant someone
    /// worked out by hand. The rigid projection raises the smallest curvature
    /// above the smallest entry of the spectrum, which is why the constant is
    /// not simply the smallest scale.
    fn dense_projected(x: ArrayView1<f64>, scales: &[f64], epsilon: f64) -> (f64, f64) {
        let dim = x.len();
        let g = quadratic_grad(scales);
        let rigid = rigid_basis(x);
        let mut h = ndarray::Array2::<f64>::zeros((dim, dim));
        for j in 0..dim {
            let mut e = Array1::<f64>::zeros(dim);
            e[j] = 1.0;
            project_rigid_with(&mut e, &rigid);
            let mut xp = x.to_owned();
            let mut xm = x.to_owned();
            for i in 0..dim {
                xp[i] += epsilon * e[i];
                xm[i] -= epsilon * e[i];
            }
            let gp = g(xp.view()).unwrap();
            let gm = g(xm.view()).unwrap();
            let mut col = Array1::<f64>::zeros(dim);
            for i in 0..dim {
                col[i] = (gp[i] - gm[i]) / (2.0 * epsilon);
            }
            project_rigid_with(&mut col, &rigid);
            for i in 0..dim {
                h[[i, j]] = col[i];
            }
        }
        let (vals, _) = crate::spectral::symmetric_eigen(h.view(), 256);
        // Six eigenvalues at zero belong to the projected-out rigid subspace.
        let mut nz: Vec<f64> = vals.iter().copied().filter(|v| v.abs() > 1e-6).collect();
        nz.sort_by(|a, b| a.partial_cmp(b).unwrap());
        (nz[0], nz[1])
    }

    /// The case the three-value spectrum above cannot fail on. Lanczos ends a
    /// short run knowing the top of a spectrum well and the bottom badly, and a
    /// test whose operator has three distinct eigenvalues converges in three
    /// steps whatever the method does.
    #[test]
    fn a_short_run_still_reaches_the_bottom_of_a_dense_spectrum() {
        let scales: Vec<f64> = (0..90).map(|i| 1.5 + 1.65 * i as f64).collect();
        let x = scattered(30);
        let (truth, _) = dense_projected(x.view(), &scales, 1e-4);
        let f = curvature_features(x.view(), quadratic_grad(&scales), 16, 1e-4).unwrap();
        assert!(
            (f.lambda_min - truth).abs() < 0.3,
            "sixteen steps reported {} against a true smallest curvature of {truth}",
            f.lambda_min
        );
    }

    /// A direction is what the escape uses, so the mode has to be soft and not
    /// merely have a plausible eigenvalue attached to it. Its Rayleigh quotient
    /// is the test that ties the two together.
    #[test]
    fn the_mode_is_as_soft_as_its_eigenvalue_claims() {
        let scales: Vec<f64> = (0..90).map(|i| 1.5 + 1.65 * i as f64).collect();
        let x = scattered(30);
        let (truth, _) = dense_projected(x.view(), &scales, 1e-4);
        let f = curvature_features(x.view(), quadratic_grad(&scales), 16, 1e-4).unwrap();
        let g = quadratic_grad(&scales);
        let rigid = rigid_basis(x.view());
        let eps = 1e-4;
        let mut xp = x.to_owned();
        let mut xm = x.to_owned();
        for i in 0..x.len() {
            xp[i] += eps * f.mode[i];
            xm[i] -= eps * f.mode[i];
        }
        let gp = g(xp.view()).unwrap();
        let gm = g(xm.view()).unwrap();
        let mut hv = Array1::<f64>::zeros(x.len());
        for i in 0..x.len() {
            hv[i] = (gp[i] - gm[i]) / (2.0 * eps);
        }
        project_rigid_with(&mut hv, &rigid);
        let rq: f64 = hv.iter().zip(f.mode.iter()).map(|(a, b)| a * b).sum();
        assert!(
            rq < truth + 0.5,
            "the mode has curvature {rq} where the softest is {truth}"
        );
    }

    #[test]
    fn a_stiffer_point_reports_a_larger_smallest_eigenvalue() {
        let soft = [0.5, 5.0, 50.0];
        let stiff = [8.0, 20.0, 200.0];
        let x = scattered(10);
        let a = curvature_features(x.view(), quadratic_grad(&soft), 24, 1e-4).unwrap();
        let b = curvature_features(x.view(), quadratic_grad(&stiff), 24, 1e-4).unwrap();
        assert!(
            b.lambda_min > a.lambda_min * 4.0,
            "stiff {} should exceed soft {}",
            b.lambda_min,
            a.lambda_min
        );
    }

    #[test]
    fn the_rigid_basis_is_orthonormal() {
        // The property whose absence made the projection leak: projecting
        // against generators that are not mutually orthogonal leaves a residue
        // that reappears as a near-zero eigenvalue.
        // A genuinely three-dimensional arrangement. A linear one has five
        // rigid modes, not six, and the first version of this test used
        // coordinates that placed every atom on one line.
        let x = scattered(10);
        let b = rigid_basis(x.view());
        assert_eq!(b.len(), 6, "a generic cluster has six rigid modes");
        for i in 0..b.len() {
            for j in 0..b.len() {
                let dot: f64 = b[i].iter().zip(b[j].iter()).map(|(p, q)| p * q).sum();
                let want = if i == j { 1.0 } else { 0.0 };
                assert!(
                    (dot - want).abs() < 1e-12,
                    "not orthonormal at {i},{j}: {dot}"
                );
            }
        }
    }

    #[test]
    fn rigid_modes_are_projected_out() {
        // A pure translation must be annihilated, so it cannot appear as a
        // spurious zero eigenvalue.
        let x = Array1::from_shape_fn(12, |i| (i as f64) * 0.5);
        let mut v = Array1::zeros(12);
        for i in 0..4 {
            v[3 * i] = 1.0;
        }
        project_rigid(&mut v, x.view());
        let norm: f64 = v.iter().map(|z| z * z).sum::<f64>().sqrt();
        assert!(norm < 1e-12, "translation survived projection: {norm}");
    }

    #[test]
    fn rotations_are_projected_out() {
        let x = Array1::from(vec![
            1.0, 0.0, 0.0, -1.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, -1.0, 0.0,
        ]);
        let n = 4;
        // Infinitesimal rotation about z: v_i = e_z x r_i.
        let mut v = Array1::zeros(12);
        for i in 0..n {
            v[3 * i] = -x[3 * i + 1];
            v[3 * i + 1] = x[3 * i];
        }
        project_rigid(&mut v, x.view());
        let norm: f64 = v.iter().map(|z| z * z).sum::<f64>().sqrt();
        assert!(norm < 1e-10, "rotation survived projection: {norm}");
    }

    #[test]
    fn participation_separates_a_local_mode_from_a_collective_one() {
        // Built directly, since the quantity is a property of a vector.
        let n = 20;
        let mut collective = Array1::<f64>::zeros(3 * n);
        for i in 0..3 * n {
            collective[i] = 1.0;
        }
        let mut local = Array1::<f64>::zeros(3 * n);
        local[0] = 1.0;
        let pr = |v: &Array1<f64>| {
            let norm: f64 = v.iter().map(|z| z * z).sum::<f64>().sqrt();
            let u = v / norm;
            let mut p2 = 0.0;
            for i in 0..n {
                let pi =
                    u[3 * i] * u[3 * i] + u[3 * i + 1] * u[3 * i + 1] + u[3 * i + 2] * u[3 * i + 2];
                p2 += pi * pi;
            }
            1.0 / (n as f64 * p2)
        };
        assert!((pr(&collective) - 1.0).abs() < 1e-9);
        assert!(pr(&local) < 0.06, "a one-atom mode should be near 1/n");
    }

    #[test]
    fn a_budget_refusal_stops_rather_than_returning_a_partial_spectrum() {
        let scales = [2.0, 20.0];
        let x = scattered(10);
        let mut left = 3;
        let out = curvature_features(
            x.view(),
            |v| {
                if left == 0 {
                    return None;
                }
                left -= 1;
                let mut g = Array1::zeros(v.len());
                for i in 0..v.len() {
                    g[i] = scales[i % scales.len()] * v[i];
                }
                Some(g)
            },
            24,
            1e-4,
        );
        assert!(out.is_none(), "an exhausted budget must not yield features");
    }

    /// The subspace has to recover the low end of a known spectrum, pairwise
    /// orthonormal, in ascending order.
    #[test]
    fn the_soft_subspace_matches_a_known_spectrum() {
        let x = scattered(20);
        let scales: Vec<f64> = (0..60).map(|i| 0.5 + 0.25 * i as f64).collect();
        let g = quadratic_grad(&scales);
        let (l, v, _ev) = soft_subspace(x.view(), |p| g(p), 40, 1e-4, 4).expect("no subspace");
        assert!(l.len() >= 3, "only {} modes", l.len());
        for j in 1..l.len() {
            assert!(l[j] >= l[j - 1] - 1e-8, "not ascending: {l:?}");
        }
        for a in 0..v.len() {
            for b in 0..a {
                let dot: f64 = v[a].iter().zip(v[b].iter()).map(|(p, q)| p * q).sum();
                assert!(dot.abs() < 1e-6, "modes {a},{b} not orthogonal: {dot}");
            }
            let n: f64 = v[a].iter().map(|z| z * z).sum::<f64>();
            assert!((n - 1.0).abs() < 1e-8, "mode {a} not unit: {n}");
        }
        // The reference: dense projected eigenvalues of the same operator.
        let (lo, _hi) = dense_projected(x.view(), &scales, 1e-4);
        assert!(
            (l[0] - lo).abs() < 0.05 * lo.abs().max(1.0),
            "softest {} against dense {}",
            l[0],
            lo
        );
    }
}
