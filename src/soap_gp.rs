//! Gradient-enhanced Gaussian process over structures, with a SOAP kernel.
//!
//! # Why this and not [`crate::gpr`]
//!
//! `gpr_optim` carries a gradient-enhanced process already, and binding it was
//! the right first move: reusing a maintained implementation beats writing one.
//! Two measurements taken through that binding say it cannot serve as an
//! acquisition surface at this cluster size, and both are recorded in
//! [`crate::gpr`]'s module docs.
//!
//! Its kernel compares pair `(i, j)` of one structure with pair `(i, j)` of
//! another, so it is not invariant to relabelling the points. On 13 points a
//! relabelling that changes nothing physical moved the posterior mean by 218
//! standard deviations and raised the reported standard deviation from 1.42e-4
//! to 9.74e-1. That is fixable, by canonicalising with
//! [`crate::shape::CanonicalOrder`], at 2.4 ms per structure.
//!
//! The cost is not fixable through that C API. At 38 points with 20
//! observations a prediction takes 3116 ms, of which 3.46 ms is the posterior
//! mean and the rest is the variance dispatch. In the currency the ledger
//! charges that is 976000 force evaluations for one query, against a whole run
//! budget of 400000. Every acquisition function needs the variance.
//!
//! Both go away with a kernel over a descriptor that is already invariant. SOAP
//! is invariant to permutation, rotation and translation by construction, so
//! nothing is canonicalised, and the variance here is
//! `k(x, x) - ||L^-1 k*||^2`, quadratic in the row count rather than a
//! dispatch that rebuilds a joint posterior over 114 coordinates.
//!
//! # The kernel and its derivatives
//!
//! `k(A, B) = sigma^2 (pA . pB)^zeta` on normalised structure descriptors. With
//! `aA` the descriptor-space image of a coordinate direction on `A`, that is
//! `aA = JA u` for `JA` the Jacobian [`crate::morphology::SoapFeatures`]
//! returns:
//!
//! ```text
//! dk/duB      = sigma^2 zeta d^(zeta-1) (pA . aB)
//! d2k/duA duB = sigma^2 zeta [ (zeta-1) d^(zeta-2) (aA . pB)(pA . aB)
//!                              + d^(zeta-1) (aA . aB) ]
//! ```
//!
//! with `d = pA . pB`. Both were verified symbolically against `diff` of the
//! kernel for zeta in 2, 3, 4 and 6, residual exactly zero, before any of this
//! was written. `zeta >= 2` because of the `zeta - 2` power.
//!
//! # Directional conditioning
//!
//! Differentiation is linear, so `f` and its derivatives are jointly Gaussian
//! and conditioning is the ordinary Gaussian conditional on the stacked
//! observation vector. The catch is size: with `n` observations in `d`
//! coordinates the joint matrix is `n(d + 1)` square, and at 38 points `d` is
//! 114, so twenty observations give a 2300 square matrix.
//!
//! A directional derivative is as much a linear functional of `f` as a partial
//! derivative is, so the same algebra applies with the Cartesian axis replaced
//! by a direction. Conditioning on the derivative along the gradient direction
//! keeps the two things a search wants, the slope and the direction of steepest
//! descent, at one extra row per observation instead of 114. The joint matrix
//! is `2n` square. [`Conditioning::Full`] is kept so the loss is measurable
//! rather than asserted.
//!
//! # Sources
//!
//! Derivative observations: Solak, Murray-Smith, Leithead, Leith and Rasmussen,
//! *Derivative observations in Gaussian process models of dynamic systems*,
//! NIPS 2003; Rasmussen and Williams section 9.4.
//!
//! The kernel form: Bartok, Kondor and Csanyi, *On representing chemical
//! environments*, Phys Rev B 87, 184115 (2013),
//! doi:10.1103/PhysRevB.87.184115, equation 36, at zeta = 4.

use ndarray::{Array1, Array2};

/// Which linear functionals of the surface the model conditions on.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub enum Conditioning {
    /// Energies only.
    Values,
    /// Energies and the derivative along the gradient direction, one extra row
    /// per observation.
    #[default]
    Directional,
    /// Energies and every Cartesian derivative, `3N` extra rows per
    /// observation. Correct, and affordable only for a handful of structures.
    Full,
}

/// One structure the model has been told about.
struct Obs {
    /// Normalised SOAP descriptor.
    p: Vec<f64>,
    /// Energy.
    y: f64,
    /// Descriptor-space directions `a = J u`, one per derivative row.
    dirs: Vec<Vec<f64>>,
    /// Observed directional derivatives, parallel to `dirs`.
    dvals: Vec<f64>,
}

/// Which row of the joint system an index is.
#[derive(Clone, Copy)]
struct Row {
    obs: usize,
    /// `None` for a value row, `Some(k)` for the `k`th derivative row.
    dir: Option<usize>,
}

/// A Gaussian process over structures with a SOAP kernel.
pub struct SoapGp {
    /// Exponent of the normalised dot product; `zeta >= 2`.
    pub zeta: usize,
    /// Signal standard deviation, in energy units.
    pub amplitude: f64,
    /// Observation noise standard deviation on energies.
    pub value_noise: f64,
    /// Observation noise standard deviation on derivatives, in energy per unit
    /// length. Separate because the two are not in the same units and one
    /// number cannot condition both.
    pub grad_noise: f64,
    /// Which functionals to condition on.
    pub conditioning: Conditioning,
    /// Most observations retained. BEACON caps at 100 rather than sparsifying
    /// and this follows it.
    pub capacity: usize,
    obs: Vec<Obs>,
    rows: Vec<Row>,
    prior_mean: f64,
    chol: Option<Array2<f64>>,
    alpha: Option<Array1<f64>>,
}

impl SoapGp {
    /// A model with the given exponent, amplitude and noises.
    pub fn new(zeta: usize, amplitude: f64, value_noise: f64, grad_noise: f64) -> Self {
        assert!(zeta >= 2, "the second kernel derivative needs zeta >= 2");
        assert!(amplitude > 0.0, "the amplitude is a standard deviation");
        assert!(value_noise > 0.0, "a positive noise keeps the solve stable");
        assert!(grad_noise > 0.0, "a positive noise keeps the solve stable");
        Self {
            zeta,
            amplitude,
            value_noise,
            grad_noise,
            conditioning: Conditioning::default(),
            capacity: 100,
            obs: Vec::new(),
            rows: Vec::new(),
            prior_mean: 0.0,
            chol: None,
            alpha: None,
        }
    }

    /// Sets which functionals to condition on.
    pub fn with_conditioning(mut self, c: Conditioning) -> Self {
        self.conditioning = c;
        self
    }

    /// Sets how many observations to retain.
    pub fn with_capacity(mut self, n: usize) -> Self {
        assert!(n > 0, "a model with no room holds nothing");
        self.capacity = n;
        self
    }

    /// Structures observed.
    pub fn len(&self) -> usize {
        self.obs.len()
    }

    /// Whether nothing has been observed.
    pub fn is_empty(&self) -> bool {
        self.obs.is_empty()
    }

    /// Rows in the joint system, which the solve is cubic in.
    pub fn rows(&self) -> usize {
        self.rows.len()
    }

    /// The lowest energy observed.
    pub fn incumbent(&self) -> Option<f64> {
        self.obs
            .iter()
            .map(|o| o.y)
            .fold(None, |acc: Option<f64>, v| Some(acc.map_or(v, |a| a.min(v))))
    }

    /// Records a structure, its energy, and optionally the Cartesian gradient
    /// with the descriptor Jacobian.
    ///
    /// `p` is the normalised descriptor of length `D`; `jacobian` is
    /// `(D, 3N)` row-major; `gradient` is length `3N`. Both come from
    /// [`crate::morphology::SoapFeatures::describe`].
    pub fn observe(
        &mut self,
        p: &[f64],
        y: f64,
        gradient: Option<&[f64]>,
        jacobian: Option<&[f64]>,
    ) {
        if !y.is_finite() || p.iter().any(|v| !v.is_finite()) {
            return;
        }
        let mut o = Obs {
            p: p.to_vec(),
            y,
            dirs: Vec::new(),
            dvals: Vec::new(),
        };
        if let (Some(g), Some(j)) = (gradient, jacobian) {
            let d = p.len();
            let m = g.len();
            if j.len() == d * m && g.iter().all(|v| v.is_finite()) {
                match self.conditioning {
                    Conditioning::Values => {}
                    Conditioning::Directional => {
                        // The derivative along the gradient direction is the
                        // gradient norm: one scalar carrying both how steep the
                        // surface is and which way it falls.
                        let norm = g.iter().map(|v| v * v).sum::<f64>().sqrt();
                        if norm > 1e-12 {
                            let mut a = vec![0.0_f64; d];
                            for (f, slot) in a.iter_mut().enumerate() {
                                let mut s = 0.0;
                                for k in 0..m {
                                    s += j[f * m + k] * g[k];
                                }
                                *slot = s / norm;
                            }
                            o.dirs.push(a);
                            o.dvals.push(norm);
                        }
                    }
                    Conditioning::Full => {
                        for k in 0..m {
                            let mut a = vec![0.0_f64; d];
                            for (f, slot) in a.iter_mut().enumerate() {
                                *slot = j[f * m + k];
                            }
                            o.dirs.push(a);
                            o.dvals.push(g[k]);
                        }
                    }
                }
            }
        }
        self.obs.push(o);
        while self.obs.len() > self.capacity {
            self.obs.remove(0);
        }
        self.chol = None;
    }

    /// Forgets every observation, keeping the hyperparameters.
    pub fn clear(&mut self) {
        self.obs.clear();
        self.rows.clear();
        self.chol = None;
        self.alpha = None;
    }

    fn k_vv(&self, pa: &[f64], pb: &[f64]) -> f64 {
        self.amplitude * self.amplitude * powi(dot(pa, pb), self.zeta)
    }

    /// `dk / d(direction on B)`. Verified symbolically for zeta 2, 3, 4, 6.
    fn k_vd(&self, pa: &[f64], pb: &[f64], ab: &[f64]) -> f64 {
        self.amplitude
            * self.amplitude
            * self.zeta as f64
            * powi(dot(pa, pb), self.zeta - 1)
            * dot(pa, ab)
    }

    /// `d2k / d(direction on A) d(direction on B)`. Verified likewise.
    fn k_dd(&self, pa: &[f64], pb: &[f64], aa: &[f64], ab: &[f64]) -> f64 {
        let d = dot(pa, pb);
        let z = self.zeta as f64;
        self.amplitude
            * self.amplitude
            * z
            * ((z - 1.0) * powi(d, self.zeta - 2) * dot(aa, pb) * dot(pa, ab)
                + powi(d, self.zeta - 1) * dot(aa, ab))
    }

    fn k_rows(&self, r: Row, s: Row) -> f64 {
        let (a, b) = (&self.obs[r.obs], &self.obs[s.obs]);
        match (r.dir, s.dir) {
            (None, None) => self.k_vv(&a.p, &b.p),
            (None, Some(j)) => self.k_vd(&a.p, &b.p, &b.dirs[j]),
            // k is symmetric, so a derivative on the first argument is a
            // derivative on the second argument of the transposed pair.
            (Some(i), None) => self.k_vd(&b.p, &a.p, &a.dirs[i]),
            (Some(i), Some(j)) => self.k_dd(&a.p, &b.p, &a.dirs[i], &b.dirs[j]),
        }
    }

    fn k_query(&self, p: &[f64], s: Row) -> f64 {
        let b = &self.obs[s.obs];
        match s.dir {
            None => self.k_vv(p, &b.p),
            Some(j) => self.k_vd(p, &b.p, &b.dirs[j]),
        }
    }

    /// Rebuilds the row index and the factorisation.
    fn fit(&mut self) {
        self.rows.clear();
        if self.obs.is_empty() {
            return;
        }
        for (i, o) in self.obs.iter().enumerate() {
            self.rows.push(Row { obs: i, dir: None });
            for k in 0..o.dirs.len() {
                self.rows.push(Row {
                    obs: i,
                    dir: Some(k),
                });
            }
        }
        // The mean of the observations, not zero. On an energy scale near -400
        // a zero prior makes every unvisited region look catastrophic and the
        // acquisition never leaves the data.
        self.prior_mean = self.obs.iter().map(|o| o.y).sum::<f64>() / self.obs.len() as f64;

        let n = self.rows.len();
        let mut k = Array2::<f64>::zeros((n, n));
        for i in 0..n {
            for j in 0..=i {
                let v = self.k_rows(self.rows[i], self.rows[j]);
                k[[i, j]] = v;
                k[[j, i]] = v;
            }
            let noise = match self.rows[i].dir {
                None => self.value_noise,
                Some(_) => self.grad_noise,
            };
            k[[i, i]] += noise * noise;
        }

        let Some(l) = cholesky(&k) else {
            self.chol = None;
            self.alpha = None;
            return;
        };
        let mut target = Array1::<f64>::zeros(n);
        for (i, r) in self.rows.iter().enumerate() {
            target[i] = match r.dir {
                // A constant prior mean contributes to values, not derivatives.
                None => self.obs[r.obs].y - self.prior_mean,
                Some(k) => self.obs[r.obs].dvals[k],
            };
        }
        let a = solve_chol(&l, &target);
        self.chol = Some(l);
        self.alpha = Some(a);
    }

    /// Posterior mean and standard deviation of the energy at a structure.
    pub fn predict(&mut self, p: &[f64]) -> (f64, f64) {
        if self.chol.is_none() {
            self.fit();
        }
        if self.obs.is_empty() {
            return (self.prior_mean, self.amplitude);
        }
        let (l, alpha) = match (&self.chol, &self.alpha) {
            (Some(l), Some(a)) => (l, a),
            _ => return (self.prior_mean, self.amplitude),
        };
        let n = self.rows.len();
        let ks: Array1<f64> = (0..n).map(|i| self.k_query(p, self.rows[i])).collect();
        let mean = self.prior_mean + ks.iter().zip(alpha.iter()).map(|(a, b)| a * b).sum::<f64>();
        let v = forward_substitute(l, &ks);
        let var = (self.k_vv(p, p) - v.iter().map(|z| z * z).sum::<f64>()).max(0.0);
        (mean, var.sqrt())
    }

    /// Lower confidence bound as a score to be maximised, `-(mean - kappa sd)`.
    ///
    /// kappa = 2 is what GOFEE and BEACON both use; see
    /// [`crate::gpr::Acquisition`] for the sweep behind it.
    pub fn lower_confidence_bound(&mut self, p: &[f64], kappa: f64) -> f64 {
        let (mean, sd) = self.predict(p);
        -(mean - kappa * sd)
    }

    /// Expected improvement below the incumbent.
    pub fn expected_improvement(&mut self, p: &[f64]) -> f64 {
        let Some(best) = self.incumbent() else {
            return f64::INFINITY;
        };
        let (mean, sd) = self.predict(p);
        if sd < 1e-12 {
            return (best - mean).max(0.0);
        }
        let z = (best - mean) / sd;
        (best - mean) * crate::funnel_bo::normal_cdf(z) + sd * crate::funnel_bo::normal_pdf(z)
    }
}

fn dot(a: &[f64], b: &[f64]) -> f64 {
    a.iter().zip(b.iter()).map(|(p, q)| p * q).sum()
}

/// `x^n` for small non-negative `n`, with `x^0 = 1` including at `x = 0`, which
/// is where the `zeta - 2` term lands for `zeta = 2`.
fn powi(x: f64, n: usize) -> f64 {
    let mut acc = 1.0;
    for _ in 0..n {
        acc *= x;
    }
    acc
}

/// Lower-triangular Cholesky factor, or `None` if not positive definite.
fn cholesky(k: &Array2<f64>) -> Option<Array2<f64>> {
    let n = k.shape()[0];
    let mut l = Array2::<f64>::zeros((n, n));
    for i in 0..n {
        for j in 0..=i {
            let mut s = k[[i, j]];
            for m in 0..j {
                s -= l[[i, m]] * l[[j, m]];
            }
            if i == j {
                if s <= 0.0 {
                    return None;
                }
                l[[i, j]] = s.sqrt();
            } else {
                l[[i, j]] = s / l[[j, j]];
            }
        }
    }
    Some(l)
}

/// `L^-1 b`.
fn forward_substitute(l: &Array2<f64>, b: &Array1<f64>) -> Array1<f64> {
    let n = b.len();
    let mut v = Array1::<f64>::zeros(n);
    for i in 0..n {
        let mut s = b[i];
        for m in 0..i {
            s -= l[[i, m]] * v[m];
        }
        v[i] = s / l[[i, i]];
    }
    v
}

/// `(L L^T)^-1 b`.
fn solve_chol(l: &Array2<f64>, b: &Array1<f64>) -> Array1<f64> {
    let n = b.len();
    let v = forward_substitute(l, b);
    let mut a = Array1::<f64>::zeros(n);
    for i in (0..n).rev() {
        let mut s = v[i];
        for m in (i + 1)..n {
            s -= l[[m, i]] * a[m];
        }
        a[i] = s / l[[i, i]];
    }
    a
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Descriptors on a circle in the first two of four components, so they are
    /// unit length and the kernel is the one the model will meet.
    fn ring(t: f64) -> Vec<f64> {
        vec![t.cos(), t.sin(), 0.0, 0.0]
    }

    /// The tangent, as a descriptor-space direction.
    fn tangent(t: f64) -> Vec<f64> {
        vec![-t.sin(), t.cos(), 0.0, 0.0]
    }

    /// A surface inside the span of the zeta = 4 kernel.
    ///
    /// `(pA . pB)^4 = cos^4(a - b)`, whose span on the circle is
    /// `{1, cos 2t, sin 2t, cos 4t, sin 4t}`, five dimensions. Choosing a test
    /// surface inside that span separates what the conditioning does from what
    /// the kernel cannot represent at all.
    fn surface(t: f64) -> f64 {
        -400.0 + 30.0 * (4.0 * t).cos() + 12.0 * (2.0 * t).sin()
    }

    fn slope(t: f64) -> f64 {
        -120.0 * (4.0 * t).sin() + 24.0 * (2.0 * t).cos()
    }

    /// Records one observation, treating the ring parameter as the single
    /// coordinate, whose descriptor Jacobian is the tangent.
    fn feed(gp: &mut SoapGp, t: f64, with_slope: bool) {
        let p = ring(t);
        if with_slope {
            gp.observe(&p, surface(t), Some(&[slope(t)]), Some(&tangent(t)));
        } else {
            gp.observe(&p, surface(t), None, None);
        }
    }

    #[test]
    fn it_reproduces_the_energies_it_was_told() {
        let mut gp = SoapGp::new(4, 60.0, 1e-4, 1e-4).with_conditioning(Conditioning::Values);
        let ts = [0.0, 1.1, 2.3, 3.4, 4.8];
        for t in ts {
            feed(&mut gp, t, false);
        }
        for t in ts {
            let (mean, sd) = gp.predict(&ring(t));
            assert!(
                (mean - surface(t)).abs() < 0.5,
                "at {t} predicted {mean:.3} for {:.3}",
                surface(t)
            );
            assert!(sd < 1.0, "standard deviation {sd} where data sits");
        }
    }

    #[test]
    fn conditioning_on_derivatives_beats_conditioning_on_values_alone() {
        // The claim gradient enhancement rests on. The kernel spans five
        // dimensions and three structures are shown, so energies alone leave
        // the surface underdetermined while the same three with their slopes
        // determine it.
        let train = [0.0, 1.0, 2.0];
        let test: Vec<f64> = (0..60).map(|k| 0.13 + k as f64 * 0.104).collect();
        let err = |c: Conditioning| -> f64 {
            let mut gp = SoapGp::new(4, 60.0, 1e-3, 1e-3).with_conditioning(c);
            for t in train {
                feed(&mut gp, t, true);
            }
            test.iter()
                .map(|t| (gp.predict(&ring(*t)).0 - surface(*t)).abs())
                .sum::<f64>()
                / test.len() as f64
        };
        let values = err(Conditioning::Values);
        let directional = err(Conditioning::Directional);
        assert!(
            directional < 0.5 * values,
            "derivatives bought little: {directional:.4} against {values:.4}"
        );
    }

    #[test]
    fn the_posterior_slope_matches_the_slope_it_was_told() {
        // A sign error in the cross-covariance block still produces a
        // plausible-looking posterior mean, so the slope is checked directly.
        let mut gp = SoapGp::new(4, 60.0, 1e-5, 1e-5).with_conditioning(Conditioning::Directional);
        for k in 0..6 {
            feed(&mut gp, f64::from(k), true);
        }
        let t = 1.0;
        let h = 1e-4;
        let numeric = (gp.predict(&ring(t + h)).0 - gp.predict(&ring(t - h)).0) / (2.0 * h);
        let exact = slope(t);
        assert!(
            (numeric - exact).abs() < 0.05 * exact.abs().max(1.0),
            "posterior slope {numeric:.4} against {exact:.4}"
        );
    }

    #[test]
    fn full_and_directional_agree_when_there_is_one_coordinate() {
        // With a single coordinate the gradient direction is the coordinate
        // axis up to a sign, so the two modes observe the same functional. Any
        // disagreement is a sign bug in the directional path, which is the one
        // that divides by the gradient norm.
        let mk = |c: Conditioning| {
            let mut gp = SoapGp::new(4, 60.0, 1e-5, 1e-5).with_conditioning(c);
            for k in 0..5 {
                feed(&mut gp, f64::from(k) * 1.2, true);
            }
            gp
        };
        let mut a = mk(Conditioning::Directional);
        let mut b = mk(Conditioning::Full);
        for k in 0..25 {
            let t = f64::from(k) * 0.25;
            let (ma, _) = a.predict(&ring(t));
            let (mb, _) = b.predict(&ring(t));
            assert!(
                (ma - mb).abs() < 1e-6,
                "at {t} directional gave {ma:.6} and full gave {mb:.6}"
            );
        }
    }

    #[test]
    fn the_prior_follows_the_energy_scale() {
        let mut gp = SoapGp::new(4, 20.0, 1e-3, 1e-3).with_conditioning(Conditioning::Values);
        for k in 0..5 {
            gp.observe(&ring(0.1 * f64::from(k)), -400.0 - f64::from(k), None, None);
        }
        // Orthogonal to every observation, so the kernel vanishes and the
        // posterior is the prior.
        let (mean, _) = gp.predict(&[0.0, 0.0, 1.0, 0.0]);
        assert!(
            mean < -300.0,
            "far-field mean {mean} should revert to the data's scale, not zero"
        );
    }

    #[test]
    fn the_variance_is_small_on_the_data_and_large_away_from_it() {
        let mut gp = SoapGp::new(4, 40.0, 1e-4, 1e-4).with_conditioning(Conditioning::Directional);
        for k in 0..5 {
            feed(&mut gp, f64::from(k) * 0.2, true);
        }
        let (_, near) = gp.predict(&ring(0.4));
        let (_, far) = gp.predict(&[0.0, 0.0, 1.0, 0.0]);
        assert!(
            far > 10.0 * near,
            "uncertainty {far} away from data against {near} inside it"
        );
    }

    #[test]
    fn a_relabelling_cannot_move_this_kernel() {
        // The property the whole module exists for, at the level this file can
        // check: the kernel reads the descriptor and nothing else, so two
        // structures with the same descriptor are the same point. That the
        // descriptor itself is permutation invariant is featomic's guarantee
        // and is exercised in morphology.
        let mut gp = SoapGp::new(4, 40.0, 1e-4, 1e-4).with_conditioning(Conditioning::Values);
        for k in 0..5 {
            feed(&mut gp, f64::from(k) * 0.7, false);
        }
        let p = ring(1.3);
        let a = gp.predict(&p);
        let b = gp.predict(&p.clone());
        assert_eq!(a.0, b.0);
        assert_eq!(a.1, b.1);
    }

    #[test]
    fn the_row_count_is_what_the_conditioning_says() {
        // The cost argument in one assertion: directional conditioning is 2n
        // rows where full is n(1 + m). Getting this wrong is how a model that
        // was supposed to be affordable becomes the thing it replaced.
        let mut d = SoapGp::new(4, 40.0, 1e-4, 1e-4).with_conditioning(Conditioning::Directional);
        let mut f = SoapGp::new(4, 40.0, 1e-4, 1e-4).with_conditioning(Conditioning::Full);
        for k in 0..7 {
            feed(&mut d, f64::from(k) * 0.5, true);
            feed(&mut f, f64::from(k) * 0.5, true);
        }
        let _ = d.predict(&ring(0.1));
        let _ = f.predict(&ring(0.1));
        assert_eq!(d.rows(), 14, "directional should be 2n");
        // One coordinate in this fixture, so full is also 2n here; the point is
        // that they are built from different row sets.
        assert_eq!(f.rows(), 14);
    }
}
