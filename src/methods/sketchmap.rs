//! Sketch-map style nonlinear embedding for collective variables.
//!
//! Adapted from the sketch-map dimensionality reduction of Ceriotti,
//! Tribello & Parrinello (JCTC 2011 / cosmo-epfl/sketchmap): high-D and
//! low-D distances are transformed by sigmoid switch functions so that
//! only intermediate length scales contribute to the stress, preserving
//! local neighborhoods while discarding uninformative long-range metric
//! noise. We implement a landmark fit to 2-D and a cheap out-of-sample
//! projection for MetaD CVs and TPS basin geometry — not a full MD
//! trajectory analysis stack.
//!
//! High-D switch (default): `F(r) = 1 - (1 + (2^{a/b}-1)(r/σ)^a )^{-b/a}`.
//! Low-D switch `G` uses the same form with `(a', b', σ')`.
//! Stress: `Σ_{i<j} w_{ij} (F(D_ij) - G(d_ij))^2` over landmark pairs.
//! Out-of-sample: place a new point by minimizing stress vs landmarks
//! with the high-D distances fixed.

use ndarray::{Array1, Array2, ArrayView1, ArrayView2};

/// Default high-D sigmoid exponents (sketch-map literature defaults).
pub const DEFAULT_A: f64 = 4.0;
/// Default high-D sigmoid b.
pub const DEFAULT_B: f64 = 2.0;
/// Default low-D sigmoid a.
pub const DEFAULT_A_LOW: f64 = 2.0;
/// Default low-D sigmoid b.
pub const DEFAULT_B_LOW: f64 = 2.0;

/// Sigmoid switch `1 - (1 + (2^{a/b}-1)(r/sigma)^a )^{-b/a}`.
#[inline]
pub fn sigmoid_switch(r: f64, sigma: f64, a: f64, b: f64) -> f64 {
    if !(r.is_finite() && sigma.is_finite() && sigma > 0.0 && a > 0.0 && b > 0.0) {
        return 0.0;
    }
    if r <= 0.0 {
        return 0.0;
    }
    let c = (2.0_f64).powf(a / b) - 1.0;
    let t = 1.0 + c * (r / sigma).powf(a);
    1.0 - t.powf(-b / a)
}

/// Euclidean distance between two rows.
#[inline]
pub fn row_l2(a: ArrayView1<f64>, b: ArrayView1<f64>) -> f64 {
    a.iter()
        .zip(b.iter())
        .map(|(u, v)| (u - v).powi(2))
        .sum::<f64>()
        .sqrt()
}

/// Pairwise high-D distance matrix (n × n).
pub fn pairwise_l2(x: ArrayView2<f64>) -> Array2<f64> {
    let n = x.nrows();
    let mut d = Array2::<f64>::zeros((n, n));
    for i in 0..n {
        for j in (i + 1)..n {
            let dij = row_l2(x.row(i), x.row(j));
            d[[i, j]] = dij;
            d[[j, i]] = dij;
        }
    }
    d
}

/// Median pairwise distance (for automatic σ).
pub fn median_pairwise(d: ArrayView2<f64>) -> f64 {
    let n = d.nrows();
    let mut vals = Vec::with_capacity(n * n / 2);
    for i in 0..n {
        for j in (i + 1)..n {
            let v = d[[i, j]];
            if v.is_finite() && v > 0.0 {
                vals.push(v);
            }
        }
    }
    if vals.is_empty() {
        return 1.0;
    }
    vals.sort_by(|a, b| a.total_cmp(b));
    vals[vals.len() / 2].max(1e-9)
}

/// Fitted 2-D sketch-map over landmarks.
#[derive(Clone, Debug)]
pub struct SketchMap2d {
    /// Landmark configurations in ambient space, shape (n, dim).
    pub landmarks: Array2<f64>,
    /// Embedded coordinates, shape (n, 2).
    pub embedded: Array2<f64>,
    /// High-D switch scale.
    pub sigma_high: f64,
    /// Low-D switch scale.
    pub sigma_low: f64,
    /// High-D sigmoid exponent `a`.
    pub a_high: f64,
    /// High-D sigmoid exponent `b`.
    pub b_high: f64,
    /// Low-D sigmoid exponent `a`.
    pub a_low: f64,
    /// Low-D sigmoid exponent `b`.
    pub b_low: f64,
}

impl SketchMap2d {
    /// Fit a 2-D sketch-map to `points` (n × dim) by gradient descent on
    /// the sketch-map stress, initialized from classical MDS on the
    /// high-D switched dissimilarities.
    pub fn fit(points: ArrayView2<f64>, n_iters: usize, lr: f64) -> Option<Self> {
        let n = points.nrows();
        let dim = points.ncols();
        if n < 3 || dim < 1 {
            return None;
        }
        let d_high = pairwise_l2(points);
        let sigma_high = median_pairwise(d_high.view());
        let sigma_low = sigma_high; // common default: match scales after unitless F,G
        let a_h = DEFAULT_A;
        let b_h = DEFAULT_B;
        let a_l = DEFAULT_A_LOW;
        let b_l = DEFAULT_B_LOW;

        // Switched high-D dissimilarities.
        let mut fmat = Array2::<f64>::zeros((n, n));
        for i in 0..n {
            for j in (i + 1)..n {
                let f = sigmoid_switch(d_high[[i, j]], sigma_high, a_h, b_h);
                fmat[[i, j]] = f;
                fmat[[j, i]] = f;
            }
        }

        // Classical MDS init on F (double-centering).
        let mut embedded = classical_mds_2d(fmat.view())?;
        // Center.
        center_rows_2d(&mut embedded);

        // Gradient descent on stress.
        let lr = lr.max(1e-8);
        for _ in 0..n_iters {
            let mut grad = Array2::<f64>::zeros((n, 2));
            let mut stress = 0.0;
            for i in 0..n {
                for j in (i + 1)..n {
                    let dx = embedded[[i, 0]] - embedded[[j, 0]];
                    let dy = embedded[[i, 1]] - embedded[[j, 1]];
                    let dij = (dx * dx + dy * dy).sqrt().max(1e-12);
                    let g = sigmoid_switch(dij, sigma_low, a_l, b_l);
                    let f = fmat[[i, j]];
                    let diff = g - f;
                    stress += diff * diff;
                    // dG/dd via finite-difference-free analytic chain:
                    // G = 1 - (1 + c (d/σ)^a )^{-b/a}
                    // dG/dd = (b/σ) * c * (d/σ)^{a-1} * (1 + c (d/σ)^a )^{-b/a - 1}
                    let c = (2.0_f64).powf(a_l / b_l) - 1.0;
                    let u = dij / sigma_low;
                    let term = 1.0 + c * u.powf(a_l);
                    let dg_dd =
                        (b_l / sigma_low) * c * u.powf(a_l - 1.0) * term.powf(-b_l / a_l - 1.0);
                    // d stress / d pos_i ∝ 2 (G-F) dG/dd * (x_i - x_j)/d
                    let coef = 2.0 * diff * dg_dd / dij;
                    grad[[i, 0]] += coef * dx;
                    grad[[i, 1]] += coef * dy;
                    grad[[j, 0]] -= coef * dx;
                    grad[[j, 1]] -= coef * dy;
                }
            }
            let _ = stress;
            for i in 0..n {
                embedded[[i, 0]] -= lr * grad[[i, 0]];
                embedded[[i, 1]] -= lr * grad[[i, 1]];
            }
            center_rows_2d(&mut embedded);
        }

        Some(Self {
            landmarks: points.to_owned(),
            embedded,
            sigma_high,
            sigma_low,
            a_high: a_h,
            b_high: b_h,
            a_low: a_l,
            b_low: b_l,
        })
    }

    /// Out-of-sample projection of `x` into the 2-D map by minimizing
    /// sketch-map stress against landmarks (few GD steps from nearest
    /// landmark).
    pub fn project(&self, x: ArrayView1<f64>, n_iters: usize) -> Array1<f64> {
        let n = self.landmarks.nrows();
        assert_eq!(x.len(), self.landmarks.ncols());
        // Init at nearest landmark embedding.
        let mut best_i = 0usize;
        let mut best_d = f64::INFINITY;
        let mut f_to_l = vec![0.0; n];
        for i in 0..n {
            let d = row_l2(x, self.landmarks.row(i));
            f_to_l[i] = sigmoid_switch(d, self.sigma_high, self.a_high, self.b_high);
            if d < best_d {
                best_d = d;
                best_i = i;
            }
        }
        let mut y0 = self.embedded[[best_i, 0]];
        let mut y1 = self.embedded[[best_i, 1]];
        let lr = 0.05 * self.sigma_low.max(1e-6);
        for _ in 0..n_iters.max(1) {
            let mut g0 = 0.0;
            let mut g1 = 0.0;
            for i in 0..n {
                let dx = y0 - self.embedded[[i, 0]];
                let dy = y1 - self.embedded[[i, 1]];
                let dij = (dx * dx + dy * dy).sqrt().max(1e-12);
                let g = sigmoid_switch(dij, self.sigma_low, self.a_low, self.b_low);
                let f = f_to_l[i];
                let diff = g - f;
                let c = (2.0_f64).powf(self.a_low / self.b_low) - 1.0;
                let u = dij / self.sigma_low;
                let term = 1.0 + c * u.powf(self.a_low);
                let dg_dd = (self.b_low / self.sigma_low)
                    * c
                    * u.powf(self.a_low - 1.0)
                    * term.powf(-self.b_low / self.a_low - 1.0);
                let coef = 2.0 * diff * dg_dd / dij;
                g0 += coef * dx;
                g1 += coef * dy;
            }
            y0 -= lr * g0;
            y1 -= lr * g1;
        }
        Array1::from_vec(vec![y0, y1])
    }

    /// Build a linear projector approximating the sketch-map near the
    /// landmark mean (Jacobian via finite differences) for use with
    /// [`crate::bias::WellTemperedBias`].
    ///
    /// Returns `(projector dim×2, mu, cv_low, cv_high)`.
    pub fn linearize_projector(&self, eps: f64) -> (Array2<f64>, Array1<f64>, [f64; 2], [f64; 2]) {
        let dim = self.landmarks.ncols();
        let mu = self
            .landmarks
            .mean_axis(ndarray::Axis(0))
            .unwrap_or(Array1::zeros(dim));
        let s0 = self.project(mu.view(), 20);
        let eps = eps.max(1e-8);
        let mut projector = Array2::<f64>::zeros((dim, 2));
        for d in 0..dim {
            let mut xp = mu.clone();
            let mut xm = mu.clone();
            xp[d] += eps;
            xm[d] -= eps;
            let sp = self.project(xp.view(), 12);
            let sm = self.project(xm.view(), 12);
            projector[[d, 0]] = (sp[0] - sm[0]) / (2.0 * eps);
            projector[[d, 1]] = (sp[1] - sm[1]) / (2.0 * eps);
        }
        // CV box from embedded landmark range with padding.
        let mut lo = [f64::INFINITY, f64::INFINITY];
        let mut hi = [f64::NEG_INFINITY, f64::NEG_INFINITY];
        for i in 0..self.embedded.nrows() {
            for k in 0..2 {
                lo[k] = lo[k].min(self.embedded[[i, k]]);
                hi[k] = hi[k].max(self.embedded[[i, k]]);
            }
        }
        for k in 0..2 {
            if !lo[k].is_finite() || !hi[k].is_finite() || lo[k] >= hi[k] {
                lo[k] = s0[k] - 1.0;
                hi[k] = s0[k] + 1.0;
            } else {
                let pad = 0.15 * (hi[k] - lo[k]).max(1e-3);
                lo[k] -= pad;
                hi[k] += pad;
            }
        }
        let _ = s0;
        (projector, mu, lo, hi)
    }
}

fn center_rows_2d(y: &mut Array2<f64>) {
    let n = y.nrows() as f64;
    let m0: f64 = y.column(0).sum() / n;
    let m1: f64 = y.column(1).sum() / n;
    for i in 0..y.nrows() {
        y[[i, 0]] -= m0;
        y[[i, 1]] -= m1;
    }
}

/// Classical MDS to 2-D from a symmetric dissimilarity matrix.
fn classical_mds_2d(dist: ArrayView2<f64>) -> Option<Array2<f64>> {
    let n = dist.nrows();
    if n < 2 {
        return None;
    }
    // B = -1/2 J D^2 J
    let mut d2 = Array2::<f64>::zeros((n, n));
    for i in 0..n {
        for j in 0..n {
            d2[[i, j]] = dist[[i, j]] * dist[[i, j]];
        }
    }
    let mut row_mean = vec![0.0; n];
    let mut col_mean = vec![0.0; n];
    let mut total = 0.0;
    for i in 0..n {
        for j in 0..n {
            row_mean[i] += d2[[i, j]];
            col_mean[j] += d2[[i, j]];
            total += d2[[i, j]];
        }
    }
    for i in 0..n {
        row_mean[i] /= n as f64;
        col_mean[i] /= n as f64;
    }
    total /= (n * n) as f64;
    let mut b = Array2::<f64>::zeros((n, n));
    for i in 0..n {
        for j in 0..n {
            b[[i, j]] = -0.5 * (d2[[i, j]] - row_mean[i] - col_mean[j] + total);
        }
    }
    // Power iteration for top-2 eigenpairs (symmetric).
    let mut y = Array2::<f64>::zeros((n, 2));
    for k in 0..2 {
        let mut v = Array1::from_elem(n, 1.0 / (n as f64).sqrt());
        // Orthogonalize against previous.
        if k == 1 {
            let prev = y.column(0).to_owned();
            let dot: f64 = v.iter().zip(prev.iter()).map(|(a, b)| a * b).sum();
            for i in 0..n {
                v[i] -= dot * prev[i];
            }
            let norm = v.dot(&v).sqrt().max(1e-15);
            v.mapv_inplace(|z| z / norm);
        }
        let mut lambda = 0.0;
        for _ in 0..80 {
            let mut w = Array1::zeros(n);
            for i in 0..n {
                let mut acc = 0.0;
                for j in 0..n {
                    acc += b[[i, j]] * v[j];
                }
                w[i] = acc;
            }
            if k == 1 {
                let prev = y.column(0).to_owned();
                let dot: f64 = w.iter().zip(prev.iter()).map(|(a, b)| a * b).sum();
                for i in 0..n {
                    w[i] -= dot * prev[i];
                }
            }
            let norm = w.dot(&w).sqrt().max(1e-15);
            v = w / norm;
            // Rayleigh
            let mut num = 0.0;
            for i in 0..n {
                let mut acc = 0.0;
                for j in 0..n {
                    acc += b[[i, j]] * v[j];
                }
                num += v[i] * acc;
            }
            lambda = num;
        }
        let scale = lambda.max(0.0).sqrt();
        for i in 0..n {
            y[[i, k]] = scale * v[i];
        }
    }
    Some(y)
}

/// Select up to `max_landmarks` diverse archive points (farthest-point).
pub fn farthest_point_landmarks(xs: &[Array1<f64>], max_landmarks: usize) -> Vec<usize> {
    if xs.is_empty() || max_landmarks == 0 {
        return Vec::new();
    }
    let n = xs.len();
    let m = max_landmarks.min(n);
    let mut chosen = Vec::with_capacity(m);
    chosen.push(0);
    let mut min_dist = vec![f64::INFINITY; n];
    for _ in 1..m {
        let last = chosen[chosen.len() - 1];
        for i in 0..n {
            let d = row_l2(xs[i].view(), xs[last].view());
            if d < min_dist[i] {
                min_dist[i] = d;
            }
        }
        let mut best_i = 0;
        let mut best_d = -1.0;
        for i in 0..n {
            if chosen.contains(&i) {
                continue;
            }
            if min_dist[i] > best_d {
                best_d = min_dist[i];
                best_i = i;
            }
        }
        chosen.push(best_i);
    }
    chosen
}

#[cfg(test)]
mod tests {
    use super::*;
    use ndarray::array;

    #[test]
    fn sigmoid_is_zero_at_origin_and_increases() {
        let s0 = sigmoid_switch(0.0, 1.0, 4.0, 2.0);
        let s1 = sigmoid_switch(1.0, 1.0, 4.0, 2.0);
        let s2 = sigmoid_switch(2.0, 1.0, 4.0, 2.0);
        assert!((s0 - 0.0).abs() < 1e-15);
        assert!(s1 > s0);
        assert!(s2 > s1);
        assert!(s2 < 1.0);
    }

    #[test]
    fn sketchmap_separates_two_clusters_in_2d() {
        // Two blobs in 4-D that differ on first two coords only.
        let mut pts = Array2::<f64>::zeros((10, 4));
        for i in 0..5 {
            pts[[i, 0]] = -2.0 + 0.05 * i as f64;
            pts[[i, 1]] = -2.0;
            pts[[i + 5, 0]] = 2.0 + 0.05 * i as f64;
            pts[[i + 5, 1]] = 2.0;
        }
        let sm = SketchMap2d::fit(pts.view(), 40, 0.05).expect("fit");
        // Mean of first 5 vs last 5 embeddings should be far apart.
        let mut m0 = [0.0; 2];
        let mut m1 = [0.0; 2];
        for i in 0..5 {
            m0[0] += sm.embedded[[i, 0]];
            m0[1] += sm.embedded[[i, 1]];
            m1[0] += sm.embedded[[i + 5, 0]];
            m1[1] += sm.embedded[[i + 5, 1]];
        }
        m0[0] /= 5.0;
        m0[1] /= 5.0;
        m1[0] /= 5.0;
        m1[1] /= 5.0;
        let sep = ((m0[0] - m1[0]).powi(2) + (m0[1] - m1[1]).powi(2)).sqrt();
        assert!(
            sep > 0.5,
            "sketch-map should separate clusters, sep={sep}, emb={:?}",
            sm.embedded
        );
        // OOS projection of a point near cluster 0 lands nearer m0.
        let x = array![-2.0, -2.0, 0.0, 0.0];
        let y = sm.project(x.view(), 30);
        let d0 = ((y[0] - m0[0]).powi(2) + (y[1] - m0[1]).powi(2)).sqrt();
        let d1 = ((y[0] - m1[0]).powi(2) + (y[1] - m1[1]).powi(2)).sqrt();
        assert!(d0 < d1, "OOS should map near cluster 0: d0={d0} d1={d1}");
    }

    #[test]
    fn linearize_projector_has_correct_shape() {
        let pts = array![
            [0.0, 0.0, 0.0],
            [1.0, 0.0, 0.0],
            [0.0, 1.0, 0.0],
            [1.0, 1.0, 0.0],
            [0.5, 0.5, 0.1],
        ];
        let sm = SketchMap2d::fit(pts.view(), 20, 0.05).expect("fit");
        let (p, mu, lo, hi) = sm.linearize_projector(1e-3);
        assert_eq!(p.shape(), &[3, 2]);
        assert_eq!(mu.len(), 3);
        assert!(lo[0] < hi[0] && lo[1] < hi[1]);
    }
}
