//! Two-phase relaxation on a compacted surface.
//!
//! Locatelli and Schoen relax a perturbed cluster first on a modified pair
//! potential that penalizes pair distances beyond a cutoff, then relax that
//! minimum on the plain potential and judge the plain energy. Doye's
//! compression is the centroid form of the same idea. The transformed
//! surface reorders basin areas so that compact packings own more of the
//! quench catchment, which is what separates the decahedral and tetrahedral
//! global minima from the icosahedral floors at 75, 98 and 102 to 104 points
//! by two orders of magnitude in cost.
//!
//! Nothing here names a structure. The penalty reads pair distances of the
//! coordinates being relaxed and a cutoff that is either fixed or a fraction
//! of the largest pair distance of the structure entering the quench.
//!
//! Locatelli, M.; Schoen, F. *Comput. Optim. Appl.* **2002**, *21*, 55
//! <https://doi.org/10.1023/A:1013596313166>; Grosso, A.; Locatelli, M.;
//! Schoen, F. *Math. Program.* **2007**, *110*, 373
//! <https://doi.org/10.1007/s10107-006-0006-3>; Doye, J. P. K. *Phys. Rev.
//! E* **2000**, *62*, 8753 <https://doi.org/10.1103/PhysRevE.62.8753>.

use ndarray::{Array1, ArrayView1};
use rand::SeedableRng;
use rand::rngs::StdRng;
use serde::Serialize;

use crate::allocate::DepthAllocator;

/// How the diameter cutoff is chosen for one relaxation.
#[derive(Debug, Clone, Copy, PartialEq, Serialize)]
pub enum Cutoff {
    /// A fixed pair-distance cutoff in the objective's length units.
    Fixed(f64),
    /// A fraction of the largest pair distance of the structure being relaxed.
    Relative(f64),
}

/// First-phase transform of the relaxation surface.
#[derive(Debug, Clone, Copy, PartialEq, Serialize)]
pub struct TwoPhase {
    /// Diameter cutoff; pairs further apart than this are penalized.
    pub cutoff: Cutoff,
    /// Strength of the quartic diameter penalty.
    pub beta: f64,
    /// Strength of the centroid compression; zero leaves it off.
    pub mu: f64,
}

impl TwoPhase {
    /// A diameter penalty at a fixed cutoff, no centroid compression.
    pub fn diameter(cutoff: f64, beta: f64) -> Self {
        Self {
            cutoff: Cutoff::Fixed(cutoff),
            beta,
            mu: 0.0,
        }
    }

    /// A diameter penalty at a fraction of the entering structure's diameter.
    pub fn relative(kappa: f64, beta: f64) -> Self {
        Self {
            cutoff: Cutoff::Relative(kappa),
            beta,
            mu: 0.0,
        }
    }

    /// The cutoff that applies to a relaxation starting from `x`.
    pub fn cutoff_for(&self, x: ArrayView1<f64>) -> f64 {
        match self.cutoff {
            Cutoff::Fixed(d) => d,
            Cutoff::Relative(kappa) => kappa * largest_pair_distance(x),
        }
    }

    /// Whether the first phase changes anything at all.
    pub fn is_active(&self) -> bool {
        let diameter_on = self.beta > 0.0
            && match self.cutoff {
                Cutoff::Fixed(d) => d > 0.0,
                Cutoff::Relative(kappa) => kappa > 0.0,
            };
        diameter_on || self.mu > 0.0
    }
}

/// Largest pair distance in a 3N coordinate vector; zero below two points.
pub fn largest_pair_distance(x: ArrayView1<f64>) -> f64 {
    let n = x.len() / 3;
    let mut best = 0.0_f64;
    for i in 0..n {
        for j in (i + 1)..n {
            let mut r2 = 0.0;
            for k in 0..3 {
                let d = x[3 * i + k] - x[3 * j + k];
                r2 += d * d;
            }
            best = best.max(r2);
        }
    }
    best.sqrt()
}

/// Penalty energy and gradient added to the plain surface in phase one.
///
/// `beta * sum_{i<j} max(0, r_ij^2 - cutoff^2)^2 + mu * sum_i |r_i - r_cm|^2`.
/// The centroid contributes no gradient of its own because displacements
/// from it sum to zero.
pub fn penalty(x: ArrayView1<f64>, cutoff: f64, beta: f64, mu: f64) -> (f64, Array1<f64>) {
    let n = x.len() / 3;
    let mut e = 0.0;
    let mut g = Array1::zeros(x.len());
    if beta > 0.0 && cutoff > 0.0 {
        let d2 = cutoff * cutoff;
        for i in 0..n {
            for j in (i + 1)..n {
                let d = [
                    x[3 * i] - x[3 * j],
                    x[3 * i + 1] - x[3 * j + 1],
                    x[3 * i + 2] - x[3 * j + 2],
                ];
                let excess = d[0] * d[0] + d[1] * d[1] + d[2] * d[2] - d2;
                if excess > 0.0 {
                    e += beta * excess * excess;
                    let coef = 4.0 * beta * excess;
                    for k in 0..3 {
                        g[3 * i + k] += coef * d[k];
                        g[3 * j + k] -= coef * d[k];
                    }
                }
            }
        }
    }
    if mu > 0.0 && n > 0 {
        let mut cm = [0.0_f64; 3];
        for i in 0..n {
            for k in 0..3 {
                cm[k] += x[3 * i + k];
            }
        }
        for value in cm.iter_mut() {
            *value /= n as f64;
        }
        for i in 0..n {
            for k in 0..3 {
                let d = x[3 * i + k] - cm[k];
                e += mu * d * d;
                g[3 * i + k] += 2.0 * mu * d;
            }
        }
    }
    (e, g)
}

/// A learned choice of relaxation surface per hop.
///
/// Which transform helps is a property of the landscape: centroid compression
/// separates the octahedral and tetrahedral minima at 38 and 98 points, the
/// diameter penalty separates the Marks decahedron at 75, and neither is
/// worth its second relaxation on a single-funnel size. Rather than name the
/// answer per size, the arms are the plain surface and every configured
/// transform, and a Normal-Gamma Thompson allocator rewarded by the depth the
/// quench reached picks one per hop, the same reward that allocates move
/// kernels in [`crate::methods::cluster_hopping`].
///
/// A hop is a screening relaxation followed by a full one on the same trial;
/// the arm is drawn at the screen and held for the full relaxation, whose
/// reached energy against the run's best is the reward.
#[derive(Debug, Clone)]
pub struct SurfacePortfolio {
    arms: Vec<Option<TwoPhase>>,
    allocator: DepthAllocator,
    held: Option<usize>,
    rng: StdRng,
}

impl SurfacePortfolio {
    /// The plain surface plus every transform, uninformative until fed.
    pub fn new(transforms: &[TwoPhase], seed: u64) -> Self {
        let mut arms = vec![None];
        arms.extend(
            transforms
                .iter()
                .copied()
                .filter(|two| two.is_active())
                .map(Some),
        );
        Self {
            allocator: DepthAllocator::new(arms.len()),
            arms,
            held: None,
            rng: StdRng::seed_from_u64(seed ^ 0x5a2f_ace5),
        }
    }

    /// The surface for the relaxation about to start.
    ///
    /// A screening relaxation opens a hop and draws a fresh arm; a full
    /// relaxation keeps the arm its screen drew, or draws one if it stands
    /// alone.
    pub fn begin(&mut self, screening: bool) -> Option<TwoPhase> {
        if screening || self.held.is_none() {
            self.held = Some(self.allocator.select(&mut self.rng));
        }
        self.held.and_then(|arm| self.arms[arm])
    }

    /// Credits the held arm with the depth a full relaxation reached.
    pub fn observe(&mut self, screening: bool, reached: f64, best: f64) {
        if screening {
            return;
        }
        if let Some(arm) = self.held.take()
            && reached.is_finite()
            && best.is_finite()
        {
            self.allocator.update(arm, -(reached - best));
        }
    }

    /// Draws taken per arm, the plain surface first.
    pub fn draws(&self) -> &[usize] {
        &self.allocator.draws
    }

    /// Posterior mean depth reward per arm, the plain surface first.
    pub fn means(&self) -> Vec<f64> {
        self.allocator.means()
    }

    /// The arms, the plain surface first.
    pub fn arms(&self) -> &[Option<TwoPhase>] {
        &self.arms
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn cluster() -> Array1<f64> {
        Array1::from(vec![
            0.0, 0.0, 0.0, 1.1, 0.0, 0.0, 0.0, 1.2, 0.0, 0.0, 0.0, 2.9, 2.5, 2.4, 0.1,
        ])
    }

    #[test]
    fn nothing_is_penalized_inside_the_cutoff() {
        let x = cluster();
        let d = largest_pair_distance(x.view());
        let (e, g) = penalty(x.view(), d + 1e-9, 1.0, 0.0);
        assert_eq!(e, 0.0);
        assert!(g.iter().all(|v| *v == 0.0));
    }

    #[test]
    fn the_penalty_gradient_matches_finite_differences() {
        let x = cluster();
        let (cutoff, beta, mu) = (2.0, 0.7, 0.3);
        let (_, g) = penalty(x.view(), cutoff, beta, mu);
        let h = 1e-6;
        for i in 0..x.len() {
            let mut plus = x.clone();
            let mut minus = x.clone();
            plus[i] += h;
            minus[i] -= h;
            let fd = (penalty(plus.view(), cutoff, beta, mu).0
                - penalty(minus.view(), cutoff, beta, mu).0)
                / (2.0 * h);
            assert!(
                (fd - g[i]).abs() < 1e-6,
                "component {i}: finite difference {fd} against analytic {}",
                g[i]
            );
        }
    }

    #[test]
    fn the_relative_cutoff_follows_the_entering_structure() {
        let x = cluster();
        let two = TwoPhase::relative(0.8, 1.0);
        let d = largest_pair_distance(x.view());
        assert!((two.cutoff_for(x.view()) - 0.8 * d).abs() < 1e-12);
        let scaled = x.mapv(|v| 2.0 * v);
        assert!((two.cutoff_for(scaled.view()) - 1.6 * d).abs() < 1e-12);
        assert!(two.is_active());
        assert!(!TwoPhase::diameter(0.0, 1.0).is_active());
        assert!(!TwoPhase::relative(0.8, 0.0).is_active());
    }

    #[test]
    fn the_portfolio_learns_the_arm_that_reaches_deeper() {
        let deep = TwoPhase::diameter(2.0, 1.0);
        let shallow = TwoPhase::relative(0.5, 1.0);
        let mut portfolio = SurfacePortfolio::new(&[deep, shallow], 7);
        assert_eq!(portfolio.arms().len(), 3);
        for _ in 0..400 {
            let arm = portfolio.begin(true);
            let reached = match arm {
                Some(two) if two == deep => -10.0,
                Some(_) => -2.0,
                None => -5.0,
            };
            assert_eq!(
                portfolio.begin(false),
                arm,
                "the full relaxation changed surface"
            );
            portfolio.observe(false, reached, -10.0);
        }
        let draws = portfolio.draws();
        assert!(
            draws[1] > draws[0] + draws[2],
            "the deeper arm was not preferred: {draws:?}"
        );
        let means = portfolio.means();
        assert!(means[1] > means[0] && means[0] > means[2], "{means:?}");
    }

    #[test]
    fn a_screened_out_hop_leaves_no_reward() {
        let mut portfolio = SurfacePortfolio::new(&[TwoPhase::diameter(2.0, 1.0)], 3);
        portfolio.begin(true);
        portfolio.observe(true, -1.0, -3.0);
        assert!(portfolio.draws().iter().all(|d| *d == 0));
        portfolio.begin(false);
        portfolio.observe(false, -1.0, -3.0);
        assert_eq!(portfolio.draws().iter().sum::<usize>(), 1);
    }
}
