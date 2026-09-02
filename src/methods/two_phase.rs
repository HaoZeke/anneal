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

use std::sync::{Arc, Mutex};

use ndarray::{Array1, ArrayView1};
use rand::SeedableRng;
use rand::rngs::StdRng;
use serde::Serialize;

use crate::allocate::DepthAllocator;

/// A surface allocator posterior several chains update together.
///
/// Pooled evidence is the cooperative channel that never touches a walk:
/// every chain draws its arm from the same Normal-Gamma posterior and
/// credits its block back to it, so an ensemble of `n` chains learns which
/// surface pays `n` times faster than one chain does, and no chain is
/// steered, relocated or interrupted to get that.
pub type SharedSurfaceAllocator = Arc<Mutex<DepthAllocator>>;

/// A fresh shared posterior over the plain surface plus `transforms`.
pub fn shared_surface_allocator(transforms: &[TwoPhase]) -> SharedSurfaceAllocator {
    let arms = 1 + transforms.iter().filter(|two| two.is_active()).count();
    Arc::new(Mutex::new(DepthAllocator::new(arms)))
}

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
/// An arm is held for a block of hops, not one: a walk on the compacted
/// surface and a walk on the plain one visit different minima, and
/// alternating them every hop is a walk on neither. A screening relaxation
/// opens a hop; every `block` hops the arm is redrawn, and the block's reward
/// is the energy its walk took off the run's best, zero when it took none.
#[derive(Debug, Clone)]
pub struct SurfacePortfolio {
    arms: Vec<Option<TwoPhase>>,
    allocator: DepthAllocator,
    /// When set, draws and credits go through this posterior instead of
    /// the private one, which then only mirrors this chain's own draws.
    shared: Option<SharedSurfaceAllocator>,
    held: Option<usize>,
    block: usize,
    hops_in_block: usize,
    block_start_best: f64,
    latest_best: f64,
    /// Sum of the depth each full relaxation of the block reached against
    /// the run's best at that moment, and how many contributed.
    block_depth: f64,
    block_relaxations: usize,
    rng: StdRng,
}

/// Hops an arm is held for before the allocator redraws.
pub const DEFAULT_SURFACE_BLOCK: usize = 100;

impl SurfacePortfolio {
    /// The plain surface plus every transform, uninformative until fed.
    pub fn new(transforms: &[TwoPhase], seed: u64) -> Self {
        Self::with_block(transforms, seed, DEFAULT_SURFACE_BLOCK)
    }

    /// As [`Self::new`], holding each drawn arm for `block` hops.
    pub fn with_block(transforms: &[TwoPhase], seed: u64, block: usize) -> Self {
        let mut arms = vec![None];
        arms.extend(transforms.iter().copied().filter(|two| two.is_active()).map(Some));
        Self {
            allocator: DepthAllocator::new(arms.len()),
            arms,
            shared: None,
            held: None,
            block: block.max(1),
            hops_in_block: 0,
            block_start_best: f64::INFINITY,
            latest_best: f64::INFINITY,
            block_depth: 0.0,
            block_relaxations: 0,
            rng: StdRng::seed_from_u64(seed ^ 0x5a2f_ace5),
        }
    }

    /// Draw from and credit a posterior shared with other chains.
    pub fn sharing(mut self, shared: SharedSurfaceAllocator) -> Self {
        let arms = shared.lock().expect("shared surface allocator").arms();
        assert_eq!(
            arms,
            self.arms.len(),
            "a shared surface allocator must cover the same arms"
        );
        self.shared = Some(shared);
        self
    }

    fn select_arm(&mut self) -> usize {
        match self.shared.as_ref() {
            Some(shared) => shared
                .lock()
                .expect("shared surface allocator")
                .select(&mut self.rng),
            None => self.allocator.select(&mut self.rng),
        }
    }

    fn credit_arm(&mut self, arm: usize, reward: f64) {
        if let Some(shared) = self.shared.as_ref() {
            shared
                .lock()
                .expect("shared surface allocator")
                .update(arm, reward);
        }
        self.allocator.update(arm, reward);
    }

    /// The surface for the relaxation about to start.
    ///
    /// A screening relaxation opens a hop; the block's arm is redrawn once
    /// the block is spent, with the finished block's improvement credited
    /// to the arm that walked it.
    pub fn begin(&mut self, screening: bool) -> Option<TwoPhase> {
        if screening {
            if self.hops_in_block >= self.block {
                self.settle_block();
            }
            self.hops_in_block += 1;
        }
        if self.held.is_none() {
            self.held = Some(self.select_arm());
            self.hops_in_block = self.hops_in_block.max(1);
            self.block_start_best = self.latest_best;
        }
        self.held.and_then(|arm| self.arms[arm])
    }

    /// Records what a full relaxation on the held arm reached against the
    /// run's best.
    pub fn observe(&mut self, screening: bool, reached: f64, best: f64) {
        if screening {
            return;
        }
        if reached.is_finite() && best.is_finite() {
            // The depth reward the move allocator uses, per relaxation: how
            // close this quench came to the best the run knows. Dense, so a
            // block on a size where nothing improves still says which
            // surface lands deeper.
            self.block_depth += -(reached - best.min(reached));
            self.block_relaxations += 1;
        }
        let best = best.min(reached);
        if best.is_finite() {
            self.latest_best = self.latest_best.min(best);
        }
    }

    fn settle_block(&mut self) {
        if let Some(arm) = self.held.take() {
            let improvement = if self.block_start_best.is_finite() && self.latest_best.is_finite()
            {
                (self.block_start_best - self.latest_best).max(0.0)
            } else {
                0.0
            };
            let mean_depth = if self.block_relaxations > 0 {
                self.block_depth / self.block_relaxations as f64
            } else {
                0.0
            };
            // Improvement is the event that matters and the mean depth is
            // the dense signal between events; both are energies of the
            // plain surface, so they add.
            self.credit_arm(arm, improvement + mean_depth);
        }
        self.block_depth = 0.0;
        self.block_relaxations = 0;
        self.hops_in_block = 0;
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
    fn the_portfolio_learns_the_arm_whose_blocks_improve_the_best() {
        let deep = TwoPhase::diameter(2.0, 1.0);
        let shallow = TwoPhase::relative(0.5, 1.0);
        let mut portfolio = SurfacePortfolio::with_block(&[deep, shallow], 7, 5);
        assert_eq!(portfolio.arms().len(), 3);
        let mut best = 0.0_f64;
        let mut held: Option<Option<TwoPhase>> = None;
        let mut switches = 0usize;
        for hop in 0..2000 {
            let arm = portfolio.begin(true);
            if held.is_some_and(|previous| previous != arm) {
                switches += 1;
            }
            held = Some(arm);
            // Only the deep arm ever lowers the best; the others walk in place.
            let reached = match arm {
                Some(two) if two == deep => best - 1.0,
                _ => best + 3.0,
            };
            assert_eq!(portfolio.begin(false), arm, "the full relaxation changed surface");
            best = best.min(reached);
            portfolio.observe(false, reached, best);
            if hop % 5 != 4 {
                assert_eq!(portfolio.begin(false), arm, "the arm changed inside a block");
            }
        }
        assert!(switches < 2000 / 5, "the arm is redrawn more often than once per block");
        let draws = portfolio.draws();
        assert!(
            draws[1] > draws[0] + draws[2],
            "the improving arm was not preferred: {draws:?}"
        );
        let means = portfolio.means();
        assert!(means[1] > means[0] && means[1] > means[2], "{means:?}");
    }

    #[test]
    fn chains_sharing_a_posterior_learn_from_each_other_s_blocks() {
        let deep = TwoPhase::diameter(2.0, 1.0);
        let shared = shared_surface_allocator(&[deep]);
        let mut teacher = SurfacePortfolio::with_block(&[deep], 1, 2).sharing(Arc::clone(&shared));
        let mut best = 0.0_f64;
        for _ in 0..200 {
            let arm = teacher.begin(true);
            let reached = if arm == Some(deep) { best - 1.0 } else { best + 1.0 };
            best = best.min(reached);
            teacher.observe(false, reached, best);
        }
        let mut student = SurfacePortfolio::with_block(&[deep], 2, 2).sharing(Arc::clone(&shared));
        let deep_draws = (0..40)
            .filter(|_| {
                let arm = student.begin(true);
                student.observe(false, 0.0, 0.0);
                arm == Some(deep)
            })
            .count();
        assert!(
            deep_draws >= 30,
            "a fresh chain on the shared posterior drew the learned arm {deep_draws} of 40 times"
        );
        assert!(student.draws().iter().sum::<usize>() > 0, "the private mirror records draws");
    }

    #[test]
    fn a_block_without_improvement_is_credited_by_its_depth() {
        let mut portfolio = SurfacePortfolio::with_block(&[TwoPhase::diameter(2.0, 1.0)], 3, 2);
        for _ in 0..6 {
            portfolio.begin(true);
            portfolio.begin(false);
            // Two units above the best every time: the block earns -2.
            portfolio.observe(false, -1.0, -3.0);
        }
        assert!(portfolio.draws().iter().sum::<usize>() >= 2);
        assert!(
            portfolio
                .means()
                .iter()
                .zip(portfolio.draws())
                .filter(|(_, d)| **d > 0)
                .all(|(m, _)| (m + 2.0).abs() < 1e-9),
            "{:?}",
            portfolio.means()
        );
    }
}
