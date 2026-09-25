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

use std::collections::BTreeMap;
use std::sync::{Arc, Mutex};

use ndarray::{Array1, ArrayView1};
use rand::rngs::StdRng;
use rand::SeedableRng;
use serde::Serialize;

use crate::allocate::{DepthAllocator, RewardMoments};
use crate::surface_evidence::{
    SourceTransferKey, SurfaceEvidenceBook, SurfaceEvidenceMessage, MIN_TRANSFER_OBSERVATIONS,
};

/// A surface allocator posterior several chains update together.
///
/// Evidence is keyed by the occupied validated source. Chains that share a
/// key draw from that key's posterior. Importing a peer reply does not
/// replace a walk's coordinates, held arm, local rewards, or random stream.
pub type SharedSurfaceAllocator = Arc<Mutex<SurfaceEvidenceBook>>;

/// A fresh shared posterior over the plain surface plus `transforms`.
pub fn shared_surface_allocator(transforms: &[TwoPhase]) -> SharedSurfaceAllocator {
    let arms = 1 + transforms.iter().filter(|two| two.is_active()).count();
    Arc::new(Mutex::new(SurfaceEvidenceBook::new(arms)))
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

fn group_centroid(x: ArrayView1<f64>, atoms: &[usize]) -> [f64; 3] {
    let mut c = [0.0_f64; 3];
    if atoms.is_empty() {
        return c;
    }
    let n = x.len() / 3;
    let mut count = 0.0;
    for &i in atoms {
        if i >= n {
            continue;
        }
        for k in 0..3 {
            c[k] += x[3 * i + k];
        }
        count += 1.0;
    }
    if count > 0.0 {
        for value in c.iter_mut() {
            *value /= count;
        }
    }
    c
}

fn add_centroid_gradient(g: &mut Array1<f64>, atoms: &[usize], force: [f64; 3]) {
    let n = g.len() / 3;
    let members = atoms.iter().filter(|&&i| i < n).count();
    if members == 0 {
        return;
    }
    let inv = 1.0 / members as f64;
    for &i in atoms {
        if i >= n {
            continue;
        }
        for k in 0..3 {
            g[3 * i + k] += force[k] * inv;
        }
    }
}

/// Penalty energy and gradient on rigid-group centroids.
///
/// Same diameter and compression terms as [`penalty`], evaluated between
/// and on the group centroids. Each centroid's gradient is spread equally
/// over the group's atoms, so intramolecular bonds feel no relative force.
pub fn penalty_groups(
    x: ArrayView1<f64>,
    groups: &[Vec<usize>],
    cutoff: f64,
    beta: f64,
    mu: f64,
) -> (f64, Array1<f64>) {
    let centroids: Vec<[f64; 3]> = groups
        .iter()
        .map(|atoms| group_centroid(x, atoms))
        .collect();
    let n_groups = centroids.len();
    let mut e = 0.0;
    let mut g = Array1::zeros(x.len());
    if beta > 0.0 && cutoff > 0.0 {
        let d2 = cutoff * cutoff;
        for i in 0..n_groups {
            for j in (i + 1)..n_groups {
                let d = [
                    centroids[i][0] - centroids[j][0],
                    centroids[i][1] - centroids[j][1],
                    centroids[i][2] - centroids[j][2],
                ];
                let excess = d[0] * d[0] + d[1] * d[1] + d[2] * d[2] - d2;
                if excess > 0.0 {
                    e += beta * excess * excess;
                    let coef = 4.0 * beta * excess;
                    add_centroid_gradient(
                        &mut g,
                        &groups[i],
                        [coef * d[0], coef * d[1], coef * d[2]],
                    );
                    add_centroid_gradient(
                        &mut g,
                        &groups[j],
                        [-coef * d[0], -coef * d[1], -coef * d[2]],
                    );
                }
            }
        }
    }
    if mu > 0.0 && n_groups > 0 {
        let mut cm = [0.0_f64; 3];
        for c in &centroids {
            for k in 0..3 {
                cm[k] += c[k];
            }
        }
        for value in cm.iter_mut() {
            *value /= n_groups as f64;
        }
        for (atoms, c) in groups.iter().zip(centroids.iter()) {
            let d = [c[0] - cm[0], c[1] - cm[1], c[2] - cm[2]];
            e += mu * (d[0] * d[0] + d[1] * d[1] + d[2] * d[2]);
            add_centroid_gradient(
                &mut g,
                atoms,
                [2.0 * mu * d[0], 2.0 * mu * d[1], 2.0 * mu * d[2]],
            );
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
    /// When set, draws and credits for the occupied source go through this book.
    shared: Option<SharedSurfaceAllocator>,
    /// Validated source occupied when the current block opened.
    occupied: Option<SourceTransferKey>,
    /// Source fixed at block open. Later occupancy does not rewrite it.
    block_source: Option<SourceTransferKey>,
    /// Rewards this chain credited, by source. Imports do not write here.
    own_by_source: BTreeMap<SourceTransferKey, Vec<RewardMoments>>,
    /// Peer replies, still keyed by the producer's source.
    peer_by_source: BTreeMap<SourceTransferKey, Vec<RewardMoments>>,
    held: Option<usize>,
    block: usize,
    hops_in_block: usize,
    block_start_best: f64,
    latest_best: f64,
    /// Lowest plain energy any full relaxation of the block reached.
    block_lowest: f64,
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
            shared: None,
            occupied: None,
            block_source: None,
            own_by_source: BTreeMap::new(),
            peer_by_source: BTreeMap::new(),
            held: None,
            block: block.max(1),
            hops_in_block: 0,
            block_start_best: f64::INFINITY,
            latest_best: f64::INFINITY,
            block_lowest: f64::INFINITY,
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

    /// Record the occupied validated source. A mismatched block interval is refused.
    pub fn set_occupied_source(&mut self, source: SourceTransferKey) -> Result<(), &'static str> {
        if source.block != self.block
            || source.descriptor_schema.is_empty()
            || source.descriptor_version == 0
            || source.proposal.is_empty()
            || source.quench_schema.is_empty()
        {
            return Err("occupied source does not match the declared block");
        }
        self.occupied = Some(source);
        Ok(())
    }

    /// Adopt a checkpoint only when its interval is this portfolio's block.
    pub fn adopt_checkpoint(
        &mut self,
        interval: usize,
        source: SourceTransferKey,
    ) -> Result<(), &'static str> {
        if interval != self.block {
            return Err("checkpoint interval is not the occupied source");
        }
        self.set_occupied_source(source)
    }

    /// The relaxation input is not a source and does not replace occupancy.
    pub fn note_perturbed_input(&mut self, _perturbed: &SourceTransferKey) {}

    fn select_arm(&mut self) -> usize {
        if let Some(key) = self.block_source.clone() {
            if let Some(shared) = self.shared.as_ref() {
                let book = shared.lock().expect("shared surface allocator");
                if let Some(allocator) = book.decision_allocator(&key) {
                    return allocator.select(&mut self.rng);
                }
            } else if let Some(allocator) = self.local_decision(&key) {
                return allocator.select(&mut self.rng);
            }
            return DepthAllocator::new(self.arms.len()).select(&mut self.rng);
        }
        self.allocator.select(&mut self.rng)
    }

    fn local_decision(&self, key: &SourceTransferKey) -> Option<DepthAllocator> {
        let mut moments = self
            .own_by_source
            .get(key)
            .cloned()
            .unwrap_or_else(|| vec![RewardMoments::default(); self.arms.len()]);
        if let Some(peers) = self.peer_by_source.get(key) {
            for (slot, peer) in moments.iter_mut().zip(peers) {
                *slot = slot.merge(*peer).ok()?;
            }
        }
        if moments.iter().map(|arm| arm.count).sum::<u64>() < MIN_TRANSFER_OBSERVATIONS {
            return None;
        }
        DepthAllocator::from_moments(&moments).ok()
    }

    fn credit_arm(&mut self, arm: usize, reward: f64) {
        self.allocator.update(arm, reward);
    }

    fn credit_source(&mut self, key: SourceTransferKey, arm: usize, reward: f64) {
        if !reward.is_finite() {
            return;
        }
        let moments = self
            .own_by_source
            .entry(key.clone())
            .or_insert_with(|| vec![RewardMoments::default(); self.arms.len()]);
        if moments[arm].observe(reward).is_err() {
            return;
        }
        let charged_work = u64::try_from(self.block).unwrap_or(u64::MAX).max(1);
        if let Some(shared) = self.shared.as_ref() {
            let _ = shared.lock().expect("shared surface allocator").observe(
                0,
                &key,
                arm,
                reward,
                reward,
                charged_work,
            );
        }
        self.allocator.draws[arm] += 1;
    }

    /// Store a peer reply under its original key.
    ///
    /// Local rewards, the held arm, and the random stream stay as they are.
    /// Coordinates are not portfolio state and are not an argument.
    pub fn import_evidence(&mut self, message: SurfaceEvidenceMessage) -> Result<(), &'static str> {
        if self.shared.is_some() || message.key.block != self.block {
            return Err("incompatible surface evidence");
        }
        message.validate(self.arms.len())?;
        self.peer_by_source.insert(message.key, message.arms);
        Ok(())
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
            self.block_source = self.occupied.clone();
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
        if reached.is_finite() {
            self.block_lowest = self.block_lowest.min(reached);
        }
        let best = best.min(reached);
        if best.is_finite() {
            self.latest_best = self.latest_best.min(best);
        }
    }

    fn settle_block(&mut self) {
        if let Some(arm) = self.held.take() {
            // The signed gap between the run's best when the block opened
            // and the lowest structure the block relaxed to: positive by the
            // improvement when the block beat it, negative by the shortfall
            // when it did not. Dense, because every block relaxes to
            // something, and it favours the surface whose blocks reach the
            // deepest structures rather than the one whose typical hop lands
            // nearest the incumbent. The first block has no incumbent to
            // measure against and is neutral.
            let reward = if self.block_start_best.is_finite() && self.block_lowest.is_finite() {
                self.block_start_best - self.block_lowest
            } else {
                0.0
            };
            if let Some(key) = self.block_source.take() {
                self.credit_source(key, arm, reward);
            } else {
                self.credit_arm(arm, reward);
            }
        }
        self.block_lowest = f64::INFINITY;
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
    use rand::Rng;

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
            assert_eq!(
                portfolio.begin(false),
                arm,
                "the full relaxation changed surface"
            );
            best = best.min(reached);
            portfolio.observe(false, reached, best);
            if hop % 5 != 4 {
                assert_eq!(
                    portfolio.begin(false),
                    arm,
                    "the arm changed inside a block"
                );
            }
        }
        assert!(
            switches < 2000 / 5,
            "the arm is redrawn more often than once per block"
        );
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
        let source = crate::surface_evidence::SourceTransferKey {
            descriptor_schema: "lj".into(),
            descriptor_version: 1,
            region: 1,
            proposal: "hop".into(),
            quench_schema: "lbfgs".into(),
            block: 2,
        };
        let mut teacher = SurfacePortfolio::with_block(&[deep], 1, 2).sharing(Arc::clone(&shared));
        teacher.set_occupied_source(source.clone()).unwrap();
        let mut best = 0.0_f64;
        for _ in 0..200 {
            let arm = teacher.begin(true);
            let reached = if arm == Some(deep) {
                best - 1.0
            } else {
                best + 1.0
            };
            best = best.min(reached);
            teacher.observe(false, reached, best);
        }
        let mut student = SurfacePortfolio::with_block(&[deep], 2, 2).sharing(Arc::clone(&shared));
        student.set_occupied_source(source).unwrap();
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
        assert!(
            student.draws().iter().sum::<usize>() > 0,
            "the private mirror records draws"
        );
    }

    #[test]
    fn imported_surface_evidence_keeps_the_held_arm_local_rewards_and_rng() {
        let deep = TwoPhase::diameter(2.0, 1.0);
        let mut portfolio = SurfacePortfolio::with_block(&[deep], 5, 4);
        let source = crate::surface_evidence::SourceTransferKey {
            descriptor_schema: "universal".into(),
            descriptor_version: 1,
            region: 2,
            proposal: "hop".into(),
            quench_schema: "lbfgs".into(),
            block: 4,
        };
        portfolio.set_occupied_source(source.clone()).unwrap();
        assert!(portfolio.adopt_checkpoint(9, source.clone()).is_err());
        let held = portfolio.begin(true);
        let perturbed = crate::surface_evidence::SourceTransferKey {
            region: 9,
            ..source.clone()
        };
        portfolio.note_perturbed_input(&perturbed);
        assert_eq!(portfolio.occupied.as_ref(), Some(&source));
        let coordinates = [0.0_f64, 1.0, 2.0];
        let before_rng = portfolio.rng.clone();
        let before_held = portfolio.held;
        let before_local = portfolio.own_by_source.clone();
        let message = crate::surface_evidence::SurfaceEvidenceMessage {
            producer: 7,
            key: source,
            arms: vec![
                crate::allocate::RewardMoments {
                    count: 20,
                    mean: 1.0,
                    m2: 0.0,
                },
                crate::allocate::RewardMoments {
                    count: 20,
                    mean: -1.0,
                    m2: 0.0,
                },
            ],
            incumbent_gap: -1.0,
            charged_work: 20,
        };
        portfolio.import_evidence(message).unwrap();
        assert_eq!(portfolio.held, before_held);
        assert_eq!(portfolio.own_by_source, before_local);
        assert_eq!(portfolio.begin(false), held);
        assert_eq!(coordinates, [0.0, 1.0, 2.0]);
        let mut expected = before_rng;
        let mut actual = portfolio.rng.clone();
        assert_eq!(
            expected.random::<u64>(),
            actual.random::<u64>(),
            "import replaced the random stream"
        );
    }

    #[test]
    fn a_block_short_of_the_best_is_credited_by_its_shortfall() {
        let mut portfolio = SurfacePortfolio::with_block(&[TwoPhase::diameter(2.0, 1.0)], 3, 2);
        for _ in 0..12 {
            portfolio.begin(true);
            portfolio.begin(false);
            // Two units above the best every time: a settled block earns -2,
            // and the opening block, with no incumbent yet, earns nothing.
            portfolio.observe(false, -1.0, -3.0);
        }
        assert!(portfolio.draws().iter().sum::<usize>() >= 4);
        for (mean, draws) in portfolio.means().iter().zip(portfolio.draws()) {
            if *draws > 1 {
                assert!(
                    (-2.0..=0.0).contains(mean) && *mean < -1.0,
                    "mean {mean} over {draws} draws"
                );
            }
        }
    }

    fn rigid_water() -> Array1<f64> {
        Array1::from(vec![
            0.0, 0.0, 0.0, 0.7572, 0.5865, 0.0, -0.7572, 0.5865, 0.0,
        ])
    }

    fn two_rigid_waters() -> (Array1<f64>, Vec<Vec<usize>>) {
        let mut x = rigid_water().to_vec();
        x.extend_from_slice(&[3.0, 0.0, 0.0, 3.7572, 0.5865, 0.0, 2.2428, 0.5865, 0.0]);
        (Array1::from(x), vec![vec![0, 1, 2], vec![3, 4, 5]])
    }

    #[test]
    fn a_single_rigid_water_receives_no_penalty_force() {
        let x = rigid_water();
        let groups = [vec![0, 1, 2]];
        let (e, g) = penalty_groups(x.view(), &groups, 1.0, 1.0, 2.5);
        assert_eq!(e, 0.0);
        assert!(
            g.iter().all(|v| *v == 0.0),
            "internal water geometry felt a penalty force: {g}"
        );
    }

    #[test]
    fn the_group_penalty_gradient_matches_finite_differences() {
        let (x, groups) = two_rigid_waters();
        let (cutoff, beta, mu) = (2.0, 0.7, 0.3);
        let (_, g) = penalty_groups(x.view(), &groups, cutoff, beta, mu);
        let h = 1e-6;
        for i in 0..x.len() {
            let mut plus = x.clone();
            let mut minus = x.clone();
            plus[i] += h;
            minus[i] -= h;
            let fd = (penalty_groups(plus.view(), &groups, cutoff, beta, mu).0
                - penalty_groups(minus.view(), &groups, cutoff, beta, mu).0)
                / (2.0 * h);
            assert!(
                (fd - g[i]).abs() < 1e-6,
                "component {i}: finite difference {fd} against analytic {}",
                g[i]
            );
        }
    }

    #[test]
    fn singleton_groups_match_the_atomic_penalty() {
        let x = cluster();
        let groups: Vec<Vec<usize>> = (0..x.len() / 3).map(|i| vec![i]).collect();
        let (cutoff, beta, mu) = (2.0, 0.7, 0.3);
        let (e_atoms, g_atoms) = penalty(x.view(), cutoff, beta, mu);
        let (e_groups, g_groups) = penalty_groups(x.view(), &groups, cutoff, beta, mu);
        assert!(
            (e_atoms - e_groups).abs() < 1e-12,
            "{e_atoms} vs {e_groups}"
        );
        for i in 0..x.len() {
            assert!(
                (g_atoms[i] - g_groups[i]).abs() < 1e-12,
                "component {i}: atomic {} against group {}",
                g_atoms[i],
                g_groups[i]
            );
        }
    }

    #[test]
    fn atoms_in_a_rigid_group_share_one_penalty_force() {
        let (x, groups) = two_rigid_waters();
        let (_, g) = penalty_groups(x.view(), &groups, 2.0, 0.7, 0.3);
        for atoms in &groups {
            let shared = [g[3 * atoms[0]], g[3 * atoms[0] + 1], g[3 * atoms[0] + 2]];
            for &i in atoms {
                for k in 0..3 {
                    assert!(
                        (g[3 * i + k] - shared[k]).abs() < 1e-12,
                        "atom {i} axis {k}: {} against group force {}",
                        g[3 * i + k],
                        shared[k]
                    );
                }
            }
        }
    }
}
