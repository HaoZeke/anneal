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

use crate::allocate::{DepthAllocator, RewardMoments};
use crate::surface_evidence::SurfaceReport;

/// A surface allocator posterior several chains update together.
///
/// Pooled evidence is the cooperative channel that never touches a walk:
/// every chain draws its arm from the same Normal-Gamma posterior and
/// credits its block back to it. Independent draws preserve distinct walks;
/// sharing evidence does not relocate a chain or guarantee a discovery gain.
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
    /// Measure pair distances in the ellipsoidal metric of the entering
    /// structure's own gyration tensor, so a prolate walker is compacted
    /// prolately; the aspect comes from the structure, never from a target.
    #[serde(default)]
    pub anisotropic: bool,
}

/// Principal axes and relative gyration radii of a structure.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct Shape {
    /// Orthonormal principal axes, rows.
    pub axes: [[f64; 3]; 3],
    /// Gyration radius along each axis divided by the geometric mean of the
    /// three, so the metric preserves volume.
    pub scale: [f64; 3],
}

/// The shape of `x` from its gyration tensor about the centroid, or none when
/// the structure is degenerate.
pub fn inertia_shape(x: ArrayView1<f64>) -> Option<Shape> {
    let n = x.len() / 3;
    if n < 4 {
        return None;
    }
    let mut c = [0.0_f64; 3];
    for i in 0..n {
        for k in 0..3 {
            c[k] += x[3 * i + k];
        }
    }
    for value in c.iter_mut() {
        *value /= n as f64;
    }
    let mut t = [[0.0_f64; 3]; 3];
    for i in 0..n {
        let d = [x[3 * i] - c[0], x[3 * i + 1] - c[1], x[3 * i + 2] - c[2]];
        for a in 0..3 {
            for b in 0..3 {
                t[a][b] += d[a] * d[b] / n as f64;
            }
        }
    }
    let (values, vectors) = symmetric_eigen(t);
    if values.iter().any(|v| !(v.is_finite() && *v > 1e-12)) {
        return None;
    }
    let radii = values.map(f64::sqrt);
    let mean = (radii[0] * radii[1] * radii[2]).cbrt();
    Some(Shape {
        axes: vectors,
        scale: radii.map(|r| r / mean),
    })
}

/// Jacobi eigen decomposition of a symmetric 3x3 matrix: eigenvalues and
/// the matching unit eigenvectors as rows.
fn symmetric_eigen(mut a: [[f64; 3]; 3]) -> ([f64; 3], [[f64; 3]; 3]) {
    let mut v = [[1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]];
    for _ in 0..50 {
        let (mut p, mut q, mut largest) = (0, 1, 0.0_f64);
        for i in 0..3 {
            for j in (i + 1)..3 {
                if a[i][j].abs() > largest {
                    largest = a[i][j].abs();
                    p = i;
                    q = j;
                }
            }
        }
        if largest < 1e-14 {
            break;
        }
        let theta = 0.5 * (a[q][q] - a[p][p]) / a[p][q];
        let t = theta.signum() / (theta.abs() + (theta * theta + 1.0).sqrt());
        let cs = 1.0 / (t * t + 1.0).sqrt();
        let sn = t * cs;
        for k in 0..3 {
            let (akp, akq) = (a[k][p], a[k][q]);
            a[k][p] = cs * akp - sn * akq;
            a[k][q] = sn * akp + cs * akq;
        }
        for k in 0..3 {
            let (apk, aqk) = (a[p][k], a[q][k]);
            a[p][k] = cs * apk - sn * aqk;
            a[q][k] = sn * apk + cs * aqk;
        }
        for k in 0..3 {
            let (vkp, vkq) = (v[k][p], v[k][q]);
            v[k][p] = cs * vkp - sn * vkq;
            v[k][q] = sn * vkp + cs * vkq;
        }
    }
    let values = [a[0][0], a[1][1], a[2][2]];
    // Columns of v are the eigenvectors; return them as rows.
    let vectors = [
        [v[0][0], v[1][0], v[2][0]],
        [v[0][1], v[1][1], v[2][1]],
        [v[0][2], v[1][2], v[2][2]],
    ];
    (values, vectors)
}

impl TwoPhase {
    /// The shape the penalty measures distances in for a relaxation entering
    /// at `x`, when the anisotropic form is on.
    pub fn shape_for(&self, x: ArrayView1<f64>) -> Option<Shape> {
        if self.anisotropic {
            inertia_shape(x)
        } else {
            None
        }
    }

    /// A diameter penalty at a fixed cutoff, no centroid compression.
    pub fn diameter(cutoff: f64, beta: f64) -> Self {
        Self {
            cutoff: Cutoff::Fixed(cutoff),
            beta,
            mu: 0.0,
            anisotropic: false,
        }
    }

    /// A diameter penalty at a fraction of the entering structure's diameter.
    pub fn relative(kappa: f64, beta: f64) -> Self {
        Self {
            cutoff: Cutoff::Relative(kappa),
            beta,
            mu: 0.0,
            anisotropic: false,
        }
    }

    /// The relative penalty measured in the entering structure's own
    /// ellipsoidal metric.
    pub fn relative_anisotropic(kappa: f64, beta: f64) -> Self {
        Self {
            cutoff: Cutoff::Relative(kappa),
            beta,
            mu: 0.0,
            anisotropic: true,
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
    penalty_shaped(x, cutoff, beta, mu, None)
}

/// [`penalty`] with the diameter term measured in an ellipsoidal metric:
/// a pair displacement `d` counts as `sum_k (d . e_k / s_k)^2`, so pairs
/// along a long axis are penalized later than pairs along a short one and
/// the structure keeps its own aspect while it compacts.
pub fn penalty_shaped(
    x: ArrayView1<f64>,
    cutoff: f64,
    beta: f64,
    mu: f64,
    shape: Option<&Shape>,
) -> (f64, Array1<f64>) {
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
                // Metric distance and its gradient with respect to d.
                let (r2, grad) = match shape {
                    None => (
                        d[0] * d[0] + d[1] * d[1] + d[2] * d[2],
                        [2.0 * d[0], 2.0 * d[1], 2.0 * d[2]],
                    ),
                    Some(shape) => {
                        let mut r2 = 0.0;
                        let mut grad = [0.0_f64; 3];
                        for k in 0..3 {
                            let axis = shape.axes[k];
                            let proj =
                                (d[0] * axis[0] + d[1] * axis[1] + d[2] * axis[2]) / shape.scale[k];
                            r2 += proj * proj;
                            for m in 0..3 {
                                grad[m] += 2.0 * proj * axis[m] / shape.scale[k];
                            }
                        }
                        (r2, grad)
                    }
                };
                let excess = r2 - d2;
                if excess > 0.0 {
                    e += beta * excess * excess;
                    let coef = 2.0 * beta * excess;
                    for k in 0..3 {
                        g[3 * i + k] += coef * grad[k];
                        g[3 * j + k] -= coef * grad[k];
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
/// is the signed gap between the entering incumbent and the block's lowest
/// energy. A block that cannot reach the incumbent receives negative credit.
#[derive(Debug, Clone)]
pub struct SurfacePortfolio {
    arms: Vec<Option<TwoPhase>>,
    allocator: DepthAllocator,
    own_moments: Vec<RewardMoments>,
    peer_moments: Option<Vec<RewardMoments>>,
    evidence_schema: String,
    /// When set, draws and credits go through this posterior instead of
    /// the private one, which then only mirrors this chain's own draws.
    shared: Option<SharedSurfaceAllocator>,
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
        let block = block.max(1);
        // Bit patterns keep every transform parameter distinct, including
        // invalid floating-point inputs that a text serializer maps to null.
        let parameters = arms.iter().map(|arm| arm.map(|two| {
            let cutoff = match two.cutoff {
                Cutoff::Fixed(value) => (0, value.to_bits()),
                Cutoff::Relative(value) => (1, value.to_bits()),
            };
            (cutoff, two.beta.to_bits(), two.mu.to_bits(), two.anisotropic)
        })).collect::<Vec<_>>();
        let evidence_schema = format!("surface-depth-v1/{block}/{parameters:?}");
        Self {
            allocator: DepthAllocator::new(arms.len()),
            own_moments: vec![RewardMoments::default(); arms.len()],
            peer_moments: None,
            evidence_schema,
            arms,
            shared: None,
            held: None,
            block,
            hops_in_block: 0,
            block_start_best: f64::INFINITY,
            latest_best: f64::INFINITY,
            block_lowest: f64::INFINITY,
            rng: StdRng::seed_from_u64(seed ^ 0x5a2f_ace5),
        }
    }

    /// Draw from and credit a posterior shared with other chains.
    pub fn sharing(mut self, shared: SharedSurfaceAllocator) -> Self {
        assert!(self.peer_moments.is_none(), "surface evidence has one sharing transport");
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
        if let Some(peers) = self.peer_moments.as_ref() {
            let moments = self.own_moments.iter().zip(peers).map(|(own, peer)| own.merge(*peer)).collect::<Result<Vec<_>, _>>();
            if let Ok(allocator) = moments.and_then(|moments| DepthAllocator::from_moments(&moments)) {
                return allocator.select(&mut self.rng);
            }
        }
        match self.shared.as_ref() {
            Some(shared) => shared
                .lock()
                .expect("shared surface allocator")
                .select(&mut self.rng),
            None => self.allocator.select(&mut self.rng),
        }
    }

    fn credit_arm(&mut self, arm: usize, reward: f64) {
        if self.own_moments[arm].observe(reward).is_err() {
            return;
        }
        if let Some(shared) = self.shared.as_ref() {
            shared
                .lock()
                .expect("shared surface allocator")
                .update(arm, reward);
        }
        self.allocator.update(arm, reward);
    }

    /// Cumulative observations produced by this chain, excluding every import.
    pub fn report(&self) -> SurfaceReport {
        SurfaceReport { schema: self.evidence_schema.clone(), arms: self.own_moments.clone() }
    }

    /// Independent peer blocks informing the portfolio's choices.
    pub fn peer_observations(&self) -> u64 {
        self.peer_moments.as_ref().map_or(0, |arms| arms.iter().map(|arm| arm.count).sum())
    }

    /// Replace peer evidence without changing the held arm, local history, or RNG.
    pub fn import_peers(&mut self, report: SurfaceReport) -> Result<(), &'static str> {
        report.validate()?;
        if self.shared.is_some() || report.schema != self.evidence_schema || report.arms.len() != self.arms.len() {
            return Err("incompatible surface evidence");
        }
        let aggregate = self.own_moments.iter().zip(&report.arms).map(|(own, peer)| own.merge(*peer)).collect::<Result<Vec<_>, _>>()?;
        DepthAllocator::from_moments(&aggregate)?;
        self.peer_moments = report.arms.iter().any(|arm| arm.count > 0).then_some(report.arms);
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
            self.credit_arm(arm, reward);
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
        let mut teacher = SurfacePortfolio::with_block(&[deep], 1, 2).sharing(Arc::clone(&shared));
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

    #[test]
    fn the_shaped_penalty_matches_finite_differences_and_reads_the_aspect() {
        // A prolate cloud along x.
        let mut x = Vec::new();
        for i in 0..12 {
            let t = i as f64;
            x.extend_from_slice(&[
                3.0 * (t * 0.7).sin() * 1.5,
                (t * 1.3).cos(),
                (t * 0.9).sin() * 0.5,
            ]);
        }
        let x = Array1::from(x);
        let shape = inertia_shape(x.view()).expect("a cloud has a shape");
        let longest = (0..3)
            .max_by(|a, b| shape.scale[*a].total_cmp(&shape.scale[*b]))
            .unwrap();
        assert!(
            shape.axes[longest][0].abs() > 0.9,
            "the longest axis follows x: {:?}",
            shape.axes[longest]
        );
        let product: f64 = shape.scale.iter().product();
        assert!(
            (product - 1.0).abs() < 1e-9,
            "volume-preserving scales: {product}"
        );
        let (cutoff, beta, mu) = (1.5, 0.6, 0.2);
        let (_, g) = penalty_shaped(x.view(), cutoff, beta, mu, Some(&shape));
        let h = 1e-6;
        for i in 0..x.len() {
            let mut plus = x.clone();
            let mut minus = x.clone();
            plus[i] += h;
            minus[i] -= h;
            let fd = (penalty_shaped(plus.view(), cutoff, beta, mu, Some(&shape)).0
                - penalty_shaped(minus.view(), cutoff, beta, mu, Some(&shape)).0)
                / (2.0 * h);
            assert!(
                (fd - g[i]).abs() < 1e-5,
                "component {i}: {fd} against {}",
                g[i]
            );
        }
        let isotropic = Shape {
            axes: [[1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]],
            scale: [1.0; 3],
        };
        let plain = penalty(x.view(), cutoff, beta, mu);
        let same = penalty_shaped(x.view(), cutoff, beta, mu, Some(&isotropic));
        assert!((plain.0 - same.0).abs() < 1e-12);
        assert!(!TwoPhase::relative(0.7, 1.0).shape_for(x.view()).is_some());
        assert!(
            TwoPhase::relative_anisotropic(0.7, 1.0)
                .shape_for(x.view())
                .is_some()
        );
    }
}
