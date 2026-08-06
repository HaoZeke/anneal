//! Delayed-acceptance basin hopping: pay for the quench only when it can matter.
//!
//! Basin hopping walks the transformed surface `E~(x) = E(Q(x))`, where `Q` is a
//! local minimisation. Every proposal costs a quench, and measured on 38 points
//! that is 87 to 93 per cent of the whole charged budget: 25 gradient
//! evaluations per proposal against one for the raw energy. Two attempts to
//! shorten the quench failed and failed informatively. Extrapolating the
//! descent geometrically mispredicts the limit by 1e4 at the step where its
//! rule fires. Simply running fewer steps solves 0 of 4 at six steps and 1 of 4
//! at fifteen, against 4 of 4 at twenty-five. The chain needs a converged
//! energy because that is the surface it walks on.
//!
//! What it does *not* need is a converged energy for proposals it is going to
//! reject. Most are: the accept rate runs near one half, and the rejected half
//! pays the same 25 evaluations as the accepted half for an answer that is
//! discarded.
//!
//! # The scheme
//!
//! Let `E^` be a cheap surrogate for `E~` and let `q` be the proposal. One step
//! from `x`:
//!
//! 1. Draw `y ~ q(x, .)` and evaluate the surrogate. Accept the first stage
//!    with
//!    ```text
//!    a1 = min(1, [q(y,x) exp(-E^(y)/T)] / [q(x,y) exp(-E^(x)/T)])
//!    ```
//!    On rejection the chain stays at `x` and no quench is paid.
//! 2. Only if the first stage accepted, quench `y` and accept with
//!    ```text
//!    a2 = min(1, exp(-[(E~(y) - E~(x)) - (E^(y) - E^(x))] / T))
//!    ```
//!
//! # Why the surrogate cannot bias the answer
//!
//! The composite kernel is reversible with respect to `pi ∝ exp(-E~/T)` for any
//! `E^` whatever, which is the Christen and Fox delayed-acceptance argument.
//! Write the first stage as a proposal in its own right: it draws from `q` and
//! keeps with `a1`, so its effective proposal density from `x` to `y != x` is
//! `Q1(x,y) = q(x,y) a1(x,y)`. That density satisfies
//! `Q1(x,y) exp(-E^(x)/T) = Q1(y,x) exp(-E^(y)/T)` by construction, since `a1`
//! is a Metropolis-Hastings ratio for the surrogate target. The second stage is
//! then a Metropolis-Hastings step with proposal `Q1` against the true target,
//! whose ratio is
//!
//! ```text
//! [Q1(y,x) exp(-E~(y)/T)] / [Q1(x,y) exp(-E~(x)/T)]
//!   = exp(-[(E~(y) - E~(x)) - (E^(y) - E^(x))] / T)
//! ```
//!
//! because the `Q1` ratio contributes exactly the surrogate difference back.
//! So detailed balance holds for the pair, and a poor surrogate costs
//! acceptance rate rather than correctness: a surrogate that is constant
//! reduces the scheme to ordinary basin hopping, and a perfect one makes the
//! second stage accept with probability one.
//!
//! This is the property the crate's existing screen does not have. That screen
//! decides on a partial quench with a hand-set margin and lets the trial into
//! the chain on the partial energy, which is a different chain from the one it
//! reports.
//!
//! # What the surrogate is
//!
//! The run labels its own training data. Every quench it pays produces a pair
//! of an unrelaxed structure and the energy it relaxed to, and a 38-point run
//! produces about twelve thousand of them. The surrogate regresses the quench
//! depth `E~(x) - E(x)` on features of the unrelaxed structure, so the
//! prediction needs one energy evaluation and no gradient.
//!
//! Depth rather than energy because the raw energy carries most of the signal
//! already and is available for one evaluation; asking the model only for what
//! the relaxation adds is asking it for the part that is actually hard.
//!
//! # Measured, and not yet paying
//!
//! The scheme is exact and the saving is real; the surrogate is not good
//! enough for the saving to be worth what it costs. On 38 points at 2e5:
//!
//! | variant | stage-1 reject | stage-2 reject | hops | acceptance | accepted moves |
//! |---------|----------------|----------------|------|------------|----------------|
//! | screen (control) | -- | -- | 5561 | 0.560 | 3114 |
//! | delayed, no abstention | 0.485 | 0.571 | 10058 | 0.223 | 2243 |
//! | delayed, abstention at 0.5 T | 0.868 | 0.960 | 21138 | 0.014 | 296 |
//!
//! Nearly half of all proposals avoid a quench and the hop count nearly
//! doubles, which is the mechanism working. What it buys is spent again at the
//! second stage, where the surrogate's error rejects more moves than the extra
//! proposals supply, and abstention on the predictive spread makes it worse
//! rather than better: the posterior's own uncertainty is not calibrated
//! against its error, so raising the bar filtered out the cases it was right
//! about along with the ones it was wrong about.
//!
//! The bottleneck is now a specific and learnable quantity rather than a
//! structural one. The features here are the raw energy, mean coordination and
//! closest contact, and none of them describes how far a structure sits off
//! the manifold of relaxed structures, which is what sets how far it will
//! fall. The gradient at the unrelaxed point is computed and discarded by the
//! evaluation the first stage already pays for, and for a locally quadratic
//! basin the depth goes as `|g|^2 / 2 lambda`, so it is the feature the model
//! is missing and the one that costs nothing to add.

use crate::screen::Screen;
use ndarray::{Array1, ArrayView1};

/// Features the surrogate regresses on.
pub const FEATURES: usize = 5;

/// Cheap structural summary of an unrelaxed structure.
///
/// Coordination counts and the closest contact, which is what says whether a
/// structure has room to relax: a proposal with an overlapping pair has a large
/// depth ahead of it, and a proposal already near a packing has little.
pub fn features(x: ArrayView1<f64>, n: usize, raw: f64) -> Array1<f64> {
    if n < 2 {
        return Array1::from(vec![1.0, raw, 0.0, 0.0, 0.0]);
    }
    let mut nearest = vec![f64::INFINITY; n];
    for i in 0..n {
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
            if d < nearest[i] {
                nearest[i] = d;
            }
        }
    }
    for v in nearest.iter_mut() {
        *v = v.sqrt();
    }
    let scale = nearest.iter().sum::<f64>() / n as f64;
    let closest = nearest.iter().cloned().fold(f64::INFINITY, f64::min);
    let cut = 1.2 * scale;
    let mut coord = 0usize;
    for i in 0..n {
        for j in 0..n {
            if i == j {
                continue;
            }
            let d: f64 = (0..3)
                .map(|k| {
                    let v = x[3 * i + k] - x[3 * j + k];
                    v * v
                })
                .sum::<f64>()
                .sqrt();
            if d < cut {
                coord += 1;
            }
        }
    }
    let mean_coord = coord as f64 / n as f64;
    // The raw energy is the strongest single predictor and costs the one
    // evaluation the stage is allowed. The rest describe how much room the
    // structure has to fall.
    Array1::from(vec![
        1.0,
        raw,
        mean_coord,
        closest / scale.max(1e-12),
        scale,
    ])
}

/// A learned stand-in for the quenched energy, with its own uncertainty.
#[derive(Debug)]
pub struct Surrogate {
    model: Screen,
    /// Quenches required before the first stage is allowed to reject.
    ///
    /// Enforced here rather than left to the model: `Screen::predict` answers
    /// from whatever posterior it holds, including a prior fitted to nothing,
    /// and the warmup it carries gates a different decision. Acting on a
    /// posterior fitted to a handful of quenches would reject good proposals
    /// for no reason. The scheme is exact either way, so waiting costs only
    /// savings not yet being made.
    pub warmup: usize,
    /// First-stage tests run.
    pub stage_one: usize,
    /// First-stage rejections, each of which saved a quench.
    pub stage_one_rejected: usize,
    /// First stages skipped because the posterior was too uncertain to speak.
    ///
    /// An abstention pays the quench the run would have paid anyway, so it is
    /// the cheap way of being unsure.
    pub abstained: usize,
    /// Second-stage tests run.
    pub stage_two: usize,
    /// Second-stage rejections, which are the surrogate's mistakes.
    pub stage_two_rejected: usize,
}

impl Default for Surrogate {
    fn default() -> Self {
        Self::new()
    }
}

impl Surrogate {
    /// A surrogate with no evidence, which accepts every first stage.
    pub fn new() -> Self {
        Self {
            // A long warmup: acting on a posterior fitted to a handful of
            // quenches would reject good proposals for no reason, and the
            // scheme is exact either way, so waiting costs only the savings it
            // has not started making.
            model: Screen::new(FEATURES, 64, 0.0, 0.5),
            warmup: 64,
            abstained: 0,
            stage_one: 0,
            stage_one_rejected: 0,
            stage_two: 0,
            stage_two_rejected: 0,
        }
    }

    /// Quenches the surrogate has been trained on.
    pub fn seen(&self) -> usize {
        self.model.observations()
    }

    /// Predicted quenched energy of an unrelaxed structure, or `None` while the
    /// posterior has too little evidence to be worth consulting.
    ///
    /// `raw` is the structure's own energy, which the caller has already paid
    /// for and which the model corrects rather than replaces.
    pub fn predict(&self, x: ArrayView1<f64>, n: usize, raw: f64) -> Option<f64> {
        self.predict_at(x, n, raw, f64::INFINITY)
    }

    /// As [`Surrogate::predict`], abstaining when the predictive spread exceeds
    /// `tolerance`.
    ///
    /// Abstention is what an uncertainty is for here. The two ways of being
    /// wrong do not cost the same: abstaining pays one quench, which is what
    /// the run would have paid anyway, while guessing wrongly passes a proposal
    /// the second stage then rejects, and that costs an accepted move. So the
    /// first stage should only speak where the posterior is sharp relative to
    /// the temperature that scales the acceptance ratio.
    ///
    /// Measured without it, the second stage rejected 57 per cent of what the
    /// first passed, and composite acceptance fell from 0.56 to 0.223: the
    /// surrogate was confidently wrong often enough to lose more moves than
    /// the extra proposals bought.
    pub fn predict_at(
        &self,
        x: ArrayView1<f64>,
        n: usize,
        raw: f64,
        tolerance: f64,
    ) -> Option<f64> {
        if self.model.observations() < self.warmup {
            return None;
        }
        let f = features(x, n, raw);
        let (depth, sd) = self.model.predict(f.view())?;
        if !sd.is_finite() || sd.sqrt() > tolerance {
            return None;
        }
        Some(raw + depth)
    }

    /// Records a quench: the structure before it and the energy after.
    pub fn observe(&mut self, x: ArrayView1<f64>, n: usize, raw: f64, quenched: f64) {
        let depth = quenched - raw;
        if depth.is_finite() && raw.is_finite() {
            let f = features(x, n, raw);
            self.model.observe(f.view(), depth);
        }
    }

    /// First-stage acceptance probability for a symmetric proposal.
    ///
    /// Symmetric because every move in this crate's library is: a displacement,
    /// a relocation, a twin and a symmetrisation all propose `y` from `x` with
    /// the same density as `x` from `y`. Under asymmetry the ratio `q(y,x) /
    /// q(x,y)` multiplies in, and this returns the wrong number rather than a
    /// slightly wrong one, which is why it says so here.
    pub fn stage_one_probability(&self, surrogate_x: f64, surrogate_y: f64, t: f64) -> f64 {
        let d = surrogate_y - surrogate_x;
        if d <= 0.0 {
            1.0
        } else {
            (-d / t.max(1e-12)).exp()
        }
    }

    /// Second-stage acceptance probability, which corrects the first.
    ///
    /// The surrogate difference is subtracted back out, so what remains is the
    /// error the surrogate made on this pair. A surrogate that was right about
    /// the difference gives one.
    pub fn stage_two_probability(
        &self,
        surrogate_x: f64,
        surrogate_y: f64,
        true_x: f64,
        true_y: f64,
        t: f64,
    ) -> f64 {
        let d = (true_y - true_x) - (surrogate_y - surrogate_x);
        if d <= 0.0 {
            1.0
        } else {
            (-d / t.max(1e-12)).exp()
        }
    }

    /// Quenches avoided as a share of proposals, which is the whole point.
    pub fn saved_share(&self) -> f64 {
        if self.stage_one == 0 {
            return 0.0;
        }
        self.stage_one_rejected as f64 / self.stage_one as f64
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use rand::rngs::StdRng;
    use rand::{Rng, SeedableRng};

    /// The claim the whole scheme rests on, checked by simulation rather than
    /// asserted from the algebra: the composite kernel has the right
    /// stationary distribution whatever the surrogate says.
    ///
    /// A five-state chain with a deliberately bad surrogate, run long, and its
    /// empirical occupancy compared against `exp(-E/T)` normalised. If the
    /// derivation were wrong this would come out skewed toward wherever the
    /// surrogate is optimistic.
    fn stationary_under(surrogate: &[f64], truth: &[f64], t: f64, steps: usize) -> Vec<f64> {
        let n = truth.len();
        let mut rng = StdRng::seed_from_u64(20260806);
        let s = Surrogate::new();
        let mut at = 0usize;
        let mut count = vec![0usize; n];
        for _ in 0..steps {
            // Symmetric proposal: any other state, uniformly.
            let mut y = rng.random_range(0..n);
            while y == at {
                y = rng.random_range(0..n);
            }
            let a1 = s.stage_one_probability(surrogate[at], surrogate[y], t);
            if rng.random::<f64>() < a1 {
                let a2 = s.stage_two_probability(
                    surrogate[at],
                    surrogate[y],
                    truth[at],
                    truth[y],
                    t,
                );
                if rng.random::<f64>() < a2 {
                    at = y;
                }
            }
            count[at] += 1;
        }
        count.iter().map(|c| *c as f64 / steps as f64).collect()
    }

    fn boltzmann(e: &[f64], t: f64) -> Vec<f64> {
        let w: Vec<f64> = e.iter().map(|v| (-v / t).exp()).collect();
        let z: f64 = w.iter().sum();
        w.into_iter().map(|v| v / z).collect()
    }

    #[test]
    fn the_composite_kernel_targets_the_true_distribution() {
        let truth = [0.0, 0.5, 1.0, 1.5, 2.0];
        let t = 0.8;
        let want = boltzmann(&truth, t);
        // A surrogate that is wrong in both directions and by a lot.
        let bad = [2.0, -1.0, 3.0, 0.0, -2.0];
        let got = stationary_under(&bad, &truth, t, 400_000);
        for i in 0..truth.len() {
            assert!(
                (got[i] - want[i]).abs() < 0.02,
                "state {i}: occupancy {:.4} against target {:.4}; a wrong surrogate has biased the chain",
                got[i],
                want[i]
            );
        }
    }

    /// And with a perfect surrogate, so that a passing test above is not
    /// passing for want of the surrogate mattering at all.
    #[test]
    fn a_perfect_surrogate_also_targets_it() {
        let truth = [0.0, 0.5, 1.0, 1.5, 2.0];
        let t = 0.8;
        let want = boltzmann(&truth, t);
        let got = stationary_under(&truth, &truth, t, 400_000);
        for i in 0..truth.len() {
            assert!((got[i] - want[i]).abs() < 0.02, "state {i}");
        }
    }

    /// A perfect surrogate must make the second stage certain, which is what
    /// says the correction is a correction and not a second filter.
    #[test]
    fn a_perfect_surrogate_never_rejects_at_the_second_stage() {
        let s = Surrogate::new();
        for (a, b) in [(0.0, 1.0), (1.0, 0.0), (-3.0, 2.5)] {
            let p = s.stage_two_probability(a, b, a, b, 0.8);
            assert!((p - 1.0).abs() < 1e-12, "second stage rejected a perfect surrogate: {p}");
        }
    }

    /// A constant surrogate must reduce the scheme to ordinary basin hopping:
    /// the first stage always passes and the second carries the real ratio.
    #[test]
    fn a_constant_surrogate_reduces_to_plain_metropolis() {
        let s = Surrogate::new();
        let t = 0.8;
        assert_eq!(s.stage_one_probability(1.0, 1.0, t), 1.0);
        let p = s.stage_two_probability(1.0, 1.0, 0.0, 0.7, t);
        assert!((p - (-0.7f64 / t).exp()).abs() < 1e-12, "got {p}");
    }

    /// The surrogate has to learn a depth relationship it is shown, or the
    /// first stage never rejects anything and the scheme saves nothing.
    #[test]
    fn the_surrogate_learns_a_depth_it_is_shown() {
        let mut rng = StdRng::seed_from_u64(5);
        let mut s = Surrogate::new();
        let n = 8;
        for _ in 0..300 {
            let mut x = Array1::zeros(3 * n);
            for v in x.iter_mut() {
                *v = rng.random_range(-2.0..2.0);
            }
            let raw: f64 = rng.random_range(-10.0..10.0);
            // Depth falls with mean coordination, which is feature 2.
            let f = features(x.view(), n, raw);
            let quenched = raw - 2.0 * f[2];
            s.observe(x.view(), n, raw, quenched);
        }
        assert!(s.seen() >= 200, "only {} observations", s.seen());
        let mut x = Array1::zeros(3 * n);
        for v in x.iter_mut() {
            *v = rng.random_range(-2.0..2.0);
        }
        let p = s.predict(x.view(), n, 0.0).expect("no prediction after 300 quenches");
        assert!(p < 0.0, "predicted quenched energy {p} is not below the raw energy");
    }

    /// Abstention has to actually abstain: a tolerance of zero means the
    /// first stage never speaks, whatever the posterior holds.
    #[test]
    fn a_zero_tolerance_abstains_always() {
        let mut rng = StdRng::seed_from_u64(17);
        let mut s = Surrogate::new();
        let n = 8;
        for _ in 0..300 {
            let mut x = Array1::zeros(3 * n);
            for v in x.iter_mut() {
                *v = rng.random_range(-2.0..2.0);
            }
            let raw: f64 = rng.random_range(-10.0..10.0);
            let f = features(x.view(), n, raw);
            s.observe(x.view(), n, raw, raw - 2.0 * f[2]);
        }
        let mut x = Array1::zeros(3 * n);
        for v in x.iter_mut() {
            *v = rng.random_range(-2.0..2.0);
        }
        assert!(s.predict_at(x.view(), n, 0.0, 0.0).is_none());
        assert!(s.predict_at(x.view(), n, 0.0, f64::INFINITY).is_some());
    }

    /// A cold surrogate must decline to predict rather than guess, so the
    /// scheme starts as plain basin hopping and becomes cheaper as it learns.
    #[test]
    fn a_cold_surrogate_declines_to_predict() {
        let s = Surrogate::new();
        let x = Array1::zeros(24);
        assert!(s.predict(x.view(), 8, -1.0).is_none());
    }
}
