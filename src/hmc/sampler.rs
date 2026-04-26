//! `HmcSaSampler`: HMC inside SA, generic over the momentum kernel.
//!
//! Phase 1 used hard-coded Gaussian momentum. Phase 2 (this commit)
//! makes the sampler generic over any `Momentum` impl, so q-Gaussian
//! momentum drives the same `Sampler<f64>` impl through the trait.
//! Dropping into `run_rs` and `MultiChainSampler` is unchanged.

use eindir_core::{FPair, Objective};
use ndarray::ArrayView1;
use rand::Rng;

use crate::cool::Cooling;
use crate::grad::Gradient;
use crate::history::State;
use crate::hmc::integrator::LeapfrogIntegrator;
use crate::hmc::momentum::{GaussianMomentum, Momentum};
use crate::sampler::Sampler;

/// HMC-driven SA sampler with pluggable momentum kernel.
///
/// `M = GaussianMomentum` recovers Phase 1 standard HMC. `M =
/// QGaussianMomentum` enables the q-deformed dynamics: heavy-tailed
/// momentum draws let the chain escape local cups at the cost of
/// less efficient exploitation in smooth regions.
pub struct HmcSaSampler<O, G, C, M = GaussianMomentum>
where
    O: Objective<f64> + Send + Sync,
    G: Gradient<f64>,
    C: Cooling<f64>,
    M: Momentum,
{
    /// The objective.
    pub obj: O,
    /// The gradient (analytic or finite-difference).
    pub gradient: G,
    /// The cooling schedule.
    pub cool: C,
    /// The momentum kernel (Gaussian, q-Gaussian, or custom).
    pub momentum: M,
    /// The leapfrog integrator (epsilon, L, temp_ref).
    pub integrator: LeapfrogIntegrator,
}

impl<O, G, C> HmcSaSampler<O, G, C, GaussianMomentum>
where
    O: Objective<f64> + Send + Sync,
    G: Gradient<f64>,
    C: Cooling<f64>,
{
    /// Constructs an HMC-SA sampler with standard Gaussian momentum.
    /// Phase 1 API; equivalent to `HmcSaSampler::with_momentum` with
    /// `GaussianMomentum`.
    pub fn new(obj: O, gradient: G, cool: C, integrator: LeapfrogIntegrator) -> Self {
        Self {
            obj,
            gradient,
            cool,
            momentum: GaussianMomentum,
            integrator,
        }
    }
}

impl<O, G, C, M> HmcSaSampler<O, G, C, M>
where
    O: Objective<f64> + Send + Sync,
    G: Gradient<f64>,
    C: Cooling<f64>,
    M: Momentum,
{
    /// Constructs with a user-chosen momentum kernel.
    pub fn with_momentum(
        obj: O,
        gradient: G,
        cool: C,
        momentum: M,
        integrator: LeapfrogIntegrator,
    ) -> Self {
        Self {
            obj,
            gradient,
            cool,
            momentum,
            integrator,
        }
    }
}

impl<O, G, C, M> Sampler<f64> for HmcSaSampler<O, G, C, M>
where
    O: Objective<f64> + Send + Sync,
    G: Gradient<f64>,
    C: Cooling<f64>,
    M: Momentum,
{
    fn initial_state<R: Rng>(&self, rng: &mut R) -> State {
        let pos = self.obj.bounds().mkpoint(rng);
        let val = self.obj.eval(pos.view());
        let pair = FPair { pos, val };
        State {
            cur: pair.clone(),
            best: pair,
        }
    }

    fn step<R: Rng>(&self, state: &mut State, epoch: usize, rng: &mut R) -> bool {
        let temp = self.cool.temperature(epoch);
        let dim = state.cur.pos.len();
        let p0 = self.momentum.sample(dim, rng);
        let x0 = state.cur.pos.clone();
        let u0 = state.cur.val;

        let obj = &self.obj;
        let result = self.integrator.evolve(
            x0,
            p0,
            u0,
            temp,
            &self.gradient,
            &self.momentum,
            &|x: &ndarray::Array1<f64>| obj.eval(ArrayView1::from(x.as_slice().unwrap())),
        );

        if result.diverged {
            return false;
        }

        let alpha = (-result.delta_h).exp().min(1.0);
        let u: f64 = rng.random();
        if u < alpha {
            let new_val = obj.eval(result.x.view());
            state.cur = FPair {
                pos: result.x,
                val: new_val,
            };
            if new_val < state.best.val {
                state.best = state.cur.clone();
            }
            true
        } else {
            false
        }
    }
}
