//! `HmcSaSampler`: HMC inside SA, generic over the momentum kernel.
//!
//! Gaussian and q-Gaussian momentum drive the same `Sampler<f64>` impl
//! through the `Momentum` trait. Fixed-step integrators are likewise
//! pluggable, so leapfrog and Omelyan maps share the sampler surface.

use eindir_core::{Bounds, FPair, Objective};
use ndarray::{Array1, ArrayView1};
use rand::Rng;

use crate::cool::Cooling;
use crate::history::State;
use crate::hmc::integrator::{HmcIntegrator, LeapfrogIntegrator};
use crate::hmc::momentum::{GaussianMomentum, Momentum};
use crate::sampler::Sampler;
use eindir_core::Gradient;

/// HMC-driven SA sampler with pluggable momentum kernel.
///
/// `M = GaussianMomentum` recovers Phase 1 standard HMC. `M =
/// QGaussianMomentum` enables the q-deformed dynamics: heavy-tailed
/// momentum draws let the chain escape local cups at the cost of
/// less efficient exploitation in smooth regions.
pub struct HmcSaSampler<O, G, C, M = GaussianMomentum, I = LeapfrogIntegrator>
where
    O: Objective<f64> + Send + Sync,
    G: Gradient<f64>,
    C: Cooling<f64>,
    M: Momentum,
    I: HmcIntegrator,
{
    /// The objective.
    pub obj: O,
    /// The gradient (analytic or finite-difference).
    pub gradient: G,
    /// The cooling schedule.
    pub cool: C,
    /// The momentum kernel (Gaussian, q-Gaussian, or custom).
    pub momentum: M,
    /// The fixed-step HMC integrator (epsilon, L, temp_ref).
    pub integrator: I,
    /// Optional deterministic starting position.
    pub initial_pos: Option<Array1<f64>>,
}

impl<O, G, C, I> HmcSaSampler<O, G, C, GaussianMomentum, I>
where
    O: Objective<f64> + Send + Sync,
    G: Gradient<f64>,
    C: Cooling<f64>,
    I: HmcIntegrator,
{
    /// Constructs an HMC-SA sampler with standard Gaussian momentum.
    /// Equivalent to `HmcSaSampler::with_momentum` with `GaussianMomentum`.
    pub fn new(obj: O, gradient: G, cool: C, integrator: I) -> Self {
        Self {
            obj,
            gradient,
            cool,
            momentum: GaussianMomentum,
            integrator,
            initial_pos: None,
        }
    }
}

impl<O, G, C, M, I> HmcSaSampler<O, G, C, M, I>
where
    O: Objective<f64> + Send + Sync,
    G: Gradient<f64>,
    C: Cooling<f64>,
    M: Momentum,
    I: HmcIntegrator,
{
    /// Constructs with a user-chosen momentum kernel.
    pub fn with_momentum(obj: O, gradient: G, cool: C, momentum: M, integrator: I) -> Self {
        Self {
            obj,
            gradient,
            cool,
            momentum,
            integrator,
            initial_pos: None,
        }
    }

    /// Returns a sampler that starts from the supplied position.
    pub fn with_initial_pos(mut self, pos: Array1<f64>) -> Self {
        assert_eq!(
            pos.len(),
            self.obj.dim(),
            "initial position dimension must match objective dimension"
        );
        self.initial_pos = Some(pos);
        self
    }
}

impl<O, G, C, M, I> Sampler<f64> for HmcSaSampler<O, G, C, M, I>
where
    O: Objective<f64> + Send + Sync,
    G: Gradient<f64>,
    C: Cooling<f64>,
    M: Momentum,
    I: HmcIntegrator,
{
    fn initial_state<R: Rng>(&self, rng: &mut R) -> State {
        let pos = self
            .initial_pos
            .clone()
            .unwrap_or_else(|| self.obj.bounds().mkpoint(rng));
        let val = self.obj.eval(pos.view());
        let pair = FPair { pos, val };
        State {
            cur: pair.clone(),
            best: pair,
        }
    }

    fn qmc_bounds(&self) -> Option<&Bounds<f64>> {
        Some(self.obj.bounds())
    }

    fn initial_state_from_position(&self, pos: Array1<f64>) -> Option<State> {
        let pos = self.obj.bounds().clip(pos.view());
        let val = self.obj.eval(pos.view());
        let pair = FPair { pos, val };
        Some(State {
            cur: pair.clone(),
            best: pair,
        })
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
