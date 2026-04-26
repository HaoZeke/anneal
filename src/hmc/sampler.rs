//! `HmcSaSampler`: HMC inside SA. Implements `Sampler<f64>` so it
//! drops into `run_rs` and `MultiChainSampler` without further code.
//!
//! Phase 1: Gaussian momentum, identity metric, explicit leapfrog.
//! The cooling temperature comes from the underlying `Cool` trait
//! object; the sampler holds a borrow of it via the `Cooling`
//! trait at construction time.

use eindir_core::{FPair, Objective};
use ndarray::{Array1, ArrayView1};
use rand::Rng;

use crate::cool::Cooling;
use crate::grad::Gradient;
use crate::hmc::integrator::LeapfrogIntegrator;
use crate::history::State;
use crate::sampler::Sampler;

/// HMC-driven SA sampler. The `step` method draws a fresh momentum,
/// integrates the Hamiltonian dynamics for `L` leapfrog steps, and
/// accepts/rejects via the standard HMC Metropolis criterion
/// `alpha = min(1, exp(-delta_H))`.
pub struct HmcSaSampler<O, G, C>
where
    O: Objective<f64> + Send + Sync,
    G: Gradient<f64>,
    C: Cooling<f64>,
{
    /// The objective.
    pub obj: O,
    /// The gradient (analytic or finite-difference).
    pub gradient: G,
    /// The cooling schedule.
    pub cool: C,
    /// The leapfrog integrator (encapsulates epsilon, L, temp_ref).
    pub integrator: LeapfrogIntegrator,
}

impl<O, G, C> HmcSaSampler<O, G, C>
where
    O: Objective<f64> + Send + Sync,
    G: Gradient<f64>,
    C: Cooling<f64>,
{
    /// Constructs an HMC-SA sampler. `temp_ref` of the integrator
    /// should typically equal `cool.temperature(0)` so the cooling
    /// rescaling kicks in correctly.
    pub fn new(obj: O, gradient: G, cool: C, integrator: LeapfrogIntegrator) -> Self {
        Self {
            obj,
            gradient,
            cool,
            integrator,
        }
    }
}

fn sample_gaussian_momentum<R: Rng>(dim: usize, rng: &mut R) -> Array1<f64> {
    Array1::from_iter((0..dim).map(|_| {
        let u1: f64 = rng.random();
        let u2: f64 = rng.random();
        (-2.0 * u1.max(1e-300).ln()).sqrt() * (2.0 * std::f64::consts::PI * u2).cos()
    }))
}

impl<O, G, C> Sampler<f64> for HmcSaSampler<O, G, C>
where
    O: Objective<f64> + Send + Sync,
    G: Gradient<f64>,
    C: Cooling<f64>,
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
        let p0 = sample_gaussian_momentum(dim, rng);
        let x0 = state.cur.pos.clone();
        let u0 = state.cur.val;

        let obj = &self.obj;
        let result = self.integrator.evolve(
            x0,
            p0,
            u0,
            temp,
            &self.gradient,
            &|x: &Array1<f64>| obj.eval(ArrayView1::from(x.as_slice().unwrap())),
        );

        if result.diverged {
            // Treat divergence as rejection. The user-facing diagnostic
            // is the `delta_h` field on `LeapfrogResult`; future epochs
            // can fall back to MultiChainSampler<SaVariant> on repeated
            // divergence (Phase 2 wiring).
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
