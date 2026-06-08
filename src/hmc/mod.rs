//! Hamiltonian Monte Carlo inside SA (Method B Phase 1).
//!
//! Phase 1 ships explicit Gaussian-momentum HMC with an unscaled
//! identity metric. Phase 2 (q-Gaussian momentum) and Phase 3
//! (NUTS-style trajectory adaptation) build on top.
//!
//! See `the design notes`.

pub mod integrator;
pub mod momentum;
pub mod nuts;
pub mod sampler;

pub use integrator::{
    HmcIntegrator, LeapfrogIntegrator, LeapfrogResult, OMELYAN_LAMBDA, OmelyanIntegrator,
};
pub use momentum::{GaussianMomentum, Momentum, QGaussianMomentum};
pub use nuts::{NutsSaSampler, NutsTransition, nuts_step};
pub use sampler::HmcSaSampler;
