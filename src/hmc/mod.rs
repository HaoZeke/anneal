//! Hamiltonian Monte Carlo inside SA (Method B Phase 1).
//!
//! The module exposes Gaussian and q-Gaussian momentum kernels, leapfrog and
//! Omelyan integrators, fixed-length HMC steps, and NUTS-style trajectory
//! adaptation behind the shared sampler trait.

pub mod integrator;
pub mod momentum;
pub mod nuts;
pub mod sampler;

pub use integrator::{
    HmcIntegrator, LeapfrogIntegrator, LeapfrogResult, OmelyanIntegrator, OMELYAN_LAMBDA,
};
pub use momentum::{GaussianMomentum, Momentum, QGaussianMomentum};
pub use nuts::{nuts_step, NutsSaSampler, NutsTransition};
pub use sampler::HmcSaSampler;
