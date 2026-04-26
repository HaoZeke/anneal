//! Hamiltonian Monte Carlo inside SA (Method B Phase 1).
//!
//! Phase 1 ships explicit Gaussian-momentum HMC with an unscaled
//! identity metric. Phase 2 (q-Gaussian momentum) and Phase 3
//! (NUTS-style trajectory adaptation) build on top.
//!
//! See `~/Git/Gitlab/obsidian-notes/Software/anneal/design_pass_09_method_b_hmc.org`.

pub mod integrator;
pub mod sampler;

pub use integrator::{LeapfrogIntegrator, LeapfrogResult};
pub use sampler::HmcSaSampler;
