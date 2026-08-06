//! Hamiltonian Monte Carlo inside SA (Method B Phase 1).
//!
//! The module exposes Gaussian and q-Gaussian momentum kernels, leapfrog and
//! Omelyan integrators, fixed-length HMC steps, and NUTS-style trajectory
//! adaptation behind the shared sampler trait.
//!
//! [`hop`] is a separate consumer of the same theory and does not share this
//! stack. It proposes for the basin-hopping chain rather than stepping a
//! continuous SA chain, so it charges an evaluation ledger, carries Stan's
//! full adaptation (dual-averaged step size, windowed metric estimation) per
//! chain, and hands its endpoint to a quench instead of to an acceptance test.
//! None of that fits [`crate::sampler::Sampler`], whose `step` has nowhere to
//! put a ledger.

pub mod dual_average;
pub mod hop;
pub mod integrator;
pub mod metric;
pub mod momentum;
pub mod nuts;
pub mod sampler;

pub use dual_average::{DualAverage, WarmupSchedule};
pub use hop::{HopChain, HopConfig, HopDiagnostics, HopProposal, MAX_DELTA_H};
pub use metric::{Metric, MetricAdaptation, MetricKind, RIGID_MASS, RigidModes};

pub use integrator::{
    HmcIntegrator, LeapfrogIntegrator, LeapfrogResult, OMELYAN_LAMBDA, OmelyanIntegrator,
};
pub use momentum::{GaussianMomentum, Momentum, QGaussianMomentum};
pub use nuts::{NutsSaSampler, NutsTransition, nuts_step};
pub use sampler::HmcSaSampler;
