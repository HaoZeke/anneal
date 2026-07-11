//! Symplectic integrators for HMC dynamics.
//!
//! The three-stage update mirrors Stan's `base_leapfrog.hpp:17-22`:
//! half-kick momentum update, position drift, then the closing half-kick
//! momentum update.
//!
//! The Omelyan minimum-norm update keeps the same reversible,
//! volume-preserving HMC map while reducing Hamiltonian error for a
//! comparable trajectory length (doi:10.1016/S0010-4655(02)00754-3).

use ndarray::Array1;

use crate::hmc::momentum::Momentum;
use eindir_core::Gradient;

/// Omelyan minimum-norm coefficient for the second-order PQPQP update.
pub const OMELYAN_LAMBDA: f64 = 0.193_183_327_503_783_6;

/// One leapfrog trajectory (L steps) at fixed temperature `temp`.
/// Returns the final `(x, p)` and the integrator energy error
/// `delta_H = H_new - H_old` (small positive on stable trajectories).
#[derive(Clone, Debug)]
pub struct LeapfrogResult {
    /// Final position.
    pub x: Array1<f64>,
    /// Final momentum.
    pub p: Array1<f64>,
    /// Hamiltonian change: `(U_new/T + K_new) - (U_old/T + K_old)`.
    pub delta_h: f64,
    /// `true` if the trajectory diverged (`|delta_h| > max_delta_h`).
    pub diverged: bool,
}

/// Common interface for fixed-step reversible HMC integrators.
pub trait HmcIntegrator: Send + Sync {
    /// Evolves `(x, p)` for the configured number of steps at temperature
    /// `temp`. Returns the final state plus the integrator energy error.
    #[allow(clippy::too_many_arguments)]
    fn evolve<G, M, Obj>(
        &self,
        x0: Array1<f64>,
        p0: Array1<f64>,
        u0: f64,
        temp: f64,
        gradient: &G,
        momentum: &M,
        objective: &Obj,
    ) -> LeapfrogResult
    where
        G: Gradient<f64>,
        M: Momentum + ?Sized,
        Obj: Fn(&Array1<f64>) -> f64;
}

/// Explicit-leapfrog integrator. Cooling-aware: `epsilon_eff =
/// epsilon * sqrt(temp / temp_ref)` keeps the linear-stability margin
/// epoch-invariant under SA cooling.
///
/// For q-Gaussian momentum the drift `dK/dp` is computed by the
/// momentum kernel from the current `p`. The kinetic contribution to
/// the Hamiltonian likewise comes from `momentum.kinetic(&p)`. The
/// integrator stays explicit because `dK/dp` only depends on `p`
/// (constant during the drift step).
pub struct LeapfrogIntegrator {
    /// Base step size; rescaled by `sqrt(temp / temp_ref)` per call.
    pub epsilon: f64,
    /// Number of leapfrog steps per `evolve` call.
    pub l_steps: usize,
    /// Reference temperature used to normalise the cooling rescaling
    /// (typically `T_0`, the initial temperature).
    pub temp_ref: f64,
    /// Divergence threshold; trajectory marked as diverged when abs(delta_h)
    /// exceeds max_delta_h. Mirrors Stan base_nuts.hpp line 113 default of
    /// 1000.
    pub max_delta_h: f64,
}

impl LeapfrogIntegrator {
    /// Constructs a leapfrog integrator. Requires epsilon > 0, l_steps >= 1,
    /// and temp_ref > 0.
    pub fn new(epsilon: f64, l_steps: usize, temp_ref: f64) -> Self {
        assert!(epsilon > 0.0, "epsilon must be positive");
        assert!(l_steps >= 1, "l_steps must be at least 1");
        assert!(temp_ref > 0.0, "temp_ref must be positive");
        Self {
            epsilon,
            l_steps,
            temp_ref,
            max_delta_h: 1000.0,
        }
    }

    /// Effective step size at temperature `temp`.
    fn effective_epsilon(&self, temp: f64) -> f64 {
        self.epsilon * (temp / self.temp_ref).sqrt()
    }
}

impl HmcIntegrator for LeapfrogIntegrator {
    #[allow(clippy::too_many_arguments)]
    fn evolve<G, M, Obj>(
        &self,
        x0: Array1<f64>,
        p0: Array1<f64>,
        u0: f64,
        temp: f64,
        gradient: &G,
        momentum: &M,
        objective: &Obj,
    ) -> LeapfrogResult
    where
        G: Gradient<f64>,
        M: Momentum + ?Sized,
        Obj: Fn(&Array1<f64>) -> f64,
    {
        let eps = self.effective_epsilon(temp);
        let mut x = x0;
        let mut p = p0;

        let h0 = u0 / temp + momentum.kinetic(&p);

        // Half-kick using the initial gradient.
        let mut grad = gradient.grad(x.view());
        for i in 0..p.len() {
            p[i] -= 0.5 * eps * grad[i] / temp;
        }

        for step in 0..self.l_steps {
            // Drift using the momentum kernel's dK/dp (constant during this step).
            let dk = momentum.dk_dp(&p);
            for i in 0..x.len() {
                x[i] += eps * dk[i];
            }
            // Re-evaluate gradient.
            grad = gradient.grad(x.view());
            // Half-kick.
            let half = if step + 1 == self.l_steps { 0.5 } else { 1.0 };
            for i in 0..p.len() {
                p[i] -= half * eps * grad[i] / temp;
            }
        }

        let u_new = objective(&x);
        let h_new = u_new / temp + momentum.kinetic(&p);
        let delta_h = h_new - h0;
        let diverged = delta_h.abs() > self.max_delta_h || !delta_h.is_finite();
        LeapfrogResult {
            x,
            p,
            delta_h,
            diverged,
        }
    }
}

/// Omelyan minimum-norm second-order integrator.
///
/// One step applies the sequence P(lambda eps), Q(eps/2),
/// P((1 - 2lambda) eps), Q(eps/2), P(lambda eps), where P is a momentum kick
/// and Q is a position drift. It has the same reversible Metropolis correction as
/// leapfrog and is the default HMC trajectory map for BGSA-style runs.
pub struct OmelyanIntegrator {
    /// Base step size; rescaled by `sqrt(temp / temp_ref)` per call.
    pub epsilon: f64,
    /// Number of Omelyan steps per `evolve` call.
    pub l_steps: usize,
    /// Reference temperature used to normalise the cooling rescaling.
    pub temp_ref: f64,
    /// Divergence threshold; trajectory marked as diverged when abs(delta_h)
    /// exceeds max_delta_h.
    pub max_delta_h: f64,
    /// Minimum-norm kick coefficient.
    pub lambda: f64,
}

impl OmelyanIntegrator {
    /// Constructs an Omelyan integrator. Requires epsilon > 0, l_steps >= 1,
    /// and temp_ref > 0.
    pub fn new(epsilon: f64, l_steps: usize, temp_ref: f64) -> Self {
        assert!(epsilon > 0.0, "epsilon must be positive");
        assert!(l_steps >= 1, "l_steps must be at least 1");
        assert!(temp_ref > 0.0, "temp_ref must be positive");
        Self {
            epsilon,
            l_steps,
            temp_ref,
            max_delta_h: 1000.0,
            lambda: OMELYAN_LAMBDA,
        }
    }

    /// Effective step size at temperature `temp`.
    fn effective_epsilon(&self, temp: f64) -> f64 {
        self.epsilon * (temp / self.temp_ref).sqrt()
    }
}

impl HmcIntegrator for OmelyanIntegrator {
    #[allow(clippy::too_many_arguments)]
    fn evolve<G, M, Obj>(
        &self,
        x0: Array1<f64>,
        p0: Array1<f64>,
        u0: f64,
        temp: f64,
        gradient: &G,
        momentum: &M,
        objective: &Obj,
    ) -> LeapfrogResult
    where
        G: Gradient<f64>,
        M: Momentum + ?Sized,
        Obj: Fn(&Array1<f64>) -> f64,
    {
        let eps = self.effective_epsilon(temp);
        let mut x = x0;
        let mut p = p0;

        let h0 = u0 / temp + momentum.kinetic(&p);
        let mut grad = gradient.grad(x.view());

        for _ in 0..self.l_steps {
            for i in 0..p.len() {
                p[i] -= self.lambda * eps * grad[i] / temp;
            }

            let dk = momentum.dk_dp(&p);
            for i in 0..x.len() {
                x[i] += 0.5 * eps * dk[i];
            }
            grad = gradient.grad(x.view());

            for i in 0..p.len() {
                p[i] -= (1.0 - 2.0 * self.lambda) * eps * grad[i] / temp;
            }

            let dk = momentum.dk_dp(&p);
            for i in 0..x.len() {
                x[i] += 0.5 * eps * dk[i];
            }
            grad = gradient.grad(x.view());

            for i in 0..p.len() {
                p[i] -= self.lambda * eps * grad[i] / temp;
            }
        }

        let u_new = objective(&x);
        let h_new = u_new / temp + momentum.kinetic(&p);
        let delta_h = h_new - h0;
        let diverged = delta_h.abs() > self.max_delta_h || !delta_h.is_finite();
        LeapfrogResult {
            x,
            p,
            delta_h,
            diverged,
        }
    }
}
