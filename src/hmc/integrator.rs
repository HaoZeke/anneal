//! Leapfrog symplectic integrator for HMC dynamics.
//!
//! The three-stage update mirrors Stan's `base_leapfrog.hpp:17-22`:
//!   begin_update_p  (half-kick: p -= 0.5*epsilon * grad U / T)
//!   update_q        (drift:     x += epsilon * dK/dp)
//!   end_update_p    (half-kick: p -= 0.5*epsilon * grad U / T)
//!
//! For Phase 1 with Gaussian momentum and identity metric, dK/dp = p
//! so the drift simplifies to `x += epsilon * p`. Phase 2 with
//! q-Gaussian momentum will need the implicit `update_q` of
//! `impl_leapfrog.hpp:25-39`.

use ndarray::Array1;

use crate::grad::Gradient;

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

/// Explicit-leapfrog integrator with Gaussian momentum and identity metric.
/// Cooling-aware: `epsilon_eff = epsilon * sqrt(temp / temp_ref)` keeps
/// the linear-stability margin epoch-invariant under SA cooling.
pub struct LeapfrogIntegrator {
    /// Base step size; rescaled by `sqrt(temp / temp_ref)` per call.
    pub epsilon: f64,
    /// Number of leapfrog steps per `evolve` call.
    pub l_steps: usize,
    /// Reference temperature used to normalise the cooling rescaling
    /// (typically `T_0`, the initial temperature).
    pub temp_ref: f64,
    /// Divergence threshold; trajectory marked as diverged when
    /// `|delta_h| > max_delta_h`. Mirrors Stan `base_nuts.hpp:113`
    /// default of 1000.
    pub max_delta_h: f64,
}

impl LeapfrogIntegrator {
    /// Constructs a leapfrog integrator. Asserts `epsilon > 0`,
    /// `l_steps >= 1`, `temp_ref > 0`.
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

    /// Evolves `(x, p)` for `l_steps` leapfrog steps at temperature
    /// `temp`. Returns the final state plus the integrator energy
    /// error.
    pub fn evolve<G, Obj>(
        &self,
        x0: Array1<f64>,
        p0: Array1<f64>,
        u0: f64,
        temp: f64,
        gradient: &G,
        objective: &Obj,
    ) -> LeapfrogResult
    where
        G: Gradient<f64>,
        Obj: Fn(&Array1<f64>) -> f64,
    {
        let eps = self.epsilon * (temp / self.temp_ref).sqrt();
        let mut x = x0;
        let mut p = p0;

        let h0 = u0 / temp + 0.5 * p.iter().map(|pi| pi * pi).sum::<f64>();

        // Half-kick using the initial gradient.
        let mut grad = gradient.grad(x.view());
        for i in 0..p.len() {
            p[i] -= 0.5 * eps * grad[i] / temp;
        }

        for step in 0..self.l_steps {
            // Drift.
            for i in 0..x.len() {
                x[i] += eps * p[i];
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
        let h_new = u_new / temp + 0.5 * p.iter().map(|pi| pi * pi).sum::<f64>();
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
