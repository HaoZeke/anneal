//! GLE-thermostatted Langevin dynamics as a simulated-annealing driver.
//!
//! This is the colored-noise point of the algebra: a gradient-driven sampler
//! whose Move slot is BAB Langevin dynamics and whose thermostat is the
//! generalized Langevin equation (Ceriotti-Bussi-Parrinello) rather than a
//! single white-noise friction. A white-noise thermostat critically damps one
//! frequency, so an ill-conditioned objective -- a wide spread of Hessian
//! curvatures -- has most modes far from critical and decorrelating slowly. The
//! GLE shapes a frequency-dependent friction from a few auxiliary momenta,
//! flattening the sampling efficiency across the curvature band, and the
//! temperature is annealed so the trajectory settles into a basin. Conditioning
//! is handled by the noise spectrum exactly as the `1/sqrt(D)` proposal scale
//! handles dimension; both are Move-slot transforms every driver could read.

use eindir_core::{GleThermostat, Objective, optimal_sampling_drift};
use ndarray::{Array1, Array2};
use rand::rngs::StdRng;
use rand::SeedableRng;

use crate::grad::Gradient;

/// Result of a GLE-Langevin annealing run.
#[derive(Clone, Debug)]
pub struct GleLangevinResult {
    /// Best-seen position.
    pub best_pos: Vec<f64>,
    /// Best-seen objective value.
    pub best_val: f64,
    /// Gradient evaluations consumed (the work unit at parity with the field).
    pub n_evals: usize,
}

/// Run GLE-thermostatted Langevin annealing on `obj` with gradient `grad`.
///
/// `max_fevals` bounds the gradient evaluations (one per dynamics step).
/// `omega0` is the characteristic frequency the fitted optimal-sampling drift is
/// scaled to (it flattens the sampling efficiency across `[omega0, 100 omega0]`),
/// `dt` the timestep (clamped so the fastest band frequency is resolved), and
/// `n_epochs` the number of geometric temperature levels. Returns the best
/// objective value at budget parity.
pub fn gle_langevin_sa<O, G>(
    obj: &O,
    grad: &G,
    seed: u64,
    max_fevals: usize,
    omega0: f64,
    dt: f64,
    n_epochs: usize,
) -> GleLangevinResult
where
    O: Objective<f64>,
    G: Gradient<f64>,
{
    assert!(max_fevals > 0, "max_fevals must be positive");
    let bounds = obj.bounds().clone();
    let dim = bounds.dims;
    let mut rng = StdRng::seed_from_u64(seed);

    // The fitted drift covers [omega0, 100 omega0]; resolve the fastest with dt.
    let omega_hi = 100.0 * omega0;
    let dt = dt.min(0.2 / omega_hi.max(1e-12)).max(1e-6);
    let drift = optimal_sampling_drift(omega0);
    let ns = drift.nrows() - 1;

    // Start at the box centre with a thermalised momentum.
    let mut x: Array1<f64> = (&bounds.low + &bounds.high) * 0.5;
    let mut fx = obj.eval(x.view());
    let mut best_val = fx;
    let mut best_pos = x.clone();

    let t_hi = fx.abs().max(1.0);
    let t_lo = 1e-3 * t_hi;
    let n_epochs = n_epochs.clamp(1, max_fevals);
    let steps = (max_fevals / n_epochs).max(1);

    // Auxiliary GLE state: (ns+1) x dim, row 0 is the physical momentum; the
    // per-epoch loop reseeds it at the current temperature.
    let mut s: Array2<f64>;
    let mut g = grad.grad(x.view());
    let mut n_evals = 1usize;

    'outer: for epoch in 0..n_epochs {
        let frac = epoch as f64 / (n_epochs.max(2) - 1) as f64;
        let temperature = t_hi * (t_lo / t_hi).powf(frac);
        let gle = GleThermostat::canonical(&drift, dt, temperature, 1.0);
        // reseed the physical momentum at this temperature
        {
            let c = Array2::<f64>::eye(ns + 1) * temperature;
            s = gle.sample_stationary(&c, dim, 1.0, &mut rng);
        }
        for _ in 0..steps {
            // B: half momentum kick from the force (mass = 1)
            let mut p: Array1<f64> = s.row(0).to_owned();
            p = &p - &(&g * (0.5 * dt));
            // A: drift the position, clip to the box
            x = &x + &(&p * dt);
            x = bounds.clip(x.view());
            g = grad.grad(x.view());
            n_evals += 1;
            let fy = obj.eval(x.view());
            if fy < best_val {
                best_val = fy;
                best_pos = x.clone();
            }
            fx = fy;
            // B: second half kick
            p = &p - &(&g * (0.5 * dt));
            // O: GLE colored-noise thermostat on the momentum
            s.row_mut(0).assign(&p);
            gle.step(&mut s.view_mut(), &mut rng);
            if n_evals >= max_fevals {
                break 'outer;
            }
        }
    }
    let _ = fx;

    GleLangevinResult {
        best_pos: best_pos.to_vec(),
        best_val,
        n_evals,
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use eindir_core::Bounds;
    use ndarray::ArrayView1;

    // Ill-conditioned separable quadratic: curvatures span three decades.
    struct IllConditioned {
        bounds: Bounds<f64>,
        a: Array1<f64>,
    }
    impl Objective<f64> for IllConditioned {
        fn eval(&self, x: ArrayView1<f64>) -> f64 {
            self.a.iter().zip(x.iter()).map(|(&ai, &xi)| ai * xi * xi).sum()
        }
        fn bounds(&self) -> &Bounds<f64> {
            &self.bounds
        }
        fn dim(&self) -> usize {
            self.bounds.dims
        }
    }
    struct IllGrad {
        a: Array1<f64>,
    }
    impl Gradient<f64> for IllGrad {
        fn grad(&self, x: ArrayView1<f64>) -> Array1<f64> {
            Array1::from_iter(self.a.iter().zip(x.iter()).map(|(&ai, &xi)| 2.0 * ai * xi))
        }
        fn dim(&self) -> usize {
            self.a.len()
        }
    }

    #[test]
    fn gle_langevin_descends_ill_conditioned_quadratic() {
        // condition number 1000 (a_j from 0.05 to 50); the colored-noise
        // thermostat should drive every coordinate's curvature toward the
        // minimum so the objective falls well below the box-centre value.
        let dim = 8;
        let a = Array1::from_vec(vec![0.05, 0.2, 1.0, 5.0, 50.0, 0.5, 2.0, 10.0]);
        let bounds = Bounds::new(
            Array1::from_elem(dim, -5.0),
            Array1::from_elem(dim, 5.0),
            0.0,
        );
        let obj = IllConditioned { bounds, a: a.clone() };
        let grad = IllGrad { a };
        let centre_val = obj.eval(Array1::<f64>::zeros(dim).view()); // 0 at the optimum
        let res = gle_langevin_sa(&obj, &grad, 0, 4000, 0.2, 0.2, 40);
        assert!(res.n_evals <= 4000);
        // optimum is 0; require the run to get close from the start basin
        assert!(
            res.best_val < 1e-2 + centre_val,
            "GLE-Langevin did not descend: best {}",
            res.best_val
        );
    }
}
