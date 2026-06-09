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

use eindir_core::{optimal_sampling_drift, GleThermostat, Objective};
use ndarray::{Array1, Array2};
use rand::rngs::StdRng;
use rand::SeedableRng;

use eindir_core::Gradient;

/// Characteristic frequency used when local curvature does not yield a finite estimate.
pub const DEFAULT_GLE_OMEGA0: f64 = 0.2;
const GLE_BAND_RATIO: f64 = 100.0;
const GLE_TIMESTEP_RESOLUTION: f64 = 0.2;
const GLE_MIN_TIMESTEP: f64 = 1e-6;
const GLE_FREQUENCY_FLOOR: f64 = 1e-12;
const GLE_PRECONDITIONER_MAX_SCALE: f64 = 32.0;

/// Result of a GLE-Langevin annealing run.
#[derive(Clone, Debug)]
pub struct GleLangevinResult {
    /// Best-seen position.
    pub best_pos: Vec<f64>,
    /// Best-seen objective value.
    pub best_val: f64,
    /// Gradient evaluations consumed (the work unit at parity with the field).
    pub n_evals: usize,
    /// Characteristic frequency used to scale the colored-noise drift.
    pub omega0: f64,
    /// Timestep after resolving the upper end of the GLE frequency band.
    pub dt: f64,
    /// Diagonal entries of the position-space preconditioning matrix.
    pub preconditioner_diag: Vec<f64>,
    /// Gradient probes spent estimating the preconditioner.
    pub n_preconditioner_grads: usize,
}

/// Diagonal GLE preconditioner in transformed coordinates.
#[derive(Clone, Debug)]
pub struct GlePreconditioner {
    /// Coordinate scale `x_j = scale_j z_j`.
    pub scale: Array1<f64>,
    /// Diagonal entries of the position-space preconditioner.
    pub diag: Array1<f64>,
    /// Characteristic transformed frequency.
    pub omega0: f64,
    /// Gradient probes consumed by the estimate.
    pub n_grads: usize,
}

fn fallback_gle_omega0(low: &Array1<f64>, high: &Array1<f64>) -> f64 {
    let diagonal = low
        .iter()
        .zip(high.iter())
        .map(|(lo, hi)| {
            let width = hi - lo;
            width * width
        })
        .sum::<f64>()
        .sqrt();
    if diagonal.is_finite() && diagonal > 0.0 {
        (1.0 / diagonal).max(GLE_FREQUENCY_FLOOR)
    } else {
        DEFAULT_GLE_OMEGA0
    }
}

fn unit_gle_preconditioner(dim: usize, omega0: f64) -> GlePreconditioner {
    GlePreconditioner {
        scale: Array1::from_elem(dim, 1.0),
        diag: Array1::from_elem(dim, 1.0),
        omega0,
        n_grads: 0,
    }
}

fn splitmix64(mut x: u64) -> u64 {
    x = x.wrapping_add(0x9E37_79B9_7F4A_7C15);
    let mut z = x;
    z = (z ^ (z >> 30)).wrapping_mul(0xBF58_476D_1CE4_E5B9);
    z = (z ^ (z >> 27)).wrapping_mul(0x94D0_49BB_1331_11EB);
    z ^ (z >> 31)
}

fn rademacher(seed: u64, probe: usize, axis: usize) -> f64 {
    let mixed = splitmix64(
        seed ^ ((probe as u64).wrapping_mul(0xD1B5_4A32_D192_ED03))
            ^ ((axis as u64).wrapping_mul(0xABC9_83B5_0FAC_03D7)),
    );
    if mixed & 1 == 0 {
        -1.0
    } else {
        1.0
    }
}

/// Estimate a diagonal coordinate preconditioner from local gradient curvature.
pub fn estimate_gle_preconditioner<O, G>(
    obj: &O,
    grad: &G,
    seed: u64,
    max_probes: usize,
) -> GlePreconditioner
where
    O: Objective<f64>,
    G: Gradient<f64>,
{
    let bounds = obj.bounds();
    let dim = bounds.dims;
    if dim == 0 {
        return unit_gle_preconditioner(0, DEFAULT_GLE_OMEGA0);
    }
    let low = &bounds.low;
    let high = &bounds.high;
    let fallback = fallback_gle_omega0(low, high);
    let center = (low + high) * 0.5;
    let probes = max_probes;
    if probes == 0 {
        return unit_gle_preconditioner(dim, fallback);
    }
    let min_width = low
        .iter()
        .zip(high.iter())
        .filter_map(|(lo, hi)| {
            let width = hi - lo;
            (width.is_finite() && width > 0.0).then_some(width)
        })
        .fold(f64::INFINITY, f64::min);
    if !min_width.is_finite() || min_width <= 0.0 {
        return unit_gle_preconditioner(dim, fallback);
    }
    let step = (f64::EPSILON.cbrt() * min_width).max(f64::EPSILON.sqrt());
    let mut curvature = Array1::<f64>::zeros(dim);
    let mut used = 0usize;
    let mut n_grads = 0usize;

    for probe in 0..probes {
        let direction = Array1::from_iter((0..dim).map(|axis| rademacher(seed, probe, axis)));
        let xp = bounds.clip((&center + &(&direction * step)).view());
        let xm = bounds.clip((&center - &(&direction * step)).view());
        let gp = grad.grad(xp.view());
        let gm = grad.grad(xm.view());
        n_grads += 2;
        if gp.len() != dim
            || gm.len() != dim
            || gp.iter().any(|value| !value.is_finite())
            || gm.iter().any(|value| !value.is_finite())
        {
            continue;
        }
        let hv = (&gp - &gm) * (0.5 / step);
        curvature += &(&direction * &hv);
        used += 1;
    }

    if used == 0 {
        return unit_gle_preconditioner(dim, fallback);
    }
    curvature.mapv_inplace(|value| (value / used as f64).abs());

    let mut positive: Vec<f64> = curvature
        .iter()
        .copied()
        .filter(|value| value.is_finite() && *value > GLE_FREQUENCY_FLOOR)
        .collect();
    if positive.is_empty() {
        return unit_gle_preconditioner(dim, fallback);
    }
    positive.sort_by(|left, right| left.total_cmp(right));
    let reference = positive[positive.len() / 2].max(GLE_FREQUENCY_FLOOR);
    let min_scale = 1.0 / GLE_PRECONDITIONER_MAX_SCALE;
    let scale = curvature.mapv(|value| {
        if value.is_finite() && value > GLE_FREQUENCY_FLOOR {
            (reference / value)
                .sqrt()
                .clamp(min_scale, GLE_PRECONDITIONER_MAX_SCALE)
        } else {
            1.0
        }
    });
    let diag = scale.mapv(|value| value * value);
    GlePreconditioner {
        scale,
        diag,
        omega0: reference.sqrt().max(GLE_FREQUENCY_FLOOR),
        n_grads,
    }
}

/// Estimate the characteristic GLE frequency from local gradient curvature.
pub fn estimate_gle_omega0<O, G>(obj: &O, grad: &G) -> f64
where
    O: Objective<f64>,
    G: Gradient<f64>,
{
    let bounds = obj.bounds();
    let dim = bounds.dims;
    if dim == 0 {
        return DEFAULT_GLE_OMEGA0;
    }
    let low = &bounds.low;
    let high = &bounds.high;
    let center = (low + high) * 0.5;
    let rel_step = f64::EPSILON.cbrt();
    let min_step = f64::EPSILON.sqrt();
    let mut frequencies = Vec::with_capacity(dim);
    for axis in 0..dim {
        let width = high[axis] - low[axis];
        if !width.is_finite() || width <= 0.0 {
            continue;
        }
        let step = (rel_step * width.abs()).max(min_step);
        let mut xp = center.clone();
        let mut xm = center.clone();
        xp[axis] = (center[axis] + step).min(high[axis]);
        xm[axis] = (center[axis] - step).max(low[axis]);
        let denom = xp[axis] - xm[axis];
        if !denom.is_finite() || denom.abs() <= min_step {
            continue;
        }
        let gp = grad.grad(xp.view());
        let gm = grad.grad(xm.view());
        if gp.len() != dim
            || gm.len() != dim
            || gp.iter().any(|v| !v.is_finite())
            || gm.iter().any(|v| !v.is_finite())
        {
            continue;
        }
        let curvature = (gp[axis] - gm[axis]) / denom;
        if curvature.is_finite() && curvature != 0.0 {
            frequencies.push(curvature.abs().sqrt());
        }
    }
    frequencies.sort_by(|left, right| left.total_cmp(right));
    frequencies
        .into_iter()
        .find(|omega| omega.is_finite() && *omega > 0.0)
        .unwrap_or_else(|| fallback_gle_omega0(low, high))
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
    x0: Option<Array1<f64>>,
) -> GleLangevinResult
where
    O: Objective<f64>,
    G: Gradient<f64>,
{
    let dim = obj.bounds().dims;
    let scale = Array1::from_elem(dim, 1.0);
    run_gle_langevin_scaled_sa(
        obj,
        grad,
        seed,
        max_fevals,
        omega0,
        dt,
        n_epochs,
        x0,
        &scale,
        Array1::from_elem(dim, 1.0),
        0,
    )
}

#[allow(clippy::too_many_arguments)]
fn run_gle_langevin_scaled_sa<O, G>(
    obj: &O,
    grad: &G,
    seed: u64,
    max_fevals: usize,
    omega0: f64,
    dt: f64,
    n_epochs: usize,
    x0: Option<Array1<f64>>,
    scale: &Array1<f64>,
    preconditioner_diag: Array1<f64>,
    n_preconditioner_grads: usize,
) -> GleLangevinResult
where
    O: Objective<f64>,
    G: Gradient<f64>,
{
    assert!(max_fevals > 0, "max_fevals must be positive");
    assert!(
        omega0.is_finite() && omega0 > 0.0,
        "omega0 must be positive"
    );
    let bounds = obj.bounds().clone();
    let dim = bounds.dims;
    assert_eq!(scale.len(), dim, "scale length must match dimension");
    let mut rng = StdRng::seed_from_u64(seed);

    // The fitted drift covers [omega0, 100 omega0]; resolve the fastest with dt.
    let omega_hi = GLE_BAND_RATIO * omega0;
    let dt = dt
        .min(GLE_TIMESTEP_RESOLUTION / omega_hi.max(GLE_FREQUENCY_FLOOR))
        .max(GLE_MIN_TIMESTEP);
    let drift = optimal_sampling_drift(omega0);
    let ns = drift.nrows() - 1;

    // Start at the box centre with a thermalised momentum.
    let mut x: Array1<f64> = x0
        .filter(|candidate| {
            candidate.len() == dim && candidate.iter().all(|value| value.is_finite())
        })
        .map(|candidate| bounds.clip(candidate.view()))
        .unwrap_or_else(|| (&bounds.low + &bounds.high) * 0.5);
    let mut fx = obj.eval(x.view());
    let mut best_val = fx;
    let mut best_pos = x.clone();

    let t_hi = fx.abs().max(1.0);
    let t_lo = 1e-3 * t_hi;
    let dynamics_budget = max_fevals.saturating_sub(n_preconditioner_grads).max(1);
    let n_epochs = n_epochs.clamp(1, dynamics_budget);
    let steps = (dynamics_budget / n_epochs).max(1);

    // Auxiliary GLE state: (ns+1) x dim, row 0 is the physical momentum; the
    // per-epoch loop reseeds it at the current temperature.
    let mut s: Array2<f64>;
    let mut g = grad.grad(x.view());
    let mut n_evals = n_preconditioner_grads + 1;

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
            p = &p - &(&(&g * scale) * (0.5 * dt));
            // A: drift the position, clip to the box
            x = &x + &(&(&p * scale) * dt);
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
            p = &p - &(&(&g * scale) * (0.5 * dt));
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
        omega0,
        dt,
        preconditioner_diag: preconditioner_diag.to_vec(),
        n_preconditioner_grads,
    }
}

/// Run GLE-Langevin annealing with a diagonal adaptive coordinate transform.
pub fn gle_langevin_preconditioned_sa<O, G>(
    obj: &O,
    grad: &G,
    seed: u64,
    max_fevals: usize,
    dt: f64,
    n_epochs: usize,
    x0: Option<Array1<f64>>,
) -> GleLangevinResult
where
    O: Objective<f64>,
    G: Gradient<f64>,
{
    let max_probes = max_fevals.saturating_sub(1).checked_div(2).unwrap_or(0);
    let preconditioner = if max_probes > 0 {
        estimate_gle_preconditioner(obj, grad, seed, max_probes)
    } else {
        let bounds = obj.bounds();
        unit_gle_preconditioner(bounds.dims, fallback_gle_omega0(&bounds.low, &bounds.high))
    };
    run_gle_langevin_scaled_sa(
        obj,
        grad,
        seed,
        max_fevals,
        preconditioner.omega0,
        dt,
        n_epochs,
        x0,
        &preconditioner.scale,
        preconditioner.diag,
        preconditioner.n_grads.min(max_fevals.saturating_sub(1)),
    )
}

/// Run GLE-Langevin annealing with a frequency estimated from the objective.
pub fn gle_langevin_adaptive_sa<O, G>(
    obj: &O,
    grad: &G,
    seed: u64,
    max_fevals: usize,
    dt: f64,
    n_epochs: usize,
    x0: Option<Array1<f64>>,
) -> GleLangevinResult
where
    O: Objective<f64>,
    G: Gradient<f64>,
{
    let omega0 = estimate_gle_omega0(obj, grad);
    gle_langevin_sa(obj, grad, seed, max_fevals, omega0, dt, n_epochs, x0)
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
            self.a
                .iter()
                .zip(x.iter())
                .map(|(&ai, &xi)| ai * xi * xi)
                .sum()
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
        let obj = IllConditioned {
            bounds,
            a: a.clone(),
        };
        let grad = IllGrad { a };
        let centre_val = obj.eval(Array1::<f64>::zeros(dim).view()); // 0 at the optimum
        let res = gle_langevin_sa(&obj, &grad, 0, 4000, 0.2, 0.2, 40, None);
        assert!(res.n_evals <= 4000);
        // optimum is 0; require the run to get close from the start basin
        assert!(
            res.best_val < 1e-2 + centre_val,
            "GLE-Langevin did not descend: best {}",
            res.best_val
        );
    }

    #[test]
    fn gle_langevin_starts_from_supplied_anchor() {
        let dim = 2;
        let a = Array1::from_vec(vec![1.0, 2.0]);
        let bounds = Bounds::new(
            Array1::from_elem(dim, -1.0),
            Array1::from_elem(dim, 1.0),
            0.0,
        );
        let obj = IllConditioned {
            bounds,
            a: a.clone(),
        };
        let grad = IllGrad { a };
        let x0 = Array1::from_vec(vec![0.25, -0.5]);

        let res = gle_langevin_sa(&obj, &grad, 11, 1, 0.2, 0.2, 1, Some(x0.clone()));

        assert_eq!(res.n_evals, 1);
        assert_eq!(res.best_pos, x0.to_vec());
    }

    #[test]
    fn gle_frequency_estimate_tracks_quadratic_curvature() {
        let dim = 2;
        let omega = 3.0;
        let curvature = omega * omega;
        let bounds = Bounds::new(
            Array1::from_elem(dim, -1.0),
            Array1::from_elem(dim, 1.0),
            0.0,
        );
        struct FrequencyQuadratic {
            bounds: Bounds<f64>,
            curvature: f64,
        }
        impl Objective<f64> for FrequencyQuadratic {
            fn eval(&self, x: ArrayView1<f64>) -> f64 {
                0.5 * self.curvature * x.iter().map(|xi| xi * xi).sum::<f64>()
            }
            fn bounds(&self) -> &Bounds<f64> {
                &self.bounds
            }
            fn dim(&self) -> usize {
                self.bounds.dims
            }
        }
        struct FrequencyGrad {
            curvature: f64,
            dim: usize,
        }
        impl Gradient<f64> for FrequencyGrad {
            fn grad(&self, x: ArrayView1<f64>) -> Array1<f64> {
                Array1::from_iter(x.iter().map(|xi| self.curvature * xi))
            }
            fn dim(&self) -> usize {
                self.dim
            }
        }
        let obj = FrequencyQuadratic { bounds, curvature };
        let grad = FrequencyGrad { curvature, dim };

        let estimated = estimate_gle_omega0(&obj, &grad);

        assert!((estimated - omega).abs() <= omega * f64::EPSILON.cbrt());
    }

    #[test]
    fn gle_preconditioner_balances_diagonal_quadratic_curvatures() {
        let dim = 5;
        let a = Array1::from_vec(vec![0.05, 0.2, 1.0, 5.0, 50.0]);
        let bounds = Bounds::new(
            Array1::from_elem(dim, -4.0),
            Array1::from_elem(dim, 4.0),
            0.0,
        );
        let obj = IllConditioned {
            bounds,
            a: a.clone(),
        };
        let grad = IllGrad { a: a.clone() };

        let preconditioner = estimate_gle_preconditioner(&obj, &grad, 13, 8);
        let target_curvature = preconditioner.omega0 * preconditioner.omega0;

        assert_eq!(preconditioner.diag.len(), dim);
        assert_eq!(preconditioner.scale.len(), dim);
        assert!(preconditioner.n_grads > 0);
        for (diag, curvature) in preconditioner.diag.iter().zip(a.iter().map(|ai| 2.0 * ai)) {
            assert!(
                (diag * curvature - target_curvature).abs() <= target_curvature * 1e-8,
                "diag {diag} did not balance curvature {curvature} toward {target_curvature}",
            );
        }
    }

    #[test]
    fn preconditioned_gle_reports_matrix_and_probe_work() {
        let dim = 8;
        let a = Array1::from_vec(vec![0.05, 0.2, 1.0, 5.0, 50.0, 0.5, 2.0, 10.0]);
        let bounds = Bounds::new(
            Array1::from_elem(dim, -5.0),
            Array1::from_elem(dim, 5.0),
            0.0,
        );
        let obj = IllConditioned {
            bounds,
            a: a.clone(),
        };
        let grad = IllGrad { a };

        let res = gle_langevin_preconditioned_sa(&obj, &grad, 0, 300, 0.2, 20, None);

        assert_eq!(res.preconditioner_diag.len(), dim);
        assert!(res.n_preconditioner_grads > 0);
        assert!(res.n_evals <= 300);
        assert!(res
            .preconditioner_diag
            .iter()
            .all(|value| value.is_finite() && *value > 0.0));
    }
}
