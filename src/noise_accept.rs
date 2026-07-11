//! Noise-aware acceptance: the sequential rule of Ball, Branke & Meisel (2018),
//! "Optimal Sampling for Simulated Annealing under Noise," INFORMS Journal on
//! Computing 30(1):200-215 (doi:10.1287/ijoc.2017.0774).
//!
//! The objective difference is observed only through noisy samples
//! `delta_i ~ Normal(Delta, sigma^2)` with known `sigma`. For each proposed
//! move the rule accumulates `c_n = c_{n-1} + delta_n` and, at every draw,
//! makes a three-way decision (accept, reject, or sample again), stopping at
//! the first accept or reject. Their universally optimal per-step acceptance
//! rule (Eq. 19) is
//!
//! ```text
//! A(c_n, c_{n-1}) = min(1, exp(-2 (c_n + beta sigma^2 / 2)
//!                              (c_{n-1} + beta sigma^2 / 2) / sigma^2)),
//! ```
//!
//! with the simple optimal rejection threshold `c* = 0`. The procedure obeys
//! detailed balance at each step while maximizing the acceptance probability
//! per sample, so it is the principled acceptance rule when `Delta` is known
//! only up to noise -- exactly the regime of the finite-precision audit, where
//! the rounding error on the energy difference is a bounded noise channel.
//!
//! Unlike [`AcceptRule`](crate::accept::AcceptRule), whose `(delta_e, T) -> p`
//! shape assumes an exact `delta_e`, OSA consumes a *sampler* of noisy energy
//! differences plus a known noise scale, so it is its own component rather
//! than an `AcceptRule` impl. This Rust component is the typed counterpart of
//! the reference `experiments/osa.py`.

use rand::Rng;
use rand_distr::{Distribution, Normal as NormalDist};

/// Outcome of one OSA decision: whether the move was accepted and how many
/// noisy energy-difference samples the decision consumed.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct OsaResult {
    /// `true` iff the move was accepted.
    pub accepted: bool,
    /// Number of noisy samples drawn before the accept/reject decision.
    pub n_samples: usize,
}

/// The noise-aware OSA acceptance component.
///
/// `c_star` is the rejection threshold on the cumulative difference (`0.0` is
/// the simple optimal strategy of the paper); `max_samples` caps the samples
/// per decision so the inner chain cannot run unbounded.
#[derive(Clone, Copy, Debug)]
pub struct OsaAccept {
    /// Rejection threshold on the cumulative cost difference. `0.0` is the
    /// simple optimal strategy of Ball, Branke & Meisel (2018).
    pub c_star: f64,
    /// Cap on the number of samples drawn for a single decision.
    pub max_samples: usize,
}

impl Default for OsaAccept {
    fn default() -> Self {
        Self {
            c_star: 0.0,
            max_samples: 100_000,
        }
    }
}

impl OsaAccept {
    /// Constructs an OSA component with the simple optimal threshold `c* = 0`
    /// and a generous sample cap.
    pub fn new() -> Self {
        Self::default()
    }

    /// Constructs an OSA component with an explicit threshold and sample cap.
    pub fn with_params(c_star: f64, max_samples: usize) -> Self {
        assert!(max_samples >= 1, "max_samples must be at least 1");
        Self {
            c_star,
            max_samples,
        }
    }

    /// Decides accept/reject for one move from noisy cost-difference samples.
    ///
    /// `sample_delta(rng)` returns one observation
    /// `delta_i ~ Normal(Delta, sigma^2)`; `temp` and `sigma` must be positive.
    /// Returns the decision and the number of samples drawn.
    pub fn decide<F, R>(&self, mut sample_delta: F, temp: f64, sigma: f64, rng: &mut R) -> OsaResult
    where
        F: FnMut(&mut R) -> f64,
        R: Rng,
    {
        assert!(temp > 0.0, "temp must be positive");
        assert!(sigma > 0.0, "sigma must be positive");
        let beta = 1.0 / temp;
        let half = 0.5 * beta * sigma * sigma;
        let inv_var = 1.0 / (sigma * sigma);
        let mut c_prev = 0.0_f64; // c_0
        let mut c = 0.0_f64;
        for n in 1..=self.max_samples {
            c += sample_delta(rng);
            if !c.is_finite() {
                // A non-finite cost-difference sample (e.g. an exhausted budget
                // surfacing as an infinite objective) cannot be accepted.
                return OsaResult {
                    accepted: false,
                    n_samples: n,
                };
            }
            let exponent = -2.0 * (c + half) * (c_prev + half) * inv_var;
            let a = if exponent >= 0.0 { 1.0 } else { exponent.exp() };
            if rng.random::<f64>() < a {
                return OsaResult {
                    accepted: true,
                    n_samples: n,
                };
            }
            if c > self.c_star {
                return OsaResult {
                    accepted: false,
                    n_samples: n,
                };
            }
            c_prev = c;
        }
        OsaResult {
            accepted: false,
            n_samples: self.max_samples,
        }
    }

    /// Empirical OSA acceptance rate and mean samples per decision for a fixed
    /// true difference `delta` observed through `Normal(delta, sigma^2)` noise.
    ///
    /// Mirrors `acceptance_rate` in `experiments/osa.py`; used by the tests and
    /// exposed to Python so the Rust port can be checked against the reference.
    pub fn acceptance_rate<R: Rng>(
        &self,
        delta: f64,
        temp: f64,
        sigma: f64,
        trials: usize,
        rng: &mut R,
    ) -> (f64, f64) {
        let noise = NormalDist::new(delta, sigma).expect("sigma > 0");
        let mut accepts = 0usize;
        let mut total_samples = 0usize;
        for _ in 0..trials {
            let res = self.decide(|r: &mut R| noise.sample(r), temp, sigma, rng);
            accepts += usize::from(res.accepted);
            total_samples += res.n_samples;
        }
        (
            accepts as f64 / trials as f64,
            total_samples as f64 / trials as f64,
        )
    }
}
