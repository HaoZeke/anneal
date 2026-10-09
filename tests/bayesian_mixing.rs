//! Witness for automatic Bayesian chain mixing: callers provide only an
//! inner sampler and a proposal budget; chain count and allocation are
//! inferred online from posterior improvement evidence.

use std::sync::{Arc, Mutex};

use anneal_core::variant::{boltzmann, boltzmann_in_box};
use anneal_core::{BayesianMixingSampler, Sampler, State};

use eindir_core::objectives::StybTang2D;
use eindir_core::{Bounds, FPair, Objective};
use ndarray::{Array1, ArrayView1, array};
use rand::Rng;

#[test]
fn bayesian_mixing_uses_single_budget_knob() {
    let variant = boltzmann(StybTang2D::new(), 5.0, 0.5).expect("variant");
    let sampler = BayesianMixingSampler::new(variant, 128);
    let result = sampler.run(42);

    assert_eq!(result.total_proposals(), 128);
    assert!(result.n_chains >= 2);
    assert_eq!(result.proposal_counts.len(), result.n_chains);
    assert!(result.best_val.is_finite());
    assert!(
        result.proposal_counts.iter().copied().max().unwrap_or(0) > 64,
        "posterior allocation should protect one incumbent chain"
    );
}

#[derive(Clone)]
struct FixedQmcSampler {
    bounds: Bounds<f64>,
}

impl FixedQmcSampler {
    fn state_at(&self, pos: Array1<f64>) -> State {
        let val = pos.iter().copied().sum::<f64>();
        let pair = FPair { pos, val };
        State {
            cur: pair.clone(),
            best: pair,
        }
    }
}

impl Sampler<f64> for FixedQmcSampler {
    fn initial_state<R: Rng>(&self, _rng: &mut R) -> State {
        self.state_at(array![0.0, 0.0, 0.0, 0.0])
    }

    fn qmc_bounds(&self) -> Option<&Bounds<f64>> {
        Some(&self.bounds)
    }

    fn initial_state_from_position(&self, pos: Array1<f64>) -> Option<State> {
        Some(self.state_at(pos))
    }

    fn step<R: Rng>(&self, _state: &mut State, _epoch: usize, _rng: &mut R) -> bool {
        false
    }
}

#[test]
fn bayesian_mixing_uses_low_discrepancy_initial_states_when_available() {
    let bounds = Bounds::new(
        array![-1.0, -1.0, -1.0, -1.0],
        array![1.0, 1.0, 1.0, 1.0],
        0.0,
    );
    let sampler = FixedQmcSampler {
        bounds: bounds.clone(),
    };
    let mixer = BayesianMixingSampler::new(sampler, 128);
    let result = mixer.run(7);

    let expected = eindir_core::low_discrepancy_points(
        &bounds,
        result.n_chains,
        anneal_core::qmc_skip_from_seed(7),
    );
    assert_eq!(result.n_chains, 2);
    for (history, expected_pos) in result.chain_histories.iter().zip(expected.outer_iter()) {
        assert_eq!(history.best.pos, expected_pos.to_owned());
    }
}

/// Records every point it evaluates; the value is the coordinate sum.
struct Recorder {
    bounds: Bounds<f64>,
    seen: Arc<Mutex<Vec<Array1<f64>>>>,
}

impl Objective<f64> for Recorder {
    fn dim(&self) -> usize {
        self.bounds.dims
    }

    fn bounds(&self) -> &Bounds<f64> {
        &self.bounds
    }

    fn eval(&self, x: ArrayView1<f64>) -> f64 {
        self.seen.lock().unwrap().push(x.to_owned());
        x.sum()
    }
}

#[test]
fn bayesian_mixing_starts_every_chain_at_a_finite_point_inside_the_box() {
    // The Halton points of an axis with an infinite wall are infinite or NaN,
    // and those of an axis whose width overflows are +inf, which the box
    // clips onto `high`. The chains start as `run_rs_qmc_variant` does.
    for (low, high) in [
        (array![0.0, -1.0], array![f64::INFINITY, 1.0]),
        (array![-1e308, -1.0], array![1e308, 1.0]),
    ] {
        let seen = Arc::<Mutex<Vec<Array1<f64>>>>::default();
        let obj = Recorder {
            bounds: Bounds::new(low.clone(), high.clone(), 0.0),
            seen: Arc::clone(&seen),
        };
        let variant = boltzmann_in_box(obj, 1.0, 0.5).unwrap();
        let result = BayesianMixingSampler::new(variant, 128).run(7);
        let seen = seen.lock().unwrap();
        let starts = &seen[..result.n_chains];
        for (c, x) in starts.iter().enumerate() {
            assert!(x[0] < high[0], "chain {c} starts on high: {x}");
            assert!(starts[..c].iter().all(|y| y != x), "repeated start {x}");
        }
        for x in seen.iter() {
            assert!(x.iter().all(|v| v.is_finite()), "evaluated {x}");
            assert!((0..2).all(|k| low[k] <= x[k] && x[k] <= high[k]), "{x}");
        }
        assert!(result.best_val.is_finite());
    }
}
