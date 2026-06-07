//! Witness for automatic Bayesian chain mixing: callers provide only an
//! inner sampler and a proposal budget; chain count and allocation are
//! inferred online from posterior improvement evidence.

use anneal_core::variant::boltzmann;
use anneal_core::{BayesianMixingSampler, Sampler, State};

use eindir_core::{Bounds, FPair};
use eindir_core::objectives::StybTang2D;
use ndarray::{array, Array1};
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
    let bounds = Bounds::new(array![-1.0, -1.0, -1.0, -1.0], array![1.0, 1.0, 1.0, 1.0]);
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
