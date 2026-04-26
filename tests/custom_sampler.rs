//! Witness that `trait Sampler<f64>` is open: a custom sampler that
//! does NOT come from `SaVariant` plugs into the same `run_rs` driver.
//! This is the Stan-style payoff of the A1 trait extraction.

use anneal_core::cool::ReciprocalCool;
use anneal_core::history::State;
use anneal_core::run_rs;
use anneal_core::sampler::Sampler;

use eindir_core::objectives::StybTang2D;
use eindir_core::{FPair, Objective};
use ndarray::Array1;
use rand::Rng;

/// A trivial random-walk sampler that ignores temperature. Demonstrates
/// the trait surface: anyone with their own propose/accept logic can
/// drop straight into `run_rs` without touching the SaVariant types.
struct RandomWalkSampler {
    obj: StybTang2D,
    sigma: f64,
}

impl Sampler<f64> for RandomWalkSampler {
    fn initial_state<R: Rng>(&self, rng: &mut R) -> State {
        let pos = self.obj.bounds().mkpoint(rng);
        let val = self.obj.eval(pos.view());
        let pair = FPair { pos, val };
        State {
            cur: pair.clone(),
            best: pair,
        }
    }

    fn step<R: Rng>(&self, state: &mut State, _epoch: usize, rng: &mut R) -> bool {
        let dim = state.cur.pos.len();
        let proposal_pos: Array1<f64> = Array1::from_iter(
            state
                .cur
                .pos
                .iter()
                .map(|&xi| xi + self.sigma * (rng.random::<f64>() - 0.5)),
        );
        let proposal_val = self.obj.eval(proposal_pos.view());
        // Greedy accept: only downhill moves.
        if proposal_val < state.cur.val {
            state.cur = FPair {
                pos: proposal_pos,
                val: proposal_val,
            };
            if state.cur.val < state.best.val {
                state.best = state.cur.clone();
            }
            return true;
        }
        let _ = dim;
        false
    }
}

#[test]
fn custom_sampler_drives_run_rs() {
    let sampler = RandomWalkSampler {
        obj: StybTang2D::new(),
        sigma: 0.5,
    };
    let cooling = ReciprocalCool::new(1.0_f64);
    let h = run_rs(sampler, &cooling, 50, 100, 1234);
    // Greedy walk is monotone: best_val can only decrease.
    let bests: Vec<f64> = h.epochs.iter().map(|e| e.best_val).collect();
    for w in bests.windows(2) {
        assert!(w[0] >= w[1], "best_val non-monotone: {} -> {}", w[0], w[1]);
    }
    assert!(h.best.val < h.epochs[0].best_val.max(0.0) + 100.0);
}
