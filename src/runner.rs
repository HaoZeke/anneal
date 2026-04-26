//! `run_rs`: the pure-Rust SA driver loop. Consumes a typed `SaVariant`
//! over `f64` components and returns a `History`.

use eindir_core::{FPair, Objective};
use rand::rngs::StdRng;
use rand::{Rng, SeedableRng};

use crate::accept::AcceptRule;
use crate::cool::Cooling;
use crate::history::{EpochLine, History, State};
use crate::movekernel::MoveKernel;
use crate::neigh::Neighborhood;
use crate::variant::SaVariant;

/// Runs the Metropolis-Hastings SA driver for `n_epochs` epochs of
/// `steps_per_epoch` proposals each, seeded from `seed`.
///
/// Initial position is drawn uniformly from `variant.obj.bounds()`.
/// Per epoch: temperature `T_k = variant.cool.temperature(epoch)`;
/// each proposal goes through `variant.mover.propose -> variant.obj.eval ->
/// variant.accept.accept_prob`; bookkeeping per epoch is collected into an
/// `EpochLine`. Best-seen `(pos, val)` is updated after every accepted move.
///
/// Determinism: same `seed` produces bitwise-identical `History.best.pos`
/// and per-epoch `accepted` / `rejected` counters.
pub fn run_rs<O, C, N, M, A>(
    variant: SaVariant<f64, O, C, N, M, A>,
    n_epochs: usize,
    steps_per_epoch: usize,
    seed: u64,
) -> History
where
    O: Objective<f64> + Send + Sync,
    C: Cooling<f64>,
    N: Neighborhood<f64>,
    M: MoveKernel<f64>,
    A: AcceptRule<f64>,
{
    let mut rng = StdRng::seed_from_u64(seed);
    let init_pos = variant.obj.bounds().mkpoint(&mut rng);
    let init_val = variant.obj.eval(init_pos.view());
    let init_pair = FPair {
        pos: init_pos,
        val: init_val,
    };

    let mut state = State {
        cur: init_pair.clone(),
        best: init_pair.clone(),
    };
    let mut history = History::with_capacity(n_epochs, init_pair);

    for epoch in 0..n_epochs {
        let temp = variant.cool.temperature(epoch);
        let mut accepted: usize = 0;
        let mut rejected: usize = 0;

        for _ in 0..steps_per_epoch {
            let proposal_pos = variant.mover.propose(state.cur.pos.view(), temp, &mut rng);
            let proposal_val = variant.obj.eval(proposal_pos.view());
            let delta = proposal_val - state.cur.val;
            let p = variant.accept.accept_prob(delta, temp);

            let u: f64 = rng.random();
            if u < p {
                state.cur = FPair {
                    pos: proposal_pos,
                    val: proposal_val,
                };
                if proposal_val < state.best.val {
                    state.best = state.cur.clone();
                }
                accepted += 1;
            } else {
                rejected += 1;
            }
        }

        history.epochs.push(EpochLine {
            epoch,
            temp,
            accepted,
            rejected,
            best_val: state.best.val,
        });
    }

    history.best = state.best;
    history
}
