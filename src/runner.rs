//! `run_rs`: the pure-Rust SA driver loop. Drives any `Sampler<f64>`
//! and returns a `History`. Pre-A1 this took the five-parameter
//! `SaVariant` directly; the `Sampler` trait extraction (Stan-style)
//! lets the loop treat preset variants and future adaptive samplers
//! through a single dispatch point. See `src/sampler.rs` for rationale.

use rand::rngs::StdRng;
use rand::SeedableRng;

use crate::cool::Cooling;
use crate::history::{EpochLine, History};
use crate::sampler::Sampler;
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
pub fn run_rs<S: Sampler<f64>>(
    sampler: S,
    cooling: &dyn Cooling<f64>,
    n_epochs: usize,
    steps_per_epoch: usize,
    seed: u64,
) -> History {
    let mut rng = StdRng::seed_from_u64(seed);
    let mut state = sampler.initial_state(&mut rng);
    let init_pair = state.best.clone();
    let mut history = History::with_capacity(n_epochs, init_pair);

    for epoch in 0..n_epochs {
        let temp = cooling.temperature(epoch);
        let mut accepted: usize = 0;
        let mut rejected: usize = 0;

        for _ in 0..steps_per_epoch {
            if sampler.step(&mut state, epoch, &mut rng) {
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

/// Convenience wrapper: drives a `SaVariant` through `run_rs`, supplying
/// the variant's own cooling schedule. Equivalent to the pre-A1 API.
pub fn run_rs_variant<O, C, N, M, A>(
    variant: SaVariant<f64, O, C, N, M, A>,
    n_epochs: usize,
    steps_per_epoch: usize,
    seed: u64,
) -> History
where
    O: eindir_core::Objective<f64> + Send + Sync,
    C: Cooling<f64> + Clone,
    N: crate::neigh::Neighborhood<f64>,
    M: crate::movekernel::MoveKernel<f64>,
    A: crate::accept::AcceptRule<f64>,
{
    let cooling = variant.cool.clone();
    run_rs(variant, &cooling, n_epochs, steps_per_epoch, seed)
}
