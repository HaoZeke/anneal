//! `run_rs`: the pure-Rust SA driver loop. Drives any `Sampler<f64>`
//! and returns a `History`. The `Sampler` trait keeps the driver loop
//! independent of the concrete proposal and acceptance machinery.

use rand::SeedableRng;
use rand::rngs::StdRng;

use crate::cool::Cooling;
use crate::history::{EpochLine, History, State};
use crate::sampler::Sampler;
use crate::variant::SaVariant;

/// Deterministic positive Halton skip derived from a run seed.
pub fn qmc_skip_from_seed(seed: u64) -> u64 {
    seed.wrapping_add(1).max(1)
}

fn drive_rs<S: Sampler<f64>>(
    sampler: &S,
    cooling: &dyn Cooling<f64>,
    mut state: State,
    n_epochs: usize,
    steps_per_epoch: usize,
    rng: &mut StdRng,
) -> History {
    let init_pair = state.best.clone();
    let mut history = History::with_capacity(n_epochs, init_pair);

    for epoch in 0..n_epochs {
        let temp = cooling.temperature(epoch);
        let mut accepted: usize = 0;
        let mut rejected: usize = 0;

        for _ in 0..steps_per_epoch {
            if sampler.step(&mut state, epoch, rng) {
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
    let state = sampler.initial_state(&mut rng);
    drive_rs(
        &sampler,
        cooling,
        state,
        n_epochs,
        steps_per_epoch,
        &mut rng,
    )
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

/// Resumable variant driver: runs epochs `[start_epoch, start_epoch + n_epochs)`
/// of the variant's own cooling schedule, continuing from a prior chain
/// position when one is supplied, and returns the history together with the
/// final chain state so a caller can extend the same annealing trajectory
/// across allocation slices. Slice-restarted SA never anneals; a persistent
/// chain recovers the long-schedule behaviour of the classical presets.
pub fn run_rs_variant_resumed<O, C, N, M, A>(
    variant: SaVariant<f64, O, C, N, M, A>,
    start_epoch: usize,
    n_epochs: usize,
    steps_per_epoch: usize,
    seed: u64,
    resume: Option<eindir_core::FPair<f64>>,
) -> (History, eindir_core::FPair<f64>)
where
    O: eindir_core::Objective<f64> + Send + Sync,
    C: Cooling<f64> + Clone,
    N: crate::neigh::Neighborhood<f64>,
    M: crate::movekernel::MoveKernel<f64>,
    A: crate::accept::AcceptRule<f64>,
{
    let mut rng =
        StdRng::seed_from_u64(seed ^ (start_epoch as u64).wrapping_mul(0x9E37_79B9_7F4A_7C15));
    let mut state = match resume {
        Some(cur) => State {
            best: cur.clone(),
            cur,
        },
        None => variant.initial_state(&mut rng),
    };
    let init_pair = state.best.clone();
    let mut history = History::with_capacity(n_epochs, init_pair);
    for epoch in start_epoch..start_epoch.saturating_add(n_epochs) {
        let temp = variant.cool.temperature(epoch);
        let mut accepted: usize = 0;
        let mut rejected: usize = 0;
        for _ in 0..steps_per_epoch {
            if variant.step(&mut state, epoch, &mut rng) {
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
    history.best = state.best.clone();
    (history, state.cur)
}

/// Runs the same `SaVariant` from a bounded low-discrepancy start set and
/// returns the best history across starts.
pub fn run_rs_qmc_variant<O, C, N, M, A>(
    variant: SaVariant<f64, O, C, N, M, A>,
    n_starts: usize,
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
    let starts = eindir_core::low_discrepancy_points(
        variant.obj.bounds(),
        n_starts.max(1),
        qmc_skip_from_seed(seed),
    );
    let mut best_history = None;

    for (idx, start) in starts.outer_iter().enumerate() {
        let pos = variant.obj.bounds().clip(start);
        let val = variant.obj.eval(pos.view());
        let pair = eindir_core::FPair { pos, val };
        let state = State {
            cur: pair.clone(),
            best: pair,
        };
        let chain_seed = seed.wrapping_add(0x9e37_79b9_7f4a_7c15_u64.wrapping_mul(idx as u64 + 1));
        let mut rng = StdRng::seed_from_u64(chain_seed);
        let history = drive_rs(
            &variant,
            &cooling,
            state,
            n_epochs,
            steps_per_epoch,
            &mut rng,
        );
        if best_history
            .as_ref()
            .map_or(true, |best: &History| history.best.val < best.best.val)
        {
            best_history = Some(history);
        }
    }

    best_history.expect("n_starts.max(1) guarantees at least one chain")
}
