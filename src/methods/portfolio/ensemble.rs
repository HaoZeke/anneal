//! Replica checkpoints preserve complete local portfolio invocations.

use std::collections::VecDeque;
use std::sync::{Arc, Condvar, Mutex};

use eindir_core::{Bounds, Gradient, Objective};
use ndarray::{Array1, ArrayView1};
use rand::rngs::StdRng;
use rand::{Rng, SeedableRng};

use crate::methods::box_hopping::coverage::{Coverage, RepulsionSnapshot};
use crate::methods::box_hopping::{BoxCoverageConfig, CoverageStats};

use super::{PortfolioPolicy, PortfolioResult, portfolio_optimize_interacting};

/// Complete portfolio replicas under one aggregate work allowance.
#[derive(Clone, Debug)]
pub struct PortfolioEnsembleConfig {
    /// Number of independent controller states, limited by funded work.
    pub replicas: usize,
    /// Combined actual objective and native-gradient callback allowance.
    pub budget: usize,
    /// Scheduling policy used unchanged by every replica.
    pub policy: PortfolioPolicy,
    /// Declared objective noise, with the same acceptance policy as one portfolio.
    pub noise_sigma: Option<f64>,
    /// Sample interaction geometry. No well-depth acceptance bias is applied;
    /// `height` gates separation and `well_tempering` does not affect samples.
    pub coverage: BoxCoverageConfig,
}

impl Default for PortfolioEnsembleConfig {
    fn default() -> Self {
        Self {
            replicas: 4,
            budget: 8_000,
            policy: PortfolioPolicy::Auto,
            noise_sigma: None,
            coverage: BoxCoverageConfig::default(),
        }
    }
}

/// Raw best and auditable work from the funded portfolio replicas.
#[derive(Clone, Debug)]
pub struct PortfolioEnsembleResult {
    /// Best finite feasible position actually evaluated by a replica.
    pub best_pos: Vec<f64>,
    /// Raw objective value at `best_pos`, without a coverage penalty.
    pub best_val: f64,
    /// Actual objective callbacks across all replicas.
    pub n_evals: usize,
    /// Actual native-gradient callbacks across all replicas.
    pub n_grads: usize,
    /// Funded replicas in index order, including their arm allocation statistics.
    pub replicas: Vec<PortfolioResult>,
    /// Sample communication, independent of minimum certificates and visit counts.
    pub coverage: CoverageStats,
}

struct UnavailableGradient;

impl Gradient<f64> for UnavailableGradient {
    fn dim(&self) -> usize {
        unreachable!("scalar portfolios have no native gradient capability")
    }

    fn grad(&self, _: ArrayView1<f64>) -> Array1<f64> {
        unreachable!("scalar portfolios have no native gradient capability")
    }
}

/// Run the same portfolio ensemble with only an objective capability.
///
/// Callers need no gradient implementation or placeholder type. Numerical
/// local-refinement probes are charged objective calls; `n_grads` is zero.
/// All configuration, domain and sharing rules of [`portfolio_ensemble_optimize`]
/// apply, including the single-replica case.
pub fn portfolio_values_ensemble_optimize<O: Objective<f64>>(
    obj: &O,
    seed: u64,
    x0: Option<ArrayView1<f64>>,
    config: &PortfolioEnsembleConfig,
) -> PortfolioEnsembleResult {
    portfolio_ensemble_optimize::<_, UnavailableGradient>(obj, None, seed, x0, config)
}

/// Run persistent portfolios on the same finite, positive-width box.
///
/// Sharing does not reset a controller or install a peer's incumbent. The
/// aggregate allowance divides evenly, with the remainder assigned to low
/// indices. Seeds use `seed ^ (replica * 0x9E37_79B9)`. With no `x0`, each
/// replica exactly retains its ordinary portfolio initialization. With `x0`,
/// replica zero uses it and other replicas use seeded uniform box starts.
///
/// Callbacks run on scoped replica threads and outside exchange locks. A
/// language binding holding an interpreter lock must release it around this
/// call. `None` for `grad` requires no caller Jacobian or force implementation;
/// internal scalar refinement charges its probes as objective calls.
pub fn portfolio_ensemble_optimize<O, G>(
    obj: &O,
    grad: Option<&G>,
    seed: u64,
    x0: Option<ArrayView1<f64>>,
    config: &PortfolioEnsembleConfig,
) -> PortfolioEnsembleResult
where
    O: Objective<f64>,
    G: Gradient<f64>,
{
    assert!(config.replicas > 0, "replicas must be positive");
    assert!(config.budget > 0, "budget must be positive");
    let count = config.replicas.min(config.budget);
    let bounds = obj.bounds();
    let enabled = config.coverage.shared
        && config.coverage.height > 0.0
        && config.coverage.peer_weight > 0.0
        && count > 1;
    let coordinator = Arc::new(Coordinator::new(bounds, count, &config.coverage));
    let replicas = std::thread::scope(|scope| {
        let handles: Vec<_> = (0..count)
            .map(|replica| {
                let replica_seed = seed ^ (replica as u64).wrapping_mul(0x9E37_79B9);
                let allowance = config.budget / config.replicas
                    + usize::from(replica < config.budget % config.replicas);
                let peer = enabled.then(|| {
                    let geometry = coordinator
                        .state
                        .lock()
                        .expect("portfolio exchange lock")
                        .coverage
                        .repulsion_snapshot(replica);
                    Peer {
                        coordinator: Arc::clone(&coordinator),
                        replica,
                        local: Mutex::new(LocalPeer {
                            rng: StdRng::seed_from_u64(replica_seed ^ 0x5045_4552_5f47_454f),
                            prepared: VecDeque::new(),
                            geometry,
                        }),
                    }
                });
                scope.spawn(move || {
                    let start = x0.map(|x| {
                        if replica == 0 {
                            x.to_owned()
                        } else {
                            let mut rng = StdRng::seed_from_u64(replica_seed);
                            Array1::from_shape_fn(bounds.dims, |j| {
                                bounds.low[j]
                                    + (bounds.high[j] - bounds.low[j]) * rng.random::<f64>()
                            })
                        }
                    });
                    portfolio_optimize_interacting(
                        obj,
                        grad,
                        allowance,
                        replica_seed,
                        config.noise_sigma,
                        config.policy,
                        start.as_ref().map(|x| x.view()),
                        peer,
                    )
                })
            })
            .collect();
        handles
            .into_iter()
            .map(|handle| match handle.join() {
                Ok(result) => result,
                Err(payload) => std::panic::resume_unwind(payload),
            })
            .collect::<Vec<_>>()
    });
    let best = replicas
        .iter()
        .min_by(|a, b| a.best_val.total_cmp(&b.best_val))
        .expect("at least one funded replica");
    let coverage = Arc::try_unwrap(coordinator)
        .unwrap_or_else(|_| panic!("replica handles must release the coordinator"))
        .state
        .into_inner()
        .expect("portfolio exchange lock")
        .coverage
        .finish()
        .0;
    PortfolioEnsembleResult {
        best_pos: best.best_pos.clone(),
        best_val: best.best_val,
        n_evals: replicas.iter().map(|r| r.n_evals).sum(),
        n_grads: replicas.iter().map(|r| r.n_grads).sum(),
        replicas,
        coverage,
    }
}

struct State {
    coverage: Coverage,
    active: Vec<bool>,
    arrived: Vec<bool>,
    epoch: usize,
}

struct Coordinator {
    state: Mutex<State>,
    changed: Condvar,
}

impl Coordinator {
    fn new(bounds: &Bounds<f64>, count: usize, config: &BoxCoverageConfig) -> Self {
        Self {
            state: Mutex::new(State {
                coverage: Coverage::new(bounds, count, config, 1),
                active: vec![true; count],
                arrived: vec![false; count],
                epoch: 0,
            }),
            changed: Condvar::new(),
        }
    }

    fn advance(&self, state: &mut State) {
        if state
            .active
            .iter()
            .zip(&state.arrived)
            .all(|(&a, &r)| !a || r)
        {
            // Every producer has finished its slice. Drain in recipient and
            // producer order before allowing any next-slice publication.
            for replica in 0..state.active.len() {
                if state.active[replica] {
                    state.coverage.hear(replica, 1.0);
                }
            }
            state.arrived.fill(false);
            state.epoch += 1;
            self.changed.notify_all();
        }
    }
}

pub(super) struct Peer {
    coordinator: Arc<Coordinator>,
    replica: usize,
    local: Mutex<LocalPeer>,
}

struct LocalPeer {
    rng: StdRng,
    prepared: VecDeque<Array1<f64>>,
    geometry: RepulsionSnapshot,
}

impl Peer {
    pub(super) fn checkpoint(&self, position: ArrayView1<f64>, value: f64) {
        // Unfunded prepared points are not observations and cannot survive
        // into a different arm's line searches or derivative stencils.
        let stats = {
            let mut local = self.local.lock().expect("local peer geometry lock");
            local.prepared.clear();
            local.geometry.take_stats()
        };
        let mut state = self
            .coordinator
            .state
            .lock()
            .expect("portfolio exchange lock");
        state.coverage.record_repulsion(stats);
        state.coverage.sample(self.replica, position, value);
        let epoch = state.epoch;
        state.arrived[self.replica] = true;
        self.coordinator.advance(&mut state);
        while state.epoch == epoch {
            state = self
                .coordinator
                .changed
                .wait(state)
                .expect("portfolio exchange lock");
        }
        self.local
            .lock()
            .expect("local peer geometry lock")
            .geometry = state.coverage.repulsion_snapshot(self.replica);
    }

    pub(super) fn prepare(
        &self,
        anchor: ArrayView1<f64>,
        proposal: &mut Array1<f64>,
        axis: Option<usize>,
    ) -> bool {
        let original = proposal.clone();
        let mut local = self.local.lock().expect("local peer geometry lock");
        let LocalPeer {
            rng,
            geometry,
            prepared,
        } = &mut *local;
        match axis {
            Some(axis) => geometry.repel_coordinate(anchor, proposal, axis, rng),
            None => geometry.repel(anchor, proposal, rng),
        }
        let changed = *proposal != original;
        prepared.push_back(proposal.clone());
        changed
    }

    pub(super) fn evaluated(&self, position: ArrayView1<f64>, value: f64) {
        let paid_proposal = {
            let mut local = self.local.lock().expect("local peer geometry lock");
            let prepared = &mut local.prepared;
            prepared
                .iter()
                .position(|x| x.view() == position)
                .and_then(|index| prepared.remove(index))
        };
        if let Some(position) = paid_proposal {
            self.coordinator
                .state
                .lock()
                .expect("portfolio exchange lock")
                .coverage
                .sample(self.replica, position.view(), value);
        }
    }
}

impl Drop for Peer {
    fn drop(&mut self) {
        // A callback unwind ends this reader's obligation without preventing
        // peers from completing their funded slices and checkpoint exchange.
        if let Ok(mut state) = self.coordinator.state.lock() {
            if let Ok(local) = self.local.get_mut() {
                state.coverage.record_repulsion(local.geometry.take_stats());
            }
            state.active[self.replica] = false;
            state.coverage.retire_reader(self.replica);
            self.coordinator.advance(&mut state);
        }
    }
}
