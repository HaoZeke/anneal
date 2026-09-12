use std::collections::HashMap;
use std::sync::Mutex;
use std::thread::ThreadId;

use anneal_core::movekernel::{MoveKernel, TsallisVisit, reflect_into_box};
use anneal_core::{
    PortfolioEnsembleConfig, PortfolioEnsembleResult, portfolio_values_ensemble_optimize,
};
use eindir_core::{Bounds, Objective, shifted_low_discrepancy_points};
use ndarray::{Array1, ArrayView1};
use rand::{Rng, SeedableRng, rngs::StdRng};

const DIM: usize = 16;
const SEED: u64 = 17;

fn bounds() -> Bounds<f64> {
    Bounds::new(
        Array1::from_elem(DIM, -2.0),
        Array1::from_elem(DIM, 2.0),
        0.0,
    )
}

struct ScalarTrace {
    bounds: Bounds<f64>,
    traces: Mutex<HashMap<ThreadId, Vec<Array1<f64>>>>,
}

impl Objective<f64> for ScalarTrace {
    fn dim(&self) -> usize {
        DIM
    }

    fn bounds(&self) -> &Bounds<f64> {
        &self.bounds
    }

    fn eval(&self, position: ArrayView1<f64>) -> f64 {
        assert!(self.bounds.contains(position));
        self.traces
            .lock()
            .unwrap()
            .entry(std::thread::current().id())
            .or_default()
            .push(position.to_owned());
        // A constant loss accepts every GSA proposal and aligns paid phases.
        1.0
    }
}

fn run(shared: bool) -> (PortfolioEnsembleResult, Vec<Vec<Array1<f64>>>) {
    let objective = ScalarTrace {
        bounds: bounds(),
        traces: Mutex::new(HashMap::new()),
    };
    let mut config = PortfolioEnsembleConfig {
        replicas: 2,
        budget: 2_048,
        ..PortfolioEnsembleConfig::default()
    };
    config.coverage.shared = shared;
    config.coverage.radius = 0.8;
    let start = Array1::from_elem(DIM, -0.125);
    let result = portfolio_values_ensemble_optimize(&objective, SEED, Some(start.view()), &config);
    let traces: Vec<_> = objective
        .traces
        .into_inner()
        .unwrap()
        .into_values()
        .collect();
    assert_eq!(traces.len(), config.replicas);
    assert!(
        traces
            .iter()
            .all(|trace| trace.len() == config.budget / config.replicas)
    );
    assert_eq!(result.n_evals, config.budget);
    assert_eq!(result.n_grads, 0);
    assert_eq!(result.best_val, 1.0);
    (result, traces)
}

fn strategy_start(traces: &[Vec<Array1<f64>>], replica: usize) -> (usize, usize, u64) {
    // DE and GSA use separate seeded front-load streams. Locate the paid
    // GSA initialization by its generator, not a fixed callback ordinal.
    let replica_seed = SEED ^ (replica as u64).wrapping_mul(0x9E37_79B9);
    let front_seed = (replica_seed ^ 0xBEEF).wrapping_add(2);
    let gsa_seed = StdRng::seed_from_u64(front_seed).random::<u64>() ^ 2;
    let starts = shifted_low_discrepancy_points(
        &bounds(),
        1,
        anneal_core::qmc_skip_from_seed(gsa_seed),
        gsa_seed,
    );
    let found: Vec<_> = traces
        .iter()
        .enumerate()
        .flat_map(|(trace, positions)| {
            let start = starts.row(0);
            positions
                .iter()
                .enumerate()
                .filter_map(move |(index, position)| {
                    (position.view() == start).then_some((trace, index))
                })
        })
        .collect();
    assert_eq!(
        found.len(),
        1,
        "GSA initialization must be uniquely identified"
    );
    let (trace, index) = found[0];
    assert!(index + 2 * DIM < traces[trace].len());
    (trace, index, gsa_seed)
}

fn replay_strategy(trace: &[Array1<f64>], start: usize, seed: u64) -> Vec<Array1<f64>> {
    let mut rng = StdRng::seed_from_u64(seed);
    let kernel = TsallisVisit::new(2.62);
    let domain = bounds();
    (0..2 * DIM)
        .map(|step| {
            let anchor = &trace[start + step];
            let raw = if step < DIM {
                kernel.propose(anchor.view(), 5230.0, &mut rng)
            } else {
                let axis = step - DIM;
                let visited = kernel.propose(
                    ArrayView1::from(std::slice::from_ref(&anchor[axis])),
                    5230.0,
                    &mut rng,
                );
                let mut position = anchor.clone();
                position[axis] = visited[0];
                position
            };
            let reflected = reflect_into_box(raw.view(), &domain);
            let _: f64 = rng.random(); // Raw equal-energy acceptance consumes one draw.
            reflected
        })
        .collect()
}

#[test]
fn private_scalar_gsa_replays_full_and_coordinate_strategy() {
    let (_, traces) = run(false);
    for replica in 0..2 {
        let (which, start, seed) = strategy_start(&traces, replica);
        let expected = replay_strategy(&traces[which], start, seed);
        for (step, proposal) in expected.iter().enumerate() {
            assert_eq!(
                traces[which][start + step + 1],
                *proposal,
                "replica={replica}, step={step}"
            );
        }
    }
}

#[test]
fn shared_scalar_gsa_preserves_coordinate_support_with_effective_separation() {
    let (_, private) = run(false);
    let (result, shared) = run(true);
    assert!(result.coverage.applied_foreign_samples > 0);
    assert!(result.coverage.repelled_proposals > 0);
    let mut coordinate_corrections = 0;
    for replica in 0..2 {
        let (which, start, seed) = strategy_start(&private, replica);
        let matching: Vec<_> = shared
            .iter()
            .filter(|trace| trace[0] == private[which][0])
            .collect();
        assert_eq!(
            matching.len(),
            1,
            "replicas must retain their explicit starts"
        );
        let trace = matching[0];
        let raw = replay_strategy(trace, start, seed);
        for axis in 0..DIM {
            let anchor = &trace[start + DIM + axis];
            let proposal = &trace[start + DIM + axis + 1];
            for inactive in 0..DIM {
                if inactive != axis {
                    assert_eq!(
                        proposal[inactive], anchor[inactive],
                        "replica={replica}, active={axis}: peer correction must not move axis {inactive}"
                    );
                }
            }
            coordinate_corrections += usize::from(proposal[axis] != raw[DIM + axis][axis]);
        }
    }
    assert!(
        coordinate_corrections > 0,
        "coordinate separation must not be disabled"
    );
}
