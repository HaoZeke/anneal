use std::collections::HashMap;
use std::sync::Mutex;
use std::thread::ThreadId;

use anneal_core::cool::{Cooling, TsallisCool};
use anneal_core::movekernel::{MoveKernel, TsallisVisit, reflect_into_box};
use anneal_core::{
    PortfolioEnsembleConfig, PortfolioEnsembleResult, portfolio_values_ensemble_optimize,
};
use eindir_core::{Bounds, Objective};
use ndarray::{Array1, ArrayView1};
use rand::{Rng, SeedableRng, rngs::StdRng};

const DIM: usize = 16;
const SEED: u64 = 17;
const VISIT_Q: f64 = 2.62;
const INITIAL_TEMP: f64 = 5230.0;
/// Recorded quenched moves the replay builds. The warm-up spends two GSA
/// slices and then yields to another arm, so the replay runs past that
/// yield into the resumed chain.
const REPLAY_LEN: usize = 12 * DIM;

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
        // A constant loss accepts every Metropolis proposal and rejects
        // every quenched one, so the chain stays on its explicit start.
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

fn replica_origin(replica: usize) -> Array1<f64> {
    if replica == 0 {
        return Array1::from_elem(DIM, -0.125);
    }
    let replica_seed = SEED ^ (replica as u64).wrapping_mul(0x9E37_79B9);
    let mut rng = StdRng::seed_from_u64(replica_seed);
    Array1::from_shape_fn(DIM, |axis| -2.0 + 4.0 * rng.random::<f64>())
}

fn gsa_seed(replica: usize) -> u64 {
    // The values-only GSA chain is this mix of the replica seed. It does
    // not follow the shared arm generator.
    let replica_seed = SEED ^ (replica as u64).wrapping_mul(0x9E37_79B9);
    let front = (replica_seed ^ 0xBEEF).wrapping_add(2);
    StdRng::seed_from_u64(front).random::<u64>() ^ 2
}

/// Quenched dual-annealing strategy: the all-coordinate half is skipped,
/// equal energy is rejected, and the chain therefore stays at `origin`.
/// A finished temperature whose index is a positive multiple of five
/// records one box-wide reseed and does not install it.
fn quenched_replay(origin: &Array1<f64>, seed: u64, steps: usize) -> Vec<Array1<f64>> {
    let mut rng = StdRng::seed_from_u64(seed);
    let kernel = TsallisVisit::new(VISIT_Q);
    let cooling = TsallisCool::new(INITIAL_TEMP, VISIT_Q);
    let domain = bounds();
    let mut epoch = 0usize;
    let mut cursor = 0usize;
    let n_strategy = 2 * DIM;
    let mut out = Vec::with_capacity(steps);
    while out.len() < steps {
        let mut temp = cooling.temperature(epoch).max(1e-300);
        if cursor == 0 {
            let floor = INITIAL_TEMP * 2.0e-5;
            if temp < floor {
                epoch = 0;
                temp = cooling.temperature(0).max(1e-300);
            }
        }
        let mut finished = true;
        while cursor < n_strategy {
            if out.len() >= steps {
                finished = false;
                break;
            }
            let j = cursor;
            if j < DIM {
                cursor += 1;
                continue;
            }
            let axis = j - DIM;
            let visited = kernel.propose(
                ArrayView1::from(std::slice::from_ref(&origin[axis])),
                temp,
                &mut rng,
            );
            let mut position = origin.clone();
            position[axis] = visited[0];
            out.push(reflect_into_box(position.view(), &domain));
            cursor += 1;
        }
        if !finished || out.len() >= steps {
            break;
        }
        if epoch > 0 && epoch.is_multiple_of(5) {
            let mut x = Array1::zeros(DIM);
            for axis in 0..DIM {
                let low = domain.low[axis];
                let high = domain.high[axis];
                x[axis] = low + (high - low) * rng.random::<f64>();
            }
            out.push(domain.clip(x.view()));
            if out.len() >= steps {
                break;
            }
        }
        epoch += 1;
        cursor = 0;
    }
    out
}

fn strategy_start(
    traces: &[Vec<Array1<f64>>],
    replica: usize,
) -> (usize, usize, u64, Vec<Array1<f64>>) {
    let origin = replica_origin(replica);
    let seed = gsa_seed(replica);
    let replay = quenched_replay(&origin, seed, REPLAY_LEN);
    let found: Vec<_> = traces
        .iter()
        .enumerate()
        .filter_map(|(trace, positions)| {
            if positions.first() != Some(&origin) {
                return None;
            }
            positions
                .windows(DIM)
                .position(|window| window == &replay[..DIM])
                .map(|index| (trace, index))
        })
        .collect();
    assert_eq!(
        found.len(),
        1,
        "replica {replica}: the quenched GSA epoch must be uniquely identified"
    );
    let (trace, index) = found[0];
    assert!(index + REPLAY_LEN < traces[trace].len());
    (trace, index, seed, replay)
}

#[test]
fn private_scalar_gsa_replays_quenched_coordinate_strategy() {
    let (_, traces) = run(false);
    for replica in 0..2 {
        let (which, start, _, replay) = strategy_start(&traces, replica);
        for (step, proposal) in replay.iter().take(DIM).enumerate() {
            assert_eq!(
                traces[which][start + step],
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
        let (which, start, _, raw) = strategy_start(&private, replica);
        let origin = &private[which][0];
        let matching: Vec<_> = shared.iter().filter(|trace| trace[0] == *origin).collect();
        assert_eq!(
            matching.len(),
            1,
            "replicas must retain their explicit starts"
        );
        let trace = matching[0];
        for axis in 0..DIM {
            let proposal = &trace[start + axis];
            for inactive in 0..DIM {
                if inactive != axis {
                    assert_eq!(
                        proposal[inactive], origin[inactive],
                        "replica={replica}, active={axis}: peer correction must not move axis {inactive}"
                    );
                }
            }
            coordinate_corrections += usize::from(proposal[axis] != raw[axis][axis]);
        }
    }
    assert!(
        coordinate_corrections > 0,
        "coordinate separation must not be disabled"
    );
}

#[test]
fn private_scalar_gsa_resumes_its_partial_strategy_after_other_arms() {
    let (result, traces) = run(false);
    for replica in 0..2 {
        let (which, start, _, replay) = strategy_start(&traces, replica);
        let trace = &traces[which];
        let gsa_pulls = result.replicas[replica]
            .arm_stats
            .iter()
            .find(|arm| arm.name == "gsa")
            .unwrap()
            .pulls;
        assert!(
            gsa_pulls > 1,
            "the quenched GSA must be pulled again after its first slice"
        );

        let prefix = replay
            .iter()
            .zip(trace[start..].iter())
            .take_while(|(expected, observed)| expected == observed)
            .count();
        assert!(prefix >= DIM, "the first quenched epoch must replay");
        assert!(prefix < replay.len(), "another arm must intervene");
        assert_ne!(prefix % DIM, 0, "the strategy must be partial");
        let continuations = trace[start + prefix..]
            .windows(replay.len() - prefix)
            .filter(|positions| *positions == &replay[prefix..])
            .count();
        assert_eq!(
            continuations, 1,
            "replica={replica}, prefix={prefix}, gsa_pulls={gsa_pulls}: the unfinished strategy must resume"
        );
    }
}
