use std::sync::atomic::{AtomicUsize, Ordering};

use anneal_core::accept::Metropolis;
use anneal_core::cool::LogCool;
use anneal_core::history::State;
use anneal_core::movekernel::MoveKernel;
use anneal_core::neigh::Neighborhood;
use anneal_core::sampler::Sampler;
use anneal_core::variant::SaVariant;
use eindir_core::{Bounds, FPair, Objective};
use ndarray::{Array1, ArrayView1, array};
use rand::Rng;
use rand::SeedableRng;

static EVALUATIONS: AtomicUsize = AtomicUsize::new(0);

struct CountingObjective {
    bounds: Bounds<f64>,
}

impl CountingObjective {
    fn new() -> Self {
        Self {
            bounds: Bounds::new(array![0.0], array![1.0], 0.0),
        }
    }
}

impl Objective<f64> for CountingObjective {
    fn dim(&self) -> usize {
        1
    }

    fn bounds(&self) -> &Bounds<f64> {
        &self.bounds
    }

    fn eval(&self, x: ArrayView1<f64>) -> f64 {
        EVALUATIONS.fetch_add(1, Ordering::SeqCst);
        x[0] * x[0]
    }
}

struct UnitInterval;

impl Neighborhood<f64> for UnitInterval {
    fn contains(&self, i: ArrayView1<f64>, j: ArrayView1<f64>) -> bool {
        i.len() == 1 && j.len() == 1 && (0.0..=1.0).contains(&i[0]) && (0.0..=1.0).contains(&j[0])
    }
}

struct EscapingMove;

impl MoveKernel<f64> for EscapingMove {
    fn propose<R: Rng>(&self, _i: ArrayView1<f64>, _t: f64, _rng: &mut R) -> Array1<f64> {
        array![2.0]
    }
}

#[test]
fn proposal_outside_neighborhood_is_rejected_without_evaluation() {
    EVALUATIONS.store(0, Ordering::SeqCst);
    let sampler = SaVariant::unchecked(
        CountingObjective::new(),
        LogCool::new(1.0, 2.0),
        UnitInterval,
        EscapingMove,
        Metropolis,
    );
    let pair = FPair {
        pos: array![0.5],
        val: 0.25,
    };
    let mut state = State {
        cur: pair.clone(),
        best: pair,
    };
    let mut rng = rand::rngs::StdRng::seed_from_u64(9);

    assert!(!sampler.step(&mut state, 0, &mut rng));
    assert_eq!(EVALUATIONS.load(Ordering::SeqCst), 0);
    assert_eq!(state.cur.pos, array![0.5]);
    assert_eq!(state.best.pos, array![0.5]);
}
