//! An overlapping rigid quench must not become the reported incumbent.
//!
//! The public search is `cluster_search::search_from`. The objective is a
//! harmonic well on a separated tetrahedron, and a much lower well wherever
//! two centres sit inside half a contact. A local orientation move leaves
//! the centres where they are. The stalled shell turn does not, and the
//! lower well is what that turn quenches into.

use std::sync::atomic::{AtomicUsize, Ordering};

use anneal_core::methods::cluster_hopping::{Config, Ledger, MoveLibrary};
use anneal_core::methods::cluster_search::search_from;
use eindir_core::Objective;
use eindir_core::bounds::Bounds;
use eindir_core::gradient::{DifferentiableObjective, Gradient};
use ndarray::{Array1, ArrayView1};

struct OverlapTrap {
    start: Array1<f64>,
    n: usize,
    gap: f64,
    bounds: Bounds<f64>,
    overlaps: AtomicUsize,
}

impl OverlapTrap {
    fn new(start: Array1<f64>, n: usize, gap: f64) -> Self {
        let extent = 20.0;
        Self {
            start,
            n,
            gap,
            bounds: Bounds::new(
                Array1::from_elem(6 * n, -extent),
                Array1::from_elem(6 * n, extent),
                0.0,
            ),
            overlaps: AtomicUsize::new(0),
        }
    }

    fn centres_clash(&self, x: ArrayView1<f64>) -> bool {
        let gap2 = self.gap * self.gap;
        let n = self.n;
        if x.len() < 3 * n {
            return false;
        }
        for i in 0..n {
            for j in (i + 1)..n {
                let mut distance = 0.0;
                for k in 0..3 {
                    let delta = x[3 * i + k] - x[3 * j + k];
                    distance += delta * delta;
                }
                if distance < gap2 {
                    return true;
                }
            }
        }
        false
    }

    fn evaluate(&self, x: ArrayView1<f64>) -> (f64, Array1<f64>) {
        if self.centres_clash(x) {
            self.overlaps.fetch_add(1, Ordering::Relaxed);
            return (-40.0, Array1::zeros(x.len()));
        }
        let mut energy = -1.0;
        let mut gradient = Array1::zeros(x.len());
        for (index, (value, origin)) in x.iter().zip(self.start.iter()).enumerate() {
            let delta = value - origin;
            energy += 0.5 * delta * delta;
            gradient[index] = delta;
        }
        (energy, gradient)
    }
}

impl Objective<f64> for OverlapTrap {
    fn dim(&self) -> usize {
        6 * self.n
    }

    fn bounds(&self) -> &Bounds<f64> {
        &self.bounds
    }

    fn eval(&self, x: ArrayView1<f64>) -> f64 {
        self.evaluate(x).0
    }
}

impl Gradient<f64> for OverlapTrap {
    fn grad(&self, x: ArrayView1<f64>) -> Array1<f64> {
        self.evaluate(x).1
    }

    fn dim(&self) -> usize {
        6 * self.n
    }
}

impl DifferentiableObjective<f64> for OverlapTrap {
    fn value_and_gradient(&self, x: ArrayView1<f64>) -> (f64, Array1<f64>) {
        self.evaluate(x)
    }
}

fn tetra_centres() -> Array1<f64> {
    let raw = [
        [1.0, 1.0, 1.0],
        [1.0, -1.0, -1.0],
        [-1.0, 1.0, -1.0],
        [-1.0, -1.0, 1.0],
    ];
    let edge: f64 = raw[0]
        .iter()
        .zip(raw[1])
        .map(|(a, b)| {
            let delta = a - b;
            delta * delta
        })
        .sum();
    let scale = 2.8 / edge.sqrt();
    let mut state = Array1::zeros(24);
    for (body, point) in raw.iter().enumerate() {
        for (offset, component) in point.iter().enumerate() {
            state[3 * body + offset] = component * scale;
        }
    }
    state
}

#[test]
fn an_overlapping_rigid_candidate_does_not_become_the_incumbent() {
    let n = 4;
    let mut cfg = Config::recommended_tip4p(n);
    cfg.restart_patience = 1;
    cfg.relax_steps = 0;
    cfg.screen_steps = 0;
    cfg.move_library = MoveLibrary::RigidBody {
        n_molecules: n,
        translate_step: 1.0e-6,
        rotate_step: std::f64::consts::PI / 3.0,
        local: true,
        reverse_cycles: false,
        relocate: false,
    };
    cfg.rigid_body_geometry = None;
    let start = tetra_centres();
    let trap = OverlapTrap::new(start.clone(), n, 0.5 * cfg.min_separation);
    let patience = cfg.restart_patience.saturating_mul(40);
    let mut ledger = Ledger::new(patience.saturating_mul(8));
    let (outcome, _) = search_from(&trap, &cfg, &mut ledger, start.view(), 11).expect("search");
    assert!(
        trap.overlaps.load(Ordering::Relaxed) > 0,
        "the search never evaluated an overlapping rigid body"
    );
    assert_eq!(outcome.best, -1.0);
    assert_eq!(ledger.best, -1.0);
    let held = outcome
        .best_state
        .expect("the separated cage stays the incumbent");
    assert!(
        !trap.centres_clash(held.view()),
        "overlapping centres are the reported incumbent"
    );
}
