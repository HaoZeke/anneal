use anneal_core::methods::cluster_hopping::{
    ChainCheckpoint, CheckpointAction, Config, Ledger, run_with_history_at_checkpoints,
};
use anneal_core::methods::minima_hopping::{HistoryHook, HistoryReport};
use ndarray::{Array1, ArrayView1, array};
use rand::SeedableRng;
use rand::rngs::StdRng;
use std::sync::atomic::{AtomicBool, AtomicUsize, Ordering};

fn energy_gradient(state: ArrayView1<f64>, a: &Array1<f64>, b: &Array1<f64>) -> (f64, Array1<f64>) {
    let da = &state - a;
    let db = &state - b;
    let ea = da.dot(&da) - 1.0;
    let eb = db.dot(&db) - 2.0;
    if ea <= eb {
        (ea, 2.0 * da)
    } else {
        (eb, 2.0 * db)
    }
}

struct ScriptedHistory {
    a: Array1<f64>,
    b: Array1<f64>,
    observations: Vec<usize>,
    accepted: Vec<usize>,
    a_observations: usize,
}

impl HistoryHook for ScriptedHistory {
    fn observe(
        &mut self,
        energy: f64,
        state: ArrayView1<f64>,
        gradient: ArrayView1<f64>,
    ) -> Option<HistoryReport> {
        let (expected_energy, expected_gradient) = energy_gradient(state, &self.a, &self.b);
        assert_eq!(energy, expected_energy);
        assert_eq!(gradient, expected_gradient.view());
        assert!(gradient.iter().all(|value| *value == 0.0));
        let (minimum, visits, observed_visits) = if state == self.a.view() {
            self.a_observations += 1;
            let (visits, observed) = match self.a_observations {
                1 => (0, 1),
                2 => (2, 2),
                3 => (8, 8),
                _ => panic!("the three-hop fixture has exactly three observations of A"),
            };
            (7, visits, observed)
        } else {
            assert_eq!(state, self.b.view());
            (9, 0, 1)
        };
        self.observations.push(minimum);
        Some(HistoryReport {
            minimum,
            is_new: visits == 0,
            visits,
            observed_visits,
            first_observation: observed_visits == 1,
        })
    }

    fn mark_accepted(&mut self, minimum: usize) {
        self.accepted.push(minimum);
    }

    fn cost(&self) -> (usize, usize, f64) {
        (self.observations.len(), 0, 0.0)
    }
}

#[test]
fn external_adoption_replaces_the_occupied_shared_identity_before_a_return() {
    let a = array![-1.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0, 0.0];
    let b = array![-1.5, 0.0, 0.0, 1.5, 0.0, 0.0, 0.0, 2.5, 0.0];
    let mut cfg = Config::for_cluster(3);
    cfg.minima_hopping = true;
    cfg.replicas = 1;
    cfg.max_hops = Some(3);
    cfg.displacement_only = true;
    cfg.md_escape = false;
    cfg.return_screen = false;
    cfg.bayes_screen = false;
    cfg.screen_margin = 10.0;
    cfg.screen_steps = 1;
    cfg.relax_steps = 1;
    cfg.polish_records = 0;
    cfg.merge_radius = 0.1;
    cfg.bias_height = 0.0;
    cfg.shared_deposits = 0;
    cfg.restart_on_stall = false;
    cfg.escape_on_stall = false;
    cfg.trail_on_stall = false;
    cfg.jump_on_stall = false;
    cfg.path_on_stall = false;
    cfg.symmetrise_on_stall = false;
    cfg.point_symmetrise_on_new = false;
    cfg.orbit_complete_on_new = false;
    cfg.superbasin_escape = false;

    let external_pending = AtomicBool::new(false);
    let relax_calls = AtomicUsize::new(0);
    let gradient_calls = AtomicUsize::new(0);
    let mut relax = |ledger: &mut Ledger, _state: ArrayView1<f64>, steps: usize| {
        assert_eq!(steps, 1, "the fixture permits only one-step quenches");
        assert!(ledger.charge());
        relax_calls.fetch_add(1, Ordering::Relaxed);
        let destination = if external_pending.swap(false, Ordering::Relaxed) {
            &b
        } else {
            &a
        };
        (
            energy_gradient(destination.view(), &a, &b).0,
            destination.clone(),
        )
    };
    let mut gradient = |ledger: &mut Ledger, state: ArrayView1<f64>| {
        assert!(ledger.charge());
        gradient_calls.fetch_add(1, Ordering::Relaxed);
        Some(energy_gradient(state, &a, &b).1)
    };
    let mut offered = false;
    let mut checkpoint = |snapshot: ChainCheckpoint<'_>| {
        if offered {
            return CheckpointAction::Continue;
        }
        assert_eq!(snapshot.hops(), 1);
        assert_eq!(snapshot.current_state(), a.view());
        offered = true;
        external_pending.store(true, Ordering::Relaxed);
        CheckpointAction::ExternalAdopt {
            state: b.clone(),
            action: "history-identity-adoption".to_owned(),
            external_calls: 0,
        }
    };
    let mut history = ScriptedHistory {
        a: a.clone(),
        b: b.clone(),
        observations: Vec::new(),
        accepted: Vec::new(),
        a_observations: 0,
    };
    let mut ledger = Ledger::new(32);
    let mut rng = StdRng::seed_from_u64(0x51a1e);
    let out = run_with_history_at_checkpoints(
        &cfg,
        a.view(),
        &mut ledger,
        &mut relax,
        Some(&mut gradient),
        None,
        Some(&mut history),
        &mut rng,
        1,
        &mut checkpoint,
    );

    assert!(offered);
    assert!(!external_pending.load(Ordering::Relaxed));
    assert_eq!(out.hops, 3);
    let adoption = out
        .accepted_transitions
        .iter()
        .find(|transition| transition.action == "history-identity-adoption")
        .expect("the checkpoint action must enter the actual adoption path");
    assert!(adoption.adopted && adoption.validated);
    assert_eq!(adoption.from_state, a);
    assert_eq!(adoption.to_state, b);
    assert_eq!(adoption.to_gradient, Some(Array1::zeros(b.len())));
    assert_eq!(out.final_state.as_ref(), Some(&b));
    assert_eq!(history.a_observations, 3);
    assert_eq!(history.accepted.first(), Some(&7));
    assert_eq!(
        (
            relax_calls.load(Ordering::Relaxed),
            gradient_calls.load(Ordering::Relaxed)
        ),
        (6, 4)
    );
    assert_eq!(
        ledger.spent(),
        relax_calls.load(Ordering::Relaxed) + gradient_calls.load(Ordering::Relaxed)
    );
    assert_eq!(out.charged, ledger.spent());
    assert!(ledger.spent() <= ledger.budget());

    assert_eq!(
        out.visit_counts,
        (1, 1, 1),
        "A self-return, A-to-B adoption, and B-to-A return are Same, New, and Known"
    );
    let expected_escape = 1.05 * (1.0 + 0.1 * 7.0_f64.ln());
    assert!((out.escape_scale - expected_escape).abs() < 1e-12);
}
