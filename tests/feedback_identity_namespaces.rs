use anneal_core::methods::cluster_hopping::{
    ChainCheckpoint, CheckpointAction, Config, Ledger, run_with_history_at_checkpoints,
};
use anneal_core::methods::minima_hopping::{HistoryHook, HistoryReport};
use ndarray::{Array1, ArrayView1, array};
use rand::SeedableRng;
use rand::rngs::StdRng;

fn energy_gradient(state: ArrayView1<f64>, minima: &[Array1<f64>; 3]) -> (f64, Array1<f64>) {
    minima
        .iter()
        .zip([-1.0, -0.5, -2.0])
        .map(|(minimum, offset)| {
            let displacement = &state - minimum;
            (offset + displacement.dot(&displacement), 2.0 * displacement)
        })
        .min_by(|left, right| left.0.total_cmp(&right.0))
        .unwrap()
}

struct PartlyAvailableHistory {
    minima: [Array1<f64>; 3],
    observations: Vec<usize>,
    accepted: Vec<usize>,
}

impl HistoryHook for PartlyAvailableHistory {
    fn observe(
        &mut self,
        energy: f64,
        state: ArrayView1<f64>,
        gradient: ArrayView1<f64>,
    ) -> Option<HistoryReport> {
        let (expected_energy, expected_gradient) = energy_gradient(state, &self.minima);
        assert_eq!(energy, expected_energy);
        assert_eq!(gradient, expected_gradient.view());
        assert!(gradient.iter().all(|component| *component == 0.0));
        let local_order = self
            .minima
            .iter()
            .position(|minimum| minimum.view() == state)
            .expect("every observation must be one of the exact fixture minima");
        self.observations.push(local_order);
        match local_order {
            0 => Some(HistoryReport {
                minimum: 100,
                is_new: true,
                visits: 0,
                observed_visits: 1,
                first_observation: true,
            }),
            1 => Some(HistoryReport {
                minimum: 2,
                is_new: false,
                visits: 8,
                observed_visits: 8,
                first_observation: false,
            }),
            2 => None,
            _ => unreachable!(),
        }
    }

    fn mark_accepted(&mut self, minimum: usize) {
        self.accepted.push(minimum);
    }

    fn cost(&self) -> (usize, usize, f64) {
        (self.observations.len(), 0, 0.0)
    }
}

#[test]
fn a_shared_numeric_id_does_not_make_a_distinct_local_fallback_basin_known() {
    let minima = [
        array![-1.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0, 0.0],
        array![-1.5, 0.0, 0.0, 1.5, 0.0, 0.0, 0.0, 2.5, 0.0],
        array![-2.0, 0.0, 0.0, 2.0, 0.0, 0.0, 0.0, 4.0, 0.0],
    ];
    let mut cfg = Config::for_cluster(3);
    cfg.minima_hopping = true;
    cfg.replicas = 1;
    cfg.max_hops = Some(2);
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

    let mut relax_calls = 0;
    let mut gradient_calls = 0;
    let mut relax = |ledger: &mut Ledger, _state: ArrayView1<f64>, steps: usize| {
        assert_eq!(steps, 1);
        assert!(ledger.charge());
        relax_calls += 1;
        let minimum = match relax_calls {
            1 => &minima[0],
            2 | 3 => &minima[1],
            4 | 5 => &minima[2],
            _ => panic!("the two-hop fixture has one initial and two screen/full quenches"),
        };
        (energy_gradient(minimum.view(), &minima).0, minimum.clone())
    };
    let mut gradient = |ledger: &mut Ledger, state: ArrayView1<f64>| {
        assert!(ledger.charge());
        gradient_calls += 1;
        Some(energy_gradient(state, &minima).1)
    };
    let mut history = PartlyAvailableHistory {
        minima: minima.clone(),
        observations: Vec::new(),
        accepted: Vec::new(),
    };
    let mut checkpoint_states = Vec::new();
    let mut checkpoint = |snapshot: ChainCheckpoint<'_>| {
        checkpoint_states.push((snapshot.hops(), snapshot.current_state().to_owned()));
        CheckpointAction::Continue
    };
    let mut ledger = Ledger::new(32);
    let mut rng = StdRng::seed_from_u64(17);
    let out = run_with_history_at_checkpoints(
        &cfg,
        minima[0].view(),
        &mut ledger,
        &mut relax,
        Some(&mut gradient),
        None,
        Some(&mut history),
        &mut rng,
        1,
        &mut checkpoint,
    );

    assert_eq!(out.hops, 2);
    assert_eq!(history.observations, vec![0, 1, 2]);
    assert_eq!(history.accepted, vec![100]);
    assert_eq!((relax_calls, gradient_calls), (5, 3));
    assert_eq!(ledger.spent(), relax_calls + gradient_calls);
    assert_eq!(out.charged, ledger.spent());
    assert!(ledger.spent() < ledger.budget());
    let (_, after_shared) = checkpoint_states
        .iter()
        .find(|(hop, _)| *hop == 1)
        .expect("the checkpoint must observe the occupied state after the shared candidate");
    assert_eq!(after_shared, &minima[0]);
    assert_eq!(out.accepted_transitions.len(), 1);
    let fallback = out
        .accepted_transitions
        .iter()
        .find(|transition| transition.to_state == minima[2])
        .expect("the local fallback must pass through the actual hopping decision");
    assert!(fallback.validated && fallback.adopted);
    assert_eq!(fallback.from_state, minima[0]);
    assert_eq!(fallback.to_gradient, Some(Array1::<f64>::zeros(9)));
    assert_eq!(out.final_state.as_ref(), Some(&minima[2]));
    assert_eq!(
        out.visit_counts,
        (0, 1, 1),
        "shared basin 2 is Known; distinct local basin 2 is a discovery"
    );
    let expected_escape = 1.05 * (1.0 + 0.1 * 7.0_f64.ln()) / 1.05;
    assert!((out.escape_scale - expected_escape).abs() < 1e-12);
}
