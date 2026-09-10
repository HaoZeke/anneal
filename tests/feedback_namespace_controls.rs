use anneal_core::methods::cluster_hopping::{
    ChainCheckpoint, CheckpointAction, Config, Ledger, Outcome, run_with_history_at_checkpoints,
};
use anneal_core::methods::minima_hopping::{EscapeFeedback, HistoryHook, HistoryReport, Visit};
use ndarray::{Array1, ArrayView1, array};
use rand::SeedableRng;
use rand::rngs::StdRng;

fn minima() -> [Array1<f64>; 2] {
    [
        array![-1.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0, 0.0],
        array![-1.5, 0.0, 0.0, 1.5, 0.0, 0.0, 0.0, 2.5, 0.0],
    ]
}

fn energy_gradient(state: ArrayView1<f64>, minima: &[Array1<f64>; 2]) -> (f64, Array1<f64>) {
    minima
        .iter()
        .zip([-1.0, -2.0])
        .map(|(minimum, offset)| {
            let displacement = &state - minimum;
            (offset + displacement.dot(&displacement), 2.0 * displacement)
        })
        .min_by(|left, right| left.0.total_cmp(&right.0))
        .unwrap()
}

struct SharedThenUnavailable {
    minima: [Array1<f64>; 2],
    observations: Vec<usize>,
    accepted: Vec<usize>,
}

impl HistoryHook for SharedThenUnavailable {
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
        let local = self
            .minima
            .iter()
            .position(|minimum| minimum.view() == state)
            .expect("the history receives an exact fixture minimum");
        self.observations.push(local);
        match self.observations.len() {
            1 => Some(HistoryReport {
                minimum: 100,
                is_new: true,
                visits: 0,
                observed_visits: 1,
                first_observation: true,
            }),
            2 => Some(HistoryReport {
                minimum: 9,
                is_new: false,
                visits: 8,
                observed_visits: 8,
                first_observation: false,
            }),
            3 => None,
            _ => panic!("the two-hop shared trace contains exactly three observations"),
        }
    }

    fn mark_accepted(&mut self, minimum: usize) {
        self.accepted.push(minimum);
    }

    fn cost(&self) -> (usize, usize, f64) {
        (self.observations.len(), 0, 0.0)
    }
}

fn run_trace(
    minima: &[Array1<f64>; 2],
    sequence: &[usize],
    history: Option<&mut SharedThenUnavailable>,
) -> (Outcome, Vec<(usize, Array1<f64>)>) {
    let hops = (sequence.len() - 1) / 2;
    let mut cfg = Config::for_cluster(3);
    cfg.minima_hopping = true;
    cfg.temperature = 10.0;
    cfg.replicas = 1;
    cfg.max_hops = Some(hops);
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
        let destination = sequence[relax_calls];
        relax_calls += 1;
        let minimum = &minima[destination];
        (energy_gradient(minimum.view(), minima).0, minimum.clone())
    };
    let mut gradient = |ledger: &mut Ledger, state: ArrayView1<f64>| {
        assert!(ledger.charge());
        gradient_calls += 1;
        Some(energy_gradient(state, minima).1)
    };
    let mut checkpoints = Vec::new();
    let mut checkpoint = |snapshot: ChainCheckpoint<'_>| {
        checkpoints.push((snapshot.hops(), snapshot.current_state().to_owned()));
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
        history,
        &mut rng,
        1,
        &mut checkpoint,
    );

    assert_eq!(out.hops, hops);
    assert_eq!(relax_calls, sequence.len());
    assert_eq!(gradient_calls, hops + 1);
    assert_eq!(ledger.spent(), relax_calls + gradient_calls);
    assert_eq!(out.charged, ledger.spent());
    assert!(ledger.spent() < ledger.budget());
    for transition in &out.accepted_transitions {
        assert!(transition.adopted && transition.validated);
        let (energy, gradient) = energy_gradient(transition.to_state.view(), minima);
        assert_eq!(transition.to_energy, energy);
        assert_eq!(transition.to_gradient.as_ref(), Some(&gradient));
    }
    (out, checkpoints)
}

#[test]
fn a_shared_backed_geometry_is_known_when_its_next_report_is_unavailable() {
    let minima = minima();
    let mut history = SharedThenUnavailable {
        minima: minima.clone(),
        observations: Vec::new(),
        accepted: Vec::new(),
    };
    let (out, checkpoints) = run_trace(&minima, &[0, 1, 1, 1, 1], Some(&mut history));

    assert_eq!(history.observations, vec![0, 1, 1]);
    assert_eq!(history.accepted, vec![100]);
    let (_, after_shared) = checkpoints.iter().find(|(hop, _)| *hop == 1).unwrap();
    assert_eq!(after_shared, &minima[0]);
    assert_eq!(out.accepted_transitions.len(), 1);
    assert_eq!(out.accepted_transitions[0].from_state, minima[0]);
    assert_eq!(out.accepted_transitions[0].to_state, minima[1]);
    assert_eq!(out.final_state.as_ref(), Some(&minima[1]));
    assert_eq!(out.visit_counts, (0, 2, 0));
    let expected_escape = 1.05 * (1.0 + 0.1 * 7.0_f64.ln()) * 1.05;
    assert!((out.escape_scale - expected_escape).abs() < 1e-12);
}

#[test]
fn local_only_hops_preserve_the_public_controllers_feedback_and_threshold() {
    let minima = minima();
    let (out, _) = run_trace(&minima, &[0, 1, 1, 1, 1, 0, 0], None);
    let mut expected = EscapeFeedback::new(1.0, 10.0);
    expected.register_initial(0);
    assert_eq!(expected.observe(Some(0), 1), Visit::New);
    assert!(expected.accept(-1.0));
    assert_eq!(expected.observe(Some(1), 1), Visit::Same);
    assert_eq!(expected.observe(Some(1), 0), Visit::Known);
    assert!(expected.accept(1.0));

    assert_eq!(expected.known_basins(), 2);
    assert_eq!(expected.visits(0), 2);
    assert_eq!(expected.visits(1), 2);
    assert_eq!(out.accepted_transitions.len(), 2);
    assert_eq!(out.final_state.as_ref(), Some(&minima[0]));
    assert_eq!(
        out.visit_counts,
        (expected.n_same, expected.n_known, expected.n_new)
    );
    assert_eq!(out.escape_scale.to_bits(), expected.escape().to_bits());
    assert_eq!(
        out.escape_threshold.to_bits(),
        expected.threshold().to_bits()
    );
}
