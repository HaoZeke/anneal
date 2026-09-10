use anneal_core::descriptor_space::{DescriptorGeometry, universal_descriptor_space};
use anneal_core::methods::cluster_hopping::{
    ChainCheckpoint, CheckpointAction, Config, LadderMode, Ledger, run_with_history_at_checkpoints,
};
use anneal_core::methods::minima_hopping::{
    HistoryHook, HistoryMembership, MinimumHistory, SharedMinimumHistory,
};
use anneal_core::pes_exploration::{ExactStructureWitness, StructureContext};
use ndarray::{ArrayView1, array};
use rand::{SeedableRng, rngs::StdRng};
use std::sync::Mutex;
use std::sync::atomic::{AtomicUsize, Ordering};

struct SameCoordinates;

impl ExactStructureWitness for SameCoordinates {
    fn equivalent(&self, left: ArrayView1<f64>, right: ArrayView1<f64>) -> bool {
        left == right
    }
}

#[test]
fn each_rung_accounts_for_foreign_history_visits_in_its_own_bias() {
    let mut results = Vec::new();
    for membership in [HistoryMembership::Accepted, HistoryMembership::Observed] {
        let minimum = array![-0.6, 0.0, 0.0, 0.6, 0.0, 0.0];
        let mut cfg = Config::for_cluster(2);
        cfg.replicas = 2;
        cfg.ladder_mode = LadderMode::Independent;
        cfg.swap_period = 1;
        cfg.ladder_pilot = 1;
        cfg.max_hops = Some(4);
        cfg.minima_hopping = false;
        cfg.shared_deposits = 16;
        cfg.bias_height = 0.0;
        cfg.displacement_only = true;
        cfg.md_escape = false;
        cfg.screen_steps = 1;
        cfg.relax_steps = 1;
        cfg.screen_margin = f64::INFINITY;
        cfg.return_screen = false;
        cfg.bayes_screen = false;
        cfg.polish_records = 0;
        cfg.restart_on_stall = false;
        cfg.escape_on_stall = false;
        cfg.trail_on_stall = false;
        cfg.jump_on_stall = false;
        cfg.path_on_stall = false;
        cfg.symmetrise_on_stall = false;
        cfg.point_symmetrise_on_new = false;
        cfg.orbit_complete_on_new = false;
        cfg.superbasin_escape = false;

        let history = Mutex::new(MinimumHistory::new(cfg.record_gradient).unwrap());
        let geometry = DescriptorGeometry::finite(1.0).unwrap();
        let descriptor = universal_descriptor_space(geometry.clone());
        let context = StructureContext::new(
            Some(vec![1; 2]),
            Some(geometry),
            Some("rung-history-quadratic".into()),
        );
        let mut hook =
            SharedMinimumHistory::new(&history, &descriptor, context, &SameCoordinates, membership);
        let relax_calls = AtomicUsize::new(0);
        let gradient_calls = AtomicUsize::new(0);
        // Every quench reaches the exact minimum of ||x - minimum||^2 - 1.
        // Zero-height hills retain visit counters without changing acceptance.
        let mut relax = |ledger: &mut Ledger, _state: ArrayView1<f64>, steps: usize| {
            assert_eq!(steps, 1);
            assert!(ledger.charge());
            relax_calls.fetch_add(1, Ordering::Relaxed);
            (-1.0, minimum.clone())
        };
        let mut gradient = |ledger: &mut Ledger, state: ArrayView1<f64>| {
            assert!(ledger.charge());
            gradient_calls.fetch_add(1, Ordering::Relaxed);
            Some(2.0 * (&state - &minimum))
        };
        let mut checkpoints = Vec::new();
        let mut checkpoint = |snapshot: ChainCheckpoint<'_>| {
            assert_eq!(snapshot.current_state(), minimum.view());
            assert_eq!(snapshot.current_energy(), -1.0);
            assert!(
                snapshot
                    .current_gradient()
                    .expect("each active rung retains its paid certificate")
                    .iter()
                    .all(|value| *value == 0.0)
            );
            let bias = snapshot
                .bias()
                .expect("atomic checkpoints expose the rung bias");
            assert!(bias.n_basins() <= 1);
            checkpoints.push((
                snapshot.hops(),
                history.lock().unwrap().total_visits(),
                bias.index().visits(0),
            ));
            CheckpointAction::Continue
        };
        let mut ledger = Ledger::new(64);
        let mut rng = StdRng::seed_from_u64(0x7275_6e67);
        let outcome = run_with_history_at_checkpoints(
            &cfg,
            minimum.view(),
            &mut ledger,
            &mut relax,
            Some(&mut gradient),
            None,
            Some(&mut hook),
            &mut rng,
            1,
            &mut checkpoint,
        );

        assert_eq!(outcome.hops, 4);
        assert_eq!(outcome.rungs.len(), 2);
        assert_eq!(outcome.swaps_tried, 0);
        assert_eq!(outcome.swaps_accepted, 0);
        assert_eq!(outcome.accepted_transitions.len(), 4);
        assert!(
            outcome
                .accepted_transitions
                .iter()
                .all(|transition| transition.adopted && transition.validated)
        );
        let callbacks =
            relax_calls.load(Ordering::Relaxed) + gradient_calls.load(Ordering::Relaxed);
        assert_eq!(ledger.spent(), callbacks);
        assert_eq!(outcome.charged, callbacks);
        assert!(callbacks <= 64);
        assert_eq!((hook.cost().0, hook.cost().1), (6, 0));
        let archive = history.lock().unwrap();
        assert_eq!(archive.minimum_count(), 1);
        assert_eq!(archive.accepted_count(), 1);
        assert_eq!(archive.total_visits(), 6);
        assert_eq!(archive.accepted_visits(0), Some(6));
        results.push((membership, checkpoints, outcome.shared_deposits));
    }

    for (membership, checkpoints, shared_deposits) in results {
        // Checkpoints expose the destination rung after each slice. The hot
        // initialization has no own hill; its first hop owes the cold rung's
        // initialization and first hop. On returning, each rung also owes
        // visits made by the other rung while its own bias was parked.
        assert_eq!(
            checkpoints,
            vec![(1, 3, 0), (2, 4, 1), (3, 5, 3), (4, 6, 4)],
            "{membership:?}: each bias must retain its own history watermark"
        );
        assert_eq!(
            shared_deposits, 5,
            "{membership:?}: foreign deposits per hop are 0, 2, 2, and 1"
        );
    }
}
