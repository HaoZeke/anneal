use anneal_core::methods::cluster_hopping::{Config, Ledger, Outcome, run};
use ndarray::{ArrayView1, array};
use rand::{SeedableRng, rngs::StdRng};

fn one_basin_run() -> (Config, Outcome) {
    let mut config = Config::for_cluster(2);
    config.minima_hopping = true;
    config.max_hops = Some(8);
    config.angular_moves = false;
    config.screen_steps = 1;
    config.relax_steps = 2;
    config.bias_height = 0.0;
    let minimum = array![-0.6, 0.0, 0.0, 0.6, 0.0, 0.0];
    let mut ledger = Ledger::new(1_000);
    let mut rng = StdRng::seed_from_u64(37);
    let mut relax = |ledger: &mut Ledger, _: ArrayView1<f64>, _: usize| {
        assert!(ledger.charge());
        (-1.0, minimum.clone())
    };
    let outcome = run(&config, minimum.view(), &mut ledger, &mut relax, &mut rng);
    (config, outcome)
}

#[test]
fn returning_to_the_occupied_basin_does_not_tighten_the_acceptance_threshold() {
    let (config, outcome) = one_basin_run();
    assert_eq!(outcome.hops, 8);
    assert_eq!(outcome.visit_counts, (8, 0, 0));
    assert_eq!(outcome.escape_threshold, config.temperature);
    assert!(outcome.escape_scale > 1.0);
}

#[test]
fn returning_to_the_occupied_basin_is_not_an_adopted_escape() {
    let (_, outcome) = one_basin_run();
    assert_eq!(outcome.hops, 8);
    assert_eq!(outcome.visit_counts, (8, 0, 0));
    assert_eq!(outcome.accepted, 0);
    assert_eq!(outcome.best, -1.0);
    assert_eq!(outcome.final_energy, -1.0);
}
