//! Label metadata never opens a structure-specific side budget.

use anneal_core::methods::cluster_hopping::{Config, Ledger, run};
use ndarray::{ArrayView1, array};
use rand::SeedableRng;
use rand::rngs::StdRng;

fn quiet(n: usize) -> Config {
    let mut cfg = Config::for_cluster(n);
    cfg.replicas = 1;
    cfg.max_hops = Some(3);
    cfg.angular_moves = false;
    cfg.soft_perturb = false;
    cfg.cov_perturb = false;
    cfg.hmc = None;
    cfg.two_phase = None;
    cfg.surfaces.clear();
    cfg.return_screen = false;
    cfg.bayes_screen = false;
    cfg.screen_margin = f64::INFINITY;
    cfg.screen_steps = 2;
    cfg.relax_steps = 3;
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
    cfg
}

fn observed_budgets(labelled: bool) -> (Vec<usize>, usize) {
    const BUDGET: usize = 40;
    let mut cfg = quiet(4);
    if labelled {
        cfg.species = Some(vec![29, 29, 79, 79]);
    }
    let mut budgets = Vec::new();
    let mut relax = |ledger: &mut Ledger, x: ArrayView1<f64>, _steps: usize| {
        budgets.push(ledger.budget());
        let _ = ledger.charge();
        (0.0, x.to_owned())
    };
    let mut ledger = Ledger::new(BUDGET);
    let mut rng = StdRng::seed_from_u64(5);
    let start = array![
        1.0, 1.0, 1.0, 1.0, -1.0, -1.0, -1.0, 1.0, -1.0, -1.0, -1.0, 1.0
    ];
    let outcome = run(&cfg, start.view(), &mut ledger, &mut relax, &mut rng).expect("finite start");
    (budgets, outcome.charged)
}

#[test]
fn labels_use_only_the_callers_ledger() {
    let (budgets, charged) = observed_budgets(true);
    assert!(!budgets.is_empty());
    assert!(budgets.iter().all(|budget| *budget == 40));
    assert!(charged <= 40);
}

#[test]
fn one_species_and_labelled_walks_share_the_ledger_contract() {
    let (labelled, labelled_charged) = observed_budgets(true);
    let (plain, plain_charged) = observed_budgets(false);
    assert!(labelled.iter().chain(&plain).all(|budget| *budget == 40));
    assert!(labelled_charged <= 40);
    assert!(plain_charged <= 40);
}
