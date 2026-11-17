//! General stall handling remains bounded without structure-specific growth.

use anneal_core::methods::cluster_hopping::{Config, HopStop, Ledger, run};
use ndarray::{Array1, ArrayView1};
use rand::{SeedableRng, rngs::StdRng};

fn line(n: usize) -> Array1<f64> {
    let mut x = Array1::zeros(3 * n);
    for i in 0..n {
        x[3 * i] = 1.3 * i as f64;
    }
    x
}

#[test]
fn a_flat_stall_obeys_the_general_hop_cap() {
    let mut cfg = Config::recommended(13);
    cfg.replicas = 1;
    cfg.max_hops = Some(6);
    let mut seen_budgets = Vec::new();
    let mut relax = |ledger: &mut Ledger, x: ArrayView1<f64>, _steps: usize| {
        seen_budgets.push(ledger.budget());
        let _ = ledger.charge();
        (0.0, x.to_owned())
    };
    let mut ledger = Ledger::new(1000);
    let mut rng = StdRng::seed_from_u64(17);
    let outcome =
        run(&cfg, line(13).view(), &mut ledger, &mut relax, &mut rng).expect("finite start");
    assert_eq!(outcome.stop, Some(HopStop::MaxHops));
    assert_eq!(outcome.hops, 6);
    assert!(seen_budgets.iter().all(|budget| *budget == 1000));
}

#[test]
fn a_stalled_walk_cannot_run_past_a_dry_ledger() {
    let mut cfg = Config::recommended(13);
    cfg.replicas = 1;
    cfg.max_hops = Some(100);
    let mut relax = |ledger: &mut Ledger, x: ArrayView1<f64>, _steps: usize| {
        while ledger.charge() {}
        (0.0, x.to_owned())
    };
    let mut ledger = Ledger::new(5);
    let mut rng = StdRng::seed_from_u64(18);
    let outcome =
        run(&cfg, line(13).view(), &mut ledger, &mut relax, &mut rng).expect("finite start");
    assert_eq!(outcome.stop, Some(HopStop::Budget));
    assert_eq!(outcome.charged, 5);
}
