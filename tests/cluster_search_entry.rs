//! Caller-free Lennard-Jones `cluster_search` through `optimize()`.
//!
//! The Python surface takes a user energy and gradient. This helper is the
//! in-crate example that runs the same driver with the crate's own potential,
//! and reports `{best, best_energy, hops}` without a solved flag.

use anneal_core::methods::cluster_hopping::{Config, Ledger, MoveLibrary, optimize};
use anneal_core::methods::warm_lbfgs::WarmLbfgs;
use anneal_core::potentials::PairPotential;
use ndarray::{Array1, ArrayView1};

struct SearchReport {
    best: Array1<f64>,
    best_energy: f64,
    hops: usize,
}

fn cluster_search(n: usize, budget: usize, seed: u64, recommended: bool) -> SearchReport {
    let cfg = if recommended {
        Config::recommended(n)
    } else {
        Config::for_cluster(n)
    };
    let mut ledger = Ledger::new(budget);
    let pot = PairPotential::lennard_jones(n);
    let mut opt = WarmLbfgs::default();
    let mut relax = |led: &mut Ledger, x: ArrayView1<f64>, iters: usize| {
        opt.forget();
        let (f, xr, _) = opt.minimize_watched(
            x,
            iters,
            |v| {
                if !led.charge() {
                    return None;
                }
                Some(pot.value_and_gradient(v))
            },
            |_, _| true,
        );
        (f, xr)
    };
    let out = optimize(&cfg, &mut ledger, &mut relax, seed).expect("finite start");
    SearchReport {
        best: out.best_state.unwrap_or_else(|| Array1::zeros(3 * n)),
        best_energy: out.best,
        hops: out.hops,
    }
}

#[test]
fn cluster_search_reports_best_energy_and_hops() {
    let out = cluster_search(13, 8_000, 0, true);
    assert_eq!(out.best.len(), 3 * 13);
    assert!(
        out.best_energy.is_finite(),
        "best_energy {}",
        out.best_energy
    );
    assert!(out.hops > 0, "no hops were taken");
}

#[test]
fn cluster_search_for_cluster_also_hops() {
    let out = cluster_search(13, 4_000, 1, false);
    assert_eq!(out.best.len(), 3 * 13);
    assert!(out.best_energy.is_finite());
    assert!(out.hops > 0);
}

#[test]
fn ledger_exposes_the_budget() {
    let ledger = Ledger::new(400_000);
    assert_eq!(ledger.budget(), 400_000);
    assert_eq!(ledger.spent(), 0);
    assert_eq!(ledger.remaining(), 400_000);
}

#[test]
fn recommended_differs_from_for_cluster() {
    let rec = Config::recommended(38);
    let base = Config::for_cluster(38);
    assert_eq!(rec.n_points, 38);
    assert_eq!(base.n_points, 38);
    assert!(matches!(rec.move_library, MoveLibrary::LeanBurst));
    assert_eq!(rec.surfaces.len(), 1);
    assert!(rec.grow_shells);
    assert!(matches!(
        Config::recommended(75).move_library,
        MoveLibrary::LeanBurst
    ));
    assert!(matches!(
        Config::derived(38).move_library,
        MoveLibrary::LeanBurst
    ));
    assert!(rec.allocate_moves && rec.depth_reward && rec.tabu_on_stall);
    assert!(rec.trail_on_stall && rec.escape_on_stall && rec.restart_on_stall);
    assert!(rec.bayes_screen && base.bayes_screen);
    assert!(matches!(base.move_library, MoveLibrary::Atomic));
    assert!(!(base.allocate_moves || base.depth_reward || base.tabu_on_stall));
    assert!(!base.trail_on_stall && !base.escape_on_stall && !base.restart_on_stall);
}

#[test]
fn recommended_installs_the_same_surface_at_every_size() {
    for n in [38usize, 75, 97, 98, 104] {
        let rec = Config::recommended(n);
        assert!(rec.two_phase.is_none());
        assert_eq!(rec.surfaces.len(), 1, "n={n}");
        assert!(rec.grow_shells, "n={n}");
        assert_ne!(rec.surfaces.len(), Config::communicating(n).surfaces.len());
        assert!(rec.orbit_complete_on_new);
        assert_eq!(rec.relax_steps, Config::FOR_CLUSTER_RELAX_STEPS);
    }
}
