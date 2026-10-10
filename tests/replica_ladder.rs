//! Every rung of a replica ladder hops at its own temperature.
//!
//! The hot end of a ladder is there to take the rises a cold chain declines,
//! and the swap brings what it finds down to the cold end. A ladder whose
//! temperatures reached only the swap test would adopt what a flat ladder
//! adopts. Counted here as adopted steps that climb more than one well depth,
//! on LJ13 at a temperature where a single chain seldom climbs that far.

use anneal_core::methods::cluster_hopping::{Config, Ledger, Outcome, optimize};
use anneal_core::methods::warm_lbfgs::WarmLbfgs;
use anneal_core::potentials::PairPotential;
use ndarray::ArrayView1;

const BUDGET: usize = 30_000;
const SEEDS: u64 = 8;

fn lj_run(cfg: &Config, seed: u64) -> Outcome {
    let pot = PairPotential::lennard_jones(cfg.n_points);
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
    let mut ledger = Ledger::new(BUDGET);
    optimize(cfg, &mut ledger, &mut relax, seed)
}

/// Adopted steps that climbed more than one well depth, over the seeds.
fn large_rises(base: &Config, ladder_top: f64) -> usize {
    let mut cfg = base.clone();
    cfg.replicas = 2;
    cfg.ladder_top = ladder_top;
    (0..SEEDS)
        .map(|seed| {
            lj_run(&cfg, seed)
                .accepted_transitions
                .iter()
                .filter(|t| t.adopted && t.to_energy - t.from_energy > 1.0)
                .count()
        })
        .sum()
}

fn assert_the_hot_rung_climbs(base: &Config) {
    let flat = large_rises(base, 1.0);
    let hot = large_rises(base, 10.0);
    assert!(
        hot >= 3 * flat.max(10),
        "a rung ten times hotter adopted {hot} rises above a well depth \
         against {flat} on a flat ladder"
    );
}

#[test]
fn a_hot_rung_adopts_more_large_rises_than_a_flat_ladder() {
    let mut cfg = Config::recommended(13);
    assert!(!cfg.budget_window && !cfg.statistical_temperature);
    cfg.temperature = 0.2;
    assert_the_hot_rung_climbs(&cfg);
}

/// The law sets the coldest rung's temperature and the hot rung hops at its
/// ratio times it, so the ladder survives the budget window.
#[test]
fn the_ladder_multiplies_the_budget_window_temperature() {
    let cfg = Config::derived(13);
    assert!(cfg.budget_window);
    assert_the_hot_rung_climbs(&cfg);
}

/// The clamped estimate is the coldest rung's temperature and the hot rung
/// hops at its ratio times it.
#[test]
fn the_ladder_multiplies_the_statistical_temperature() {
    let mut cfg = Config::recommended(13);
    cfg.temperature = 0.2;
    cfg.statistical_temperature = true;
    assert_the_hot_rung_climbs(&cfg);
}
