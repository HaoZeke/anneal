//! GOAT's ladder of self-regulating walkers on the cluster hop.

use anneal_core::goat::{TEMPERATURE_FACTOR, ladder_temperatures};
use anneal_core::methods::cluster_hopping::{
    Config, EnergyUnit, LadderMode, Ledger, MoveLibrary, run,
};
use anneal_core::methods::cluster_search::search;
use anneal_core::methods::ensemble::{EnsembleConfig, HistoryMode};
use anneal_core::methods::minima_hopping::HistoryMembership;
use anneal_core::potentials::PairPotential;
use ndarray::{Array1, ArrayView1, array};
use rand::SeedableRng;
use rand::rngs::StdRng;

/// One chain whose hops are a random displacement, a screen and a full
/// relaxation and nothing else, with every trial its own basin.
fn bare() -> Config {
    let mut cfg = Config::for_cluster(3);
    cfg.move_library = MoveLibrary::WalesDoye;
    cfg.angular_moves = false;
    cfg.return_screen = false;
    cfg.bayes_screen = false;
    cfg.calibrated_screen = false;
    cfg.screen_margin = f64::INFINITY;
    cfg.screen_steps = 1;
    cfg.relax_steps = 1;
    cfg.polish_records = 0;
    cfg.orbit_complete_on_new = false;
    cfg.surfaces.clear();
    cfg.merge_radius = 1e-9;
    cfg.max_hops = Some(12);
    cfg
}

fn triangle() -> Array1<f64> {
    array![-1.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0, 0.0]
}

#[test]
fn the_ladder_sits_where_each_step_is_accepted_half_the_time() {
    for scale in [1.0, 0.0104, 4.184] {
        let cfg = Config::with_scales(13, 1.0, scale).with_goat_ladder();
        let t = ladder_temperatures(scale);
        assert_eq!(cfg.replicas, 4);
        assert_eq!(cfg.ladder_mode, LadderMode::Independent);
        assert!(cfg.self_regulating_temperature);
        assert!((cfg.temperature - t[0]).abs() < 1e-15 * scale);
        assert!((cfg.temperature * cfg.ladder_top - t[3]).abs() < 1e-12 * scale);
        assert!(((-0.5 * scale / cfg.temperature).exp() - 0.5).abs() < 1e-12);
    }
    let water = Config::for_tip4p(4).with_goat_ladder();
    assert!(
        !water.minima_hopping,
        "the ladder needs the Metropolis rule"
    );
}

#[test]
fn a_tip4p_ladder_reads_its_kcal_steps_in_kilojoules() {
    let water = Config::for_tip4p(4);
    assert_eq!(water.energy_unit, EnergyUnit::KilojoulePerMole);
    assert_eq!(water.kcal_per_mol(), 4.184);
    let ladder = water.with_goat_ladder();
    // Half a kcal/mol is 2.092 kJ/mol, taken half the time at the bottom.
    assert!(((-2.092 / ladder.temperature).exp() - 0.5).abs() < 1e-12);
    let top = ladder.temperature * ladder.ladder_top;
    assert!(((-4.0 * 4.184 / top).exp() - 0.5).abs() < 1e-12);
    let reduced = Config::for_cluster(13);
    assert_eq!(reduced.energy_unit, EnergyUnit::Reduced);
    assert_eq!(reduced.kcal_per_mol(), reduced.energy_scale);
}

#[test]
fn rejections_between_distinct_minima_heat_the_walker() {
    let mut cfg = bare();
    cfg.self_regulating_temperature = true;
    let start = triangle();
    // The start is the only low state, so every trial is a rise the
    // base temperature refuses.
    let mut relax = |ledger: &mut Ledger, x: ArrayView1<f64>, _: usize| {
        let _ = ledger.charge();
        let energy = if x == start.view() { 0.0 } else { 1.0e3 };
        (energy, x.to_owned())
    };
    let mut rng = StdRng::seed_from_u64(7);
    let mut ledger = Ledger::new(10_000);
    let out = run(&cfg, start.view(), &mut ledger, &mut relax, &mut rng).expect("finite start");
    assert_eq!(out.accepted, 0);
    assert_eq!(out.temperature_factors.len(), 1);
    let expected = TEMPERATURE_FACTOR.powi(out.hops as i32);
    assert!(
        (out.temperature_factors[0] - expected).abs() < 1e-12 * expected,
        "{} hops left the factor at {}",
        out.hops,
        out.temperature_factors[0]
    );
}

#[test]
fn acceptances_keep_the_walker_at_its_base() {
    let mut cfg = bare();
    cfg.self_regulating_temperature = true;
    let mut depth = 0.0;
    // Every trial is lower than the last, so every hop is accepted.
    let mut relax = |ledger: &mut Ledger, x: ArrayView1<f64>, _: usize| {
        let _ = ledger.charge();
        depth -= 1.0;
        (depth, x.to_owned())
    };
    let mut rng = StdRng::seed_from_u64(7);
    let mut ledger = Ledger::new(10_000);
    let out =
        run(&cfg, triangle().view(), &mut ledger, &mut relax, &mut rng).expect("finite start");
    assert!(out.accepted > 0);
    assert_eq!(out.temperature_factors, vec![1.0]);
}

#[test]
fn the_flag_off_reports_no_factors() {
    let cfg = bare();
    let mut relax = |ledger: &mut Ledger, x: ArrayView1<f64>, _: usize| {
        let _ = ledger.charge();
        (0.0, x.to_owned())
    };
    let mut rng = StdRng::seed_from_u64(7);
    let mut ledger = Ledger::new(10_000);
    let out =
        run(&cfg, triangle().view(), &mut ledger, &mut relax, &mut rng).expect("finite start");
    assert!(out.temperature_factors.is_empty());
}

#[test]
fn a_ladder_search_reports_four_rungs_at_or_above_their_bases() {
    let pot = PairPotential::lennard_jones(13);
    let cfg = Config::for_cluster(13).with_goat_ladder();
    let mut ledger = Ledger::new(20_000);
    let (out, _) = search(&pot, &cfg, &mut ledger, 3).expect("finite start");
    let bases = ladder_temperatures(1.0);
    assert_eq!(out.rungs.len(), 4);
    for (k, (t, _, _)) in out.rungs.iter().enumerate() {
        assert!((t - bases[k]).abs() < 1e-12, "rung {k} at {t}");
    }
    assert_eq!(out.temperature_factors.len(), 4);
    for f in &out.temperature_factors {
        let level = f.ln() / TEMPERATURE_FACTOR.ln();
        assert!(*f >= 1.0 && (level - level.round()).abs() < 1e-9);
    }
    assert!(out.best < -44.0);
}

#[test]
fn an_ensemble_refuses_the_ladder() {
    let ens = EnsembleConfig {
        replicas: 4,
        budget: 40_000,
        history: HistoryMode::Shared,
        membership: HistoryMembership::Accepted,
        shared_bias: None,
        gossip: None,
        two_choice_stall: None,
        checkpoint_interval: 1_000,
        target: None,
    };
    let chain = Config::communicating(13);
    assert!(ens.validate(&chain).is_ok());
    let ladder = chain.with_goat_ladder();
    assert!(ens.validate(&ladder).is_err());
}
