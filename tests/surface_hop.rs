// Copyright (c) anneal developers

use anneal_core::methods::cluster_hopping::{Config, Ledger, run};
use anneal_core::methods::warm_lbfgs::WarmLbfgs;
use anneal_core::packing::surface_hops;
use anneal_core::potentials::{SuttonChen, SuttonChenFamily};
use anneal_core::soap::CONTACT_CUTOFF;
use ndarray::{Array1, ArrayView1};
use rand::SeedableRng;
use rand::rngs::StdRng;

/// Points of the 9-6 shelf fixture.
const SHELF_POINTS: usize = 78;
/// Gradient tolerance of the fixture quenches.
const QUENCH_GTOL: f64 = 1e-6;
/// Most steps of a fixture quench.
const QUENCH_STEPS: usize = 5000;
/// Quenched energy of the shelf fixture in Cambridge units.
const SHELF_ENERGY: f64 = -19053.98767;
/// Quenched energy of the 9-6 N=78 global minimum in Cambridge units.
const GLOBAL_MINIMUM: f64 = -19054.397655;
/// Error allowed against [`SHELF_ENERGY`] and [`GLOBAL_MINIMUM`].
const ENERGY_TOLERANCE: f64 = 1e-4;
/// Bound on the count of the shelf's surface hops.
const MAX_CANDIDATES: usize = 40;
/// Lennard-Jones temperature that `Config::with_scales` multiplies by the energy scale.
const LJ_TEMPERATURE: f64 = 0.8;
/// Potential calls of each search. Enough for the opening quench, one
/// improvement, and the surface hops that improvement asks for.
const SEARCH_BUDGET: usize = 400;

fn structure(text: &str) -> Array1<f64> {
    text.lines()
        .skip(2)
        .flat_map(|line| line.split_whitespace().skip(1).take(3))
        .map(|value| value.parse::<f64>().unwrap())
        .collect()
}

/// The shelf two of every three missed 9-6 N=78 development runs ended on.
/// It has the global minimum's core and differs on the surface only. One
/// surface hop of its single four-coordinated atom quenches to the global
/// minimum, -19054.397655 in Cambridge units.
#[test]
fn one_surface_hop_takes_the_9_6_78_shelf_to_the_global_minimum() {
    let x = structure(include_str!("fixtures/sc96_78_shelf.xyz"));
    let n = SHELF_POINTS;
    let pot = SuttonChen::new(n, SuttonChenFamily::Sc9_6);
    let quench = |y: &Array1<f64>| {
        let mut opt = WarmLbfgs::default();
        opt.gtol = QUENCH_GTOL;
        let (energy, _, _) =
            opt.minimize(y.view(), QUENCH_STEPS, |v| Some(pot.value_and_gradient(v)));
        energy
    };
    let shelf = quench(&x);
    assert!((shelf - SHELF_ENERGY).abs() < ENERGY_TOLERANCE, "{shelf}");
    let length = std::f64::consts::FRAC_1_SQRT_2 / 2f64.powf(1.0 / 6.0);
    let candidates = surface_hops(x.view(), n, CONTACT_CUTOFF * length, 0.0);
    assert!(!candidates.is_empty() && candidates.len() < MAX_CANDIDATES);
    let lowest = candidates.iter().map(quench).fold(f64::INFINITY, f64::min);
    assert!(
        (lowest - GLOBAL_MINIMUM).abs() < ENERGY_TOLERANCE,
        "{lowest}"
    );
}

/// The legacy flag does not feed surface analysis back into the search.
#[test]
fn surface_hops_run_from_new_bests_only_when_asked() {
    let shelf = structure(include_str!("fixtures/sc96_78_shelf.xyz"));
    let n = SHELF_POINTS;
    let family = SuttonChenFamily::Sc12_6;
    let length = std::f64::consts::FRAC_1_SQRT_2 / 2f64.powf(1.0 / 6.0);
    let start = shelf.clone();
    let search = |on: bool| {
        let mut cfg = Config::with_scales(
            n,
            length,
            family.basin_hopping_temperature() / LJ_TEMPERATURE,
        );
        cfg.surface_hop_on_best = on;
        let mut ledger = Ledger::new(SEARCH_BUDGET);
        let mut calls = 0usize;
        let mut relax = |led: &mut Ledger, v: ArrayView1<f64>, _iters: usize| {
            let _ = v;
            if !led.charge() {
                return (1.0, shelf.clone());
            }
            calls += 1;
            // The opening quench sits above the next one, so the hop records
            // a new best and the surface hop has a structure to read. Once
            // the ledger is spent the energy rises, so the descent stops.
            let energy = if calls == 1 { 0.0 } else { -(calls as f64) };
            (energy, shelf.clone())
        };
        let mut rng = StdRng::seed_from_u64(1);
        let outcome = run(&cfg, start.view(), &mut ledger, &mut relax, &mut rng).unwrap();
        (cfg, outcome)
    };
    let (off, off_outcome) = search(false);
    let (on, on_outcome) = search(true);
    assert_eq!(off_outcome.surface_hops, (0, 0, 0.0));
    assert_eq!(on_outcome.surface_hops, (0, 0, 0.0));
    assert_eq!(on_outcome.best, off_outcome.best);
    assert_eq!(on_outcome.charged, off_outcome.charged);
    let plain = Config::with_scales(
        n,
        length,
        family.basin_hopping_temperature() / LJ_TEMPERATURE,
    );
    assert_eq!(
        off.resolved_sha256().unwrap(),
        plain.resolved_sha256().unwrap()
    );
    assert_ne!(
        on.resolved_sha256().unwrap(),
        plain.resolved_sha256().unwrap()
    );
}
