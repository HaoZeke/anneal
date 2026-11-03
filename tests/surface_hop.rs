use anneal_core::methods::cluster_hopping::{Config, Ledger, run};
use anneal_core::methods::warm_lbfgs::WarmLbfgs;
use anneal_core::packing::surface_hops;
use anneal_core::potentials::{SuttonChen, SuttonChenFamily};
use ndarray::{Array1, ArrayView1};
use rand::rngs::StdRng;
use rand::{Rng, SeedableRng};

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
    let n = 78;
    let pot = SuttonChen::new(n, SuttonChenFamily::Sc9_6);
    let quench = |y: &Array1<f64>| {
        let mut opt = WarmLbfgs::default();
        opt.gtol = 1e-6;
        let (energy, _, _) = opt.minimize(y.view(), 5000, |v| Some(pot.value_and_gradient(v)));
        energy
    };
    let shelf = quench(&x);
    assert!((shelf - -19053.98767).abs() < 1e-4, "{shelf}");
    let length = std::f64::consts::FRAC_1_SQRT_2 / 2f64.powf(1.0 / 6.0);
    let candidates = surface_hops(x.view(), n, 1.35 * length, 0.0);
    assert!(!candidates.is_empty() && candidates.len() < 40);
    let lowest = candidates.iter().map(quench).fold(f64::INFINITY, f64::min);
    assert!((lowest - -19054.397655).abs() < 1e-4, "{lowest}");
}

/// The hops run only when asked, and asking for them changes the resolved
/// configuration while leaving it unchanged when they are off.
#[test]
fn surface_hops_run_from_new_bests_only_when_asked() {
    let n = 13;
    let pot = SuttonChen::new(n, SuttonChenFamily::Sc12_6);
    let length = std::f64::consts::FRAC_1_SQRT_2 / 2f64.powf(1.0 / 6.0);
    let mut start_rng = StdRng::seed_from_u64(7);
    let start: Array1<f64> = (0..3 * n)
        .map(|_| start_rng.random_range(-0.9..0.9))
        .collect();
    let search = |on: bool| {
        let mut cfg = Config::with_scales(n, length, 30.0 / 0.8);
        cfg.surface_hop_on_best = on;
        let mut ledger = Ledger::new(20_000);
        let mut opt = WarmLbfgs::default();
        let mut relax = |led: &mut Ledger, v: ArrayView1<f64>, iters: usize| {
            opt.forget();
            opt.gtol = 1e-3;
            let (energy, x, _) = opt.minimize(v, iters.max(1), |w| {
                led.charge().then(|| pot.value_and_gradient(w))
            });
            (energy, x)
        };
        let mut rng = StdRng::seed_from_u64(1);
        let outcome = run(&cfg, start.view(), &mut ledger, &mut relax, &mut rng).unwrap();
        (cfg, outcome)
    };
    let (off, off_outcome) = search(false);
    let (on, on_outcome) = search(true);
    assert_eq!(off_outcome.surface_hops, (0, 0, 0.0));
    assert!(on_outcome.surface_hops.0 > 0);
    let plain = Config::with_scales(n, length, 30.0 / 0.8);
    assert_eq!(
        off.resolved_sha256().unwrap(),
        plain.resolved_sha256().unwrap()
    );
    assert_ne!(
        on.resolved_sha256().unwrap(),
        plain.resolved_sha256().unwrap()
    );
}
