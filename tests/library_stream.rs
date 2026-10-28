//! Paper and example hops go through `optimize_with_gradient`.
//!
//! That entry draws the start and continues the search on one `StdRng`.
//! Reseeding a second stream at the same integer, which `paper_controls`
//! and `ljrunner` used to do, is a different search.

use anneal_core::methods::cluster_hopping::{
    Config, Ledger, optimize_with_gradient, random_cluster_in_radius, run_with_gradient,
};
use anneal_core::methods::warm_lbfgs::WarmLbfgs;
use anneal_core::potentials::PairPotential;
use ndarray::ArrayView1;
use rand::SeedableRng;
use rand::rngs::StdRng;

fn source(name: &str) -> String {
    let path = format!("{}/examples/{name}", env!("CARGO_MANIFEST_DIR"));
    std::fs::read_to_string(&path).unwrap_or_else(|err| panic!("read {path}: {err}"))
}

#[test]
fn paper_controls_and_ljrunner_call_the_library_stream() {
    for name in ["paper_controls.rs", "ljrunner.rs"] {
        let text = source(name);
        assert!(
            text.contains("optimize_with_gradient"),
            "{name} does not call the library entry"
        );
        assert!(
            !text.contains("start_rng"),
            "{name} still draws the start on a second stream"
        );
    }
}

fn drive(seed: u64, shared: bool) -> (u64, usize, usize) {
    let cfg = Config::recommended(13);
    let mut ledger = Ledger::new(400);
    let pot = PairPotential::lennard_jones(13);
    let mut opt = WarmLbfgs::default();
    let mut relax = |led: &mut Ledger, x: ArrayView1<f64>, iters: usize| {
        opt.forget();
        let (f, xr, _) = opt.minimize(x, iters, |v| {
            if !led.charge() {
                return None;
            }
            Some(pot.value_and_gradient(v))
        });
        (f, xr)
    };
    let mut grad = |led: &mut Ledger, v: ArrayView1<f64>| {
        if !led.charge() {
            return None;
        }
        Some(pot.value_and_gradient(v).1)
    };
    let out = if shared {
        optimize_with_gradient(&cfg, &mut ledger, &mut relax, Some(&mut grad), seed)
    } else {
        let mut start_rng = StdRng::seed_from_u64(seed);
        let start = random_cluster_in_radius(
            cfg.n_points,
            cfg.start_radius(),
            cfg.min_separation,
            &mut start_rng,
        );
        let mut rng = StdRng::seed_from_u64(seed);
        run_with_gradient(
            &cfg,
            start.view(),
            &mut ledger,
            &mut relax,
            Some(&mut grad),
            &mut rng,
        )
    }
    .expect("finite start");
    (out.best.to_bits(), out.charged, out.hops)
}

#[test]
fn one_shared_stream_matches_the_library_and_a_reseed_does_not() {
    let library = drive(7, true);
    let again = drive(7, true);
    assert_eq!(library, again, "the library stream is not repeatable");
    let reseeded = drive(7, false);
    assert_ne!(
        library, reseeded,
        "reseeding a second stream at the same integer matched the library"
    );
}
