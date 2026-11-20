// Copyright (c) anneal developers
//! `Ledger::trace_accepted` observes the chain without steering it. With
//! the sink off and on, `Config::recommended` reaches bit-identical best
//! energies, best states and charged counts on three seeds of each
//! double-funnel cluster: Lennard-Jones 38, 75, 98 and 150 and Morse 80 at
//! rho 6. Each run stops at the Cambridge energy plus the hit tolerance, as
//! the `h2h_gmin_proof` and `h2h_morse` runs do.
//!
//! The default tests spend 40000 calls. The ignored test spends the paper
//! budgets, 4e5 calls at 38 atoms and 4e6 above:
//!
//! ```text
//! cargo test --release --test accept_trace_identity -- --ignored --nocapture
//! ```

use anneal_core::methods::cluster_hopping::{
    Config, Ledger, Outcome, random_cluster_in_radius, run_with_gradient,
};
use anneal_core::methods::warm_lbfgs::WarmLbfgs;
use anneal_core::potentials::PairPotential;
use rand::SeedableRng;
use rand::rngs::StdRng;
use std::sync::{Arc, Mutex};

const SEEDS: [u64; 3] = [2000, 2001, 2002];
const SHORT_BUDGET: usize = 40_000;

struct System {
    name: &'static str,
    n: usize,
    rho: Option<f64>,
    reference: f64,
    paper_budget: usize,
}

const SYSTEMS: [System; 5] = [
    System {
        name: "lj38",
        n: 38,
        rho: None,
        reference: -173.928427,
        paper_budget: 400_000,
    },
    System {
        name: "lj75",
        n: 75,
        rho: None,
        reference: -397.492331,
        paper_budget: 4_000_000,
    },
    System {
        name: "lj98",
        n: 98,
        rho: None,
        reference: -543.665361,
        paper_budget: 4_000_000,
    },
    System {
        name: "lj150",
        n: 150,
        rho: None,
        reference: -893.310258,
        paper_budget: 4_000_000,
    },
    System {
        name: "m80r6",
        n: 80,
        rho: Some(6.0),
        reference: -378.333471,
        paper_budget: 4_000_000,
    },
];

#[derive(Debug, Clone, PartialEq)]
struct Event {
    hop: usize,
    spent: usize,
    action: String,
    energy_bits: u64,
    atoms: usize,
}

fn search(
    system: &System,
    budget: usize,
    seed: u64,
    events: Option<Arc<Mutex<Vec<Event>>>>,
) -> Outcome {
    let n = system.n;
    let pair = match system.rho {
        Some(rho) => PairPotential::morse(n, rho),
        None => PairPotential::lennard_jones(n),
    };
    let cfg = Config::recommended(n);
    let mut rng = StdRng::seed_from_u64(seed);
    let start = random_cluster_in_radius(n, cfg.start_radius(), cfg.min_separation, &mut rng);
    let mut ledger = Ledger::new(budget);
    ledger.stop_below(system.reference + anneal_core::DEFAULT_HIT_TOL);
    if let Some(events) = events {
        ledger.trace_accepted(move |m| {
            events.lock().unwrap().push(Event {
                hop: m.hop,
                spent: m.spent,
                action: m.action.to_owned(),
                energy_bits: m.energy.to_bits(),
                atoms: m.state.len() / 3,
            });
        });
    }
    let mut optimiser = WarmLbfgs::default();
    let mut relax = |led: &mut Ledger, x: ndarray::ArrayView1<f64>, iterations: usize| {
        optimiser.forget();
        let (value, relaxed, _) = optimiser.minimize(x, iterations, |point| {
            if !led.charge() {
                return None;
            }
            Some(pair.value_and_gradient(point))
        });
        (value, relaxed)
    };
    let mut grad = |led: &mut Ledger, point: ndarray::ArrayView1<f64>| {
        if !led.charge() {
            return None;
        }
        Some(pair.value_and_gradient(point).1)
    };
    let mut rng = StdRng::seed_from_u64(seed);
    run_with_gradient(
        &cfg,
        start.view(),
        &mut ledger,
        &mut relax,
        Some(&mut grad),
        &mut rng,
    )
    .expect("finite start")
}

fn state_bits(outcome: &Outcome) -> Vec<u64> {
    outcome
        .best_state
        .as_ref()
        .map(|state| state.iter().map(|v| v.to_bits()).collect())
        .unwrap_or_default()
}

fn check(name: &str, budget_of: impl Fn(&System) -> usize) {
    let system = SYSTEMS.iter().find(|s| s.name == name).expect("system");
    let budget = budget_of(system);
    for seed in SEEDS {
        let off = search(system, budget, seed, None);
        let events = Arc::new(Mutex::new(Vec::new()));
        let on = search(system, budget, seed, Some(Arc::clone(&events)));
        assert_eq!(
            off.best.to_bits(),
            on.best.to_bits(),
            "{name} seed {seed}: best_bits"
        );
        assert_eq!(off.charged, on.charged, "{name} seed {seed}: calls");
        assert_eq!(off.hops, on.hops, "{name} seed {seed}: hops");
        assert_eq!(
            state_bits(&off),
            state_bits(&on),
            "{name} seed {seed}: best state"
        );
        let events = events.lock().unwrap();
        assert!(
            events.len() > 1,
            "{name} seed {seed}: {} adopted minima",
            events.len()
        );
        assert_eq!(events[0].action, "start", "{name} seed {seed}");
        assert!(
            events.iter().all(|event| event.atoms == system.n),
            "{name} seed {seed}: atoms"
        );
        assert!(
            events
                .windows(2)
                .all(|w| w[0].spent <= w[1].spent && w[0].hop <= w[1].hop),
            "{name} seed {seed}: trace out of order"
        );
        assert!(
            events
                .iter()
                .all(|event| event.spent <= on.charged
                    && f64::from_bits(event.energy_bits).is_finite()),
            "{name} seed {seed}: trace past the ledger"
        );
        let unattributed = events
            .iter()
            .filter(|event| event.action == "unattributed")
            .count();
        println!(
            "{name} seed {seed} budget {budget} best_bits {:016x} calls {} hops {} adopted {} \
             unattributed {unattributed} identical",
            on.best.to_bits(),
            on.charged,
            on.hops,
            events.len()
        );
    }
}

#[test]
fn tracing_leaves_lj38_unchanged() {
    check("lj38", |_| SHORT_BUDGET);
}

#[test]
fn tracing_leaves_lj75_unchanged() {
    check("lj75", |_| SHORT_BUDGET);
}

#[test]
fn tracing_leaves_lj98_unchanged() {
    check("lj98", |_| SHORT_BUDGET);
}

#[test]
fn tracing_leaves_lj150_unchanged() {
    check("lj150", |_| SHORT_BUDGET);
}

#[test]
fn tracing_leaves_m80_rho6_unchanged() {
    check("m80r6", |_| SHORT_BUDGET);
}

#[test]
#[ignore = "paper budgets; minutes per seed"]
fn tracing_leaves_paper_budget_runs_unchanged() {
    std::thread::scope(|scope| {
        for system in &SYSTEMS {
            scope.spawn(|| check(system.name, |s| s.paper_budget));
        }
    });
}

#[test]
fn the_sink_is_off_by_default() {
    assert!(!Ledger::new(10).traces_accepted());
    let mut ledger = Ledger::new(10);
    ledger.trace_accepted(|_| {});
    assert!(ledger.traces_accepted());
}
