// Copyright (c) anneal developers
//! `Ledger::trace_accepted` and `Ledger::trace_incumbent` observe the chain
//! without steering it. With both sinks off and on, `Config::recommended` reaches bit-identical best
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

#[derive(Debug, Clone, PartialEq)]
struct Improvement {
    hop: usize,
    spent: usize,
    energy_bits: u64,
    state_bits: Vec<u64>,
}

type Sinks = (Arc<Mutex<Vec<Event>>>, Arc<Mutex<Vec<Improvement>>>);

fn search(system: &System, budget: usize, seed: u64, sinks: Option<Sinks>) -> Outcome {
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
    if let Some((events, improvements)) = sinks {
        ledger.trace_incumbent(move |m| {
            improvements.lock().unwrap().push(Improvement {
                hop: m.hop,
                spent: m.spent,
                energy_bits: m.energy.to_bits(),
                state_bits: m.state.iter().map(|v| v.to_bits()).collect(),
            });
        });
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
        let improvements = Arc::new(Mutex::new(Vec::new()));
        let on = search(
            system,
            budget,
            seed,
            Some((Arc::clone(&events), Arc::clone(&improvements))),
        );
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
        let improvements = improvements.lock().unwrap();
        let last = improvements.last().expect("an incumbent");
        assert!(
            improvements.windows(2).all(|w| w[0].hop <= w[1].hop
                && w[0].spent <= w[1].spent
                && f64::from_bits(w[1].energy_bits) < f64::from_bits(w[0].energy_bits)),
            "{name} seed {seed}: incumbents out of order"
        );
        assert!(
            last.hop <= on.hops,
            "{name} seed {seed}: incumbent past the hops"
        );
        assert_eq!(
            last.energy_bits,
            on.best.to_bits(),
            "{name} seed {seed}: last incumbent"
        );
        assert_eq!(
            last.state_bits,
            state_bits(&on),
            "{name} seed {seed}: last incumbent state"
        );
        let unattributed = events
            .iter()
            .filter(|event| event.action == "unattributed")
            .count();
        println!(
            "{name} seed {seed} budget {budget} best_bits {:016x} calls {} hops {} adopted {} \
             unattributed {unattributed} incumbents {} last at hop {} identical",
            on.best.to_bits(),
            on.charged,
            on.hops,
            events.len(),
            improvements.len(),
            last.hop
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

#[test]
fn the_incumbent_sink_sees_each_lowering() {
    let seen = Arc::new(Mutex::new(Vec::new()));
    let sink = Arc::clone(&seen);
    let mut ledger = Ledger::new(10);
    ledger.trace_incumbent(move |m| sink.lock().unwrap().push((m.hop, m.energy)));
    let state = ndarray::Array1::<f64>::zeros(3);
    for value in [1.0, 2.0, -1.0, -1.0, -3.0] {
        ledger.record(value, state.view());
    }
    assert_eq!(*seen.lock().unwrap(), vec![(0, 1.0), (0, -1.0), (0, -3.0)]);
}
