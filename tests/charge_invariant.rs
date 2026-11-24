// Copyright (c) anneal developers
//! Every public search entry point reports every call it settled.
//!
//! A run's charge is `Ledger::charged()`: `spent` plus whatever settled past
//! the budget. Two things have to hold at the top of each entry point.
//!
//! * `outcome.charged == ledger.spent() + ledger.settled_beyond_cap()` and
//!   the outcome carries the overrun itself.
//! * The units a child ledger priced through `charge_frac` are on the
//!   parent's `priced` once the child is settled, so `priced` is summed over
//!   every child a run settled.
//!
//! The overrun is forced two ways. A relaxation that settles a fixed
//! number of calls in one step, as an external evaluator does, runs the
//! ledger past its budget inside the search, on the main ledger and on the
//! slices and side ledgers the search hands it. An entry point whose
//! relaxation is its own, and charges one call at a time, is run on a
//! ledger that already carries a nested overrun, which reaches its early
//! returns.
//!
//! `binary_water::search_*`, `lattice_ensemble::run` and
//! `ensemble::run_ensemble` build a ledger of their own from a budget, so a
//! caller cannot overrun them.

#[cfg(feature = "graphkey")]
use anneal_core::methods::archive_search::{Archive, archive_search};
use anneal_core::methods::cluster_hopping::{
    Config, Ledger, random_cluster_in_radius, run, run_with_gradient, run_with_gradient_settle,
    run_with_history,
};
use anneal_core::methods::cluster_search::{search_from, search_lennard_jones};
use anneal_core::methods::csa_cluster::{self, BankConfig, spectrum_distance};
use anneal_core::methods::dls_published::{self, TieCut};
use anneal_core::methods::gmin_hop;
use anneal_core::methods::lattice_search::{self, LatticeSearchConfig, Quench};
use anneal_core::potentials::PairPotential;
use ndarray::{Array1, ArrayView1};
use rand::SeedableRng;
use rand::rngs::StdRng;
use std::sync::atomic::{AtomicUsize, Ordering};

const ATOMS: usize = 8;
const PRICED_PER_QUENCH: usize = 2;
const SETTLED_PER_QUENCH: usize = 8;

/// Work the relaxation settled, counted outside the ledger.
#[derive(Default)]
struct Tally {
    priced: AtomicUsize,
    settled: AtomicUsize,
}

/// A parent of thirty with room for none: a child of ten settled ten calls
/// and forty past its own cap, so thirty are spent and twenty are beyond.
fn ledger_carrying_an_overrun() -> Ledger {
    let mut parent = Ledger::new(30);
    let mut child = parent.slice(10);
    assert!(child.charge_many(10));
    assert!(!child.charge_many(40));
    parent.settle_child(&child, 0);
    assert_eq!(parent.spent(), 30);
    assert_eq!(parent.settled_beyond_cap(), 20);
    assert_eq!(parent.remaining(), 0);
    parent
}

fn assert_carries_the_overrun(label: &str, charged: usize, beyond: usize, ledger: &Ledger) {
    assert_eq!(
        charged,
        ledger.spent() + ledger.settled_beyond_cap(),
        "{label}: charged is not spent plus the overrun"
    );
    assert_eq!(
        beyond,
        ledger.settled_beyond_cap(),
        "{label}: the outcome dropped the overrun"
    );
}

fn assert_ledger_carries_the_prepared_overrun(label: &str, charged: usize, beyond: usize) {
    assert_eq!(charged, 50, "{label}: charged");
    assert_eq!(beyond, 20, "{label}: settled beyond the cap");
}

/// With no room left the search cannot quench a start and returns the
/// error, which has no outcome to carry. The ledger keeps the overrun.
fn refused_without_a_start(label: &str, error: &impl std::fmt::Debug, ledger: &Ledger) {
    assert!(
        format!("{error:?}").contains("NoFiniteStart"),
        "{label}: {error:?}"
    );
    assert_eq!(ledger.charged(), 50, "{label}: the ledger lost the overrun");
}

fn start(seed: u64) -> Array1<f64> {
    let cfg = Config::recommended(ATOMS);
    let mut rng = StdRng::seed_from_u64(seed);
    random_cluster_in_radius(ATOMS, cfg.start_radius(), cfg.min_separation, &mut rng)
}

/// A relaxation that prices two units and settles eight calls in one step,
/// whatever room is left, and reports the energy of the point it was given.
fn settling_relax<'a>(
    potential: &'a PairPotential,
    tally: &'a Tally,
) -> impl FnMut(&mut Ledger, ArrayView1<f64>, usize) -> (f64, Array1<f64>) + Send + 'a {
    move |led: &mut Ledger, x: ArrayView1<f64>, _steps: usize| {
        let before = led.priced();
        for _ in 0..PRICED_PER_QUENCH {
            led.charge_frac(1.0);
        }
        let paid = led.priced() - before;
        tally.priced.fetch_add(paid, Ordering::Relaxed);
        led.charge_many(SETTLED_PER_QUENCH);
        tally
            .settled
            .fetch_add(paid + SETTLED_PER_QUENCH, Ordering::Relaxed);
        (potential.value_and_gradient(x).0, x.to_owned())
    }
}

fn assert_forced_overrun(label: &str, ledger: &Ledger, charged: usize, beyond: usize, t: &Tally) {
    assert!(
        ledger.settled_beyond_cap() > 0,
        "{label}: the relaxation did not run the ledger past its budget"
    );
    assert!(ledger.spent() <= ledger.budget(), "{label}: I1 broke");
    assert_carries_the_overrun(label, charged, beyond, ledger);
    assert_eq!(
        ledger.charged(),
        t.settled.load(Ordering::Relaxed),
        "{label}: a settled call is on no field of the ledger"
    );
    assert_eq!(
        ledger.priced(),
        t.priced.load(Ordering::Relaxed),
        "{label}: priced is not summed over the children"
    );
}

fn forced_cfg() -> Config {
    let mut cfg = Config::recommended(ATOMS);
    cfg.packing_deferral = 0;
    cfg.relax_steps = 10;
    // The lattice arm prices its site scans on the ledger itself, outside
    // the tally. The packing probe still runs without it.
    cfg.displacement_only = true;
    cfg
}

#[test]
fn the_walk_entry_points_settle_a_forced_overrun_exactly() {
    let potential = PairPotential::lennard_jones(ATOMS);
    let cfg = forced_cfg();
    for entry in 0..4 {
        let tally = Tally::default();
        let mut ledger = Ledger::new(1_005);
        let mut relax = settling_relax(&potential, &tally);
        let mut grad = |led: &mut Ledger, x: ArrayView1<f64>| {
            led.charge().then(|| {
                tally.settled.fetch_add(1, Ordering::Relaxed);
                potential.value_and_gradient(x).1
            })
        };
        let mut rng = StdRng::seed_from_u64(7);
        let s = start(7);
        let out = match entry {
            0 => run(&cfg, s.view(), &mut ledger, &mut relax, &mut rng),
            1 => run_with_gradient(
                &cfg,
                s.view(),
                &mut ledger,
                &mut relax,
                Some(&mut grad),
                &mut rng,
            ),
            2 => run_with_gradient_settle(
                &cfg,
                s.view(),
                &mut ledger,
                &mut relax,
                Some(&mut grad),
                None,
                &mut rng,
            ),
            _ => run_with_history(
                &cfg,
                s.view(),
                &mut ledger,
                &mut relax,
                Some(&mut grad),
                None,
                &mut rng,
            ),
        }
        .expect("finite start");
        let label = ["run", "run_with_gradient", "settle", "history"][entry];
        assert_forced_overrun(label, &ledger, out.charged, out.settled_beyond_cap, &tally);
    }
}

#[test]
fn a_stall_branch_settles_what_it_priced() {
    let potential = PairPotential::lennard_jones(ATOMS);
    let mut cfg = forced_cfg();
    cfg.branch_on_stall = true;
    cfg.restart_patience = 40;
    let tally = Tally::default();
    let mut ledger = Ledger::new(3_005);
    let mut relax = settling_relax(&potential, &tally);
    let mut rng = StdRng::seed_from_u64(11);
    let out = run_with_gradient(
        &cfg,
        start(11).view(),
        &mut ledger,
        &mut relax,
        None,
        &mut rng,
    )
    .expect("finite start");
    assert_forced_overrun(
        "branch",
        &ledger,
        out.charged,
        out.settled_beyond_cap,
        &tally,
    );
}

#[cfg(feature = "graphkey")]
#[test]
fn the_archive_search_settles_a_forced_overrun_exactly() {
    let potential = PairPotential::lennard_jones(ATOMS);
    let cfg = forced_cfg();
    let tally = Tally::default();
    let mut ledger = Ledger::new(1_005);
    let mut relax = settling_relax(&potential, &tally);
    let mut archive = Archive::new();
    let mut rng = StdRng::seed_from_u64(5);
    let out = archive_search(
        &cfg,
        start(5).view(),
        &mut ledger,
        &mut relax,
        None,
        &mut archive,
        &mut rng,
    );
    assert_forced_overrun(
        "archive_search",
        &ledger,
        out.charged,
        out.settled_beyond_cap,
        &tally,
    );
}

#[test]
fn the_bank_search_settles_a_forced_overrun_exactly() {
    let potential = PairPotential::lennard_jones(ATOMS);
    let cfg = forced_cfg();
    let bank_cfg = BankConfig {
        capacity: 3,
        slice: 90,
        seeding: 3,
        ..BankConfig::default()
    };
    let tally = Tally::default();
    let mut ledger = Ledger::new(1_500);
    let mut relax = settling_relax(&potential, &tally);
    let out = csa_cluster::run(
        &cfg,
        &bank_cfg,
        &mut ledger,
        &mut relax,
        None,
        spectrum_distance(ATOMS),
        3,
    );
    assert_forced_overrun(
        "csa_cluster",
        &ledger,
        out.charged,
        out.settled_beyond_cap,
        &tally,
    );
}

#[test]
fn the_lattice_search_settles_a_forced_overrun_exactly() {
    let potential = PairPotential::lennard_jones(ATOMS);
    let mut cfg = LatticeSearchConfig::lennard_jones(ATOMS);
    cfg.relax_steps = 10;
    let tally = Tally::default();
    let mut ledger = Ledger::new(603);
    let mut relax = settling_relax(&potential, &tally);
    let out = lattice_search::run(&cfg, &mut ledger, &mut relax, 4);
    assert!(ledger.settled_beyond_cap() > 0, "no overrun was forced");
    assert_carries_the_overrun(
        "lattice_search",
        out.charged,
        out.settled_beyond_cap,
        &ledger,
    );
    assert!(
        ledger.charged() >= tally.settled.load(Ordering::Relaxed),
        "the ledger holds every call the relaxation settled and the site energies"
    );
}

#[test]
fn every_entry_point_reports_an_overrun_the_ledger_already_carries() {
    let potential = PairPotential::lennard_jones(ATOMS);
    let cfg = Config::for_cluster(ATOMS);
    let s = start(2);

    let mut ledger = ledger_carrying_an_overrun();
    match search_lennard_jones(&potential, &cfg, &mut ledger, 1) {
        Ok((out, _)) => assert_ledger_carries_the_prepared_overrun(
            "search_lennard_jones",
            out.charged,
            out.settled_beyond_cap,
        ),
        Err(error) => refused_without_a_start("search_lennard_jones", &error, &ledger),
    }

    let mut ledger = ledger_carrying_an_overrun();
    match search_from(&potential, &cfg, &mut ledger, s.view(), 1) {
        Ok((out, _)) => assert_ledger_carries_the_prepared_overrun(
            "search_from",
            out.charged,
            out.settled_beyond_cap,
        ),
        Err(error) => refused_without_a_start("search_from", &error, &ledger),
    }

    let mut ledger = ledger_carrying_an_overrun();
    let hop_cfg = gmin_hop::Config::morse(ATOMS);
    let morse = PairPotential::morse(ATOMS, 6.0);
    let out = gmin_hop::hop(&morse, &hop_cfg, &mut ledger, s.view(), 1);
    assert_ledger_carries_the_prepared_overrun("gmin_hop", out.charged, out.settled_beyond_cap);

    let mut ledger = ledger_carrying_an_overrun();
    let quench = Quench::default();
    let mut rng = StdRng::seed_from_u64(2);
    let out = dls_published::run(
        ATOMS,
        &mut ledger,
        &quench,
        s.as_slice().expect("contiguous"),
        2.2,
        0.5,
        &mut rng,
        TieCut::LowestIndex,
        None,
    );
    assert_ledger_carries_the_prepared_overrun(
        "dls_published",
        out.charged,
        out.settled_beyond_cap,
    );

    let tally = Tally::default();
    let mut ledger = ledger_carrying_an_overrun();
    let mut relax = settling_relax(&potential, &tally);
    let mut rng = StdRng::seed_from_u64(2);
    let out = run(&cfg, s.view(), &mut ledger, &mut relax, &mut rng).expect("run");
    assert_carries_the_overrun("run", out.charged, out.settled_beyond_cap, &ledger);
    assert!(out.charged >= 50, "run: the carried overrun was dropped");

    let mut ledger = ledger_carrying_an_overrun();
    let mut relax = settling_relax(&potential, &tally);
    let lattice = LatticeSearchConfig::lennard_jones(ATOMS);
    let out = lattice_search::run(&lattice, &mut ledger, &mut relax, 2);
    assert_carries_the_overrun(
        "lattice_search",
        out.charged,
        out.settled_beyond_cap,
        &ledger,
    );
    assert!(
        out.charged >= 50,
        "lattice_search: the carried overrun was dropped"
    );
}
