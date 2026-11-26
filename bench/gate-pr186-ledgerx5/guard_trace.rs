// Copyright (c) anneal developers
//! One default-preset guard seed, with a bit-level trajectory hash.
//!
//! Lennard-Jones and Morse use `PairPotential` and `Config::recommended`.
//! The Morse 38 rho 14 and LJ98 cells use the same driver.
//! Au55 uses `Gupta::copper_gold` and the species the metal example installs.
//! The hexamer uses `GminTip4p` and `Config::recommended_tip4p`.
//! Each run stops at the published minimum plus `DEFAULT_HIT_TOL`, or at
//! the paper budget. Nothing else is changed.

use anneal_core::methods::cluster_hopping::{
    Config, Ledger, Outcome, random_cluster_in_radius, run_with_gradient,
};
#[cfg(ledger_terms)]
use anneal_core::methods::cluster_hopping::{ChargeName, ChargeSplit};
use anneal_core::methods::cluster_search::search_from;
use anneal_core::methods::warm_lbfgs::WarmLbfgs;
use anneal_core::potentials::{
    GminTip4p, Gupta, PairPotential, gmin_water_geometry, random_tip4p_cluster,
};
use eindir_core::Objective;
use eindir_core::bounds::Bounds;
use eindir_core::gradient::{DifferentiableObjective, Gradient};
use ndarray::{Array1, ArrayView1};
use rand::SeedableRng;
use rand::rngs::StdRng;
use sha2::{Digest, Sha256};
use std::sync::atomic::{AtomicU64, Ordering};
use std::time::Instant;

const HIT_TOL: f64 = anneal_core::DEFAULT_HIT_TOL;

struct Done {
    budget: usize,
    hit: bool,
    calls: usize,
    charged: usize,
    branch_on_stall: bool,
    branch_draws: Option<u64>,
    branch_spent: Option<u64>,
    wall_s: f64,
    inside_s: f64,
    best_bits: String,
    trajectory: String,
    rss_anon_kb: u64,
    peak_rss_anon_kb: u64,
    screen: String,
    full: String,
    check: String,
    closure: String,
    quench: String,
    settle: String,
    lattice: String,
    reoccupy: String,
    probe: String,
    probe_unbacked: String,
    unnamed: String,
    oracle_calls: usize,
    call_budget: usize,
    ledger_error: String,
}

const RSS_SAMPLE_MASK: u64 = 1023;

struct RssSampler {
    calls: AtomicU64,
    peak_kb: AtomicU64,
}

impl RssSampler {
    fn new() -> Self {
        Self {
            calls: AtomicU64::new(0),
            peak_kb: AtomicU64::new(rss_anon_kb()),
        }
    }

    fn record(&self, kb: u64) {
        self.peak_kb.fetch_max(kb, Ordering::Relaxed);
    }

    fn sample_now(&self) {
        self.record(rss_anon_kb());
    }

    fn maybe_sample(&self) {
        if (self.calls.fetch_add(1, Ordering::Relaxed) & RSS_SAMPLE_MASK) == 0 {
            self.sample_now();
        }
    }

    fn peak(&self) -> u64 {
        self.peak_kb.load(Ordering::Relaxed)
    }
}

/// Nanoseconds inside potential energy and gradient calls.
struct PesClock {
    inside_ns: AtomicU64,
}

impl PesClock {
    fn new() -> Self {
        Self {
            inside_ns: AtomicU64::new(0),
        }
    }

    fn time<T>(&self, body: impl FnOnce() -> T) -> T {
        let started = Instant::now();
        let out = body();
        let dt = u64::try_from(started.elapsed().as_nanos()).unwrap_or(u64::MAX);
        self.inside_ns.fetch_add(dt, Ordering::Relaxed);
        out
    }

    fn seconds(&self) -> f64 {
        self.inside_ns.load(Ordering::Relaxed) as f64 * 1e-9
    }
}

/// Times `value_and_gradient` at the potential boundary and forwards it.
struct TimedPes<'a, P> {
    inner: &'a P,
    clock: &'a PesClock,
    rss: &'a RssSampler,
}

impl<P> Objective<f64> for TimedPes<'_, P>
where
    P: Objective<f64>,
{
    fn dim(&self) -> usize {
        Objective::dim(self.inner)
    }

    fn bounds(&self) -> &Bounds<f64> {
        self.inner.bounds()
    }

    fn eval(&self, x: ArrayView1<f64>) -> f64 {
        self.rss.maybe_sample();
        self.clock.time(|| self.inner.eval(x))
    }
}

impl<P> Gradient<f64> for TimedPes<'_, P>
where
    P: Gradient<f64>,
{
    fn grad(&self, x: ArrayView1<f64>) -> Array1<f64> {
        self.rss.maybe_sample();
        self.clock.time(|| self.inner.grad(x))
    }

    fn dim(&self) -> usize {
        Gradient::dim(self.inner)
    }
}

impl<P> DifferentiableObjective<f64> for TimedPes<'_, P>
where
    P: DifferentiableObjective<f64>,
{
    fn value_and_gradient(&self, x: ArrayView1<f64>) -> (f64, Array1<f64>) {
        self.rss.maybe_sample();
        self.clock.time(|| self.inner.value_and_gradient(x))
    }
}


#[cfg(ledger_terms)]
fn pay(ledger: &mut Ledger, kind: u8) -> bool {
    let name = if kind == 0 {
        ChargeName::Quench
    } else {
        ChargeName::Check
    };
    ledger.with_charge_name(name, |ledger| ledger.charge())
}

#[cfg(not(ledger_terms))]
fn pay(ledger: &mut Ledger, kind: u8) -> bool {
    let _ = kind;
    ledger.charge()
}

struct LedgerReport {
    screen: String,
    full: String,
    check: String,
    closure: String,
    quench: String,
    settle: String,
    lattice: String,
    reoccupy: String,
    probe: String,
    probe_unbacked: String,
    unnamed: String,
    oracle_calls: usize,
    ledger_error: String,
}

fn ledger_report(ledger: &Ledger) -> LedgerReport {
    #[cfg(ledger_terms)]
    {
        let ledger_error = match ledger.assert_named_charges() {
            Ok(()) => String::new(),
            Err(err) => err.to_string(),
        };
        let charges: ChargeSplit = ledger.charges();
        return LedgerReport {
            screen: charges.screen.to_string(),
            full: charges.full.to_string(),
            check: charges.check.to_string(),
            closure: charges.closure.to_string(),
            quench: charges.quench.to_string(),
            settle: charges.settle.to_string(),
            lattice: charges.lattice.to_string(),
            reoccupy: charges.reoccupy.to_string(),
            probe: charges.probe.to_string(),
            probe_unbacked: "0".to_string(),
            unnamed: charges.unnamed.to_string(),
            oracle_calls: ledger.spent(),
            ledger_error,
        };
    }
    #[cfg(not(ledger_terms))]
    {
        LedgerReport {
            screen: "na".to_string(),
            full: "na".to_string(),
            check: "na".to_string(),
            closure: "na".to_string(),
            quench: "na".to_string(),
            settle: "na".to_string(),
            lattice: "na".to_string(),
            reoccupy: "na".to_string(),
            probe: "na".to_string(),
            probe_unbacked: "na".to_string(),
            unnamed: "na".to_string(),
            oracle_calls: ledger.spent(),
            ledger_error: String::new(),
        }
    }
}

fn error_token(text: &str) -> &'static str {
    if text.is_empty() {
        "ok"
    } else if text.contains("unnamed") {
        "unnamed"
    } else if text.contains("finite start") {
        "nofinite"
    } else {
        "error"
    }
}

fn main() {
    anneal_core::install_rgpot_kernels!();
    let core = affinity();
    if core != "0" {
        eprintln!("affinity {core}");
        std::process::exit(3);
    }
    let mut args = std::env::args().skip(1);
    let Some(system) = args.next() else {
        eprintln!("need a system");
        std::process::exit(2);
    };
    let Some(seed) = args.next().and_then(|text| text.parse::<u64>().ok()) else {
        eprintln!("need a seed");
        std::process::exit(2);
    };
    if args.next().is_some() {
        eprintln!("too many arguments");
        std::process::exit(2);
    }
    if let Err(err) = Config::check_removed_env() {
        eprintln!("removed config: {err}");
        std::process::exit(2);
    }
    let done = match system.as_str() {
        "lj38" => run_lj(38, 400_000, -173.928427, seed),
        "m38r14d" => run_morse(38, 14.0, 400_000, -144.321054, seed),
        "lj75" => run_lj(75, 4_000_000, -397.492331, seed),
        "lj98" => run_lj(98, 4_000_000, -543.665361, seed),
        "m80r6" => run_morse(80, 6.0, 4_000_000, -378.333471, seed),
        "au55" => run_gold(55, 1_000_000, -191.03085280, seed),
        "hexamer" => run_hexamer(seed),
        other => {
            eprintln!("unknown system {other}");
            std::process::exit(2);
        }
    };
    println!(
        "GATE system={system} seed={seed} budget={} hit={} calls={} charged={} branch_on_stall={} branch_draws={} branch_spent={} wall_s={:.9} inside_s={:.9} best_bits={} trajectory={} rss_anon_kb={} peak_rss_anon_kb={} core={core} screen={} full={} check={} closure={} quench={} settle={} lattice={} reoccupy={} probe={} probe_unbacked={} unnamed={} oracle_calls={} call_budget={} ledger_error={}",
        done.budget,
        u8::from(done.hit),
        done.calls,
        done.charged,
        u8::from(done.branch_on_stall),
        optional_count(done.branch_draws),
        optional_count(done.branch_spent),
        done.wall_s,
        done.inside_s,
        done.best_bits,
        done.trajectory,
        done.rss_anon_kb,
        done.peak_rss_anon_kb,
        done.screen,
        done.full,
        done.check,
        done.closure,
        done.quench,
        done.settle,
        done.lattice,
        done.reoccupy,
        done.probe,
        done.probe_unbacked,
        done.unnamed,
        done.oracle_calls,
        done.call_budget,
        done.ledger_error,
    );
}

fn affinity() -> String {
    let text = std::fs::read_to_string("/proc/self/status").unwrap_or_default();
    text.lines()
        .find_map(|line| line.strip_prefix("Cpus_allowed_list:"))
        .map(|rest| rest.trim().to_string())
        .unwrap_or_default()
}

fn optional_count(value: Option<u64>) -> String {
    value.map_or_else(|| "na".to_owned(), |count| count.to_string())
}

fn outcome_counter(debug: &str, field: &str) -> Option<u64> {
    let marker = format!("{field}: ");
    let value = debug.split_once(&marker)?.1;
    let digits = value
        .split_once(|ch: char| !ch.is_ascii_digit())
        .map_or(value, |(head, _)| head);
    digits.parse().ok()
}

/// `RssAnon` of this process in kB, read while the search state is alive.
fn rss_anon_kb() -> u64 {
    let text = std::fs::read_to_string("/proc/self/status").unwrap_or_default();
    text.lines()
        .find_map(|line| line.strip_prefix("RssAnon:"))
        .and_then(|rest| rest.split_whitespace().next())
        .and_then(|num| num.parse::<u64>().ok())
        .unwrap_or(0)
}

fn run_lj(n: usize, budget: usize, reference: f64, seed: u64) -> Done {
    let cfg = Config::recommended(n);
    let potential = PairPotential::lennard_jones(n);
    run_lbfgs("lj", n, budget, reference, seed, cfg, &potential)
}

fn run_morse(n: usize, rho: f64, budget: usize, reference: f64, seed: u64) -> Done {
    let cfg = Config::recommended(n);
    let potential = PairPotential::morse(n, rho);
    run_lbfgs("morse", n, budget, reference, seed, cfg, &potential)
}

fn run_gold(n: usize, budget: usize, reference: f64, seed: u64) -> Done {
    let cfg = Config::recommended(n).with_species(Gupta::copper_gold_species(0, n), None);
    let potential = Gupta::copper_gold(0, n);
    run_lbfgs("au", n, budget, reference, seed, cfg, &potential)
}

fn run_lbfgs<P>(
    _label: &str,
    n: usize,
    budget: usize,
    reference: f64,
    seed: u64,
    cfg: Config,
    potential: &P,
) -> Done
where
    P: DifferentiableObjective<f64>,
{
    let mut rng = StdRng::seed_from_u64(seed);
    let start = random_cluster_in_radius(n, cfg.start_radius(), cfg.min_separation, &mut rng);
    let mut ledger = Ledger::new(budget);
    ledger.stop_below(reference + HIT_TOL);
    let mut optimiser = WarmLbfgs::default();
    let clock = PesClock::new();
    let rss = RssSampler::new();
    let mut relax = |led: &mut Ledger, x: ArrayView1<f64>, iterations: usize| {
        optimiser.forget();
        let (value, relaxed, _) = optimiser.minimize(x, iterations, |point| {
            if !pay(led, 0) {
                return None;
            }
            rss.maybe_sample();
            Some(clock.time(|| potential.value_and_gradient(point)))
        });
        (value, relaxed)
    };
    let mut grad = |led: &mut Ledger, point: ArrayView1<f64>| {
        if !pay(led, 1) {
            return None;
        }
        rss.maybe_sample();
        Some(clock.time(|| potential.value_and_gradient(point)).1)
    };
    let mut rng = StdRng::seed_from_u64(seed);
    let started = Instant::now();
    let outcome = match run_with_gradient(
        &cfg,
        start.view(),
        &mut ledger,
        &mut relax,
        Some(&mut grad),
        &mut rng,
    ) {
        Ok(outcome) => outcome,
        Err(err) => {
            return finish_error(
                budget,
                cfg.branch_on_stall,
                started,
                clock.seconds(),
                &rss,
                &ledger,
                &err.to_string(),
            );
        }
    };
    finish(
        budget,
        reference,
        cfg.branch_on_stall,
        started,
        clock.seconds(),
        &rss,
        &ledger,
        outcome,
    )
}

fn run_hexamer(seed: u64) -> Done {
    let n = 6usize;
    let budget = 500_000usize;
    let reference = -197.7805260034;
    let mut cfg = Config::recommended_tip4p(n);
    cfg.rigid_body_geometry = Some(gmin_water_geometry());
    let mut rng = StdRng::seed_from_u64(seed.wrapping_mul(0x9E37_79B9_7F4A_7C15).wrapping_add(3));
    let start = random_tip4p_cluster(n, &mut rng);
    let potential = GminTip4p::new(n);
    let clock = PesClock::new();
    let rss = RssSampler::new();
    let timed = TimedPes {
        inner: &potential,
        clock: &clock,
        rss: &rss,
    };
    let mut ledger = Ledger::new(budget);
    ledger.stop_below(reference + HIT_TOL);
    let started = Instant::now();
    let outcome = match search_from(&timed, &cfg, &mut ledger, start.view(), seed) {
            Ok((outcome, _)) => outcome,
            Err(err) => {
                return finish_error(
                    budget,
                    cfg.branch_on_stall,
                    started,
                    clock.seconds(),
                    &rss,
                    &ledger,
                    &err.to_string(),
                );
            }
        };
    finish(
        budget,
        reference,
        cfg.branch_on_stall,
        started,
        clock.seconds(),
        &rss,
        &ledger,
        outcome,
    )
}

fn finish(
    budget: usize,
    reference: f64,
    branch_on_stall: bool,
    started: Instant,
    inside_s: f64,
    rss: &RssSampler,
    ledger: &Ledger,
    outcome: Outcome,
) -> Done {
    let report = ledger_report(ledger);
    let wall_s = started.elapsed().as_secs_f64();
    let end_rss = rss_anon_kb();
    rss.record(end_rss);
    let hit = anneal_core::energy_hits(outcome.best, reference, HIT_TOL);
    let at_hit = outcome
        .improvements
        .iter()
        .find(|(_, _, _, energy)| anneal_core::energy_hits(*energy, reference, HIT_TOL))
        .map(|(_, spent, _, _)| *spent);
    let calls = if hit {
        at_hit.unwrap_or(outcome.charged).min(budget)
    } else {
        budget
    };
    let outcome_debug = format!("{outcome:?}");
    Done {
        budget,
        hit,
        calls,
        charged: outcome.charged,
        branch_on_stall,
        branch_draws: outcome_counter(&outcome_debug, "branch_draws"),
        branch_spent: outcome_counter(&outcome_debug, "branch_spent"),
        wall_s,
        inside_s,
        best_bits: format!("{:016x}", outcome.best.to_bits()),
        trajectory: trajectory_hash(&outcome),
        rss_anon_kb: end_rss,
        peak_rss_anon_kb: rss.peak(),
        screen: report.screen,
        full: report.full,
        check: report.check,
        closure: report.closure,
        quench: report.quench,
        settle: report.settle,
        lattice: report.lattice,
        reoccupy: report.reoccupy,
        probe: report.probe,
        probe_unbacked: report.probe_unbacked,
        unnamed: report.unnamed,
        oracle_calls: report.oracle_calls,
        call_budget: budget,
        ledger_error: error_token(&report.ledger_error).to_string(),
    }
}

fn finish_error(
    budget: usize,
    branch_on_stall: bool,
    started: Instant,
    inside_s: f64,
    rss: &RssSampler,
    ledger: &Ledger,
    message: &str,
) -> Done {
    let report = ledger_report(ledger);
    let wall_s = started.elapsed().as_secs_f64();
    let end_rss = rss_anon_kb();
    rss.record(end_rss);
    let token = if message.contains("unnamed") {
        "unnamed"
    } else if message.contains("finite start") {
        "nofinite"
    } else {
        "error"
    };
    Done {
        budget,
        hit: false,
        calls: budget,
        charged: report.oracle_calls,
        branch_on_stall,
        branch_draws: None,
        branch_spent: None,
        wall_s,
        inside_s,
        best_bits: "na".to_string(),
        trajectory: String::new(),
        rss_anon_kb: end_rss,
        peak_rss_anon_kb: rss.peak(),
        screen: report.screen,
        full: report.full,
        check: report.check,
        closure: report.closure,
        quench: report.quench,
        settle: report.settle,
        lattice: report.lattice,
        reoccupy: report.reoccupy,
        probe: report.probe,
        probe_unbacked: report.probe_unbacked,
        unnamed: report.unnamed,
        oracle_calls: report.oracle_calls,
        call_budget: budget,
        ledger_error: token.to_string(),
    }
}

fn trajectory_hash(outcome: &Outcome) -> String {
    let mut hasher = Sha256::new();
    push_f64(&mut hasher, outcome.best);
    push_f64(&mut hasher, outcome.final_energy);
    push_usize(&mut hasher, outcome.charged);
    push_usize(&mut hasher, outcome.hops);
    push_usize(&mut hasher, outcome.basins);
    push_usize(&mut hasher, outcome.accepted_transitions.len());
    for edge in &outcome.accepted_transitions {
        push_usize(&mut hasher, edge.hop);
        hasher.update(edge.action.as_bytes());
        hasher.update([0]);
        push_f64(&mut hasher, edge.from_energy);
        push_f64(&mut hasher, edge.to_energy);
        hasher.update([u8::from(edge.adopted), u8::from(edge.validated)]);
        for value in edge.from_state.iter() {
            push_f64(&mut hasher, *value);
        }
        for value in edge.to_state.iter() {
            push_f64(&mut hasher, *value);
        }
    }
    push_usize(&mut hasher, outcome.improvements.len());
    for (hop, spent, basins, energy) in &outcome.improvements {
        push_usize(&mut hasher, *hop);
        push_usize(&mut hasher, *spent);
        push_usize(&mut hasher, *basins);
        push_f64(&mut hasher, *energy);
    }
    let digest = hasher.finalize();
    let mut text = String::with_capacity(digest.len() * 2);
    for byte in digest {
        text.push_str(&format!("{byte:02x}"));
    }
    text
}

fn push_f64(hasher: &mut Sha256, value: f64) {
    hasher.update(value.to_le_bytes());
}

fn push_usize(hasher: &mut Sha256, value: usize) {
    hasher.update((value as u64).to_le_bytes());
}
