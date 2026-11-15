// Copyright (c) anneal developers
//! One seed of `Config::recommended` on a Sutton–Chen cluster.
//!
//! ```text
//! h2h_sc run <system> <seed> [coords]
//! h2h_sc coords <system> <seed> <path>
//! h2h_sc energy <system> <path>
//! h2h_sc grad <system> <path>
//! h2h_sc budget <system>
//! ```
//!
//! The energy and the forces come from rgpot's `rgpot_sutton_chen_energy_forces`.
//! The library is `RGPOT_SC_LIB`, or rgpot's own candidate list
//! (`RGPOT_LIBRARY`, `RGPOT_RGPOT_ENGINE`, `librgpot.so`). The parameters are the reduced
//! Doye–Wales rows, ε = 1 and a = 1, with the c, n and m on the first line
//! of the Cambridge point files. The search stops within 0.001 of the
//! database energy. A miss is the full budget. `pot_s` is the time inside
//! that call. Lennard-Jones and TIP4P stay on `Config::recommended`.
//! `H2H_PORTFOLIO=1` interleaves that chain with a pair. Anything else
//! leaves the chain alone.

use anneal_core::bias::BasinBias;
use anneal_core::methods::cluster_hopping::{
    CheckpointAction, ClusterFingerprint, Config, Ledger, Outcome, random_cluster_in_radius,
    run_with_bias, run_with_energy_gradient_at_checkpoints, run_with_gradient,
};
use anneal_core::methods::splice::cut_and_splice;
use anneal_core::methods::warm_lbfgs::WarmLbfgs;
use ndarray::{Array1, ArrayView1};
use rand::SeedableRng;
use rand::rngs::StdRng;
use std::sync::atomic::{AtomicBool, AtomicU64, AtomicUsize, Ordering};
use std::sync::{Arc, Mutex};
use std::time::Instant;

const HIT_TOL: f64 = 0.001;

#[repr(C)]
struct ScConfig {
    epsilon: f64,
    c: f64,
    a: f64,
    n: i32,
    m: i32,
    cutoff: f64,
    periodic: i32,
}

struct Family {
    n_exp: i32,
    m_exp: i32,
    c: f64,
}

struct Spec {
    n: usize,
    budget: usize,
    reference: f64,
    family: Family,
}

fn main() {
    anneal_core::install_rgpot_kernels!();
    let args: Vec<String> = std::env::args().skip(1).collect();
    let cmd = args.first().map(String::as_str).unwrap_or("run");
    let status = match cmd {
        "run" => cmd_run(&args),
        "coords" => cmd_coords(&args),
        "energy" => cmd_energy(&args),
        "grad" => cmd_grad(&args),
        "budget" => cmd_budget(&args),
        _ => {
            eprintln!("unknown command {cmd}");
            2
        }
    };
    std::process::exit(status);
}

fn family(name: &str) -> Option<Family> {
    // c, n, m from the first line of the Cambridge point files.
    // 12-6 is silver and rhodium, 10-8 gold and platinum, 9-6 nickel and copper.
    let (n_exp, m_exp, c) = match name {
        "126" => (12, 6, 144.41),
        "108" => (10, 8, 34.408),
        "96" => (9, 6, 39.432),
        _ => return None,
    };
    Some(Family { n_exp, m_exp, c })
}

fn system_spec(name: &str) -> Option<Spec> {
    let (stem, n) = name.rsplit_once('-')?;
    let stem = stem.strip_prefix("sc")?;
    let n: usize = n.parse().ok()?;
    let family = family(stem)?;
    // Printed Cambridge table, four digits. Doye and Wales, New J. Chem. 22,
    // 733 (1998). 38 and 55 are the LJ paper-map sizes. 80 is not in that map.
    let (reference, budget) = match (stem, n) {
        ("126", 38) => (-35419.9804, 400_000),
        ("126", 55) => (-52884.6806, 1_000_000),
        ("126", 80) => (-78414.6271, 1_000_000),
        ("108", 38) => (-10117.2454, 400_000),
        ("108", 55) => (-14814.5225, 1_000_000),
        ("108", 80) => (-21819.9952, 1_000_000),
        ("96", 38) => (-8917.7056, 400_000),
        ("96", 55) => (-13217.8963, 1_000_000),
        ("96", 80) => (-19560.8966, 1_000_000),
        _ => return None,
    };
    Some(Spec {
        n,
        budget,
        reference,
        family,
    })
}

fn sutton_chen_candidates() -> Vec<String> {
    // Same order as rgpot's library lookup: an explicit path, then
    // `RGPOT_LIBRARY` / `RGPOT_RGPOT_ENGINE`, then the bare soname. The
    // profile module that implements that list is feature-gated, so the
    // example keeps the list itself. The old absolute path is only one
    // more place to try.
    let mut candidates = Vec::new();
    for key in ["RGPOT_SC_LIB", "RGPOT_LIBRARY", "RGPOT_RGPOT_ENGINE"] {
        if let Ok(path) = std::env::var(key)
            && !path.is_empty()
            && !candidates.contains(&path)
        {
            candidates.push(path);
        }
    }
    for path in [
        "librgpot.so",
        "./librgpot.so",
        "librgpot.dylib",
        "./librgpot.dylib",
        "rgpot.dll",
        "/work/rgpot/bdir-xtb/CppCore/librgpot.so",
    ] {
        if !candidates.iter().any(|candidate| candidate == path) {
            candidates.push(path.to_string());
        }
    }
    candidates
}

fn library() -> Result<libloading::Library, String> {
    let mut failures = Vec::new();
    for candidate in sutton_chen_candidates() {
        match unsafe { libloading::Library::new(&candidate) } {
            Ok(library) => {
                let symbol: Result<libloading::Symbol<*mut std::ffi::c_void>, _> =
                    unsafe { library.get(b"rgpot_sutton_chen_energy_forces\0") };
                if symbol.is_ok() {
                    return Ok(library);
                }
                failures.push(format!(
                    "{candidate}: missing rgpot_sutton_chen_energy_forces"
                ));
            }
            Err(error) => failures.push(format!("{candidate}: {error}")),
        }
    }
    Err(format!(
        "rgpot Sutton-Chen library not found ({})",
        failures.join("; ")
    ))
}

fn evaluate(lib: &libloading::Library, family: &Family, x: ArrayView1<f64>) -> (f64, Array1<f64>) {
    let n = x.len() / 3;
    let cfg = ScConfig {
        epsilon: 1.0,
        c: family.c,
        a: 1.0,
        n: family.n_exp,
        m: family.m_exp,
        cutoff: 0.0,
        periodic: 0,
    };
    let mut energy = 0.0;
    let mut forces = vec![0.0; x.len()];
    let positions = x.as_slice().expect("coordinates are contiguous");
    type Eval = unsafe extern "C" fn(
        *const ScConfig,
        i32,
        *const f64,
        *const f64,
        *mut f64,
        *mut f64,
    ) -> i32;
    let func: libloading::Symbol<Eval> = unsafe {
        lib.get(b"rgpot_sutton_chen_energy_forces\0")
            .expect("rgpot_sutton_chen_energy_forces")
    };
    let status = unsafe {
        func(
            &cfg,
            n as i32,
            positions.as_ptr(),
            std::ptr::null(),
            &mut energy,
            forces.as_mut_ptr(),
        )
    };
    assert_eq!(status, 0, "rgpot Sutton-Chen rejected the geometry");
    // The C entry returns forces. The quench wants dE/dx.
    let gradient = Array1::from_vec(forces.iter().map(|force| -force).collect());
    (energy, gradient)
}

fn config(spec: &Spec) -> Config {
    // Stock atomic preset. The basin stride stays 1, which is what
    // Lennard-Jones uses. This example does not set it.
    Config::recommended(spec.n)
}

fn cmd_run(args: &[String]) -> i32 {
    let Some(spec) = spec_of(args, "run") else {
        return 2;
    };
    let Some(seed) = args.get(2).and_then(|value| value.parse::<u64>().ok()) else {
        eprintln!("run needs a seed");
        return 2;
    };
    let budget = env_budget().unwrap_or(spec.budget);
    let cfg = config(&spec);
    let start = start_of(&cfg, spec.n, seed);
    if std::env::var("H2H_PORTFOLIO").ok().as_deref() == Some("1") {
        let lib = match library() {
            Ok(lib) => lib,
            Err(error) => {
                eprintln!("{error}");
                return 1;
            }
        };
        portfolio_search(&lib, &cfg, &spec, &args[1], &start, seed, budget);
        return 0;
    }
    if let Some(path) = args.get(3)
        && let Err(err) = write_coords(path, &start)
    {
        eprintln!("coords: {err}");
        return 1;
    }
    let lib = match library() {
        Ok(lib) => lib,
        Err(error) => {
            eprintln!("{error}");
            return 1;
        }
    };
    let (outcome, wall, pot_s) = search(&lib, &cfg, &spec, &start, seed, budget);
    report(
        &args[1],
        seed,
        budget,
        &outcome,
        wall,
        pot_s,
        spec.reference,
    );
    0
}

fn cmd_coords(args: &[String]) -> i32 {
    let Some(spec) = spec_of(args, "coords") else {
        return 2;
    };
    let Some(seed) = args.get(2).and_then(|value| value.parse::<u64>().ok()) else {
        eprintln!("coords needs a seed");
        return 2;
    };
    let Some(path) = args.get(3) else {
        eprintln!("coords needs a path");
        return 2;
    };
    let cfg = config(&spec);
    let start = start_of(&cfg, spec.n, seed);
    match write_coords(path, &start) {
        Ok(()) => 0,
        Err(err) => {
            eprintln!("coords: {err}");
            1
        }
    }
}

fn cmd_budget(args: &[String]) -> i32 {
    let Some(spec) = spec_of(args, "budget") else {
        return 2;
    };
    println!("{}", spec.budget);
    println!("{:.10}", spec.reference);
    0
}

fn cmd_energy(args: &[String]) -> i32 {
    let Some((spec, coords)) = geometry_of(args, "energy") else {
        return 2;
    };
    let lib = match library() {
        Ok(lib) => lib,
        Err(error) => {
            eprintln!("{error}");
            return 1;
        }
    };
    let (energy, _) = evaluate(&lib, &spec.family, coords.view());
    println!("{energy:.12}");
    0
}

fn cmd_grad(args: &[String]) -> i32 {
    let Some((spec, coords)) = geometry_of(args, "grad") else {
        return 2;
    };
    let lib = match library() {
        Ok(lib) => lib,
        Err(error) => {
            eprintln!("{error}");
            return 1;
        }
    };
    let (energy, gradient) = evaluate(&lib, &spec.family, coords.view());
    println!("{energy:.12}");
    for atom in gradient.exact_chunks(3) {
        println!("{:.12} {:.12} {:.12}", atom[0], atom[1], atom[2]);
    }
    0
}

fn spec_of(args: &[String], command: &str) -> Option<Spec> {
    let Some(name) = args.get(1).map(String::as_str) else {
        eprintln!("{command} needs a system");
        return None;
    };
    let Some(spec) = system_spec(name) else {
        eprintln!("unknown system {name}");
        return None;
    };
    Some(spec)
}

fn geometry_of(args: &[String], command: &str) -> Option<(Spec, Array1<f64>)> {
    let spec = spec_of(args, command)?;
    let Some(path) = args.get(2) else {
        eprintln!("{command} needs a coordinate file");
        return None;
    };
    let coords = match read_coords(path) {
        Ok(coords) => coords,
        Err(err) => {
            eprintln!("{command}: {err}");
            return None;
        }
    };
    if coords.len() != 3 * spec.n {
        eprintln!("{command}: expected {} numbers", 3 * spec.n);
        return None;
    }
    Some((spec, coords))
}

fn env_budget() -> Option<usize> {
    let text = std::env::var("H2H_BUDGET").ok()?;
    let budget = text.parse::<usize>().ok()?;
    (budget > 0).then_some(budget)
}

fn start_of(cfg: &Config, n: usize, seed: u64) -> Array1<f64> {
    let mut rng = StdRng::seed_from_u64(seed);
    random_cluster_in_radius(n, cfg.start_radius(), cfg.min_separation, &mut rng)
}

/// Record counts and the best geometry shared by the chain and the pair.
///
/// Each arm starts with one record, so the first share is defined. A new
/// finite minimum on an arm adds one record. The pair's calls are counted
/// apart from the chain's so the cap can be read without a second pass.
struct Portfolio {
    stop: Arc<AtomicBool>,
    budget: usize,
    threshold: f64,
    spent: AtomicUsize,
    best: AtomicU64,
    best_state: Mutex<Option<Array1<f64>>>,
    arm_best: [AtomicU64; 2],
    records: [AtomicUsize; 2],
    pair_calls: AtomicUsize,
    hit_at: AtomicUsize,
}

impl Portfolio {
    fn new(budget: usize, threshold: f64) -> Arc<Self> {
        let inf = f64::INFINITY.to_bits();
        Arc::new(Self {
            stop: Arc::new(AtomicBool::new(false)),
            budget,
            threshold,
            spent: AtomicUsize::new(0),
            best: AtomicU64::new(inf),
            best_state: Mutex::new(None),
            arm_best: [AtomicU64::new(inf), AtomicU64::new(inf)],
            records: [AtomicUsize::new(1), AtomicUsize::new(1)],
            pair_calls: AtomicUsize::new(0),
            hit_at: AtomicUsize::new(0),
        })
    }

    fn note(&self, arm: usize, energy: f64, state: ArrayView1<f64>) {
        if !energy.is_finite() {
            return;
        }
        if atomic_min(&self.arm_best[arm], energy) {
            self.records[arm].fetch_add(1, Ordering::Relaxed);
        }
        if !atomic_min(&self.best, energy) {
            return;
        }
        *self
            .best_state
            .lock()
            .unwrap_or_else(|poison| poison.into_inner()) = Some(state.to_owned());
        if energy <= self.threshold
            && self
                .hit_at
                .compare_exchange(
                    0,
                    self.spent.load(Ordering::Relaxed).max(1),
                    Ordering::Relaxed,
                    Ordering::Relaxed,
                )
                .is_ok()
        {
            self.stop.store(true, Ordering::Relaxed);
        }
    }

    fn finished(&self) -> bool {
        self.stop.load(Ordering::Relaxed) || self.spent.load(Ordering::Relaxed) >= self.budget
    }

    /// Calls for the next pair slice.
    ///
    /// The share is the pair's records over the chain's, times the
    /// checkpoint interval, and never more than that interval. While the
    /// chain has at least as many records, the pair's calls stay within a
    /// fifth of the budget. Once the pair leads, the cap is half.
    fn pair_burst(&self, interval: usize) -> usize {
        let single = self.records[0].load(Ordering::Relaxed).max(1);
        let pair_raw = self.records[1].load(Ordering::Relaxed);
        let pair = pair_raw.max(1);
        let cap = if pair_raw > single {
            self.budget / 2
        } else {
            self.budget / 5
        };
        let room = cap.saturating_sub(self.pair_calls.load(Ordering::Relaxed));
        let share = (pair.saturating_mul(interval) / single)
            .min(interval)
            .min(room);
        let remaining = self
            .budget
            .saturating_sub(self.spent.load(Ordering::Relaxed));
        share.min(remaining)
    }
}

fn atomic_min(cell: &AtomicU64, value: f64) -> bool {
    let mut current = cell.load(Ordering::Relaxed);
    loop {
        let now = f64::from_bits(current);
        if value.partial_cmp(&now) != Some(std::cmp::Ordering::Less) {
            return false;
        }
        match cell.compare_exchange_weak(
            current,
            value.to_bits(),
            Ordering::Relaxed,
            Ordering::Relaxed,
        ) {
            Ok(_) => return true,
            Err(seen) => current = seen,
        }
    }
}

/// One evaluation against the shared budget. The ledger still counts the
/// slice; the portfolio counts the campaign.
fn begin_eval(portfolio: &Portfolio, ledger: &mut Ledger) -> bool {
    if portfolio.finished() {
        portfolio.stop.store(true, Ordering::Relaxed);
        return false;
    }
    ledger.charge()
}

fn account(portfolio: &Portfolio, pair: bool) {
    let spent = portfolio.spent.fetch_add(1, Ordering::Relaxed) + 1;
    if pair {
        portfolio.pair_calls.fetch_add(1, Ordering::Relaxed);
    }
    if spent >= portfolio.budget {
        portfolio.stop.store(true, Ordering::Relaxed);
    }
}

fn descriptor_distance(left: ArrayView1<f64>, right: ArrayView1<f64>) -> f64 {
    left.iter()
        .zip(right.iter())
        .map(|(p, q)| (p - q) * (p - q))
        .sum::<f64>()
        .sqrt()
}

fn offer_archive(
    archive: &mut Vec<(Array1<f64>, Array1<f64>, f64)>,
    bias: &BasinBias<ClusterFingerprint>,
    state: &Array1<f64>,
    energy: f64,
    distinct: f64,
) {
    if !energy.is_finite() {
        return;
    }
    let desc = bias.index().describe(state.view());
    let mut nearest = None;
    let mut nearest_d = f64::MAX;
    for (index, (held, _, _)) in archive.iter().enumerate() {
        let distance = descriptor_distance(held.view(), desc.view());
        if distance < nearest_d {
            nearest_d = distance;
            nearest = Some(index);
        }
    }
    if nearest_d <= distinct
        && let Some(index) = nearest
    {
        if energy < archive[index].2 {
            archive[index] = (desc, state.clone(), energy);
        }
        return;
    }
    if archive.len() < 32 {
        archive.push((desc, state.clone(), energy));
    } else if let Some(index) = nearest
        && energy < archive[index].2
    {
        archive[index] = (desc, state.clone(), energy);
    }
}

fn farthest(
    archive: &[(Array1<f64>, Array1<f64>, f64)],
    here: ArrayView1<f64>,
    distinct: f64,
) -> Option<(Array1<f64>, f64)> {
    archive
        .iter()
        .filter_map(|(desc, state, _)| {
            let distance = descriptor_distance(desc.view(), here);
            (distance > distinct).then_some((state.clone(), distance))
        })
        .max_by(|left, right| {
            left.1
                .partial_cmp(&right.1)
                .unwrap_or(std::cmp::Ordering::Equal)
        })
}

struct PairWalk {
    states: Vec<(Array1<f64>, f64)>,
    rngs: Vec<StdRng>,
    bias: Option<BasinBias<ClusterFingerprint>>,
    archive: Vec<(Array1<f64>, Array1<f64>, f64)>,
    turn: usize,
}

/// Two walkers for `burst` calls. The walkers, the bias and the archive
/// stay put between slices. A cut-and-splice is taken from the archive
/// member farthest from the live walker.
#[allow(clippy::too_many_arguments)]
fn pair_segment(
    lib: &libloading::Library,
    cfg: &Config,
    spec: &Spec,
    start: &Array1<f64>,
    seed: u64,
    portfolio: &Arc<Portfolio>,
    walk: &mut PairWalk,
    burst: usize,
    pot_ns: &AtomicU64,
) {
    if burst == 0 || portfolio.finished() {
        return;
    }
    let mut ledger = Ledger::with_stop(burst, Arc::clone(&portfolio.stop));
    ledger.stop_below(portfolio.threshold);
    let mut optimiser = WarmLbfgs::default();
    let mut relax = |led: &mut Ledger, x: ArrayView1<f64>, iterations: usize| {
        optimiser.forget();
        let (value, relaxed, _) = optimiser.minimize(x, iterations, |v| {
            if !begin_eval(portfolio, led) {
                return None;
            }
            let mark = Instant::now();
            let pair = evaluate(lib, &spec.family, v);
            pot_ns.fetch_add(mark.elapsed().as_nanos() as u64, Ordering::Relaxed);
            account(portfolio, true);
            Some(pair)
        });
        portfolio.note(1, value, relaxed.view());
        (value, relaxed)
    };
    let mut grad = |led: &mut Ledger, v: ArrayView1<f64>| {
        if !begin_eval(portfolio, led) {
            return None;
        }
        let mark = Instant::now();
        let gradient = evaluate(lib, &spec.family, v).1;
        pot_ns.fetch_add(mark.elapsed().as_nanos() as u64, Ordering::Relaxed);
        account(portfolio, true);
        Some(gradient)
    };
    if walk.states.is_empty() {
        let walkers = 2;
        walk.rngs = (0..walkers)
            .map(|index| {
                StdRng::seed_from_u64(if index == 0 {
                    seed
                } else {
                    seed.wrapping_add(index as u64)
                })
            })
            .collect();
        for index in 0..walkers {
            if ledger.remaining() == 0 || portfolio.finished() {
                break;
            }
            let drawn = if index == 0 {
                start.clone()
            } else {
                random_cluster_in_radius(
                    spec.n,
                    cfg.start_radius(),
                    cfg.min_separation,
                    &mut walk.rngs[index],
                )
            };
            let (energy, quenched) = relax(&mut ledger, drawn.view(), cfg.relax_steps);
            if energy.is_finite() {
                walk.states.push((quenched, energy));
            }
        }
        if walk.states.is_empty() {
            return;
        }
        walk.bias = Some(BasinBias::new(
            ClusterFingerprint::of_config(cfg, &walk.states[0].0),
            cfg.merge_radius,
            cfg.bias_height,
            cfg.bias_gamma,
        ));
        let distinct = 4.0 * cfg.merge_radius;
        let bias = walk.bias.as_ref().expect("bias");
        for (state, energy) in &walk.states {
            offer_archive(&mut walk.archive, bias, state, *energy, distinct);
        }
    }
    let distinct = 4.0 * cfg.merge_radius;
    let slice_len = (burst / (walk.states.len() * 2)).max(1);
    while ledger.remaining() > 0 && !ledger.target_reached() && !portfolio.finished() {
        let index = walk.turn % walk.states.len();
        walk.turn += 1;
        let take = ledger.remaining().min(slice_len);
        if take == 0 {
            break;
        }
        let mut slice = ledger.slice(take);
        if let Some(target) = ledger.target() {
            slice.stop_below(target);
        }
        let bias = walk.bias.as_mut().expect("bias");
        let walked = run_with_bias(
            cfg,
            walk.states[index].0.view(),
            &mut slice,
            &mut relax,
            Some(&mut grad),
            bias,
            &mut walk.rngs[index],
        );
        ledger.charge_many(slice.spent());
        if let Some(state) = slice.best_state.as_ref() {
            ledger.record(slice.best, state.view());
            offer_archive(&mut walk.archive, bias, state, slice.best, distinct);
        }
        if let Ok(outcome) = &walked
            && let Some(state) = outcome.final_state.as_ref()
        {
            walk.states[index] = (state.clone(), outcome.final_energy);
            offer_archive(
                &mut walk.archive,
                bias,
                state,
                outcome.final_energy,
                distinct,
            );
        }
        if ledger.target_reached() || ledger.remaining() == 0 || portfolio.finished() {
            break;
        }
        let here = bias.index().describe(walk.states[index].0.view());
        if let Some((partner, _)) = farthest(&walk.archive, here.view(), distinct) {
            let child = cut_and_splice(
                walk.states[index].0.view(),
                partner.view(),
                cfg.species.as_deref(),
                cfg.min_separation,
                &mut walk.rngs[index],
            );
            let (energy, quenched) = relax(&mut ledger, child.view(), cfg.relax_steps);
            if energy.is_finite() {
                let bias = walk.bias.as_ref().expect("bias");
                offer_archive(&mut walk.archive, bias, &quenched, energy, distinct);
                if energy < walk.states[index].1 {
                    walk.states[index] = (quenched, energy);
                }
            }
        }
    }
}

/// One chain and a pair share the budget.
///
/// The pair opens on an eighth of the budget. The chain then runs, and at
/// each eighth a pair slice takes its record count over the chain's.
fn portfolio_search(
    lib: &libloading::Library,
    cfg: &Config,
    spec: &Spec,
    system: &str,
    start: &Array1<f64>,
    seed: u64,
    budget: usize,
) {
    let portfolio = Portfolio::new(budget, spec.reference + HIT_TOL);
    let mut walk = PairWalk {
        states: Vec::new(),
        rngs: Vec::new(),
        bias: None,
        archive: Vec::new(),
        turn: 0,
    };
    let pot_ns = AtomicU64::new(0);
    let started = Instant::now();
    let interval = (budget / 8).max(1);
    let opening = budget.min(interval);
    pair_segment(
        lib, cfg, spec, start, seed, &portfolio, &mut walk, opening, &pot_ns,
    );
    if !portfolio.finished() {
        let mut ledger = Ledger::with_stop(budget, Arc::clone(&portfolio.stop));
        ledger.stop_below(portfolio.threshold);
        let mut optimiser = WarmLbfgs::default();
        let mut relax = |led: &mut Ledger, x: ArrayView1<f64>, iterations: usize| {
            optimiser.forget();
            let (value, relaxed, _) = optimiser.minimize(x, iterations, |v| {
                if !begin_eval(&portfolio, led) {
                    return None;
                }
                let mark = Instant::now();
                let pair = evaluate(lib, &spec.family, v);
                pot_ns.fetch_add(mark.elapsed().as_nanos() as u64, Ordering::Relaxed);
                account(&portfolio, false);
                Some(pair)
            });
            portfolio.note(0, value, relaxed.view());
            (value, relaxed)
        };
        let mut grad = |led: &mut Ledger, v: ArrayView1<f64>| {
            if !begin_eval(&portfolio, led) {
                return None;
            }
            let mark = Instant::now();
            let gradient = evaluate(lib, &spec.family, v).1;
            pot_ns.fetch_add(mark.elapsed().as_nanos() as u64, Ordering::Relaxed);
            account(&portfolio, false);
            Some(gradient)
        };
        let mut rng = StdRng::seed_from_u64(seed);
        let _ = run_with_energy_gradient_at_checkpoints(
            cfg,
            start.view(),
            &mut ledger,
            &mut relax,
            Some(&mut grad),
            None,
            &mut rng,
            interval,
            &mut |_checkpoint| {
                if portfolio.finished() {
                    return CheckpointAction::Retire {
                        reason: "portfolio".to_string(),
                    };
                }
                let single = portfolio.records[0].load(Ordering::Relaxed).max(1);
                let pair = portfolio.records[1].load(Ordering::Relaxed).max(1);
                let burst = portfolio.pair_burst(interval);
                let best = f64::from_bits(portfolio.best.load(Ordering::Relaxed));
                eprintln!("portfolio single {single} pair {pair} burst {burst} best {best:.8}");
                if burst > 0 {
                    pair_segment(
                        lib, cfg, spec, start, seed, &portfolio, &mut walk, burst, &pot_ns,
                    );
                }
                if portfolio.finished() {
                    CheckpointAction::Retire {
                        reason: "portfolio".to_string(),
                    }
                } else {
                    CheckpointAction::Continue
                }
            },
        );
    }
    let wall = started.elapsed().as_secs_f64();
    let pot_s = pot_ns.load(Ordering::Relaxed) as f64 / 1.0e9;
    report_portfolio(
        &portfolio,
        system,
        seed,
        budget,
        spec.reference,
        wall,
        pot_s,
    );
}

fn report_portfolio(
    portfolio: &Portfolio,
    system: &str,
    seed: u64,
    budget: usize,
    reference: f64,
    wall: f64,
    pot_s: f64,
) {
    let best = f64::from_bits(portfolio.best.load(Ordering::Relaxed));
    let hit = best <= reference + HIT_TOL;
    let charged = portfolio.spent.load(Ordering::Relaxed);
    let evals = if hit {
        portfolio.hit_at.load(Ordering::Relaxed).max(1)
    } else {
        budget
    };
    let outside = (wall - pot_s).max(0.0);
    let calls = charged.max(1) as f64;
    println!(
        "system,code,seed,budget,hit,evals_to_hit,charged,wall_s,best,pot_s,outside_s,pot_per_eval_s,outside_per_eval_s"
    );
    println!(
        "{system},anneal,{seed},{budget},{hit},{evals},{charged},{wall:.6},{best:.8},{pot_s:.6},{outside:.6},{:.9},{:.9}",
        pot_s / calls,
        outside / calls,
    );
    if let Ok(path) = std::env::var("H2H_BEST")
        && let Some(state) = portfolio
            .best_state
            .lock()
            .unwrap_or_else(|poison| poison.into_inner())
            .as_ref()
        && let Err(err) = write_coords(&path, state)
    {
        eprintln!("best: {err}");
    }
}

fn search(
    lib: &libloading::Library,
    cfg: &Config,
    spec: &Spec,
    start: &Array1<f64>,
    seed: u64,
    budget: usize,
) -> (Outcome, f64, f64) {
    let mut ledger = Ledger::new(budget);
    ledger.stop_below(spec.reference + HIT_TOL);
    let mut optimiser = WarmLbfgs::default();
    let pot_ns = std::sync::atomic::AtomicU64::new(0);
    let mut relax = |led: &mut Ledger, x: ArrayView1<f64>, iterations: usize| {
        optimiser.forget();
        let (value, relaxed, _) = optimiser.minimize(x, iterations, |v| {
            if !led.charge() {
                return None;
            }
            let mark = Instant::now();
            let pair = evaluate(lib, &spec.family, v);
            pot_ns.fetch_add(
                mark.elapsed().as_nanos() as u64,
                std::sync::atomic::Ordering::Relaxed,
            );
            Some(pair)
        });
        (value, relaxed)
    };
    let mut grad = |led: &mut Ledger, v: ArrayView1<f64>| {
        if !led.charge() {
            return None;
        }
        let mark = Instant::now();
        let gradient = evaluate(lib, &spec.family, v).1;
        pot_ns.fetch_add(
            mark.elapsed().as_nanos() as u64,
            std::sync::atomic::Ordering::Relaxed,
        );
        Some(gradient)
    };
    let mut rng = StdRng::seed_from_u64(seed);
    let started = Instant::now();
    let outcome = run_with_gradient(
        cfg,
        start.view(),
        &mut ledger,
        &mut relax,
        Some(&mut grad),
        &mut rng,
    )
    .expect("finite start");
    let pot_s = pot_ns.load(std::sync::atomic::Ordering::Relaxed) as f64 / 1.0e9;
    (outcome, started.elapsed().as_secs_f64(), pot_s)
}

fn report(
    system: &str,
    seed: u64,
    budget: usize,
    outcome: &Outcome,
    wall: f64,
    pot_s: f64,
    reference: f64,
) {
    let level = reference + HIT_TOL;
    let hit = outcome.best <= level;
    let at_hit = outcome
        .improvements
        .iter()
        .find(|(_, _, _, energy)| *energy <= level)
        .map(|(_, spent, _, _)| *spent);
    let evals = if hit {
        at_hit.unwrap_or(budget)
    } else {
        budget
    };
    let outside = (wall - pot_s).max(0.0);
    let calls = outcome.charged.max(1) as f64;
    println!(
        "system,code,seed,budget,hit,evals_to_hit,charged,wall_s,best,pot_s,outside_s,pot_per_eval_s,outside_per_eval_s"
    );
    println!(
        "{system},anneal,{seed},{budget},{hit},{evals},{},{wall:.6},{:.8},{pot_s:.6},{outside:.6},{:.9},{:.9}",
        outcome.charged,
        outcome.best,
        pot_s / calls,
        outside / calls,
    );
    // After the search. The row above is the result; these lines are the
    // funnel and the arms that produced it.
    if let Ok(path) = std::env::var("H2H_BEST")
        && let Some(state) = outcome.best_state.as_ref()
        && let Err(err) = write_coords(&path, state)
    {
        eprintln!("best: {err}");
    }
    if let Ok(path) = std::env::var("H2H_FINAL")
        && let Some(state) = outcome.final_state.as_ref()
        && let Err(err) = write_coords(&path, state)
    {
        eprintln!("final: {err}");
    }
    eprintln!(
        "diag final={:.8} hops={} accepted={} screened={} returned={} restarts={} stall_escapes={} trail_escapes={} paths={} path_improvements={} stall_gain={:.6}",
        outcome.final_energy,
        outcome.hops,
        outcome.accepted,
        outcome.screened_out,
        outcome.returned,
        outcome.restarts,
        outcome.stall_escapes,
        outcome.trail_escapes,
        outcome.paths,
        outcome.path_improvements,
        outcome.stall_escape_gain,
    );
    for (name, draws, accepts, best) in &outcome.arms {
        eprintln!("arm {name} draws {draws} accepts {accepts} best {best:.8}");
    }
    for count in &outcome.proposal_counts {
        eprintln!(
            "move {} proposed {} accepted {} improved {}",
            count.move_type, count.proposed, count.accepted, count.improved
        );
    }
}

fn write_coords(path: &str, coords: &Array1<f64>) -> std::io::Result<()> {
    let mut text = String::new();
    for atom in coords.exact_chunks(3) {
        text.push_str(&format!(
            "{:.10} {:.10} {:.10}\n",
            atom[0], atom[1], atom[2]
        ));
    }
    std::fs::write(path, text)
}

fn read_coords(path: &str) -> std::io::Result<Array1<f64>> {
    let text = std::fs::read_to_string(path)?;
    let mut values = Vec::new();
    for line in text.lines() {
        let fields: Vec<&str> = line.split_whitespace().collect();
        let numbers = if fields.len() == 4 && fields[0].parse::<u32>().is_ok() {
            &fields[1..]
        } else {
            fields.as_slice()
        };
        for word in numbers {
            if let Ok(value) = word.parse::<f64>() {
                values.push(value);
            }
        }
    }
    Ok(Array1::from_vec(values))
}

#[cfg(test)]
mod tests {
    use super::{evaluate, library, system_spec};
    use anneal_core::methods::cluster_hopping::Config;
    use ndarray::Array1;

    /// Two atoms at r = a. The pair term is 1 and each density square root
    /// is 1, so the energy is 1 − 2c.
    #[test]
    fn a_dimer_at_a_has_the_reduced_sutton_chen_energy() {
        let Ok(lib) = library() else {
            eprintln!("rgpot Sutton-Chen library not found; skipping the dimer check");
            return;
        };
        for name in ["sc126-38", "sc108-38", "sc96-38"] {
            let spec = system_spec(name).unwrap();
            let x = Array1::from_vec(vec![0.0, 0.0, 0.0, 1.0, 0.0, 0.0]);
            let (energy, gradient) = evaluate(&lib, &spec.family, x.view());
            let closed = 1.0 - 2.0 * spec.family.c;
            assert!(
                (energy - closed).abs() < 1e-8,
                "{name} dimer {energy} closed {closed}"
            );
            let step = 1e-5;
            let plus = Array1::from_vec(vec![0.0, 0.0, 0.0, 1.0 + step, 0.0, 0.0]);
            let minus = Array1::from_vec(vec![0.0, 0.0, 0.0, 1.0 - step, 0.0, 0.0]);
            let numeric = (evaluate(&lib, &spec.family, plus.view()).0
                - evaluate(&lib, &spec.family, minus.view()).0)
                / (2.0 * step);
            assert!(
                (gradient[3] - numeric).abs() < 1e-4,
                "{name} gradient {} numeric {numeric}",
                gradient[3]
            );
        }
    }

    #[test]
    fn the_campaign_uses_the_cambridge_table_and_the_lj_preset() {
        let sc38 = system_spec("sc108-38").unwrap();
        assert_eq!((sc38.n, sc38.budget), (38, 400_000));
        assert!((sc38.reference + 10117.2454).abs() < 1e-10);
        assert_eq!(system_spec("sc108-55").unwrap().budget, 1_000_000);
        assert_eq!(system_spec("sc126-80").unwrap().budget, 1_000_000);
        assert!(system_spec("sc108-13").is_none());
        let cfg = super::config(&sc38);
        let stock = Config::recommended(38);
        assert_eq!(cfg.temperature, stock.temperature);
        assert_eq!(cfg.basin_stride, 1);
        assert!(cfg.species.is_none());
        assert_eq!(cfg.move_library, stock.move_library);
    }
}
