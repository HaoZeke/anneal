// Copyright (c) anneal developers
//! One seed of anneal on a Stillinger-Weber or Tersoff cluster.
//!
//! ```text
//! h2h_covalent run <system> <seed> [best-coords]
//! h2h_covalent coords <system> <seed> <path>
//! h2h_covalent energy <system> <path>
//! h2h_covalent grad <system> <path>
//! h2h_covalent budget <system>
//! h2h_covalent scales <system>
//! ```
//!
//! The energy and the gradient come from rgpot's `StillingerWeberSi` and
//! `TersoffSiC` through `librgpot_covalent.so`, built by
//! `scripts/covalent_h2h/build_rgpot_covalent.sh`. `RGPOT_COVALENT_LIB`
//! is the library. Stillinger-Weber silicon runs in its reduced units,
//! sigma and epsilon, which are GMIN's `SW` units and the units of the
//! literature tables. Tersoff carbon runs in Angstrom and eV.
//!
//! `H2H_ARM` picks the search. `recommended` is `Config::recommended`.
//! `H2H_PRIORS=0` with it, or `H2H_ARM=nopriors`, switches off the
//! structure priors the preset carries for close-packed clusters: stall
//! symmetrisation, the continuous-symmetry reseed, the Lennard-Jones energy
//! lattice, the two-phase surface and the five prior gates (fivefold SOAP,
//! icosahedral reseed, lattice regrowth, labelled core, motif reseed). The
//! fixed temperature, 0.8 of the Lennard-Jones well, goes with them, and the
//! budget-window law sets it. Orbit completion stays, on every new basin and
//! as an allocator arm, and the search keeps parts of the cluster within the
//! pair's measured reach of each other in place of the container sphere.
//! `general` is the same search on scales measured from the potential's own
//! dimer, the bond length over 2^(1/6) and the well depth, so the reduced
//! constants mean what they mean for Lennard-Jones. The dimer and reach
//! probes are charged to the ledger: about 55 calls, a geometric scan and
//! false position on the bond force for the dimer, and a bisection to a
//! relative 1e-6 for the reach.
//!
//! `H2H_GTOL` sets the quench's gradient tolerance, an infinity norm, in
//! place of 1e-6, and lets a quench at that tolerance be recorded. The row
//! names the arm with `-sloppy`.
//!
//! A hit is a recorded energy at or below the reference plus 0.001 in the
//! system's energy unit. A miss is the full budget. `pot_s` is the time
//! inside the library call.

use anneal_core::methods::cluster_hopping::{
    Config, ContinuousSymmetry, Ledger, MoveLibrary, Outcome, PairRange, SoapProposalMode,
    random_cluster_in_radius, run_with_gradient,
};
use anneal_core::methods::warm_lbfgs::WarmLbfgs;
use ndarray::{Array1, ArrayView1};
use rand::SeedableRng;
use rand::rngs::StdRng;
use std::time::Instant;

const HIT_TOL: f64 = 0.001;

#[derive(Clone, Copy, PartialEq, Eq, Debug)]
enum Kind {
    StillingerWeber,
    TersoffCarbon,
}

impl Kind {
    fn code(self) -> i32 {
        match self {
            Self::StillingerWeber => 0,
            Self::TersoffCarbon => 1,
        }
    }
}

#[derive(Clone, Copy, PartialEq, Eq, Debug)]
enum Arm {
    Recommended,
    NoPriors,
    General,
}

impl Arm {
    fn from_env() -> Result<Self, String> {
        let priors = std::env::var("H2H_PRIORS").unwrap_or_else(|_| "1".to_string());
        let arm = std::env::var("H2H_ARM").unwrap_or_else(|_| "recommended".to_string());
        match (arm.as_str(), priors.as_str()) {
            ("recommended", "1") => Ok(Self::Recommended),
            ("recommended", "0") | ("nopriors", _) => Ok(Self::NoPriors),
            ("general", _) => Ok(Self::General),
            _ => Err(format!("H2H_ARM={arm} H2H_PRIORS={priors}")),
        }
    }

    fn code(self) -> String {
        let name = match self {
            Self::Recommended => "anneal",
            Self::NoPriors => "anneal-nopriors",
            Self::General => "anneal-general",
        };
        if env_f64("H2H_GTOL").is_some() {
            format!("{name}-sloppy")
        } else {
            name.to_string()
        }
    }
}

struct Spec {
    n: usize,
    kind: Kind,
    budget: usize,
    reference: f64,
}

fn system_spec(name: &str) -> Option<Spec> {
    // Si20: Yoo and Zeng, J. Chem. Phys. 119, 1442 (2003), Table II,
    // -1.5094 epsilon per atom, here to eight places. Si10, Si13 and C10 are
    // the lowest minima of bench/covalent-h2h/reference.csv. C20 is the
    // relaxed corannulene bowl, below every unbiased run. The README has
    // the sources. Budgets are 1e5 at 10 and 13 atoms and 4e5 at 20.
    let (n, kind, budget, reference) = match name {
        "sw10" => (10, Kind::StillingerWeber, 100_000, -13.79658026),
        "sw13" => (13, Kind::StillingerWeber, 100_000, -18.50399759),
        "sw20" => (20, Kind::StillingerWeber, 400_000, -30.18725174),
        "tc10" => (10, Kind::TersoffCarbon, 100_000, -55.08979464),
        "tc20" => (20, Kind::TersoffCarbon, 400_000, -120.69889105),
        _ => return None,
    };
    Some(Spec {
        n,
        kind,
        budget,
        reference,
    })
}

struct Potential {
    _lib: libloading::Library,
    eval: Eval,
    kind: Kind,
    length: f64,
    energy: f64,
}

type Eval = unsafe extern "C" fn(i32, std::ffi::c_long, *const f64, *mut f64, *mut f64) -> i32;
type Units = unsafe extern "C" fn(*mut f64, *mut f64);

impl Potential {
    fn open(kind: Kind) -> Self {
        let path = std::env::var("RGPOT_COVALENT_LIB")
            .unwrap_or_else(|_| "/work/covalent-h2h/lib/librgpot_covalent.so".to_string());
        let lib = unsafe { libloading::Library::new(&path) }
            .unwrap_or_else(|err| panic!("rgpot covalent library {path}: {err}"));
        let (length, energy) = match kind {
            Kind::StillingerWeber => {
                let units: libloading::Symbol<Units> = unsafe {
                    lib.get(b"rgpot_covalent_sw_units\0")
                        .expect("rgpot_covalent_sw_units")
                };
                let (mut sigma, mut epsilon) = (0.0, 0.0);
                unsafe { units(&mut sigma, &mut epsilon) };
                (sigma, epsilon)
            }
            Kind::TersoffCarbon => (1.0, 1.0),
        };
        let eval = unsafe {
            *lib.get::<Eval>(b"rgpot_covalent_energy_gradient\0")
                .expect("rgpot_covalent_energy_gradient")
        };
        Self {
            _lib: lib,
            eval,
            kind,
            length,
            energy,
        }
    }

    /// Energy and dE/dx in the system's units.
    fn evaluate(&self, x: ArrayView1<f64>) -> (f64, Array1<f64>) {
        let n = x.len() / 3;
        let positions: Vec<f64> = x.iter().map(|value| value * self.length).collect();
        let mut energy = 0.0;
        let mut gradient = vec![0.0; x.len()];
        let status = unsafe {
            (self.eval)(
                self.kind.code(),
                n as std::ffi::c_long,
                positions.as_ptr(),
                &mut energy,
                gradient.as_mut_ptr(),
            )
        };
        if status != 0 {
            return (f64::INFINITY, Array1::zeros(x.len()));
        }
        let scale = self.length / self.energy;
        (
            energy / self.energy,
            Array1::from_vec(gradient.into_iter().map(|g| g * scale).collect()),
        )
    }

    /// Energy of the dimer at `r` and its slope along the bond.
    fn dimer(&self, r: f64) -> (f64, f64) {
        let (energy, gradient) =
            self.evaluate(Array1::from_vec(vec![0.0, 0.0, 0.0, r, 0.0, 0.0]).view());
        (energy, gradient[3])
    }

    /// Whether the dimer at `r` has exactly zero energy and force.
    fn silent(&self, r: f64) -> bool {
        let (energy, gradient) =
            self.evaluate(Array1::from_vec(vec![0.0, 0.0, 0.0, r, 0.0, 0.0]).view());
        energy == 0.0 && gradient.iter().all(|g| *g == 0.0)
    }
}

/// Bond length and well depth of the dimer.
///
/// A geometric scan walks out until the bond force turns from repulsive to
/// attractive inside the well, and false position on the force (Illinois)
/// narrows that bracket. `charge` is called before every evaluation and stops
/// the probe when it returns false.
fn dimer_scales(potential: &Potential, mut charge: impl FnMut() -> bool) -> Option<(f64, f64)> {
    let mut probe = |r: f64| charge().then(|| potential.dimer(r));
    let mut r = 0.05_f64;
    let mut before = probe(r)?;
    let (mut lo, mut hi, mut slope_lo, mut slope_hi) = loop {
        let next = 1.2 * r;
        if next >= 1.0e3 {
            return None;
        }
        let now = probe(next)?;
        if before.1 < 0.0 && now.1 > 0.0 && before.0.min(now.0) < 0.0 {
            break (r, next, before.1, now.1);
        }
        (r, before) = (next, now);
    };
    let mut kept = 0_i32;
    while hi - lo > 1.0e-10 * hi {
        let mid = (lo * slope_hi - hi * slope_lo) / (slope_hi - slope_lo);
        let (_, slope) = probe(mid)?;
        if slope < 0.0 {
            lo = mid;
            slope_lo = slope;
            kept = kept.min(0) - 1;
            if kept <= -2 {
                slope_hi *= 0.5;
            }
        } else if slope > 0.0 {
            hi = mid;
            slope_hi = slope;
            kept = kept.max(0) + 1;
            if kept >= 2 {
                slope_lo *= 0.5;
            }
        } else {
            lo = mid;
            hi = mid;
        }
    }
    let bond = 0.5 * (lo + hi);
    let (energy, _) = probe(bond)?;
    (energy < 0.0).then(|| (bond / 2.0_f64.powf(1.0 / 6.0), -energy))
}

/// The dimer's reach, probed from its bond out to a thousand bonds.
fn pair_range(
    potential: &Potential,
    bond: f64,
    mut charge: impl FnMut() -> bool,
) -> Option<PairRange> {
    PairRange::measure(bond, 1.0e3 * bond, |r| {
        charge().then(|| potential.silent(r))
    })
}

/// `Config::recommended` without its close-packed structure priors.
///
/// The temperature is a Lennard-Jones calibration too, so the budget-window
/// law replaces it. Orbit completion is symmetry the structure itself
/// carries, so it stays, on every new basin and as an allocator arm. The
/// five prior gates go: the fivefold SOAP move, the icosahedral reseed,
/// lattice regrowth, the labelled core and the motif reseed.
fn without_priors(mut cfg: Config) -> Config {
    cfg.orbit_complete_on_new = true;
    cfg.orbit_arm = true;
    cfg.prior_soap = false;
    cfg.prior_reseed = false;
    cfg.prior_growth = false;
    cfg.prior_core = false;
    cfg.prior_motif = false;
    cfg.symmetrise_on_stall = false;
    cfg.continuous_symmetry = ContinuousSymmetry::Off;
    cfg.energy_lattice = false;
    cfg.surfaces.clear();
    cfg.budget_window = true;
    cfg
}

/// The recommended search flags on measured scales, priors off.
fn general(n: usize, length: f64, energy: f64) -> Config {
    let mut cfg = Config::with_scales(n, length, energy);
    cfg.bayes_screen = true;
    cfg.bayes_threshold =
        anneal_core::screen::cost_asymmetric_threshold(cfg.screen_steps, cfg.relax_steps);
    cfg.move_library = MoveLibrary::LeanBurst;
    cfg.allocate_moves = true;
    cfg.depth_reward = true;
    cfg.tabu_on_stall = true;
    cfg.return_screen = true;
    cfg.soap_class_residual = false;
    cfg.soap_mode = SoapProposalMode::Flexible;
    cfg.trail_on_stall = true;
    cfg.escape_on_stall = true;
    without_priors(cfg)
}

fn main() {
    let args: Vec<String> = std::env::args().skip(1).collect();
    let cmd = args.first().map(String::as_str).unwrap_or("run");
    let status = match cmd {
        "run" => cmd_run(&args),
        "coords" => cmd_coords(&args),
        "energy" => cmd_energy(&args),
        "grad" => cmd_grad(&args),
        "budget" => cmd_budget(&args),
        "scales" => cmd_scales(&args),
        _ => {
            eprintln!("unknown command {cmd}");
            2
        }
    };
    std::process::exit(status);
}

/// One start per system and seed for every arm and every code: the
/// recommended draw on the dimer scale.
fn start_of(potential: &Potential, spec: &Spec, seed: u64) -> Array1<f64> {
    let (length, energy) = dimer_scales(potential, || true).expect("a bound dimer");
    let cfg = Config::with_scales(spec.n, length, energy);
    let mut rng = StdRng::seed_from_u64(seed);
    random_cluster_in_radius(spec.n, cfg.start_radius(), cfg.min_separation, &mut rng)
}

fn cmd_run(args: &[String]) -> i32 {
    let Some(spec) = spec_of(args, "run") else {
        return 2;
    };
    let Some(seed) = args.get(2).and_then(|value| value.parse::<u64>().ok()) else {
        eprintln!("run needs a seed");
        return 2;
    };
    let arm = match Arm::from_env() {
        Ok(arm) => arm,
        Err(err) => {
            eprintln!("unknown arm {err}");
            return 2;
        }
    };
    let budget = env_u64("H2H_BUDGET").map_or(spec.budget, |value| value as usize);
    let target = env_f64("H2H_TARGET").unwrap_or(spec.reference);
    let potential = Potential::open(spec.kind);
    let start = start_of(&potential, &spec, seed);
    let (outcome, wall, pot_s) = search(&potential, arm, &spec, &start, seed, budget, target);
    if let Some(path) = args.get(3)
        && let Some(best) = outcome.best_state.as_ref()
        && let Err(err) = write_coords(path, best)
    {
        eprintln!("coords: {err}");
        return 1;
    }
    report(&args[1], arm, seed, budget, &outcome, wall, pot_s, target);
    0
}

fn search(
    potential: &Potential,
    arm: Arm,
    spec: &Spec,
    start: &Array1<f64>,
    seed: u64,
    budget: usize,
    target: f64,
) -> (Outcome, f64, f64) {
    let mut ledger = Ledger::new(budget);
    ledger.stop_below(target + HIT_TOL);
    let pot_ns = std::sync::atomic::AtomicU64::new(0);
    let timed = |x: ArrayView1<f64>| {
        let mark = Instant::now();
        let pair = potential.evaluate(x);
        pot_ns.fetch_add(
            mark.elapsed().as_nanos() as u64,
            std::sync::atomic::Ordering::Relaxed,
        );
        pair
    };
    let started = Instant::now();
    // Both arms without priors measure the dimer, the bond its reach is
    // scanned from. Only the general arm takes its scales.
    let mark = Instant::now();
    let scales = (arm != Arm::Recommended).then(|| {
        let (length, energy) =
            dimer_scales(potential, || ledger.charge()).expect("the dimer probe fits the budget");
        let bond = length * 2.0_f64.powf(1.0 / 6.0);
        (
            length,
            energy,
            pair_range(potential, bond, || ledger.charge()),
        )
    });
    pot_ns.fetch_add(
        mark.elapsed().as_nanos() as u64,
        std::sync::atomic::Ordering::Relaxed,
    );
    let mut cfg = match (arm, scales) {
        (Arm::General, Some((length, energy, _))) => general(spec.n, length, energy),
        (Arm::NoPriors, _) => without_priors(Config::recommended(spec.n)),
        _ => Config::recommended(spec.n),
    };
    cfg.pair_range = scales.and_then(|(_, _, range)| range);
    let mut optimiser = WarmLbfgs::default();
    if let Some(gtol) = env_f64("H2H_GTOL") {
        optimiser.gtol = gtol;
        cfg.record_gradient = cfg.record_gradient.max(gtol);
    }
    let mut relax = |led: &mut Ledger, x: ArrayView1<f64>, iterations: usize| {
        optimiser.forget();
        let (value, relaxed, _) = optimiser.minimize(x, iterations, |v| {
            if !led.charge() {
                return None;
            }
            Some(timed(v))
        });
        (value, relaxed)
    };
    let mut grad = |led: &mut Ledger, v: ArrayView1<f64>| {
        if !led.charge() {
            return None;
        }
        Some(timed(v).1)
    };
    let mut rng = StdRng::seed_from_u64(seed);
    let outcome = run_with_gradient(
        &cfg,
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

#[allow(clippy::too_many_arguments)]
fn report(
    system: &str,
    arm: Arm,
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
    // The ledger stops at the target, so a hit recorded off the
    // improvement list, such as an orbit completion, is the charged count.
    let evals = if hit {
        at_hit.map_or(outcome.charged, |spent| spent.min(outcome.charged))
    } else {
        budget
    };
    let outside = (wall - pot_s).max(0.0);
    let calls = outcome.charged.max(1) as f64;
    println!(
        "system,code,seed,budget,hit,evals_to_hit,charged,wall_s,best,pot_s,outside_s,pot_per_eval_s,outside_per_eval_s"
    );
    println!(
        "{system},{},{seed},{budget},{hit},{evals},{},{wall:.6},{:.8},{pot_s:.6},{outside:.6},{:.9},{:.9}",
        arm.code(),
        outcome.charged,
        outcome.best,
        pot_s / calls,
        outside / calls,
    );
    for (name, draws, accepts, best) in &outcome.arms {
        println!("arm {name} draws {draws} accepts {accepts} best {best:.6}");
    }
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
    let start = start_of(&Potential::open(spec.kind), &spec, seed);
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
    println!("{}", spec.n);
    0
}

/// The dimer length and energy scales the general arm and the ASE
/// comparison use, then the container radius on that scale, which is
/// GMIN's `RADIUS`.
fn cmd_scales(args: &[String]) -> i32 {
    let Some(spec) = spec_of(args, "scales") else {
        return 2;
    };
    let (length, energy) =
        dimer_scales(&Potential::open(spec.kind), || true).expect("a bound dimer");
    println!("{length:.12}");
    println!("{energy:.12}");
    println!(
        "{:.12}",
        Config::with_scales(spec.n, length, energy).container
    );
    0
}

fn cmd_energy(args: &[String]) -> i32 {
    let Some((spec, coords)) = geometry_of(args, "energy") else {
        return 2;
    };
    let (energy, _) = Potential::open(spec.kind).evaluate(coords.view());
    println!("{energy:.12}");
    0
}

fn cmd_grad(args: &[String]) -> i32 {
    let Some((spec, coords)) = geometry_of(args, "grad") else {
        return 2;
    };
    let (energy, gradient) = Potential::open(spec.kind).evaluate(coords.view());
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

fn env_u64(name: &str) -> Option<u64> {
    let value = std::env::var(name).ok()?.parse::<u64>().ok()?;
    (value > 0).then_some(value)
}

fn env_f64(name: &str) -> Option<f64> {
    std::env::var(name).ok()?.parse::<f64>().ok()
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
        let numbers = if fields.len() == 4 && fields[0].parse::<f64>().is_err() {
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
