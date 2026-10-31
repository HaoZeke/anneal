//! Arm 3 on a Sutton–Chen cluster: `for_cluster`'s two-core orbit completion.
//!
//! One random start, one seed, one charged budget. The Cambridge energy is a
//! stopping threshold only. Coordinates of the published minimum are never
//! read.
//!
//! ```text
//! sc_arm3 <12-6|9-6> <n> <budget> <seed> <reference-energy>
//! ```

use std::sync::atomic::{AtomicBool, AtomicUsize, Ordering};
use std::sync::Arc;
use std::time::Instant;

use anneal_core::methods::cluster_hopping::{Config, Ledger, OrbitCore, optimize};
use anneal_core::methods::warm_lbfgs::WarmLbfgs;
use anneal_core::potentials::{SuttonChen, SuttonChenFamily};
use anneal_core::symmetrise::ShellCore;
use ndarray::{Array1, ArrayView1};

/// GMIN's default basin-hopping quench: RMS force `1e-3`, at most 500 steps.
const QUENCH_RMS: f64 = 1.0e-3;
const QUENCH_ITERS: usize = 500;
/// A hit is a validated quench at or below the Cambridge energy plus this window.
const HIT_WINDOW: f64 = 1.0e-3;

fn check_points(args: &[String]) {
    if args.len() != 4 {
        eprintln!("usage: sc_arm3 check <12-6|9-6> <points-file>");
        std::process::exit(2);
    }
    let family = SuttonChenFamily::parse(&args[2]).expect("family");
    let text = std::fs::read_to_string(&args[3]).expect("points file");
    let mut coords = Vec::new();
    for (i, line) in text.lines().enumerate() {
        if i == 0 {
            continue;
        }
        let parts: Vec<f64> = line.split_whitespace().filter_map(|v| v.parse().ok()).collect();
        if parts.len() >= 3 {
            coords.extend_from_slice(&parts[..3]);
        }
    }
    let n = coords.len() / 3;
    let pot = SuttonChen::new(n, family);
    let x = Array1::from(coords);
    let (e, g) = pot.value_and_gradient(x.view());
    println!("check_family: {}", family.label());
    println!("check_n: {n}");
    println!("check_energy: {e:.10}");
    println!("check_rms: {:.6e}", rms(g.as_slice().expect("gradient")));
}

fn rms(g: &[f64]) -> f64 {
    let mut s = 0.0;
    for v in g {
        s += v * v;
    }
    (s / g.len().max(1) as f64).sqrt()
}

fn main() {
    let args: Vec<String> = std::env::args().collect();
    if args.get(1).map(String::as_str) == Some("check") {
        check_points(&args);
        return;
    }
    if args.len() != 6 {
        eprintln!(
            "usage: sc_arm3 <12-6|9-6> <n> <budget> <seed> <reference-energy>\n       sc_arm3 check <12-6|9-6> <points-file>"
        );
        std::process::exit(2);
    }
    let family = SuttonChenFamily::parse(&args[1]).unwrap_or_else(|| {
        eprintln!("unknown family {}", args[1]);
        std::process::exit(2);
    });
    let n: usize = args[2].parse().expect("n");
    let budget: usize = args[3].parse().expect("budget");
    let seed: u64 = args[4].parse().expect("seed");
    let reference: f64 = args[5].parse().expect("reference energy");
    let target = reference + HIT_WINDOW;

    let potential = SuttonChen::new(n, family);
    let nn = std::f64::consts::FRAC_1_SQRT_2;
    let lj_rmin = 2.0_f64.powf(1.0 / 6.0);
    let length_scale = nn / lj_rmin;
    let temperature = family.basin_hopping_temperature();
    // `with_scales` multiplies the Lennard-Jones temperature 0.8 by the energy
    // scale. The scale is the published Sutton–Chen basin-hopping temperature
    // divided by that 0.8, so the Metropolis temperature is T* and the bias
    // height stays in the same proportion to T* as 0.25 is to 0.8.
    let mut cfg = Config::with_scales(n, length_scale, temperature / 0.8);
    assert!(
        (cfg.temperature - temperature).abs() < 1e-9,
        "temperature {} did not become T* {temperature}",
        cfg.temperature
    );
    cfg.bayes_screen = true;
    cfg.relax_steps = QUENCH_ITERS;
    cfg.bayes_threshold =
        anneal_core::screen::cost_asymmetric_threshold(cfg.screen_steps, cfg.relax_steps);
    cfg.orbit_complete_on_new = true;
    cfg.orbit_core = OrbitCore::Both(ShellCore::scaled(cfg.length_scale));

    let stop = Arc::new(AtomicBool::new(false));
    let full_quenches = Arc::new(AtomicUsize::new(0));
    let hit_calls = Arc::new(AtomicUsize::new(0));
    let hit_quenches = Arc::new(AtomicUsize::new(0));
    let hit_energy = Arc::new(std::sync::Mutex::new(None::<f64>));

    let mut ledger = Ledger::with_stop(budget, Arc::clone(&stop));
    let mut opt = WarmLbfgs::default();
    let screen_steps = cfg.screen_steps;
    let relax_steps = cfg.relax_steps;

    let pot = potential.clone();
    let stop_r = Arc::clone(&stop);
    let full_quenches_r = Arc::clone(&full_quenches);
    let hit_calls_r = Arc::clone(&hit_calls);
    let hit_quenches_r = Arc::clone(&hit_quenches);
    let hit_energy_r = Arc::clone(&hit_energy);

    let mut relax = move |led: &mut Ledger, x: ArrayView1<f64>, iters: usize| {
        if iters == 0 || led.remaining() == 0 {
            return if led.charge() {
                let (e, _) = pot.value_and_gradient(x);
                (e, x.to_owned())
            } else {
                (f64::INFINITY, x.to_owned())
            };
        }
        let charged_before = led.spent();
        opt.forget();
        let full = iters > screen_steps;
        // Full quenches stop at an infinity-norm of the same 1e-3 GMIN uses
        // as an RMS bound (`BQMAX`). A screen runs its step count.
        opt.gtol = if full { QUENCH_RMS } else { 1.0e9 };
        let cap = if full { iters.min(relax_steps) } else { iters };
        let (mut energy, mut xr, _) = opt.minimize(x, cap, |v| {
            led.charge().then(|| pot.value_and_gradient(v))
        });
        let mut grad = Array1::zeros(0);
        if full && led.charge() {
            let (e2, g2) = pot.value_and_gradient(xr.view());
            energy = e2;
            grad = g2;
        }
        let validated = full
            && grad.len() == xr.len()
            && rms(grad.as_slice().unwrap_or(&[])) <= QUENCH_RMS;
        led.record_quench_boundary(
            charged_before,
            energy,
            xr.clone(),
            validated.then(|| grad.clone()),
        );
        if full {
            let q = full_quenches_r.fetch_add(1, Ordering::Relaxed) + 1;
            if validated && energy <= target && hit_calls_r.load(Ordering::Relaxed) == 0 {
                led.record(energy, xr.view());
                hit_calls_r.store(led.spent().max(1), Ordering::Relaxed);
                hit_quenches_r.store(q, Ordering::Relaxed);
                *hit_energy_r.lock().expect("hit energy") = Some(energy);
                stop_r.store(true, Ordering::Relaxed);
            }
        }
        (energy, xr)
    };

    println!("code: anneal");
    println!("arm: 3");
    println!("family: {}", family.label());
    println!("n: {n}");
    println!("budget: {budget}");
    println!("seed: {seed}");
    println!("reference: {reference:.10}");
    println!("hit_window: {HIT_WINDOW}");
    println!("temperature: {temperature}");
    println!("length_scale: {length_scale:.10}");
    println!("energy_scale: {:.10}", cfg.energy_scale);
    println!("orbit_complete_on_new: {}", cfg.orbit_complete_on_new);
    println!("quench_rms: {QUENCH_RMS}");
    println!("quench_iters: {QUENCH_ITERS}");
    println!(
        "resolved_sha256: {}",
        cfg.resolved_sha256().unwrap_or_else(|_| "unavailable".into())
    );

    let started = Instant::now();
    let outcome = optimize(&cfg, &mut ledger, &mut relax, seed).unwrap_or_else(|err| {
        eprintln!("search error: {err:?}");
        std::process::exit(1);
    });
    let wall = started.elapsed().as_secs_f64();
    let hit_e = hit_energy.lock().expect("hit").clone();
    let calls = hit_calls.load(Ordering::Relaxed);
    let quenches = hit_quenches.load(Ordering::Relaxed);
    let hit = hit_e.is_some() && calls > 0;
    println!("wall_s: {wall:.6}");
    println!("spent: {}", ledger.spent());
    println!("outcome_charged: {}", outcome.charged);
    println!("best: {:.10}", outcome.best);
    println!("final_energy: {:.10}", outcome.final_energy);
    println!("hops: {}", outcome.hops);
    println!("orbits_completed: {}", outcome.orbits.0);
    println!("orbit_gain: {:.6}", outcome.orbits.1);
    println!("full_quenches: {}", full_quenches.load(Ordering::Relaxed));
    println!("hit: {hit}");
    if let Some(e) = hit_e {
        println!("hit_energy: {e:.10}");
        println!("hit_calls: {calls}");
        println!("hit_minimisations: {quenches}");
        println!("below_reference: {}", e < reference - HIT_WINDOW);
    } else {
        println!("hit_energy: none");
        println!("hit_calls: none");
        println!("hit_minimisations: none");
        println!("below_reference: false");
    }
}
