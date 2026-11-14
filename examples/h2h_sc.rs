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

use anneal_core::methods::cluster_hopping::{
    Config, Ledger, Outcome, random_cluster_in_radius, run_with_gradient,
};
use anneal_core::methods::warm_lbfgs::WarmLbfgs;
use ndarray::{Array1, ArrayView1};
use rand::SeedableRng;
use rand::rngs::StdRng;
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
