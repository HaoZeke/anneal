// Driver for the rows in this directory. Copy it to examples/ to build.
//! nsc_reach <system> <seed> <search|from>
//!
//! `search` is `cluster_search::search`, the library and Python default
//! entry and `paper_controls`' path. `from` is `search_from` on the start
//! `h2h_gmin_proof`/`h2h_morse` draw. `Config::recommended(n)` unscaled,
//! h2h references, paper budgets, stop at reference + DEFAULT_HIT_TOL.
//! NSC_SURF=1 installs `Config::recommended(75).surfaces`; NSC_NOSURF=1
//! clears the surfaces. Both unset leave the preset.
use anneal_core::methods::cluster_hopping::{Config, Ledger, random_cluster_in_radius};
use anneal_core::methods::cluster_search;
use anneal_core::potentials::PairPotential;
use rand::SeedableRng;
use rand::rngs::StdRng;
use std::time::Instant;

fn main() {
    anneal_core::install_rgpot_kernels!();
    let args: Vec<String> = std::env::args().skip(1).collect();
    let system = args[0].as_str();
    let seed: u64 = args[1].parse().expect("seed");
    let mode = args.get(2).map(String::as_str).unwrap_or("search");
    let (n, rho, reference, budget) = match system {
        "lj103" => (103, None, -575.766131, 4_000_000),
        "m38r6" => (38, Some(6.0), -157.477108, 400_000),
        "m55r14" => (55, Some(14.0), -220.646208, 1_000_000),
        _ => panic!("unknown system"),
    };
    let potential = match rho {
        Some(rho) => PairPotential::morse(n, rho),
        None => PairPotential::lennard_jones(n),
    };
    let mut cfg = Config::recommended(n);
    if std::env::var("NSC_SURF").is_ok_and(|v| v == "1") {
        cfg.surfaces = Config::recommended(75).surfaces.clone();
    }
    if std::env::var("NSC_NOSURF").is_ok_and(|v| v == "1") {
        cfg.surfaces.clear();
    }
    eprintln!("NSC_SURFACES {:?}", cfg.surfaces);
    let tol = anneal_core::DEFAULT_HIT_TOL;
    let mut ledger = Ledger::new(budget);
    ledger.stop_below(reference + tol);
    let started = Instant::now();
    let (outcome, stats) = match mode {
        "search" => cluster_search::search(&potential, &cfg, &mut ledger, seed),
        "from" => {
            let mut rng = StdRng::seed_from_u64(seed);
            let start =
                random_cluster_in_radius(n, cfg.start_radius(), cfg.min_separation, &mut rng);
            cluster_search::search_from(&potential, &cfg, &mut ledger, start.view(), seed)
        }
        _ => panic!("unknown mode"),
    }
    .expect("finite start");
    let wall = started.elapsed().as_secs_f64();
    let hit = anneal_core::energy_hits(outcome.best, reference, tol);
    let at_hit = outcome
        .improvements
        .iter()
        .find(|(_, _, _, energy)| anneal_core::energy_hits(*energy, reference, tol))
        .map(|(_, spent, _, _)| *spent);
    let evals = if hit {
        at_hit.unwrap_or(outcome.charged).min(budget)
    } else {
        budget
    };
    println!(
        "ROW system={system} mode={mode} seed={seed} budget={budget} hit={hit} evals_to_hit={evals} charged={} best={:.8} wall_s={wall:.3} converged={} capped={}",
        outcome.charged, outcome.best, stats.converged, stats.capped
    );
    if let Some(x) = outcome.best_state.as_ref() {
        let (e, g) = potential.value_and_gradient(x.view());
        let gmax = g.iter().fold(0.0_f64, |m, v| m.max(v.abs()));
        let mut opt = anneal_core::methods::warm_lbfgs::WarmLbfgs::default();
        let (er, _, _) = opt.minimize(x.view(), 5000, |v| Some(potential.value_and_gradient(v)));
        println!("VERIFY recomputed_e={e:.8} gmax={gmax:.2e} relaxed_e={er:.8}");
    }
    probe_dump();
}

fn probe_dump() {
    eprintln!("NSC_PROBE n/a");
}
