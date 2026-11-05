//! One Morse run of arm 3: `Config::for_cluster` with orbit completion on
//! both cores (shell, then fraction), the lower quench offered.
//!
//! The library `for_cluster` leaves orbit completion off. Arm 3 turns that
//! completion on and reads the point group from both cores. This process
//! does that on the configuration the library builds, after scaling the
//! container, the minimum separation and the length scale by the Morse pair
//! minimum over the Lennard-Jones pair minimum, which is the adjustment
//! `cluster_bench` makes so a Morse cluster is not run in a Lennard-Jones box.
//! The run spends the whole budget, so the wall clock is the time to
//! exhaust it.
//!
//! Usage: `morse_h2h <rho> <n> <budget> <seed> <reference>`, with the hit
//! tolerance read from `HIT_TOLERANCE` (default 1e-4).
//! One JSON object on stdout.

use anneal_core::methods::cluster_hopping::{Config, Ledger, OrbitCore};
use anneal_core::methods::cluster_search::search;
use anneal_core::potentials::PairPotential;
use anneal_core::symmetrise::ShellCore;

fn main() {
    let args: Vec<String> = std::env::args().collect();
    let rho: f64 = args[1].parse().expect("rho");
    let n: usize = args[2].parse().expect("n");
    let budget: usize = args[3].parse().expect("budget");
    let seed: u64 = args[4].parse().expect("seed");
    let reference: f64 = args[5].parse().expect("reference");
    let tolerance: f64 = std::env::var("HIT_TOLERANCE")
        .ok()
        .and_then(|v| v.parse().ok())
        .unwrap_or(1e-4);

    let pot = PairPotential::morse(n, rho);
    let mut cfg = Config::for_cluster(n);
    let scale = pot.kind().r_min() / 2.0_f64.powf(1.0 / 6.0);
    cfg.min_separation *= scale;
    cfg.container *= scale;
    cfg.length_scale *= scale;
    cfg.orbit_complete_on_new = true;
    cfg.orbit_core = OrbitCore::Both(ShellCore::scaled(cfg.length_scale));
    cfg.trace_quenched = true;

    let started = std::time::Instant::now();
    let mut ledger = Ledger::new(budget);
    let (out, stats) = search(&pot, &cfg, &mut ledger, seed).expect("finite start");
    let wall_s = started.elapsed().as_secs_f64();

    let mut first_calls = None;
    let mut first_mins = None;
    for &(hops, charged, _, energy) in &out.improvements {
        if energy < reference + tolerance {
            first_calls = Some(charged);
            first_mins = Some(hops);
            break;
        }
    }
    let hit = out.best < reference + tolerance;
    let minimisations = if out.quenched.is_empty() {
        out.hops
    } else {
        out.quenched.len()
    };
    println!(
        concat!(
            "{{\"code\":\"anneal\",\"rho\":{rho},\"n\":{n},\"seed\":{seed},",
            "\"budget\":{budget},\"reference\":{reference:.10},\"tolerance\":{tolerance},",
            "\"best\":{best:.10},\"hit\":{hit},\"calls\":{calls},\"minimisations\":{mins},",
            "\"first_hit_calls\":{fc},\"first_hit_minimisations\":{fm},",
            "\"wall_s\":{wall:.6},\"hops\":{hops},\"basins\":{basins},",
            "\"converged_quenches\":{conv},\"quench_records\":{qr},",
            "\"config_sha256\":\"{digest}\"}}"
        ),
        rho = rho,
        n = n,
        seed = seed,
        budget = budget,
        reference = reference,
        tolerance = tolerance,
        best = out.best,
        hit = hit,
        calls = ledger.spent(),
        mins = minimisations,
        fc = first_calls
            .map(|v| v.to_string())
            .unwrap_or_else(|| "null".into()),
        fm = first_mins
            .map(|v| v.to_string())
            .unwrap_or_else(|| "null".into()),
        wall = wall_s,
        hops = out.hops,
        basins = out.basins,
        conv = stats.converged,
        qr = out.quenched.len(),
        digest = cfg.resolved_sha256().expect("the configuration serialises"),
    );
}
