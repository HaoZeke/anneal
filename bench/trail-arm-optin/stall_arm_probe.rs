//! Opt-in stall-arm benchmark driver.
//!
//! `stall_arm_probe <lj13|lj38> <budget> <seed>`

use anneal_core::methods::cluster_hopping::{
    Config, Ledger, random_cluster_in_radius,
};
use anneal_core::methods::cluster_search::search_from;
use anneal_core::potentials::PairPotential;
use rand::SeedableRng;
use rand::rngs::StdRng;
use std::fs;

fn rss_anon_kib() -> u64 {
    fs::read_to_string("/proc/self/status")
        .ok()
        .and_then(|status| {
            status.lines().find_map(|line| {
                let rest = line.strip_prefix("RssAnon:")?;
                rest.split_whitespace().next()?.parse().ok()
            })
        })
        .unwrap_or(0)
}

fn main() {
    anneal_core::install_rgpot_kernels!();
    let args: Vec<String> = std::env::args().skip(1).collect();
    assert_eq!(args.len(), 3, "lj13|lj38 budget seed");
    let (n, reference) = match args[0].as_str() {
        "lj13" => (13, -44.326_801),
        "lj38" => (38, -173.928_427),
        other => panic!("unknown system {other}"),
    };
    let budget: usize = args[1].parse().expect("budget");
    let seed: u64 = args[2].parse().expect("seed");
    let objective = PairPotential::lennard_jones(n);
    let mut cfg = Config::recommended(n);
    cfg.escape_stall_patience = 1;
    let sphere = Config::for_cluster(n);
    let mut rng = StdRng::seed_from_u64(seed);
    let start = random_cluster_in_radius(
        n,
        sphere.start_radius(),
        sphere.min_separation,
        &mut rng,
    );
    let mut ledger = Ledger::new(budget);
    let out = search_from(
        &objective,
        &cfg,
        &mut ledger,
        start.view(),
        seed,
    )
    .expect("finite start")
    .0;
    let calls = ledger
        .encounters()
        .iter()
        .find(|(_, energy)| {
            anneal_core::energy_hits(
                *energy,
                reference,
                anneal_core::DEFAULT_HIT_TOL,
            )
        })
        .map(|(spent, _)| *spent);
    let hit = calls.is_some();
    println!(
        concat!(
            "RESULT system={} seed={} budget={} hit={} calls={} ",
            "charged={} best={:.17e} final={:.17e} final_bits={:016x} ",
            "rss_anon_kib={}"
        ),
        args[0],
        seed,
        budget,
        usize::from(hit),
        calls.unwrap_or(budget + 1),
        ledger.spent(),
        out.best,
        out.final_energy,
        out.final_energy.to_bits(),
        rss_anon_kib(),
    );
}
