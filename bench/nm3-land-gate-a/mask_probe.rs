// Copyright (c) anneal developers

use anneal_core::methods::cluster_hopping::{Config, Ledger, MoveLibrary};
use anneal_core::methods::cluster_search::search_lennard_jones;
use anneal_core::potentials::PairPotential;

const BUDGET: usize = 30_000;
const REFERENCE: f64 = -173.928427;

fn main() {
    anneal_core::install_rgpot_kernels!();
    let seed = std::env::args()
        .nth(1)
        .and_then(|value| value.parse::<u64>().ok())
        .expect("one integer seed is required");
    let mut cfg = Config::recommended(38);
    cfg.relax_steps = 200;
    cfg.move_library = MoveLibrary::LeanBurst;
    cfg.trail_on_stall = false;
    cfg.escape_on_stall = false;
    cfg.restart_on_stall = false;
    cfg.bayes_screen = false;
    cfg.surfaces.clear();

    let potential = PairPotential::lennard_jones(cfg.n_points);
    let mut ledger = Ledger::new(BUDGET);
    let (outcome, _) =
        search_lennard_jones(&potential, &cfg, &mut ledger, seed).expect("finite start");
    let hit = anneal_core::energy_hits(outcome.best, REFERENCE, anneal_core::SEARCH_HIT_TOL);
    let calls = outcome
        .improvements
        .iter()
        .find(|(_, _, _, energy)| {
            anneal_core::energy_hits(*energy, REFERENCE, anneal_core::SEARCH_HIT_TOL)
        })
        .map(|(_, spent, _, _)| *spent)
        .unwrap_or(BUDGET);
    println!("system,seed,budget,hit,calls_to_hit,charged,best");
    println!(
        "lj38-plain-mask,{seed},{BUDGET},{hit},{calls},{},{:.12}",
        ledger.spent(),
        outcome.best
    );
}
