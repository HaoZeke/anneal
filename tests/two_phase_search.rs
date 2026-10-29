use anneal_core::methods::cluster_hopping::{Config, Ledger, MoveLibrary};
use anneal_core::methods::cluster_search::search;
use anneal_core::methods::two_phase::TwoPhase;
use anneal_core::potentials::PairPotential;

const LJ38_OCTAHEDRON: f64 = -173.928427;

fn run(cfg: &Config, budget: usize, seed: u64) -> (f64, usize, usize) {
    let pot = PairPotential::lennard_jones(cfg.n_points);
    let mut ledger = Ledger::new(budget);
    let (out, _) = search(&pot, cfg, &mut ledger, seed).expect("finite start");
    (out.best, out.hops, ledger.spent())
}

#[test]
fn the_relative_cutoff_reaches_the_lj38_octahedron_where_the_plain_walk_need_not() {
    let budget = 30_000;
    let mut plain = Config::recommended(38);
    // The surface comparison is the LeanBurst walk. The 38-point reseed
    // and the stall exits are a different measurement.
    plain.move_library = MoveLibrary::LeanBurst;
    plain.trail_on_stall = false;
    plain.escape_on_stall = false;
    plain.restart_on_stall = false;
    plain.bayes_screen = false;
    plain.surfaces.clear();
    let mut compacted = plain.clone();
    compacted.two_phase = Some(TwoPhase::relative(0.7, 1.0));
    for seed in 0..4 {
        let (best, hops, spent) = run(&compacted, budget, seed);
        assert!(
            best < LJ38_OCTAHEDRON + 1e-4,
            "seed {seed}: the compacted walk stopped at {best:.6} after {hops} hops"
        );
        assert!(spent <= budget, "the ledger overspent: {spent} of {budget}");
        let (_, plain_hops, plain_spent) = run(&plain, budget, seed);
        assert!(plain_spent <= budget);
        assert!(
            hops < plain_hops,
            "seed {seed}: two relaxations per hop must buy fewer hops than one ({hops} against {plain_hops})"
        );
    }
}

#[test]
fn a_surface_portfolio_runs_to_the_end_of_its_ledger() {
    let mut cfg = Config::recommended(13);
    cfg.surfaces = vec![
        TwoPhase::relative(0.7, 1.0),
        TwoPhase {
            cutoff: anneal_core::methods::two_phase::Cutoff::Fixed(0.0),
            beta: 0.0,
            mu: 5.0,
            anisotropic: false,
        },
    ];
    let (best, hops, spent) = run(&cfg, 6_000, 3);
    assert!(best.is_finite() && hops > 0);
    assert_eq!(spent, 6_000, "the portfolio walk did not spend its ledger");
    assert!(best < -44.3268 + 1e-3, "LJ13 icosahedron missed: {best:.6}");
}
