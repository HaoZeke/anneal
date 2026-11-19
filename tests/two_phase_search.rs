use anneal_core::methods::cluster_hopping::{Config, Ledger, MoveLibrary};
use anneal_core::methods::cluster_search::search_lennard_jones;
use anneal_core::methods::two_phase::TwoPhase;
use anneal_core::potentials::PairPotential;

const LJ38_OCTAHEDRON: f64 = -173.928427;

fn run(cfg: &Config, budget: usize, seed: u64) -> (f64, usize, usize) {
    let pot = PairPotential::lennard_jones(cfg.n_points);
    let mut ledger = Ledger::new(budget);
    let (out, _) = search_lennard_jones(&pot, cfg, &mut ledger, seed).expect("finite start");
    (out.best, out.hops, ledger.spent())
}

/// The relative cutoff reaches the LJ38 octahedron on all sixteen starts.
///
/// This is the exact Linux scalar-libm walk for seeds 0 through 15 at a
/// budget of 30,000. It uses the recommended 38-point reseed, a 200-step
/// quench, the LeanBurst library, and no stall exits. The plain walk reaches
/// five starts; the relative-cutoff walk reaches all sixteen.
#[test]
fn the_relative_cutoff_reaches_the_lj38_octahedron_where_the_plain_walk_need_not() {
    anneal_core::install_rgpot_kernels!();
    const SEEDS: u64 = 16;
    let budget = 30_000;
    let mut plain = Config::recommended(38);
    // The surface comparison is the LeanBurst walk. Stall exits are a
    // different measurement, and this walk does not grow a named template.
    // The 30_000-call bars above were measured at a 200-step quench.
    plain.relax_steps = 200;
    plain.move_library = MoveLibrary::LeanBurst;
    plain.trail_on_stall = false;
    plain.escape_on_stall = false;
    plain.restart_on_stall = false;
    plain.bayes_screen = false;
    plain.surfaces.clear();
    let mut compacted = plain.clone();
    compacted.two_phase = Some(TwoPhase::relative(0.7, 1.0));
    let mut compact_hits = 0usize;
    let mut plain_hits = 0usize;
    let mut compact_mask = 0u16;
    let mut plain_mask = 0u16;
    for seed in 0..SEEDS {
        let (best, _, spent) = run(&compacted, budget, seed);
        assert!(spent <= budget, "the ledger overspent: {spent} of {budget}");
        if anneal_core::energy_hits(best, LJ38_OCTAHEDRON, anneal_core::SEARCH_HIT_TOL) {
            compact_hits += 1;
            compact_mask |= 1 << seed;
        }
        let (plain_best, _, plain_spent) = run(&plain, budget, seed);
        assert!(plain_spent <= budget);
        if anneal_core::energy_hits(plain_best, LJ38_OCTAHEDRON, anneal_core::SEARCH_HIT_TOL) {
            plain_hits += 1;
            plain_mask |= 1 << seed;
        }
    }
    // The descriptor-cloud walk is deterministic: cutoff 0xffff, plain
    // 0x8c22 (5 seeds), with and without IRA.
    let plain_exact = 0x8c22;
    assert_eq!(
        compact_mask, 0xffff,
        "the cutoff walk reached the octahedron on {compact_hits} of {SEEDS} seeds (mask {compact_mask:#06x})"
    );
    assert_eq!(
        plain_mask, plain_exact,
        "the plain walk reached the octahedron on {plain_hits} of {SEEDS} seeds (mask {plain_mask:#06x})"
    );
}

#[test]
fn a_surface_portfolio_runs_to_the_end_of_its_ledger() {
    anneal_core::install_rgpot_kernels!();
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
    assert!(
        anneal_core::energy_hits(best, -44.3268, anneal_core::DEFAULT_HIT_TOL),
        "LJ13 icosahedron missed: {best:.6}"
    );
}
