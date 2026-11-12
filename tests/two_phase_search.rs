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

/// The relative cutoff reaches the LJ38 octahedron on starts where the plain
/// LeanBurst walk need not.
///
/// Seeds 0..16 at a budget of 30_000, from `recommended` with the 38-point
/// reseed and the stall exits off. The `ira` feature replaces the fivefold
/// permutation in `residual_about`, and the plain walk's hits move with it.
/// Counted hits of the octahedron, within `1e-4`:
///
/// ```text
/// matcher   cutoff   plain
/// local     16/16    6/16
/// ira       16/16    5/16
/// ```
///
/// Hop counts are not that comparison. Seed 0 with IRA spends 435 hops and
/// reaches the octahedron while the plain walk spends 414 and stops at
/// -173.252378. The bars sit above those two plain counts and under a full
/// miss of the cutoff walk: at least 15 cutoff hits, at most 10 plain hits
/// on glibc and 12 on macOS, and a gap of at least 5. The measured gaps are
/// 10 and 11. The same 16 seeds reach the octahedron 11 times on macOS, so
/// the macOS ceiling sits one above that count.
#[test]
fn the_relative_cutoff_reaches_the_lj38_octahedron_where_the_plain_walk_need_not() {
    const SEEDS: u64 = 16;
    let budget = 30_000;
    let mut plain = Config::recommended(38);
    // The surface comparison is the LeanBurst walk. Stall exits are a
    // different measurement, and this walk does not grow a named template.
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
    for seed in 0..SEEDS {
        let (best, _, spent) = run(&compacted, budget, seed);
        assert!(spent <= budget, "the ledger overspent: {spent} of {budget}");
        if best < LJ38_OCTAHEDRON + 1e-4 {
            compact_hits += 1;
        }
        let (plain_best, _, plain_spent) = run(&plain, budget, seed);
        assert!(plain_spent <= budget);
        if plain_best < LJ38_OCTAHEDRON + 1e-4 {
            plain_hits += 1;
        }
    }
    assert!(
        compact_hits >= 15,
        "the cutoff walk reached the octahedron on {compact_hits} of {SEEDS} seeds"
    );
    // glibc stays at or under 10. macOS reaches the octahedron on 11 of
    // these 16 seeds; the ceiling is one above that count.
    let plain_limit = if cfg!(target_os = "macos") { 12 } else { 10 };
    assert!(
        plain_hits <= plain_limit,
        "the plain walk reached the octahedron on {plain_hits} of {SEEDS} seeds (cutoff {compact_hits})"
    );
    assert!(
        compact_hits - plain_hits >= 5,
        "the cutoff walk reached {compact_hits} seeds and the plain walk {plain_hits}"
    );
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
