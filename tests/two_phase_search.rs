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
/// The hit pattern is exact, not a ceiling. Host `libm` (`exp`/`log`/`sin`
/// of glibc or libSystem) is not one function, and LLVM contracts the
/// generic search in whatever context the test crate gives it. macOS then
/// reported 11 of 16 plain hits and, after the test text changed and
/// nothing else did, 13 of 16, on the same image and the same rustc. The
/// proposals, the initial cluster and the acceptance test now call the
/// `libm` crate, the pair force is an IEEE `mul_add`, and
/// `search_lennard_jones` is monomorphized in the library and forces
/// rgmin's scalar dot product. The AVX-512 and AVX2 clones match that
/// mask on x86. The NEON clone does not: macOS reported plain mask
/// `0x9204`. Five reruns of the scalar path, at one thread and at
/// eight, all gave the same masks. The `ira` feature replaces the
/// permutation and has its own mask, measured on the feature build.
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
    let mut compact_mask = 0u16;
    let mut plain_mask = 0u16;
    for seed in 0..SEEDS {
        let (best, _, spent) = run(&compacted, budget, seed);
        assert!(spent <= budget, "the ledger overspent: {spent} of {budget}");
        if best < LJ38_OCTAHEDRON + 1e-4 {
            compact_hits += 1;
            compact_mask |= 1 << seed;
        }
        let (plain_best, _, plain_spent) = run(&plain, budget, seed);
        assert!(plain_spent <= budget);
        if plain_best < LJ38_OCTAHEDRON + 1e-4 {
            plain_hits += 1;
            plain_mask |= 1 << seed;
        }
    }
    // Without `ira`: cutoff 0xffff, plain 0x9acd (9 seeds).
    // With `ira`: cutoff 0xffff, plain 0x18f5 (8 seeds).
    let plain_exact = if cfg!(feature = "ira") {
        0x18f5
    } else {
        0x9acd
    };
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
