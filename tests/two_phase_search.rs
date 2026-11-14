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
/// local     16/16    7/16
/// ira       16/16    7/16
/// ```
///
/// The hit pattern is exact, not a ceiling. Host `libm` is not one
/// function: macOS reported 11 of 16 plain hits, then 13 of 16, then
/// `0x9204`, then `0x513a`, while Linux held a different mask on the
/// same seeds. Proposal axes, the gamma sampler, the depth-rewarded
/// Student-t draw, and the symmetry rotation already go through the
/// `libm` crate. Three calls were still the host: the fivefold turn
/// (`2π/5` and `3π/5`), the continuous-symmetry orientation (`τ/5`),
/// and the energy-lattice softmax. Those use the `libm` crate as well.
/// On that walk Linux plain was `0x517b` (9 seeds). Merging develop's
/// stall-growth hold moves it to `0x91e4` (7 seeds). The fivefold
/// residual keeps the built-in assignment when IRA is linked, so the
/// IRA build has the same mask.
///
/// Those masks were measured at a 200-step quench. `recommended` now
/// quenches for 1000 steps, and at that length the plain walk reaches
/// the octahedron on more seeds, so this comparison keeps the 200-step
/// quench the masks were measured on.
#[test]
fn the_relative_cutoff_reaches_the_lj38_octahedron_where_the_plain_walk_need_not() {
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
    // Merging develop's stall-growth hold moves which seeds reach the
    // octahedron. Cutoff 0xffff, plain 0x91e4 (7 seeds), with and
    // without IRA.
    let plain_exact = 0x91e4;
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
