use anneal_core::catalog::{BasinCensus, PRODUCTION_MINIMUM_VISITS};
use anneal_core::methods::cluster_hopping::{
    ChainCheckpoint, CheckpointAction, Config, Ledger, QuenchStatus, random_cluster,
};
use anneal_core::methods::cluster_search::search_from_with_checkpoint;
use anneal_core::potentials::PairPotential;
use rand::SeedableRng;

const BUDGET: usize = 6_000;

#[derive(Debug, Default)]
struct Census {
    checkpoints: usize,
    boundaries: usize,
    visits: u64,
    saturated: bool,
    spent: usize,
}

/// Runs the water Good-Turing checkpoint on LJ13, keyed on energy.
fn census_run(cfg: &Config, seed: u64) -> Census {
    let pot = PairPotential::lennard_jones(cfg.n_points);
    let mut rng = rand::rngs::StdRng::seed_from_u64(seed);
    let start = random_cluster(cfg.n_points, 0.7, cfg.min_separation, &mut rng);
    let mut census = BasinCensus::new(1, 1e-3).expect("census");
    let mut out = Census::default();
    let mut ledger = Ledger::new(BUDGET);
    let mut checkpoint = |snapshot: ChainCheckpoint<'_>| {
        out.checkpoints += 1;
        for boundary in snapshot.quench_boundaries() {
            out.boundaries += 1;
            if boundary.status() == QuenchStatus::Validated {
                census.observe(&[boundary.energy()]).expect("finite energy");
            }
        }
        if census.is_saturated() {
            out.saturated = true;
            return CheckpointAction::Retire {
                reason: "saturated".to_owned(),
            };
        }
        CheckpointAction::Continue
    };
    search_from_with_checkpoint(
        &pot,
        cfg,
        &mut ledger,
        start.view(),
        seed,
        1,
        &mut checkpoint,
    )
    .expect("finite start");
    out.visits = census.total_visits();
    out.spent = ledger.spent();
    out
}

/// The census counts only validated boundaries. With none recorded it
/// stays at zero visits, below the production floor, and never saturates.
#[test]
fn lj13_warm_relax_quenches_saturate_the_census() {
    for seed in 0..3 {
        let run = census_run(&Config::for_cluster(13), seed);
        assert!(run.checkpoints > 0, "seed {seed}: no checkpoint ran");
        assert!(run.boundaries > 0, "seed {seed}: no boundary: {run:?}");
        assert!(
            run.saturated,
            "seed {seed}: census never saturated: {run:?}"
        );
        assert!(run.spent < BUDGET, "seed {seed}: retire spent the budget");
    }
}

#[test]
fn lj13_recommended_quenches_reach_the_census_floor() {
    for seed in 0..3 {
        let run = census_run(&Config::recommended(13), seed);
        assert!(
            run.visits >= PRODUCTION_MINIMUM_VISITS,
            "seed {seed}: census below the visit floor: {run:?}"
        );
    }
}
