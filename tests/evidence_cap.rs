//! The run evidence a ledger keeps.
//!
//! By default, as under [`Ledger::keep_all_evidence`], every quench
//! boundary and accepted transition is kept. The digest below pins that
//! full record, checkpoint slices included, as the run produced it when
//! every ledger kept it all. Under [`Ledger::keep_evidence_until_shown`] a
//! run keeps an entry only until a checkpoint has shown it, and
//! [`Ledger::keep_recent_evidence`] keeps the newest entries. Either way
//! the rest is counted, and the run itself is the same to the bit.
//!
//! As in `lj_trajectory_pin.rs`, the scalar dot path is forced so the
//! digest is one trajectory on every target.

use anneal_core::methods::cluster_hopping::{
    AcceptedTransition, ChainCheckpoint, CheckpointAction, Config, Ledger, Outcome, QuenchBoundary,
    QuenchStatus, random_cluster_in_radius,
};
use anneal_core::methods::cluster_search::{search_from, search_from_with_checkpoint};
use anneal_core::potentials::PairPotential;
use ndarray::ArrayView1;
use rand::SeedableRng;
use rand::rngs::StdRng;
use sha2::{Digest, Sha256};

const SEED: u64 = 2000;
const ATOMS: usize = 13;
const BUDGET: usize = 30_000;
const INTERVAL: usize = 2_000;
const PINNED: &str = "b4c30260dec80005f510fb890dcc0386957dfb1a754a103d90e41d70ebb68c24";

#[derive(Clone, Copy)]
enum Keep {
    Default,
    All,
    Shown,
    Recent(usize),
}

struct Run {
    outcome: Outcome,
    boundaries: Vec<QuenchBoundary>,
    dropped_boundaries: usize,
    checkpoints: Vec<(usize, Vec<QuenchBoundary>, Vec<AcceptedTransition>)>,
}

fn ledger(keep: Keep) -> Ledger {
    let mut ledger = Ledger::new(BUDGET);
    match keep {
        Keep::Default => {}
        Keep::All => ledger.keep_all_evidence(),
        Keep::Shown => ledger.keep_evidence_until_shown(),
        Keep::Recent(cap) => ledger.keep_recent_evidence(cap),
    }
    ledger
}

fn run(keep: Keep) -> Run {
    let _ = rgmin::vecops::force_simd_path("scalar");
    let cfg = Config::recommended(ATOMS);
    let potential = PairPotential::lennard_jones(ATOMS);
    let mut rng = StdRng::seed_from_u64(SEED);
    let start = random_cluster_in_radius(ATOMS, cfg.start_radius(), cfg.min_separation, &mut rng);
    let mut ledger = ledger(keep);
    let mut checkpoints = Vec::new();
    let mut checkpoint = |snapshot: ChainCheckpoint<'_>| {
        checkpoints.push((
            snapshot.charged(),
            snapshot.quench_boundaries().to_vec(),
            snapshot.accepted_transitions().to_vec(),
        ));
        CheckpointAction::Continue
    };
    let (outcome, _) = search_from_with_checkpoint(
        &potential,
        &cfg,
        &mut ledger,
        start.view(),
        SEED,
        INTERVAL,
        &mut checkpoint,
    )
    .expect("finite start");
    Run {
        outcome,
        boundaries: ledger.quench_boundaries().to_vec(),
        dropped_boundaries: ledger.dropped_quench_boundaries(),
        checkpoints,
    }
}

fn view(hasher: &mut Sha256, values: ArrayView1<f64>) {
    hasher.update((values.len() as u64).to_le_bytes());
    for value in values {
        hasher.update(value.to_le_bytes());
    }
}

fn boundary(hasher: &mut Sha256, b: &QuenchBoundary) {
    hasher.update([u8::from(b.status() == QuenchStatus::Validated)]);
    hasher.update((b.charged_calls() as u64).to_le_bytes());
    hasher.update(b.energy().to_le_bytes());
    view(hasher, b.state());
    match b.gradient() {
        Some(g) => {
            hasher.update([1]);
            view(hasher, g);
        }
        None => hasher.update([0]),
    }
}

fn transition(hasher: &mut Sha256, t: &AcceptedTransition) {
    hasher.update((t.hop as u64).to_le_bytes());
    hasher.update(t.action.as_bytes());
    hasher.update([0]);
    hasher.update(t.from_energy.to_le_bytes());
    hasher.update(t.to_energy.to_le_bytes());
    view(hasher, t.from_state.view());
    view(hasher, t.to_state.view());
    for gradient in [&t.from_gradient, &t.to_gradient] {
        match gradient {
            Some(g) => {
                hasher.update([1]);
                view(hasher, g.view());
            }
            None => hasher.update([0]),
        }
    }
    hasher.update([u8::from(t.validated), u8::from(t.adopted)]);
}

fn trajectory(hasher: &mut Sha256, outcome: &Outcome) {
    hasher.update(outcome.best.to_le_bytes());
    hasher.update(outcome.final_energy.to_le_bytes());
    hasher.update((outcome.charged as u64).to_le_bytes());
    hasher.update((outcome.hops as u64).to_le_bytes());
    if let Some(state) = &outcome.final_state {
        view(hasher, state.view());
    }
}

fn digest(run: &Run) -> String {
    let mut hasher = Sha256::new();
    trajectory(&mut hasher, &run.outcome);
    hasher.update((run.outcome.accepted_transitions.len() as u64).to_le_bytes());
    for t in &run.outcome.accepted_transitions {
        transition(&mut hasher, t);
    }
    hasher.update((run.boundaries.len() as u64).to_le_bytes());
    for b in &run.boundaries {
        boundary(&mut hasher, b);
    }
    hasher.update((run.checkpoints.len() as u64).to_le_bytes());
    for (charged, boundaries, transitions) in &run.checkpoints {
        hasher.update((*charged as u64).to_le_bytes());
        hasher.update((boundaries.len() as u64).to_le_bytes());
        for b in boundaries {
            boundary(&mut hasher, b);
        }
        hasher.update((transitions.len() as u64).to_le_bytes());
        for t in transitions {
            transition(&mut hasher, t);
        }
    }
    format!("{:x}", hasher.finalize())
}

fn boundary_bits(b: &QuenchBoundary) -> String {
    let mut hasher = Sha256::new();
    boundary(&mut hasher, b);
    format!("{:x}", hasher.finalize())
}

fn transition_bits(t: &AcceptedTransition) -> String {
    let mut hasher = Sha256::new();
    transition(&mut hasher, t);
    format!("{:x}", hasher.finalize())
}

fn newest<T>(items: &[T], cap: usize) -> &[T] {
    &items[items.len() - cap.min(items.len())..]
}

fn same_trajectory(a: &Outcome, b: &Outcome) -> bool {
    let mut x = Sha256::new();
    let mut y = Sha256::new();
    trajectory(&mut x, a);
    trajectory(&mut y, b);
    x.finalize() == y.finalize()
}

#[test]
fn keeping_all_evidence_keeps_every_record_as_before() {
    let full = run(Keep::All);
    assert_eq!(full.dropped_boundaries, 0);
    assert_eq!(full.outcome.dropped_transitions, 0);
    assert_eq!(digest(&full), PINNED);
}

#[test]
fn by_default_a_ledger_keeps_every_record() {
    let full = run(Keep::Default);
    assert_eq!(full.dropped_boundaries, 0);
    assert_eq!(full.outcome.dropped_transitions, 0);
    assert_eq!(digest(&full), PINNED);
}

#[test]
fn until_shown_a_run_keeps_what_its_checkpoints_have_yet_to_show() {
    let full = run(Keep::All);
    let kept = run(Keep::Shown);
    assert!(
        same_trajectory(&full.outcome, &kept.outcome),
        "keeping until shown changed the trajectory"
    );
    assert_eq!(kept.checkpoints.len(), full.checkpoints.len());
    for ((charged, b_kept, t_kept), (charged_full, b_full, t_full)) in
        kept.checkpoints.iter().zip(&full.checkpoints)
    {
        assert_eq!(charged, charged_full);
        assert_eq!(
            b_kept.iter().map(boundary_bits).collect::<Vec<_>>(),
            b_full.iter().map(boundary_bits).collect::<Vec<_>>(),
            "checkpoint at {charged} saw other boundaries"
        );
        assert_eq!(
            t_kept.iter().map(transition_bits).collect::<Vec<_>>(),
            t_full.iter().map(transition_bits).collect::<Vec<_>>(),
            "checkpoint at {charged} saw other transitions"
        );
    }
    let shown_boundaries: usize = full.checkpoints.iter().map(|(_, b, _)| b.len()).sum();
    let shown_transitions: usize = full.checkpoints.iter().map(|(_, _, t)| t.len()).sum();
    assert!(shown_boundaries > 0 && shown_transitions > 0);
    assert_eq!(
        kept.boundaries
            .iter()
            .map(boundary_bits)
            .collect::<Vec<_>>(),
        full.boundaries[shown_boundaries..]
            .iter()
            .map(boundary_bits)
            .collect::<Vec<_>>(),
        "keeping until shown did not keep exactly the boundaries after the last checkpoint"
    );
    assert_eq!(
        kept.outcome
            .accepted_transitions
            .iter()
            .map(transition_bits)
            .collect::<Vec<_>>(),
        full.outcome.accepted_transitions[shown_transitions..]
            .iter()
            .map(transition_bits)
            .collect::<Vec<_>>(),
        "keeping until shown did not keep exactly the transitions after the last checkpoint"
    );
    assert_eq!(kept.dropped_boundaries, shown_boundaries);
    assert_eq!(kept.outcome.dropped_transitions, shown_transitions);
}

#[test]
fn until_shown_a_run_without_a_checkpoint_keeps_no_evidence() {
    let _ = rgmin::vecops::force_simd_path("scalar");
    let cfg = Config::recommended(ATOMS);
    let potential = PairPotential::lennard_jones(ATOMS);
    let mut rng = StdRng::seed_from_u64(SEED);
    let start = random_cluster_in_radius(ATOMS, cfg.start_radius(), cfg.min_separation, &mut rng);
    let mut all = ledger(Keep::All);
    let (full, _) =
        search_from(&potential, &cfg, &mut all, start.view(), SEED).expect("finite start");
    let mut shown = ledger(Keep::Shown);
    let (kept, _) =
        search_from(&potential, &cfg, &mut shown, start.view(), SEED).expect("finite start");
    assert!(
        same_trajectory(&full, &kept),
        "keeping until shown changed the trajectory"
    );
    assert!(!all.quench_boundaries().is_empty() && !full.accepted_transitions.is_empty());
    assert!(shown.quench_boundaries().is_empty());
    assert!(kept.accepted_transitions.is_empty());
    assert_eq!(
        shown.dropped_quench_boundaries(),
        all.quench_boundaries().len()
    );
    assert_eq!(kept.dropped_transitions, full.accepted_transitions.len());
}

#[test]
fn a_cap_keeps_the_newest_records_and_counts_the_rest() {
    let full = run(Keep::All);
    assert!(
        full.boundaries.len() > 4 && full.outcome.accepted_transitions.len() > 4,
        "the run is too short to test a cap: {} boundaries, {} transitions",
        full.boundaries.len(),
        full.outcome.accepted_transitions.len()
    );
    assert!(
        full.checkpoints
            .iter()
            .any(|(_, boundaries, _)| boundaries.len() > 3),
        "no checkpoint saw more boundaries than the smallest cap below"
    );
    let largest = full
        .boundaries
        .len()
        .max(full.outcome.accepted_transitions.len());
    for cap in [0, 1, 3, largest - 1, largest, usize::MAX] {
        let kept = run(Keep::Recent(cap));
        assert!(
            same_trajectory(&full.outcome, &kept.outcome),
            "cap {cap} changed the trajectory"
        );

        let boundaries = newest(&full.boundaries, cap);
        assert_eq!(
            kept.boundaries
                .iter()
                .map(boundary_bits)
                .collect::<Vec<_>>(),
            boundaries.iter().map(boundary_bits).collect::<Vec<_>>(),
            "cap {cap} did not keep the newest boundaries"
        );
        assert_eq!(
            kept.dropped_boundaries + kept.boundaries.len(),
            full.boundaries.len()
        );

        let transitions = newest(&full.outcome.accepted_transitions, cap);
        assert_eq!(
            kept.outcome
                .accepted_transitions
                .iter()
                .map(transition_bits)
                .collect::<Vec<_>>(),
            transitions.iter().map(transition_bits).collect::<Vec<_>>(),
            "cap {cap} did not keep the newest transitions"
        );
        assert_eq!(
            kept.outcome.dropped_transitions + kept.outcome.accepted_transitions.len(),
            full.outcome.accepted_transitions.len()
        );

        assert_eq!(kept.checkpoints.len(), full.checkpoints.len());
        for ((charged, b_kept, t_kept), (charged_full, b_full, t_full)) in
            kept.checkpoints.iter().zip(&full.checkpoints)
        {
            assert_eq!(charged, charged_full);
            assert_eq!(
                b_kept.iter().map(boundary_bits).collect::<Vec<_>>(),
                newest(b_full, cap)
                    .iter()
                    .map(boundary_bits)
                    .collect::<Vec<_>>(),
                "cap {cap} checkpoint at {charged} saw other boundaries"
            );
            assert_eq!(
                t_kept.iter().map(transition_bits).collect::<Vec<_>>(),
                newest(t_full, cap)
                    .iter()
                    .map(transition_bits)
                    .collect::<Vec<_>>(),
                "cap {cap} checkpoint at {charged} saw other transitions"
            );
        }
    }
}
