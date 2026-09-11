use anneal_core::{bias, methods, shared_bias};
use eindir_core::Bounds;
use methods::box_hopping::BoxEnsembleConfig;
use ndarray::{Array1, ArrayView1, array};
use rand::{Rng, SeedableRng, rngs::StdRng};

#[path = "../src/methods/box_hopping/coverage.rs"]
mod coverage;
#[path = "../src/methods/box_hopping/repulsion.rs"]
mod repulsion;

fn distance(a: ArrayView1<f64>, b: ArrayView1<f64>, bounds: &Bounds<f64>) -> f64 {
    let widths = &bounds.high - &bounds.low;
    let free = widths.iter().filter(|w| **w > 0.0).count();
    let scale = 1.0 / (free as f64).sqrt();
    let normalized =
        |point: ArrayView1<f64>, j: usize| ((point[j] - bounds.low[j]) / widths[j]) * scale;
    widths
        .iter()
        .enumerate()
        .filter(|(_, w)| **w > 0.0)
        .map(|(j, _)| (normalized(a, j) - normalized(b, j)).powi(2))
        .sum::<f64>()
        .sqrt()
}

fn clearance(point: ArrayView1<f64>, peers: &[Array1<f64>], bounds: &Bounds<f64>) -> f64 {
    peers
        .iter()
        .map(|peer| distance(point, peer.view(), bounds))
        .fold(f64::INFINITY, f64::min)
}

fn rounded_case(fixed: bool) -> (Array1<f64>, Array1<f64>, Vec<Array1<f64>>, Bounds<f64>, f64) {
    let low = 1e10;
    let spacing = 1.9073486328125e-6;
    let point = |a: f64, b: f64| {
        if fixed {
            array![low + a * spacing, 7.0, low + b * spacing]
        } else {
            array![low + a * spacing, low + b * spacing]
        }
    };
    let bounds = Bounds::new(point(0.0, 0.0), point(4.0, 6.0), 0.0);
    let peers = vec![
        point(1.0, 2.0),
        point(4.0, 1.0),
        point(1.0, 3.0),
        point(0.0, 1.0),
    ];
    let initial = point(3.0, 2.0);
    let radius = 0.36;
    let before = clearance(initial.view(), &peers, &bounds);
    let allowance = radius - before;
    let sideways = point(3.0, 3.0);
    assert!(clearance(sideways.view(), &peers, &bounds) > before);
    assert!(distance(initial.view(), sideways.view(), &bounds) <= allowance);
    let mut coverage = coverage::Coverage::new(
        &bounds,
        2,
        &coverage::BoxCoverageConfig {
            radius,
            ..coverage::BoxCoverageConfig::default()
        },
        8,
    );
    for peer in &peers {
        coverage.sample(1, peer.view(), 0.0);
    }
    coverage.hear(0, 1.0);
    let mut proposal = initial.clone();
    let mut rng = StdRng::seed_from_u64(9);
    let mut untouched_rng = rng.clone();
    coverage.repel(0, initial.view(), &mut proposal, &mut rng);
    assert_eq!(rng.random::<u64>(), untouched_rng.random::<u64>());
    let (stats, _) = coverage.finish();
    assert_eq!(stats.applied_foreign_samples, peers.len() as u64);
    assert_eq!(stats.sample_overlaps, 1);
    assert_eq!(stats.repelled_proposals, 1);
    assert_eq!(stats.constrained_repulsions, 0);
    assert!(bounds.contains(proposal.view()));
    if fixed {
        assert_eq!(proposal[1], 7.0);
    }
    (initial, proposal, peers, bounds, allowance)
}

#[test]
fn representable_parameter_repulsion_increases_actual_peer_clearance() {
    for fixed in [false, true] {
        let (initial, proposal, peers, bounds, _) = rounded_case(fixed);
        let before = clearance(initial.view(), &peers, &bounds);
        let after = clearance(proposal.view(), &peers, &bounds);
        assert!(
            after > before,
            "physical clearance decreased: {before} -> {after}"
        );
    }
}

#[test]
fn representable_parameter_repulsion_keeps_the_geometric_step_cap() {
    for fixed in [false, true] {
        let (initial, proposal, _, bounds, allowance) = rounded_case(fixed);
        let displacement = distance(initial.view(), proposal.view(), &bounds);
        assert!(
            displacement <= allowance + 16.0 * f64::EPSILON,
            "physical displacement {displacement} exceeds allowance {allowance}"
        );
    }
}
