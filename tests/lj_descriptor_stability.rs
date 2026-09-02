#![cfg(feature = "ira")]

use anneal_core::catalog::lj::{
    CALIBRATION_ENERGY_TOLERANCE, CALIBRATION_GRADIENT_TOLERANCE, CALIBRATION_IRA_TOLERANCE,
    descriptor_space, perturb_reference,
};
use anneal_core::descriptor_space::DescriptorVector;
use anneal_core::methods::cluster_hopping::{Config, random_cluster};
use anneal_core::methods::warm_lbfgs::WarmLbfgs;
use anneal_core::potentials::PairPotential;
use anneal_core::shape::match_shapes;
use ndarray::{Array1, ArrayView1};
use rand::SeedableRng;

struct Quenched {
    coordinates: Array1<f64>,
    energy: f64,
    gradient_norm: f64,
}

fn quench(potential: &PairPotential, start: ArrayView1<'_, f64>) -> Quenched {
    let mut optimizer = WarmLbfgs::default();
    let (_, coordinates, _) = optimizer.minimize(start, 2_000, |point| {
        Some(potential.value_and_gradient(point))
    });
    let (energy, gradient) = potential.value_and_gradient(coordinates.view());
    let gradient_norm = gradient
        .iter()
        .map(|value| value * value)
        .sum::<f64>()
        .sqrt();
    Quenched {
        coordinates,
        energy,
        gradient_norm,
    }
}

fn block_diagnostics(left: &DescriptorVector, right: &DescriptorVector) -> String {
    left.blocks()
        .iter()
        .zip(right.blocks())
        .map(|(left_block, right_block)| {
            let range = left_block.offset()..left_block.offset() + left_block.len();
            let distance = left.values()[range.clone()]
                .iter()
                .zip(&right.values()[range])
                .map(|(left, right)| (left - right).powi(2))
                .sum::<f64>()
                .sqrt();
            format!(
                "{:?}@{}: distance={distance:.9e}, raw_norms={:.9e}/{:.9e}",
                left_block.kind(),
                left_block.cutoff(),
                left_block.raw_norm(),
                right_block.raw_norm(),
            )
        })
        .collect::<Vec<_>>()
        .join("; ")
}

#[test]
fn exact_same_lj38_minimum_has_a_stable_universal_descriptor() {
    const POINTS: usize = 38;
    const SOURCE_SEED: u64 = 3_891_067;
    let potential = PairPotential::lennard_jones(POINTS);
    let config = Config::recommended(POINTS);
    let mut source_rng = rand::rngs::StdRng::seed_from_u64(SOURCE_SEED);
    let source_start = random_cluster(POINTS, 0.7, config.min_separation, &mut source_rng);
    assert_eq!(source_start.len(), 3 * POINTS);
    let source = quench(&potential, source_start.view());
    let left_start = perturb_reference(
        source.coordinates.as_slice().unwrap(),
        POINTS,
        SOURCE_SEED + 1,
        0.01,
    )
    .unwrap();
    let right_start = perturb_reference(
        source.coordinates.as_slice().unwrap(),
        POINTS,
        SOURCE_SEED + 2,
        0.01,
    )
    .unwrap();
    let left = quench(&potential, ArrayView1::from(&left_start));
    let right = quench(&potential, ArrayView1::from(&right_start));

    for candidate in [&left, &right] {
        assert!((candidate.energy - source.energy).abs() <= CALIBRATION_ENERGY_TOLERANCE);
        assert!(candidate.gradient_norm <= CALIBRATION_GRADIENT_TOLERANCE);
        assert!(
            match_shapes(source.coordinates.view(), candidate.coordinates.view(), 1.8)
                .unwrap()
                .distance
                <= CALIBRATION_IRA_TOLERANCE
        );
    }

    let space = descriptor_space();
    let species = vec![18; POINTS];
    let left_descriptor = space
        .describe(left.coordinates.view(), Some(&species))
        .unwrap();
    let right_descriptor = space
        .describe(right.coordinates.view(), Some(&species))
        .unwrap();
    let distance = left_descriptor.distance(&right_descriptor).unwrap();
    assert!(
        distance <= CALIBRATION_IRA_TOLERANCE,
        "same-minimum descriptor distance {distance:.9e}; {}",
        block_diagnostics(&left_descriptor, &right_descriptor),
    );
}
