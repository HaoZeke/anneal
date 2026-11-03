//! Criterion benchmarks for the kernels that dominate cluster optimisation.

use anneal_core::bias::{Fingerprint, SortedPairs};
use anneal_core::methods::cluster_hopping::Config;
use anneal_core::methods::warm_lbfgs::WarmLbfgs;
use anneal_core::neighbors::open_cutoff_pairs;
use anneal_core::potentials::{PairPotential, Tip4pCluster, random_tip4p_cluster};
use criterion::{BatchSize, BenchmarkId, Criterion, Throughput, criterion_group, criterion_main};
use ndarray::Array1;
use rand::SeedableRng;
use rand::rngs::StdRng;
use std::hint::black_box;

// The Lennard-Jones pair minimum 2^(1/6) sigma, rounded down to one decimal.
const FIXTURE_SPACING: f64 = 1.1;
// A 5 by 5 cross-section gives LJ75 three layers and LJ98 almost four.
const FIXTURE_SIDE: usize = 5;
// Between the axial shell at 1.1 and its diagonal at sqrt(2) * 1.1.
const FIRST_SHELL_CUTOFF: f64 = 1.5;
// A fixed seed makes the rigid-water fixture reproducible; it is not fitted.
const TIP4P_FIXTURE_SEED: u64 = 1;

fn point_fixture(n: usize) -> Array1<f64> {
    Array1::from_shape_fn(3 * n, |component| {
        let point = component / 3;
        let axis = component % 3;
        let coordinates = [
            (point % FIXTURE_SIDE) as f64,
            ((point / FIXTURE_SIDE) % FIXTURE_SIDE) as f64,
            (point / (FIXTURE_SIDE * FIXTURE_SIDE)) as f64,
        ];
        FIXTURE_SPACING * coordinates[axis]
    })
}

fn point_rows(x: &Array1<f64>) -> Vec<[f64; 3]> {
    x.as_slice()
        .expect("fixture is contiguous")
        .as_chunks::<3>()
        .0
        .to_vec()
}

fn bench_pair_kernels(c: &mut Criterion) {
    let mut group = c.benchmark_group("pair_value_and_gradient");
    for n in [75, 98] {
        let potential = PairPotential::lennard_jones(n);
        let coordinates = point_fixture(n);
        group.throughput(Throughput::Elements((n * (n - 1) / 2) as u64));
        group.bench_with_input(BenchmarkId::from_parameter(n), &n, |b, _| {
            b.iter(|| potential.value_and_gradient(black_box(coordinates.view())))
        });
    }
    group.finish();
}

fn bench_pair_descriptors(c: &mut Criterion) {
    let mut group = c.benchmark_group("sorted_pair_descriptor");
    for n in [75, 98] {
        let descriptor = SortedPairs { n_points: n };
        let coordinates = point_fixture(n);
        group.throughput(Throughput::Elements((n * (n - 1) / 2) as u64));
        group.bench_with_input(BenchmarkId::from_parameter(n), &n, |b, _| {
            b.iter(|| descriptor.describe(black_box(coordinates.view())))
        });
    }
    group.finish();
}

fn bench_open_neighbors(c: &mut Criterion) {
    let mut group = c.benchmark_group("open_cutoff_neighbors");
    for n in [75, 98] {
        let coordinates = point_fixture(n);
        let positions = point_rows(&coordinates);
        group.throughput(Throughput::Elements((n * (n - 1)) as u64));
        group.bench_with_input(BenchmarkId::from_parameter(n), &n, |b, _| {
            b.iter(|| {
                open_cutoff_pairs(black_box(&positions), black_box(FIRST_SHELL_CUTOFF), false)
            })
        });
    }
    group.finish();
}

fn bench_tip4p_kernels(c: &mut Criterion) {
    let n = 10;
    let potential = Tip4pCluster::new(n);
    let mut rng = StdRng::seed_from_u64(TIP4P_FIXTURE_SEED);
    let coordinates = random_tip4p_cluster(n, &mut rng);

    c.bench_function("tip4p_value_and_gradient/10", |b| {
        b.iter(|| potential.value_and_gradient(black_box(coordinates.view())))
    });

    let geometry = potential.rigid_geometry();
    c.bench_function("rigid_body_descriptor/10", |b| {
        b.iter(|| geometry.describe(n, black_box(coordinates.view())))
    });
}

fn bench_local_minimizer(c: &mut Criterion) {
    let n = 75;
    let potential = PairPotential::lennard_jones(n);
    let coordinates = point_fixture(n);
    let iterations = Config::for_cluster(n).screen_steps;
    c.bench_function("warm_lbfgs_screen/75", |b| {
        b.iter_batched(
            || (WarmLbfgs::default(), coordinates.clone()),
            |(mut minimizer, start)| {
                minimizer.minimize(start.view(), iterations, |x| {
                    Some(potential.value_and_gradient(x))
                })
            },
            BatchSize::SmallInput,
        )
    });
}

criterion_group!(
    globalopt_hotpaths,
    bench_pair_kernels,
    bench_pair_descriptors,
    bench_open_neighbors,
    bench_tip4p_kernels,
    bench_local_minimizer,
);
criterion_main!(globalopt_hotpaths);
