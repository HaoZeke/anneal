//! Pair spectrum and the fivefold SOAP proposal.
//!
//! `sorted_pairs` is the basin fingerprint. `soap_proposal` builds the
//! squared-distance matching that turns a cluster away from a fivefold
//! axis. The cluster is fixed outside the timed loop.

use anneal_core::bias::{Fingerprint, SortedPairs};
use criterion::{Criterion, criterion_group, criterion_main};
use ndarray::Array1;
use rand::SeedableRng;
use rand::rngs::StdRng;
use std::hint::black_box;

fn cluster(n: usize, salt: u64) -> Array1<f64> {
    let mut state = 0x2545_f491_4f6c_dd1d_u64 ^ salt;
    let mut next = || {
        state ^= state << 13;
        state ^= state >> 7;
        state ^= state << 17;
        (state >> 11) as f64 / ((1u64 << 53) as f64) - 0.5
    };
    let radius = 1.15 * (n as f64).cbrt();
    Array1::from(
        (0..3 * n)
            .map(|_| 2.0 * radius * next())
            .collect::<Vec<_>>(),
    )
}

fn kernels(c: &mut Criterion) {
    for n in [98_usize, 150] {
        let sorted = SortedPairs { n_points: n };
        let mut group = c.benchmark_group(format!("outside/n{n}"));
        group.sample_size(15);
        let mut salt = 0u64;
        group.bench_function("sorted_pairs", |b| {
            b.iter(|| {
                salt = salt.wrapping_add(1);
                let y = cluster(n, salt);
                black_box(sorted.describe(black_box(y.view())))
            })
        });
        let mut soap_salt = 10_000u64;
        group.bench_function("soap_proposal", |b| {
            b.iter(|| {
                soap_salt = soap_salt.wrapping_add(1);
                let y = cluster(n, soap_salt);
                let mut rng = StdRng::seed_from_u64(11);
                black_box(anneal_core::soap::step_away_fivefold(
                    black_box(y.view()),
                    0.2,
                    &mut rng,
                ))
            })
        });
        group.finish();
    }
}

criterion_group!(benches, kernels);
criterion_main!(benches);
