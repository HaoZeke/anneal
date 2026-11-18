//! Pair-spectrum basin fingerprint.

use anneal_core::bias::{Fingerprint, SortedPairs};
use criterion::{Criterion, criterion_group, criterion_main};
use ndarray::Array1;
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
        group.finish();
    }
}

criterion_group!(benches, kernels);
criterion_main!(benches);
