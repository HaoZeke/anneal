// Copyright (c) anneal developers

//! Symmetrise, the largest outside-potential cost after assignment on LJ150.
//!
//! `deviation` is the nearest-image residual of one fivefold turn.
//! `orbit_completion` places every atom on the core's orbits. Both see a
//! fresh cluster each call. The geometry is built outside the timed loop.

use criterion::{Criterion, criterion_group, criterion_main};
use ndarray::Array1;
use std::hint::black_box;

fn cluster(n: usize) -> Array1<f64> {
    let mut state = 0x2545_f491_4f6c_dd1d_u64;
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
    for n in [55_usize, 98, 150] {
        let x = cluster(n);
        let mut group = c.benchmark_group(format!("symmetrise/n{n}"));
        group.sample_size(20);
        group.bench_function("deviation", |b| {
            b.iter(|| {
                black_box(anneal_core::symmetrise::deviation(
                    black_box(x.view()),
                    n,
                    [0.0, 0.0, 1.0],
                    5,
                ))
            })
        });
        group.bench_function("orbit_completion", |b| {
            b.iter(|| {
                black_box(anneal_core::symmetrise::orbit_complete_core(
                    black_box(x.view()),
                    n,
                    0.15,
                    0.4,
                    0.5,
                    0.5,
                ))
            })
        });
        group.finish();
    }
}

criterion_group!(benches, kernels);
criterion_main!(benches);
