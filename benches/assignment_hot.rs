// Copyright (c) anneal developers

//! Minimum-cost assignment, the largest share of time outside the potential.
//!
//! `shortest_paths` is the scan used for a full point set of 98 atoms and
//! for every fivefold residual. `kuhn_munkres` is the column-first matching
//! used below that size, which is the Au55 and Sutton-Chen scan. The costs
//! are squared distances of a fixed cluster, the matrix a symmetry
//! projection and a fivefold residual both build. The cluster is built
//! outside the timed loop.

use anneal_core::assignment::minimum_cost_assignment;
use criterion::{Criterion, criterion_group, criterion_main};
use std::hint::black_box;

fn cluster(n: usize) -> Vec<[f64; 3]> {
    let mut state = 0x2545_f491_4f6c_dd1d_u64;
    let mut next = || {
        state ^= state << 13;
        state ^= state >> 7;
        state ^= state << 17;
        (state >> 11) as f64 / ((1u64 << 53) as f64) - 0.5
    };
    let radius = 1.15 * (n as f64).cbrt();
    (0..n)
        .map(|_| {
            [
                2.0 * radius * next(),
                2.0 * radius * next(),
                2.0 * radius * next(),
            ]
        })
        .collect()
}

fn squared_distances(points: &[[f64; 3]]) -> Vec<f64> {
    let n = points.len();
    let mut costs = vec![0.0; n * n];
    for row in 0..n {
        for column in 0..n {
            let squared = points[row]
                .iter()
                .zip(points[(column + 3) % n])
                .map(|(left, right)| {
                    let delta = left - right;
                    delta * delta
                })
                .sum();
            costs[row * n + column] = squared;
        }
    }
    costs
}

fn kernels(c: &mut Criterion) {
    for n in [55_usize, 80, 98] {
        let costs = squared_distances(&cluster(n));
        let mut group = c.benchmark_group(format!("assignment/n{n}"));
        group.sample_size(20);
        group.bench_function("minimum_cost", |b| {
            b.iter(|| black_box(minimum_cost_assignment(black_box(&costs), n)))
        });
        group.finish();
    }
}

criterion_group!(benches, kernels);
criterion_main!(benches);
