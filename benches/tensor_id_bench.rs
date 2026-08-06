//! Per-call cost of the basin descriptors, at the cluster sizes the search
//! runs at.
//!
//! The comparison that matters is against a hop rather than against each
//! other: a hop charges thirty-one energy evaluations, so the descriptor has a
//! budget of roughly that. `pair_potential` measures one gradient of the same
//! Lennard-Jones cluster to give the reading a unit.

use anneal_core::bias::{Fingerprint, SortedPairs};
use anneal_core::potentials::PairPotential;
use anneal_core::spectral::symmetric_eigen;
use anneal_core::tensor_id::{TripletSpectrum, kernel_matrix, mode_gram, triplet_matrix};
use criterion::{Criterion, criterion_group, criterion_main};
use ndarray::Array1;
use std::hint::black_box;

/// A compact random cluster at roughly Lennard-Jones density.
fn cluster(n: usize) -> Array1<f64> {
    let mut s = 0x2545F491_4F6CDD1D_u64;
    let mut next = || {
        s ^= s << 13;
        s ^= s >> 7;
        s ^= s << 17;
        (s >> 11) as f64 / ((1u64 << 53) as f64) - 0.5
    };
    let r = 1.15 * (n as f64).cbrt();
    Array1::from((0..3 * n).map(|_| 2.0 * r * next()).collect::<Vec<_>>())
}

fn descriptors(c: &mut Criterion) {
    for n in [38_usize, 75, 98] {
        let x = cluster(n);
        let mut g = c.benchmark_group(format!("descriptor/n{n}"));

        let sorted = SortedPairs { n_points: n };
        g.bench_function("sorted_pairs", |b| {
            b.iter(|| black_box(sorted.describe(x.view())))
        });

        let tri = TripletSpectrum::new(n);
        g.bench_function("triplet_spectrum", |b| {
            b.iter(|| black_box(tri.describe(x.view())))
        });
        g.bench_function("triplet_spectra_only", |b| {
            b.iter(|| black_box(tri.spectra(x.view())))
        });

        // The exact HOSVD mode Gram, order N^4, for the cost the descriptor
        // avoids by contracting a mode instead of unfolding it.
        let a = kernel_matrix(x.view(), n, tri.sigma);
        g.bench_function("mode_gram_n4", |b| {
            b.iter(|| black_box(mode_gram(a.view())))
        });

        // The same two spectra through cyclic Jacobi, in the same run as the
        // tridiagonal path so the two are comparable.
        let m = triplet_matrix(a.view());
        g.bench_function("spectra_via_jacobi", |b| {
            b.iter(|| {
                black_box(symmetric_eigen(a.view(), 30));
                black_box(symmetric_eigen(m.view(), 30));
            })
        });

        let pot = PairPotential::lennard_jones(n);
        g.bench_function("lj_gradient", |b| {
            b.iter(|| black_box(pot.value_and_gradient(x.view())))
        });

        g.finish();
    }
}

criterion_group!(benches, descriptors);
criterion_main!(benches);
