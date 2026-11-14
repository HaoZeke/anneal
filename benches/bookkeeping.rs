//! Cost of the work outside the potential.
//!
//! One reading each for the dynamic-lattice draw, the fivefold SOAP
//! proposal, the sorted pair spectrum, and the common-neighbour fractions.
//! `sorted_pairs_cold` changes the geometry every call. `sorted_pairs_hot`
//! repeats one geometry, which is what a chain does with its incumbent.

use anneal_core::bias::{Fingerprint, SortedPairs};
use anneal_core::methods::cluster_hopping::Ledger;
use anneal_core::methods::lattice_search::Lattice;
use anneal_core::soap::{fivefold_axis_table, step_away_fivefold};
use anneal_core::structure::named_triplet_fractions;
use criterion::{Criterion, criterion_group, criterion_main};
use ndarray::Array1;
use rand::SeedableRng;
use rand::rngs::StdRng;
use std::hint::black_box;

fn cluster(n: usize, salt: u64) -> Array1<f64> {
    let mut s = 0x2545F491_4F6CDD1D_u64 ^ salt;
    let mut next = || {
        s ^= s << 13;
        s ^= s >> 7;
        s ^= s << 17;
        (s >> 11) as f64 / ((1u64 << 53) as f64) - 0.5
    };
    let radius = 1.15 * (n as f64).cbrt();
    Array1::from((0..3 * n).map(|_| 2.0 * radius * next()).collect::<Vec<_>>())
}

fn kernels(c: &mut Criterion) {
    for n in [38_usize, 70, 80, 98] {
        let x = cluster(n, 1);
        let mut g = c.benchmark_group(format!("bookkeeping/n{n}"));
        g.sample_size(30);

        let sorted = SortedPairs { n_points: n };
        g.bench_function("sorted_pairs_hot", |b| {
            b.iter(|| black_box(sorted.describe(x.view())))
        });
        let mut salt = 10u64;
        g.bench_function("sorted_pairs_cold", |b| {
            b.iter(|| {
                salt = salt.wrapping_add(1);
                let y = cluster(n, salt);
                black_box(sorted.describe(y.view()))
            })
        });

        g.bench_function("common_neighbour", |b| {
            b.iter(|| black_box(named_triplet_fractions(x.view(), n, 1.35)))
        });

        let flat = x.to_vec();
        let lattice = Lattice::default();
        g.bench_function("dynamic_lattice", |b| {
            b.iter(|| {
                let mut ledger = Ledger::new(1_000_000);
                let mut rng = StdRng::seed_from_u64(7);
                black_box(lattice.propose_energy_exchange(&flat, &mut ledger, 0.8, &mut rng))
            })
        });

        g.bench_function("soap_fivefold_table", |b| {
            b.iter(|| black_box(fivefold_axis_table(x.view())))
        });
        g.bench_function("soap_proposal", |b| {
            b.iter(|| {
                let mut rng = StdRng::seed_from_u64(11);
                black_box(step_away_fivefold(x.view(), 0.2, &mut rng))
            })
        });

        g.finish();
    }
}

criterion_group!(benches, kernels);
criterion_main!(benches);
