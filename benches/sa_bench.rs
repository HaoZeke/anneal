//! Criterion benchmark: compare the three preset SA variants
//! (Boltzmann, Fast, GSA) on the Styblinski-Tang 2D objective.
//! Steady-state proposals/sec is the headline metric.

use anneal_core::run_rs_variant;
use anneal_core::variant::{boltzmann, fast, gsa};
use criterion::{Criterion, criterion_group, criterion_main};
use eindir_core::objectives::StybTang2D;
use std::hint::black_box;

const N_EPOCHS: usize = 50;
const STEPS_PER_EPOCH: usize = 200;
const SEED: u64 = 0xDEADBEEF;

fn bench_boltzmann_styb_tang(c: &mut Criterion) {
    c.bench_function("boltzmann_styb_tang_2d", |b| {
        b.iter(|| {
            let v = boltzmann(StybTang2D::new(), 5.0, 0.5).expect("variant");
            let h = run_rs_variant(v, N_EPOCHS, STEPS_PER_EPOCH, SEED);
            black_box(h.best.val)
        });
    });
}

fn bench_fast_styb_tang(c: &mut Criterion) {
    c.bench_function("fast_styb_tang_2d", |b| {
        b.iter(|| {
            let v = fast(StybTang2D::new(), 5.0, 0.5).expect("variant");
            let h = run_rs_variant(v, N_EPOCHS, STEPS_PER_EPOCH, SEED);
            black_box(h.best.val)
        });
    });
}

fn bench_gsa_styb_tang(c: &mut Criterion) {
    c.bench_function("gsa_styb_tang_2d", |b| {
        b.iter(|| {
            let v = gsa(StybTang2D::new(), 5.0, 2.62, 1.7).expect("variant");
            let h = run_rs_variant(v, N_EPOCHS, STEPS_PER_EPOCH, SEED);
            black_box(h.best.val)
        });
    });
}

criterion_group!(
    sa_variants,
    bench_boltzmann_styb_tang,
    bench_fast_styb_tang,
    bench_gsa_styb_tang,
);
criterion_main!(sa_variants);
