//! Witness for Method B Phase 3a: NUTS-driven SA + Sampler<f64>
//! composability with the existing run_rs / MultiChainSampler /
//! ParallelTemperingSampler infrastructure.

use anneal_core::cool::LogCool;
use anneal_core::geometric_ladder;
use anneal_core::hmc::{GaussianMomentum, NutsSaSampler, QGaussianMomentum};
use anneal_core::run_rs;
use anneal_core::ParallelTemperingSampler;
use eindir_core::FiniteDiffGradient;

use eindir_core::objectives::StybTang2D;

#[test]
fn nuts_sa_finds_negative_minimum_on_styb_tang() {
    let obj = StybTang2D::new();
    let grad = FiniteDiffGradient::new(StybTang2D::new());
    let cool = LogCool::new(5.0_f64, 2.0);
    let sampler = NutsSaSampler::new(
        obj,
        grad,
        cool.clone(),
        GaussianMomentum,
        0.1, // epsilon
        5.0, // temp_ref
        6,   // max_depth
    );
    let history = run_rs(sampler, &cool, 30, 30, 42);
    assert!(
        history.best.val < 0.0,
        "NUTS-SA should find a negative value on StybTang2D; got {}",
        history.best.val
    );
}

#[test]
fn nuts_sa_with_q_gaussian_momentum_runs() {
    let obj = StybTang2D::new();
    let grad = FiniteDiffGradient::new(StybTang2D::new());
    let cool = LogCool::new(5.0_f64, 2.0);
    let sampler = NutsSaSampler::new(
        obj,
        grad,
        cool.clone(),
        QGaussianMomentum::new(1.5_f64),
        0.1,
        5.0,
        6,
    );
    let history = run_rs(sampler, &cool, 20, 20, 7);
    // Just check finite + non-trivial run.
    assert!(history.best.val.is_finite());
    assert_eq!(history.epochs.len(), 20);
}

#[test]
fn nuts_sa_drops_into_parallel_tempering() {
    // The trait-composition story: NutsSaSampler impl Sampler<f64>
    // -> ParallelTemperingSampler<NutsSaSampler<...>> works without
    // any extra glue. This test confirms the typed algebra extends
    // cleanly to NUTS as another point.
    let obj = StybTang2D::new();
    let grad = FiniteDiffGradient::new(StybTang2D::new());
    let cool = LogCool::new(5.0_f64, 2.0);
    let sampler = NutsSaSampler::new(obj, grad, cool.clone(), GaussianMomentum, 0.1, 5.0, 4);
    let temps = geometric_ladder(0.5, 10.0, 3);
    let pt = ParallelTemperingSampler::new(sampler, temps, 20, 5);
    let result = pt.run(&cool, 10, 99);
    assert_eq!(result.chain_histories.len(), 3);
    assert!(result.best_val.is_finite());
}
