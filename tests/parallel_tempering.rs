//! Witness for Method PT: ParallelTemperingSampler + Exchange operator
//! drive any inner Sampler<f64> across a temperature ladder with
//! periodic Metropolis swaps.

use anneal_core::ParallelTemperingSampler;
use anneal_core::exchange::{MetropolisExchange, TsallisExchange};
use anneal_core::geometric_ladder;
use anneal_core::variant::boltzmann;

use eindir_core::objectives::StybTang2D;

#[test]
fn geometric_ladder_endpoints_match() {
    let temps = geometric_ladder(0.1, 10.0, 5);
    assert_eq!(temps.len(), 5);
    assert!((temps[0] - 0.1).abs() < 1e-9);
    assert!((temps[4] - 10.0).abs() < 1e-9);
    // Geometric: T_2 = sqrt(0.1 * 10) = 1.0
    assert!((temps[2] - 1.0).abs() < 1e-9);
}

#[test]
fn pt_drives_boltzmann_inner_kernel_on_styb_tang() {
    let variant = boltzmann(StybTang2D::new(), 5.0, 0.5).expect("variant");
    let cooling = variant.cool.clone();
    let temps = geometric_ladder(0.5, 20.0, 4);
    let pt = ParallelTemperingSampler::new(variant, temps, 50, 5);
    let result = pt.run(&cooling, 30, 42);
    assert_eq!(result.chain_histories.len(), 4);
    assert!(result.swap_attempts > 0);
    assert!(result.swap_accepts <= result.swap_attempts);
    assert!(
        result.best_val < 0.0,
        "PT should find a negative value on StybTang2D; got {}",
        result.best_val
    );
}

#[test]
fn pt_with_tsallis_exchange_runs() {
    let variant = boltzmann(StybTang2D::new(), 5.0, 0.5).expect("variant");
    let cooling = variant.cool.clone();
    let temps = geometric_ladder(0.5, 20.0, 4);
    let pt = ParallelTemperingSampler::with_exchange(
        variant,
        TsallisExchange::new(1.5_f64),
        temps,
        50,
        5,
    );
    let result = pt.run(&cooling, 20, 7);
    assert_eq!(result.chain_histories.len(), 4);
    // Tsallis exchange is stricter than Metropolis -> may have fewer
    // accepts but the run should still produce a finite best_val.
    assert!(result.best_val.is_finite());
}

#[test]
fn pt_metropolis_exchange_default_used_via_new() {
    let variant = boltzmann(StybTang2D::new(), 5.0, 0.5).expect("variant");
    let cooling = variant.cool.clone();
    let temps = geometric_ladder(1.0, 5.0, 2);
    let pt = ParallelTemperingSampler::new(variant, temps, 10, 3);
    let result = pt.run(&cooling, 5, 1);
    assert_eq!(result.chain_histories.len(), 2);
    let _ = MetropolisExchange; // type-check the default exchange exists
}
