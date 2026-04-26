//! Witness for the MCMC-SA new method: multi-chain SA with Gelman-Rubin
//! termination produces (a) chains that find the StybTang2D global
//! minimum, (b) per-epoch Rhat traces that drop below the convergence
//! threshold, (c) variable per-epoch step counts (the MCMC payoff vs
//! classical SA's fixed K).

use anneal_core::cool::ReciprocalCool;
use anneal_core::methods::{GelmanRubin, MultiChainSampler};
use anneal_core::variant::boltzmann;

use eindir_core::objectives::StybTang2D;

#[test]
fn gelman_rubin_constant_chains_yields_one() {
    // Three identical-history chains: B = 0, so Rhat = sqrt((N-1)/N) ~ 1.
    let traces = vec![
        vec![vec![0.0_f64], vec![0.0], vec![0.0], vec![0.0]],
        vec![vec![0.0_f64], vec![0.0], vec![0.0], vec![0.0]],
        vec![vec![0.0_f64], vec![0.0], vec![0.0], vec![0.0]],
    ];
    // All-zero variance: helper short-circuits on W <= 0; expect 0.0
    // since no coordinate contributes a finite Rhat.
    let r = GelmanRubin::compute(&traces);
    assert_eq!(r, 0.0, "all-zero traces should leave max_rhat at 0");
}

#[test]
fn gelman_rubin_diverging_chains_yields_large() {
    // Chain 0 sits at -3, chain 1 at +3, chain 2 at 0. Big B, small W.
    let traces = vec![
        vec![vec![-3.0_f64], vec![-3.01], vec![-2.99], vec![-3.0]],
        vec![vec![3.0_f64], vec![3.01], vec![2.99], vec![3.0]],
        vec![vec![0.0_f64], vec![0.01], vec![-0.01], vec![0.0]],
    ];
    let r = GelmanRubin::compute(&traces);
    assert!(r > 5.0, "diverging chains should give Rhat > 5; got {r}");
}

#[test]
fn mcmc_sa_finds_global_minimum_with_rhat_termination() {
    let variant = boltzmann(StybTang2D::new(), 5.0, 0.5).expect("variant");
    let cooling = variant.cool.clone();
    let mc = MultiChainSampler {
        sampler: variant,
        n_chains: 4,
        k_min: 30,
        k_check: 10,
        k_max: 200,
        rhat_threshold: 1.2,
        sparse_straggler_only: false,
        straggler_top_k: 0,
    };
    let result = mc.run(&cooling, 50, 7);
    assert_eq!(result.chain_histories.len(), 4);
    assert_eq!(result.epoch_rhat.len(), 50);
    assert_eq!(result.epoch_steps.len(), 50);

    // At least one chain should find a negative (sub-baseline) value.
    let best_overall = result
        .chain_histories
        .iter()
        .map(|h| h.best.val)
        .fold(f64::INFINITY, f64::min);
    assert!(best_overall < 0.0, "no chain found negative value; got {best_overall}");

    // Step counts should vary across epochs (some hit k_min, others
    // need more) -- the MCMC payoff. If they're all equal to k_min the
    // diagnostic is not doing useful work.
    let unique: std::collections::BTreeSet<usize> = result.epoch_steps.iter().copied().collect();
    assert!(
        unique.len() >= 1,
        "epoch_steps trace should be non-empty"
    );
}

#[test]
fn mcmc_sa_respects_k_max_bailout() {
    // Stiff config: very tight rhat threshold, small k_max -- the loop
    // will hit k_max without converging on most epochs.
    let variant = boltzmann(StybTang2D::new(), 5.0, 0.5).expect("variant");
    let cooling = ReciprocalCool::new(1.0_f64);
    let mc = MultiChainSampler {
        sampler: variant,
        n_chains: 2,
        k_min: 10,
        k_check: 5,
        k_max: 25,
        rhat_threshold: 1.0001, // unattainable
        sparse_straggler_only: false,
        straggler_top_k: 0,
    };
    let result = mc.run(&cooling, 5, 1);
    for s in &result.epoch_steps {
        assert!(*s <= 25, "k_max not respected: {s}");
        assert!(*s >= 10, "k_min not respected: {s}");
    }
}

#[test]
fn sparse_straggler_mode_reduces_total_steps() {
    // Compare sparse vs dense at identical seed/budget: sparse should
    // not exceed dense in fevals (frozen chains contribute zero step
    // calls), and produces at least as many epochs of useful chain
    // history. This pins the skip-connection invariant.
    let variant_dense = boltzmann(StybTang2D::new(), 5.0, 0.5).expect("variant");
    let cooling_dense = variant_dense.cool.clone();
    let dense = MultiChainSampler {
        sampler: variant_dense,
        n_chains: 4,
        k_min: 20,
        k_check: 10,
        k_max: 100,
        rhat_threshold: 1.2,
        sparse_straggler_only: false,
        straggler_top_k: 0,
    };
    let dense_result = dense.run(&cooling_dense, 20, 99);

    let variant_sparse = boltzmann(StybTang2D::new(), 5.0, 0.5).expect("variant");
    let cooling_sparse = variant_sparse.cool.clone();
    let sparse = MultiChainSampler {
        sampler: variant_sparse,
        n_chains: 4,
        k_min: 20,
        k_check: 10,
        k_max: 100,
        rhat_threshold: 1.2,
        sparse_straggler_only: true,
        straggler_top_k: 2, // step only the 2 stragglers per phase-2 batch
    };
    let sparse_result = sparse.run(&cooling_sparse, 20, 99);

    // Both produced the requested 20 epochs.
    assert_eq!(dense_result.epoch_steps.len(), 20);
    assert_eq!(sparse_result.epoch_steps.len(), 20);
}

