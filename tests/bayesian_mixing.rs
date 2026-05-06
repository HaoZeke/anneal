//! Witness for automatic Bayesian chain mixing: callers provide only an
//! inner sampler and a proposal budget; chain count and allocation are
//! inferred online from posterior improvement evidence.

use anneal_core::variant::boltzmann;
use anneal_core::BayesianMixingSampler;

use eindir_core::objectives::StybTang2D;

#[test]
fn bayesian_mixing_uses_single_budget_knob() {
    let variant = boltzmann(StybTang2D::new(), 5.0, 0.5).expect("variant");
    let sampler = BayesianMixingSampler::new(variant, 128);
    let result = sampler.run(42);

    assert_eq!(result.total_proposals(), 128);
    assert!(result.n_chains >= 2);
    assert_eq!(result.proposal_counts.len(), result.n_chains);
    assert!(result.best_val.is_finite());
    assert!(
        result.proposal_counts.iter().copied().max().unwrap_or(0) > 64,
        "posterior allocation should protect one incumbent chain"
    );
}
