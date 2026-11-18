//! The ordinary LJ driver hands the search a charged gradient, so an answer
//! is validated rather than assumed.

#[test]
fn ordinary_lj_driver_supplies_charged_answer_validation_without_escape_flags() {
    let source = include_str!("../examples/lj_cluster_search.rs");
    let invocation = source
        .split_once("anneal_core::methods::cluster_hopping::optimize_with_settle(")
        .unwrap()
        .1
        .split_once("if cfg.staged_quench")
        .unwrap()
        .0;
    assert!(invocation.contains("Some(&mut grad)"));
    assert!(!invocation.contains("cfg.minima_hopping"));
    assert!(!invocation.contains("None"));
}
