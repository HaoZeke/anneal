use anneal_core::methods::cluster_hopping::Config;

#[test]
fn restart_is_not_a_recommended_stall_arm() {
    for n in [38usize, 75, 98] {
        assert!(!Config::recommended(n).restart_on_stall);
        assert!(!Config::derived(n).restart_on_stall);
        assert!(Config::recommended(n).escape_on_stall);
        assert!(!Config::communicating(n).restart_on_stall);
        assert!(!Config::communicating(n).escape_on_stall);
    }
}
