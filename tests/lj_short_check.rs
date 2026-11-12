//! Species-free walks. The GFN2 session is not on this path.

use anneal_core::methods::cluster_hopping::{Config, Ledger};
use anneal_core::methods::cluster_search::search;
use anneal_core::potentials::PairPotential;

#[test]
fn lj38_and_lj98_short_searches_stay_put() {
    let cases = [
        (38usize, 1u64, 2500usize, -171.3449081396, 23usize),
        (98, 7, 2500, -522.4010032187, 16),
    ];
    for (n, seed, budget, best, hops) in cases {
        let cfg = Config::recommended(n);
        let pot = PairPotential::lennard_jones(n);
        let mut ledger = Ledger::new(budget);
        let (out, _) = search(&pot, &cfg, &mut ledger, seed).expect("finite");
        assert!(
            (out.best - best).abs() < 1e-9,
            "LJ{n} best {} moved from {best}",
            out.best
        );
        assert_eq!(out.hops, hops, "LJ{n} hops");
        assert_eq!(ledger.spent(), budget, "LJ{n} spent the budget");
    }
}
