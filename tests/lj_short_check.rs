//! Short recommended-preset searches. A molecular quench change that
//! also moves these is a change to the species-free walk.
//!
//! `Config::recommended` keeps the pair-spectrum merge. It does not
//! switch on the SOAP-packing key: that key belongs to
//! `Config::packing_superbasin`. The recorded minima moved when
//! acceptance, proposals, and the depth-rewarded arm draw switched to
//! the portable libm, and again when the recommended quench cap went
//! from 200 steps to 1000 (`d756085b`). A frozen frame is not part of
//! this walk.
//!
//! The fivefold residual uses the built-in assignment on every build
//! (`81d6bffc`). Linking IRA no longer changes this walk, so the
//! feature build keeps the Hungarian record.
//!
//! The communicating quench reuses a gradient it has already paid for.
//! On this 1000-step walk that leaves room for two more capped LJ38
//! hops inside the same 2500 calls. The minima are the ones above.
//! LJ98's counts are unchanged.

use anneal_core::methods::cluster_hopping::{Config, Ledger};
use anneal_core::methods::cluster_search::search;
use anneal_core::potentials::PairPotential;

#[test]
fn lj38_and_lj98_short_searches_stay_put() {
    // (n, seed, budget, best, hops, converged, capped)
    let cases: &[(usize, u64, usize, f64, usize, usize, usize)] = &[
        (38, 1, 2500, -170.260_483_558_385_52, 25, 16, 23),
        (98, 7, 2500, -520.107_879_491_539_2, 10, 9, 9),
    ];
    let mut records = Vec::new();
    for (n, seed, budget, best, hops, converged, capped) in cases {
        let cfg = Config::recommended(*n);
        assert!(cfg.target_energy.is_none());
        assert!(cfg.species.is_none());
        assert_ne!(
            cfg.keying,
            anneal_core::methods::cluster_hopping::Keying::SoapPacking
        );
        let pot = PairPotential::lennard_jones(*n);
        let mut ledger = Ledger::new(*budget);
        let (out, stats) = search(&pot, &cfg, &mut ledger, *seed).expect("finite");
        records.push(format!(
            "LJ{n} {:.16} {} {} {} {}",
            out.best,
            out.hops,
            stats.converged,
            stats.capped,
            ledger.spent()
        ));
        assert!(
            (out.best - best).abs() < 1e-9,
            "LJ{n} {:.16} hops {} conv {} cap {}",
            out.best,
            out.hops,
            stats.converged,
            stats.capped
        );
        assert_eq!(out.hops, *hops);
        assert_eq!(stats.converged, *converged);
        assert_eq!(stats.capped, *capped);
    }
    let expected = "\
LJ38 -170.2604835583855163 25 16 23 2500
LJ98 -520.1078794915391654 10 9 9 2500";
    assert_eq!(records.join("\n"), expected);
}
