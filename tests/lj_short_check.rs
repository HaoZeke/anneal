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
//! The packing SOAP hop follows the observed descriptor residual on
//! every build. These pins were remeasured when that residual replaced
//! the fivefold template, and again when each pair's harmonics are
//! formed once. Both trajectories stay exact.

use anneal_core::methods::cluster_hopping::{Config, Ledger};
use anneal_core::methods::cluster_search::search;
use anneal_core::potentials::PairPotential;

#[test]
fn lj38_and_lj98_short_searches_stay_put() {
    anneal_core::install_rgpot_kernels!();
    // (n, seed, budget, best, hops, converged, capped)
    let cases: &[(usize, u64, usize, f64, usize, usize, usize)] = &[
        (38, 1, 2500, -172.298_128_722_104_8, 38, 33, 26),
        (98, 7, 2500, -520.313_399_821_534_3, 10, 10, 7),
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
LJ38 -172.2981287221047921 38 33 26 2500
LJ98 -520.3133998215349720 10 10 7 2500";
    assert_eq!(records.join("\n"), expected);
}
