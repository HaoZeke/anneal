//! Exact short searches for the species-free recommended preset.
//!
//! The pins guard the minima, hop counts, converged quenches, capped
//! quenches, and spent ledgers. The packing SOAP arm follows the observed
//! descriptor cloud; the SOAP-packing key remains confined to the packing
//! superbasin preset.

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
LJ98 -520.3133998215357678 10 10 7 2500";
    assert_eq!(records.join("\n"), expected);
}
