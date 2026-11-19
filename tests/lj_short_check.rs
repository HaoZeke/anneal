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
//! Removing structure-specific proposals changes the LJ38 pin. Offering
//! the compacted surface at every size moves the short LJ98 trajectory.
//! A restraint block that does not beat its opening incumbent returns
//! the chain to where it stood when that incumbent was found, and later
//! blocks hold the plain surface. That changes which of these short
//! descents are counted as converged.

use anneal_core::methods::cluster_hopping::{Config, Ledger};
use anneal_core::methods::cluster_search::search;
use anneal_core::potentials::PairPotential;

#[test]
fn lj38_and_lj98_short_searches_stay_put() {
    anneal_core::install_rgpot_kernels!();
    // (n, seed, budget, best, hops, converged, capped)
    let cases: &[(usize, u64, usize, f64, usize, usize, usize)] = &[
        (38, 1, 2500, -170.711_308_119_462_44, 21, 14, 19),
        (98, 7, 2500, -529.681_963_509_890_6, 3, 4, 4),
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
LJ38 -170.7113081194624442 21 14 19 2500
LJ98 -529.6819635098905792 3 4 4 2500";
    assert_eq!(records.join("\n"), expected);
}
