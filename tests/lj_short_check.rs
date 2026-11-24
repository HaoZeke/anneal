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
//! Removing structure-specific proposals changes the LJ38 pin. The search
//! that replaced the structure builders completes a new basin from the
//! group its core carries, and `recommended` runs it, so the pins below
//! are the walk with it on. With it off, the walk is bit-identical to
//! the one the series started from with its orbit completion and stall
//! symmetrisation off, so the second set of pins is that walk.

use anneal_core::methods::cluster_hopping::{Config, Ledger};
use anneal_core::methods::cluster_search::search;
use anneal_core::potentials::PairPotential;

#[test]
fn lj38_and_lj98_short_searches_stay_put() {
    anneal_core::install_rgpot_kernels!();
    struct Case {
        n: usize,
        seed: u64,
        budget: usize,
        completion: bool,
        best: f64,
        hops: usize,
        converged: usize,
        capped: usize,
    }
    let case = |n, seed, completion, best, hops, converged, capped| Case {
        n,
        seed,
        budget: 2500,
        completion,
        best,
        hops,
        converged,
        capped,
    };
    let cases = [
        case(38, 1, true, -172.237_562_735_773_4, 40, 29, 33),
        case(98, 7, true, -520.107_879_491_539_2, 15, 9, 14),
        case(38, 1, false, -171.499_842_453_860_6, 46, 27, 46),
        case(98, 7, false, -531.623_165_861_692_3, 20, 13, 21),
    ];
    let mut records = Vec::new();
    for c in &cases {
        let (n, completion) = (c.n, c.completion);
        let mut cfg = Config::recommended(n);
        cfg.symmetry_completion_on_new = completion;
        assert!(cfg.target_energy.is_none());
        assert!(cfg.species.is_none());
        assert_ne!(
            cfg.keying,
            anneal_core::methods::cluster_hopping::Keying::SoapPacking
        );
        let pot = PairPotential::lennard_jones(n);
        let mut ledger = Ledger::new(c.budget);
        let (out, stats) = search(&pot, &cfg, &mut ledger, c.seed).expect("finite");
        records.push(format!(
            "LJ{n} {completion} {:.16} {} {} {} {}",
            out.best,
            out.hops,
            stats.converged,
            stats.capped,
            ledger.spent()
        ));
        assert!(
            (out.best - c.best).abs() < 1e-9,
            "LJ{n} {:.16} hops {} conv {} cap {}",
            out.best,
            out.hops,
            stats.converged,
            stats.capped
        );
        assert_eq!(out.hops, c.hops);
        assert_eq!(stats.converged, c.converged);
        assert_eq!(stats.capped, c.capped);
    }
    let expected = "\
LJ38 true -172.2375627357733947 40 29 33 2500
LJ98 true -520.1078794915391654 15 9 14 2500
LJ38 false -171.4998424538606514 46 27 46 2500
LJ98 false -531.6231658616923141 20 13 21 2500";
    assert_eq!(records.join("\n"), expected);
}
