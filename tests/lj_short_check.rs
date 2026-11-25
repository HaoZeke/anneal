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
//! The constant-free generic packing candidates are offered after 36 hops,
//! and these 2500-call walks end first, so the default pins are develop's.
//! With `packing_deferral = 0` the candidates are offered before the first
//! hop and both walks move to the pins below. Each candidate is screened
//! against the walk's best, so the probe leaves more of the 2500 calls to
//! the walk: LJ38 made 15 hops where the unscreened probe left it 10, and
//! LJ98 7 where it made 5, at the same best energies. The refused
//! candidates are then audited in full before the next hop, which takes
//! those calls back: LJ38 makes 10 hops and LJ98 5, at the same best
//! energies. With the audit never scheduled the walks are 15 and 7 again.

use anneal_core::methods::cluster_hopping::{Config, Ledger};
use anneal_core::methods::cluster_search::search;
use anneal_core::potentials::PairPotential;

struct Case {
    n: usize,
    seed: u64,
    budget: usize,
    packing_deferral: Option<usize>,
    best: f64,
    hops: usize,
    converged: usize,
    capped: usize,
}

const fn case(
    n: usize,
    seed: u64,
    packing_deferral: Option<usize>,
    best: f64,
    walk: (usize, usize, usize),
) -> Case {
    Case {
        n,
        seed,
        budget: 2500,
        packing_deferral,
        best,
        hops: walk.0,
        converged: walk.1,
        capped: walk.2,
    }
}

#[test]
fn lj38_and_lj98_short_searches_stay_put() {
    // develop's walks, then the same walks with the probe at hop 0.
    // The walk is (hops, converged, capped).
    let cases = [
        case(38, 1, None, -170.711_308_119_462_44, (21, 14, 19)),
        case(98, 7, None, -520.107_879_491_539_2, (10, 9, 9)),
        case(38, 1, Some(0), -173.928_426_584_850_03, (10, 20, 13)),
        case(98, 7, Some(0), -531.883_176_996_307_8, (5, 11, 13)),
    ];
    anneal_core::install_rgpot_kernels!();
    let mut records = Vec::new();
    for Case {
        n,
        seed,
        budget,
        packing_deferral: deferral,
        best,
        hops,
        converged,
        capped,
    } in &cases
    {
        let mut cfg = Config::recommended(*n);
        if let Some(deferral) = deferral {
            cfg.packing_deferral = *deferral;
        }
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
            "LJ{n} {deferral:?} {:.16} {} {} {} {}",
            out.best,
            out.hops,
            stats.converged,
            stats.capped,
            ledger.spent()
        ));
        assert!(
            (out.best - best).abs() < 1e-9,
            "LJ{n} {deferral:?} {:.16} hops {} conv {} cap {}",
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
LJ38 None -170.7113081194624442 21 14 19 2500
LJ98 None -520.1078794915391654 10 9 9 2500
LJ38 Some(0) -173.9284265848500297 10 20 13 2500
LJ98 Some(0) -531.8831769963078386 5 11 13 2500";
    assert_eq!(records.join("\n"), expected);
}
