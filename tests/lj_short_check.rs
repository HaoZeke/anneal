//! Short recommended-preset searches. A molecular quench change that
//! also moves these is a change to the species-free walk.
//!
//! Merging develop brings its stall-growth hold and the later proposal
//! path. That walk is not the one pinned before the merge. Linking IRA
//! still replaces the Hungarian fivefold assignment with SOFI/IRA, so
//! each build has its own exact record. Both were measured on this
//! merge and repeat.

use anneal_core::methods::cluster_hopping::{Config, Ledger};
use anneal_core::methods::cluster_search::search;
use anneal_core::potentials::PairPotential;

#[test]
fn lj38_and_lj98_short_searches_stay_put() {
    let cases: &[(usize, u64, usize, f64, usize, usize, usize)] = if cfg!(feature = "ira") {
        &[
            (38, 1, 2500, -170.695_491_988_085_14, 21, 17, 18),
            (98, 7, 2500, -527.591_386_501_495_2, 25, 12, 28),
        ]
    } else {
        &[
            (38, 1, 2500, -170.770_415_413_335_23, 33, 20, 31),
            (98, 7, 2500, -533.730_715_134_085_2, 27, 20, 31),
        ]
    };
    let mut records = Vec::new();
    for (n, seed, budget, best, hops, converged, capped) in cases {
        let cfg = Config::recommended(*n);
        assert!(cfg.target_energy.is_none());
        assert!(cfg.species.is_none());
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
        assert!((out.best - best).abs() < 1e-9);
        assert_eq!(out.hops, *hops);
        assert_eq!(stats.converged, *converged);
        assert_eq!(stats.capped, *capped);
    }
    let expected = if cfg!(feature = "ira") {
        "\
LJ38 -170.6954919880851378 21 17 18 2500
LJ98 -527.5913865014952080 25 12 28 2500"
    } else {
        "\
LJ38 -170.7704154133352290 33 20 31 2500
LJ98 -533.7307151340852442 27 20 31 2500"
    };
    assert_eq!(records.join("\n"), expected);
}
