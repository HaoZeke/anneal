// Copyright (c) anneal developers
//! The binary panel uses the recommended preset plus two labels and no diameters.

use anneal_core::methods::cluster_hopping::Config;

#[test]
fn two_species_keeps_the_recommended_preset() {
    let stock = Config::recommended(38);
    let mut labels = vec![200_u32; 38];
    for label in labels.iter_mut().skip(8) {
        *label = 201;
    }
    let cfg = Config::recommended(38).with_species(labels, None);
    assert!(cfg.species_sigma.is_none());
    assert_eq!(cfg.container, stock.container);
    assert_eq!(cfg.min_separation, stock.min_separation);
    assert_eq!(cfg.neighbour_cutoff, stock.neighbour_cutoff);
    assert_eq!(cfg.move_library, stock.move_library);
    assert_eq!(cfg.homotop_descent, stock.homotop_descent);
    assert_eq!(cfg.homotop_ranked, stock.homotop_ranked);
    assert_eq!(cfg.homotop_window, stock.homotop_window);
    assert_eq!(cfg.homotop_reorder, stock.homotop_reorder);
    assert!(!cfg.homotop_reorder);
    let species = cfg.species.expect("labels");
    assert_eq!(species.iter().filter(|z| **z == 200).count(), 8);
    assert!(species.iter().skip(8).all(|z| *z == 201));
}
