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
    assert_eq!(cfg.budget_window, stock.budget_window);
    assert!(!cfg.budget_window);
    assert_eq!(cfg.temperature, stock.temperature);
    assert_eq!(cfg.theta, stock.theta);
    assert_eq!(cfg.statistical_temperature, stock.statistical_temperature);
    let species = cfg.species.expect("labels");
    assert_eq!(species.iter().filter(|z| **z == 200).count(), 8);
    assert!(species.iter().skip(8).all(|z| *z == 201));
}

/// Labels, diameters, a rigid chart and a molecular library leave every
/// other resolved field on the preset, which is how those paths resolved
/// before a two-label chain forced the budget window.
#[test]
fn species_rigid_and_molecular_configs_keep_the_preset_law() {
    let labels = vec![29_u32, 29, 29, 29, 79, 79, 79, 79];
    let diameters = Some(vec![1.1, 1.1, 1.1, 1.1, 0.8, 0.8, 0.8, 0.8]);
    for (name, base) in [
        ("recommended", Config::recommended(8)),
        ("communicating", Config::communicating(8)),
        ("for_cluster", Config::for_cluster(8)),
        ("derived", Config::derived(8)),
    ] {
        assert_species_keeps_resolved_law(&base, labels.clone(), None, name);
        assert_species_keeps_resolved_law(&base, labels.clone(), diameters.clone(), name);
    }

    let geometry = anneal_core::potentials::gmin_water_geometry();
    for (name, base) in [
        ("recommended-rigid", Config::recommended(4)),
        ("communicating-rigid", Config::communicating(4)),
        ("for-cluster-rigid", Config::for_cluster(4)),
    ] {
        let rigid = base.clone().with_rigid_bodies(geometry.clone());
        assert_eq!(rigid.budget_window, base.budget_window, "{name}");
        assert_eq!(rigid.temperature, base.temperature, "{name}");
        assert_eq!(rigid.theta, base.theta, "{name}");
        assert_eq!(
            rigid.statistical_temperature, base.statistical_temperature,
            "{name}"
        );
        assert!(rigid.move_library.is_rigid_body(), "{name}");
        assert!(rigid.surfaces.is_empty(), "{name}");
        let body_labels = vec![8_u32, 1, 1, 8];
        let labeled = rigid.clone().with_species(body_labels, None);
        assert_eq!(labeled.budget_window, rigid.budget_window, "{name}");
        assert_eq!(labeled.temperature, rigid.temperature, "{name}");
        assert!(labeled.move_library.is_rigid_body(), "{name}");
        let json = labeled.resolved_json().expect("rigid species json");
        assert_resolved_except_species(labeled, &rigid, name);
        assert!(
            json.contains("\"budget_window\":false"),
            "{name} resolved {json}"
        );
    }

    let water = Config::recommended_molecular(
        vec![8, 1, 1, 8, 1, 1],
        vec![vec![0, 1, 2], vec![3, 4, 5]],
        1.0,
    );
    let plain = Config::for_molecular(vec![8, 1, 1], vec![vec![0, 1, 2]], 1.0);
    for (name, molecular) in [("recommended-molecular", water), ("for-molecular", plain)] {
        assert!(!molecular.budget_window, "{name}");
        assert!(molecular.move_library.is_molecular(), "{name}");
        assert!(!molecular.move_library.is_rigid_body(), "{name}");
        let again = molecular
            .clone()
            .with_species(molecular.species.clone().expect("labels"), None);
        assert_eq!(again.budget_window, molecular.budget_window, "{name}");
        assert_eq!(again.temperature, molecular.temperature, "{name}");
        assert_eq!(
            again.resolved_json().expect("molecular again"),
            molecular.resolved_json().expect("molecular"),
            "{name}"
        );
        assert!(
            molecular
                .resolved_json()
                .expect("molecular json")
                .contains("\"budget_window\":false"),
            "{name}"
        );
    }
}

fn assert_species_keeps_resolved_law(
    base: &Config,
    species: Vec<u32>,
    sigma: Option<Vec<f64>>,
    name: &str,
) {
    let adapted = base.clone().with_species(species.clone(), sigma.clone());
    assert_eq!(adapted.budget_window, base.budget_window, "{name}");
    assert_eq!(adapted.temperature, base.temperature, "{name}");
    assert_eq!(adapted.theta, base.theta, "{name}");
    assert_eq!(
        adapted.statistical_temperature, base.statistical_temperature,
        "{name}"
    );
    assert_eq!(adapted.homotop_descent, base.homotop_descent, "{name}");
    assert_eq!(adapted.homotop_ranked, base.homotop_ranked, "{name}");
    assert_eq!(adapted.homotop_window, base.homotop_window, "{name}");
    assert_eq!(adapted.homotop_reorder, base.homotop_reorder, "{name}");
    assert_eq!(adapted.allocate_moves, base.allocate_moves, "{name}");
    assert_eq!(adapted.move_library, base.move_library, "{name}");
    assert_eq!(
        adapted.species.as_deref(),
        Some(species.as_slice()),
        "{name}"
    );
    assert_eq!(adapted.species_sigma, sigma, "{name}");
    assert_resolved_except_shell(adapted.clone(), base, name);
    let flag = if adapted.budget_window {
        "true"
    } else {
        "false"
    };
    let json = adapted.resolved_json().expect("adapted json");
    assert!(
        json.contains(&format!("\"budget_window\":{flag}")),
        "{name} resolved {json}"
    );
    let (max_sigma, min_sigma) = match sigma {
        None => {
            assert_eq!(adapted.neighbour_cutoff, base.neighbour_cutoff, "{name}");
            assert_eq!(adapted.container, base.container, "{name}");
            assert_eq!(adapted.min_separation, base.min_separation, "{name}");
            return;
        }
        Some(values) => (
            values.iter().copied().fold(0.0_f64, f64::max),
            values.iter().copied().fold(f64::INFINITY, f64::min),
        ),
    };
    assert_eq!(
        adapted.neighbour_cutoff,
        base.neighbour_cutoff * max_sigma.max(1.0),
        "{name}"
    );
    assert!(
        (adapted.min_separation - base.min_separation * min_sigma).abs() <= 1e-12,
        "{name} separation {} against {}",
        adapted.min_separation,
        base.min_separation * min_sigma
    );
    if max_sigma > 1.0 {
        assert!(adapted.container > base.container, "{name}");
    }
}

/// Resolved JSON with the species shell put back, so any other field shows up.
fn assert_resolved_except_shell(mut adapted: Config, base: &Config, name: &str) {
    adapted.species = base.species.clone();
    adapted.species_sigma = base.species_sigma.clone();
    adapted.neighbour_cutoff = base.neighbour_cutoff;
    adapted.container = base.container;
    adapted.min_separation = base.min_separation;
    assert_eq!(
        adapted.resolved_json().expect("restored json"),
        base.resolved_json().expect("base json"),
        "{name} changed a resolved field outside labels and the diameter shell"
    );
}

fn assert_resolved_except_species(mut adapted: Config, base: &Config, name: &str) {
    adapted.species = base.species.clone();
    adapted.species_sigma = base.species_sigma.clone();
    assert_eq!(
        adapted.resolved_json().expect("restored rigid json"),
        base.resolved_json().expect("rigid json"),
        "{name} changed a resolved field other than the labels"
    );
}
