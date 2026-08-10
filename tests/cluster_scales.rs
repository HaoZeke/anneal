use anneal_core::methods::cluster_hopping::{ClusterMove, Config, MoveLibrary, covalent_radius};
use anneal_core::movekernel::MoveKernel;
use ndarray::Array1;
use rand::SeedableRng;
use rand::rngs::StdRng;

fn proposal_from<K: MoveKernel<f64>>(kernel: &K, n: usize) -> Array1<f64> {
    let mut rng = StdRng::seed_from_u64(11);
    kernel.propose(Array1::zeros(3 * n).view(), 1.0, &mut rng)
}

#[test]
fn lj_preset_scales_every_unit_bearing_default() {
    let reduced = Config::with_scales(13, 1.0, 1.0);
    let physical = Config::with_scales(13, 2.5, 7.0);

    assert_eq!(reduced.length_scale, 1.0);
    assert_eq!(reduced.energy_scale, 1.0);
    assert!((physical.temperature / reduced.temperature - 7.0).abs() < 1e-12);
    assert!((physical.bias_height / reduced.bias_height - 7.0).abs() < 1e-12);
    assert!((physical.screen_margin / reduced.screen_margin - 7.0).abs() < 1e-12);
    assert!((physical.merge_radius / reduced.merge_radius - 2.5).abs() < 1e-12);
    assert!((physical.neighbour_cutoff / reduced.neighbour_cutoff - 2.5).abs() < 1e-12);
    assert!((physical.symmetrise_cutoff / reduced.symmetrise_cutoff - 2.5).abs() < 1e-12);
    assert!((physical.container / reduced.container - 2.5).abs() < 1e-12);
    assert!((physical.min_separation / reduced.min_separation - 2.5).abs() < 1e-12);
    assert!((physical.record_gradient / reduced.record_gradient - 7.0 / 2.5).abs() < 1e-12);

    let expected_start = 0.9 * physical.length_scale * (physical.n_points as f64).cbrt();
    assert!((physical.start_radius() - expected_start).abs() < 1e-12);
}

#[test]
fn species_preset_derives_its_length_and_selects_one_library() {
    let species: Vec<u32> = (0..6).flat_map(|_| [8, 1, 1]).collect();
    let groups: Vec<Vec<usize>> = (0..6).map(|g| (3 * g..3 * g + 3).collect()).collect();
    let cfg = Config::for_molecular(species.clone(), groups.clone(), 1.0);

    assert!((cfg.length_scale - 2.0 * covalent_radius(8)).abs() < 1e-12);
    assert_eq!(cfg.species.as_deref(), Some(species.as_slice()));
    assert!(matches!(
        cfg.move_library,
        MoveLibrary::Molecular {
            reactive: false,
            ..
        }
    ));
    assert_eq!(cfg.move_library.declared_groups(), Some(groups.as_slice()));
}

#[test]
fn cluster_moves_are_general_move_kernels() {
    let n = 5;
    let kernel = ClusterMove::SinglePoint {
        n_points: n,
        step: 0.4,
    };
    assert_eq!(proposal_from(&kernel, n).len(), 3 * n);
}
