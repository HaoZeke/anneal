use anneal_core::methods::cluster_hopping::{
    ClusterMove, Config, MoveLibrary, covalent_radius, repack_rigid_groups,
};
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

#[test]
fn symmetry_and_rigid_repacking_follow_the_declared_length_scale() {
    let reduced = Config::with_scales(2, 1.0, 1.0);
    let physical = Config::with_scales(2, 2.5, 1.0);
    assert!((physical.symmetry_merge_radius / reduced.symmetry_merge_radius - 2.5).abs() < 1e-12);

    let groups = vec![vec![0], vec![1]];
    let reduced_template = Array1::from_vec(vec![0.0, 0.0, 0.0, 1.0, 0.0, 0.0]);
    let physical_template = reduced_template.mapv(|coordinate| coordinate * 2.5);
    let mut reduced_rng = StdRng::seed_from_u64(37);
    let mut physical_rng = StdRng::seed_from_u64(37);
    let reduced_repack = repack_rigid_groups(
        reduced_template.view(),
        &groups,
        reduced.length_scale,
        &mut reduced_rng,
    );
    let physical_repack = repack_rigid_groups(
        physical_template.view(),
        &groups,
        physical.length_scale,
        &mut physical_rng,
    );

    for (reduced_coordinate, physical_coordinate) in
        reduced_repack.iter().zip(physical_repack.iter())
    {
        assert!((physical_coordinate - 2.5 * reduced_coordinate).abs() < 1e-12);
    }
}

#[test]
fn soap_pullback_follows_the_declared_length_scale() {
    fn soap_scales(cfg: &Config) -> (f64, f64) {
        cfg.move_library
            .kernels(cfg)
            .into_iter()
            .find_map(|kernel| match kernel {
                ClusterMove::Soap { rmsd, cutoff } => Some((rmsd, cutoff)),
                _ => None,
            })
            .expect("LeanBurst must include its SOAP pullback")
    }

    let mut reduced = Config::with_scales(13, 1.0, 1.0);
    reduced.move_library = MoveLibrary::LeanBurst;
    let mut physical = Config::with_scales(13, 2.5, 1.0);
    physical.move_library = MoveLibrary::LeanBurst;

    let (reduced_rmsd, reduced_cutoff) = soap_scales(&reduced);
    let (physical_rmsd, physical_cutoff) = soap_scales(&physical);
    assert!((physical_rmsd / reduced_rmsd - 2.5).abs() < 1e-12);
    assert!((physical_cutoff / reduced_cutoff - 2.5).abs() < 1e-12);
}
