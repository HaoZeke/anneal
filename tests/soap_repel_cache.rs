use anneal_core::catalog::set_packing_references;
use anneal_core::methods::cluster_hopping::ClusterMove;
use anneal_core::soap::{SoapSpec, packing_mean_nu3, push_away_means};
use ndarray::{Array1, array};
use rand::{SeedableRng, rngs::StdRng};

#[test]
fn a_reference_mean_is_keyed_by_its_descriptor_parameters() {
    let x = array![0.0, 0.0, 0.0, 1.1, 0.0, 0.0, 0.0, 1.2, 0.0, 0.0, 0.0, 1.3];
    let peer: Array1<f64> = &x * 1.03;
    set_packing_references(vec![peer.to_vec(); 3]);
    let mut rng = StdRng::seed_from_u64(104);
    let cap = 1e-4;
    for cutoff in [2.5, 3.5, 2.5] {
        let spec = SoapSpec {
            rcut_nn: cutoff,
            ..Default::default()
        };
        let mean = packing_mean_nu3(peer.view(), spec, None, None);
        let expected = push_away_means(x.view(), &[mean.to_vec(); 3], spec, cap).unwrap();
        let actual = ClusterMove::SoapRepel { rmsd: cap, cutoff }.propose(x.view(), 1.0, &mut rng);
        assert_eq!(
            actual, expected,
            "a cutoff {cutoff} proposal must not consume a reference mean from a different map"
        );
    }
    set_packing_references(Vec::new());
}
