use anneal_core::catalog::packing::{PackingPeerScope, nearby_packing_peers, set_packing_peers};
use anneal_core::catalog::{packing_references, set_packing_references};
use anneal_core::methods::cluster_hopping::ClusterMove;
use anneal_core::soap::{SoapSpec, push_away_clouds, step_away_cloud};
use ndarray::{Array1, s};
use rand::{SeedableRng, rngs::StdRng};

fn structure(text: &str) -> Array1<f64> {
    text.lines()
        .skip(2)
        .flat_map(|line| line.split_whitespace().skip(1).take(3))
        .map(|value| value.parse::<f64>().unwrap())
        .collect()
}

#[test]
fn live_nearby_peers_supply_separation_without_archival_multiplicity() {
    let x = structure(include_str!("fixtures/lj38_ico.xyz"));
    let peer = &x * 1.0001;
    let _scope = PackingPeerScope::new(true);
    set_packing_references(vec![peer.to_vec()]);
    let spec = SoapSpec::default();
    let cap = 1e-4;
    let expected = push_away_clouds(x.view(), &[peer.to_vec()], spec, cap).unwrap();
    for count in 1..=3 {
        set_packing_peers(vec![peer.to_vec(); count]);
        assert_eq!(
            nearby_packing_peers(x.as_slice().unwrap()).unwrap().len(),
            count
        );
        assert_eq!(packing_references().len(), 1);
        let actual = ClusterMove::SoapRepel {
            rmsd: cap,
            cutoff: spec.rcut_nn,
        }
        .propose(x.view(), 1.0, &mut StdRng::seed_from_u64(71));
        assert_eq!(actual, expected);
    }
}

#[test]
fn an_empty_live_view_overrides_history_for_strided_coordinates_too() {
    let x = structure(include_str!("fixtures/lj38_ico.xyz"));
    let peer = &x * 1.0001;
    let archive = vec![peer.to_vec(); 3];
    set_packing_references(archive.clone());
    let _scope = PackingPeerScope::new(true);
    let interleaved: Array1<f64> = x.iter().flat_map(|&value| [value, 42.0]).collect();
    let strided = interleaved.slice(s![..;2]);
    assert!(strided.as_slice().is_none());
    let spec = SoapSpec::default();
    let cap = 1e-4;
    let expected = step_away_cloud(
        strided,
        spec,
        cap,
        None,
        None,
        None,
        &mut StdRng::seed_from_u64(71),
    );
    let actual = ClusterMove::SoapRepel {
        rmsd: cap,
        cutoff: spec.rcut_nn,
    }
    .propose(strided, 1.0, &mut StdRng::seed_from_u64(71));
    assert_eq!(
        actual, expected,
        "array storage cannot change the active evidence source"
    );
    assert_eq!(packing_references(), archive);
}

#[test]
fn peer_locality_tracks_the_proposals_geometry_and_replacement() {
    let ico = structure(include_str!("fixtures/lj75_ico.xyz"));
    let marks = structure(include_str!("fixtures/lj75_marks.xyz"));
    let _scope = PackingPeerScope::new(true);
    set_packing_peers(vec![ico.to_vec(); 2]);
    assert_eq!(
        nearby_packing_peers(ico.as_slice().unwrap()).unwrap().len(),
        2
    );
    assert!(
        nearby_packing_peers(marks.as_slice().unwrap())
            .unwrap()
            .is_empty()
    );
    set_packing_peers(vec![marks.to_vec()]);
    assert_eq!(
        nearby_packing_peers(marks.as_slice().unwrap())
            .unwrap()
            .len(),
        1
    );
    assert!(
        nearby_packing_peers(ico.as_slice().unwrap())
            .unwrap()
            .is_empty()
    );
    set_packing_peers(Vec::new());
    assert!(
        nearby_packing_peers(marks.as_slice().unwrap())
            .unwrap()
            .is_empty()
    );
}

#[test]
fn a_chain_scope_restores_the_callers_view() {
    let x = structure(include_str!("fixtures/lj38_ico.xyz"));
    assert!(nearby_packing_peers(x.as_slice().unwrap()).is_none());
    {
        let _scope = PackingPeerScope::new(true);
        set_packing_peers(vec![x.to_vec(); 2]);
        {
            let _independent = PackingPeerScope::new(false);
            assert!(nearby_packing_peers(x.as_slice().unwrap()).is_none());
        }
        assert_eq!(
            nearby_packing_peers(x.as_slice().unwrap()).unwrap().len(),
            2
        );
    }
    assert!(nearby_packing_peers(x.as_slice().unwrap()).is_none());
}
