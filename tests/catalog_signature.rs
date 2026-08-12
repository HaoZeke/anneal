use anneal_core::catalog::{DescriptorSignature, EngineSignature, SystemSignature};
use std::collections::BTreeMap;

fn digest(byte: u8) -> [u8; 32] {
    [byte; 32]
}

fn signature_with_map_order(reverse: bool) -> SystemSignature {
    let mut external_inputs = BTreeMap::new();
    let mut hyperparameters = BTreeMap::new();
    let input_rows = [
        ("parameters".to_owned(), digest(0x22)),
        ("weights".to_owned(), digest(0x33)),
    ];
    let hyperparameter_rows = [
        ("cutoff".to_owned(), "5.0 angstrom".to_owned()),
        ("n_max".to_owned(), "8".to_owned()),
    ];
    let order: &[usize] = if reverse { &[1, 0] } else { &[0, 1] };
    for &i in order {
        external_inputs.insert(input_rows[i].0.clone(), input_rows[i].1);
        hyperparameters.insert(
            hyperparameter_rows[i].0.clone(),
            hyperparameter_rows[i].1.clone(),
        );
    }

    SystemSignature {
        atomic_numbers: vec![8, 1, 1],
        coordinate_dim: 9,
        group_labels: vec![0, 0, 0],
        group_schema: "one-rigid-molecule-v1".to_owned(),
        frozen_mask: vec![false, false, false],
        cell: Some([12.0, 0.0, 0.0, 0.0, 13.0, 0.0, 0.0, 0.0, 14.0]),
        periodic: [true, true, false],
        length_scale: 1.0,
        energy_scale: 27.211_386_245_988,
        engine: EngineSignature {
            kind: "xtb".to_owned(),
            config_digest: digest(0x11),
            external_inputs,
        },
        descriptor: DescriptorSignature {
            schema: "multiscale-soap".to_owned(),
            version: 2,
            hyperparameters,
            species_channels: vec![1, 8],
        },
        validation_schema_version: 3,
    }
}

fn assert_distinct(left: &SystemSignature, right: &SystemSignature) {
    assert_ne!(left, right);
    assert_ne!(left.canonical_bytes(), right.canonical_bytes());
    assert_ne!(left.digest(), right.digest());
}

#[test]
fn canonical_identity_is_independent_of_map_insertion_order() {
    let forward = signature_with_map_order(false);
    let reverse = signature_with_map_order(true);

    assert_eq!(forward, reverse);
    assert_eq!(forward.canonical_bytes(), reverse.canonical_bytes());
    assert_eq!(forward.digest(), reverse.digest());
    assert_eq!(forward.digest().len(), 32);
    assert!(
        forward
            .canonical_bytes()
            .starts_with(b"ANNEAL\0SYSTEM_SIGNATURE\0")
    );
}

#[test]
fn every_contract_field_changes_identity() {
    let base = signature_with_map_order(false);

    let mut changed = base.clone();
    changed.atomic_numbers = vec![1, 8, 1];
    assert_distinct(&base, &changed);

    let mut changed = base.clone();
    changed.coordinate_dim = 12;
    assert_distinct(&base, &changed);

    let mut changed = base.clone();
    changed.group_labels = vec![0, 1, 1];
    assert_distinct(&base, &changed);

    let mut changed = base.clone();
    changed.group_schema = "independent-atoms-v1".to_owned();
    assert_distinct(&base, &changed);

    let mut changed = base.clone();
    changed.frozen_mask[2] = true;
    assert_distinct(&base, &changed);

    let mut changed = base.clone();
    changed.cell.as_mut().unwrap()[4] = 15.0;
    assert_distinct(&base, &changed);

    let mut changed = base.clone();
    changed.periodic = [false, false, false];
    assert_distinct(&base, &changed);

    let mut changed = base.clone();
    changed.length_scale = 0.529_177_210_903;
    assert_distinct(&base, &changed);

    let mut changed = base.clone();
    changed.energy_scale = 1.0;
    assert_distinct(&base, &changed);

    let mut changed = base.clone();
    changed.engine.kind = "pet-mad".to_owned();
    assert_distinct(&base, &changed);

    let mut changed = base.clone();
    changed.engine.config_digest = digest(0x44);
    assert_distinct(&base, &changed);

    let mut changed = base.clone();
    changed
        .engine
        .external_inputs
        .insert("weights".to_owned(), digest(0x55));
    assert_distinct(&base, &changed);

    let mut changed = base.clone();
    changed.descriptor.schema = "ace".to_owned();
    assert_distinct(&base, &changed);

    let mut changed = base.clone();
    changed.descriptor.version = 4;
    assert_distinct(&base, &changed);

    let mut changed = base.clone();
    changed
        .descriptor
        .hyperparameters
        .insert("n_max".to_owned(), "12".to_owned());
    assert_distinct(&base, &changed);

    let mut changed = base.clone();
    changed.descriptor.species_channels = vec![8, 1];
    assert_distinct(&base, &changed);

    let mut changed = base.clone();
    changed.validation_schema_version = 4;
    assert_distinct(&base, &changed);
}
