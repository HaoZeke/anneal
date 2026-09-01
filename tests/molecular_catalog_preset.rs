use anneal_core::catalog::molecular::{
    DESCRIPTOR_SCHEMA, ENGINE_BINARY_INPUT, ENGINE_KIND, WATER_HEXAMER_MOLECULES, descriptor_space,
    engine_binary_digest, engine_config_digest, fresh_evaluation, leftover_descriptor_dim,
    leftover_space, leftover_spec, leftover_values, length_scale, reference_coordinates,
    system_signature, validator_config, water_groups, water_species,
};
use anneal_core::descriptor_space::{UNIVERSAL_DESCRIPTOR_SCHEMA, UNIVERSAL_DESCRIPTOR_VERSION};
use anneal_core::methods::cluster_hopping::Config;
use anneal_core::soap::SoapSpec;

fn engine_digest(byte: u8) -> [u8; 32] {
    [byte; 32]
}

#[test]
fn hexamer_signature_records_gfn2_kind_and_leftover_dim() {
    let species = water_species(WATER_HEXAMER_MOLECULES).unwrap();
    let leftover_dim = leftover_descriptor_dim(&species).unwrap();
    let signature = system_signature(WATER_HEXAMER_MOLECULES, engine_digest(0x11)).unwrap();
    let spec = leftover_spec(&species).unwrap();

    assert_eq!(signature.engine.kind, ENGINE_KIND);
    assert_eq!(signature.engine.kind, "gfn2-xtb-rgpot-v1");
    assert_eq!(signature.engine.config_digest, engine_config_digest());
    assert_eq!(
        signature.engine.external_inputs.get(ENGINE_BINARY_INPUT),
        Some(&engine_digest(0x11))
    );
    assert_eq!(signature.coordinate_dim, 54);
    assert_eq!(signature.atomic_numbers, species);
    assert_eq!(signature.descriptor.schema, DESCRIPTOR_SCHEMA);
    assert_eq!(signature.descriptor.species_channels, vec![1, 8]);
    assert_eq!(
        signature.descriptor.hyperparameters.get("leftover_dim"),
        Some(&leftover_dim.to_string())
    );
    assert_eq!(leftover_dim, species.len() * spec.feat_dim(Some(&species)));
    let leftover_spec = SoapSpec {
        n_max: 3,
        l_max: 6,
        rcut_nn: spec.rcut_nn,
    };
    assert_eq!(leftover_dim, 18 * leftover_spec.feat_dim(Some(&species)));
    assert_eq!(signature.length_scale, length_scale(&species).unwrap());
    let molecular = Config::for_molecular(
        species.clone(),
        water_groups(WATER_HEXAMER_MOLECULES).unwrap(),
        1.0,
    );
    assert_eq!(signature.length_scale, molecular.length_scale);
    assert!((spec.rcut_nn - 3.5 * signature.length_scale).abs() < 1e-15);
}

#[test]
fn leftover_space_describe_has_the_leftover_dimension() {
    let species = water_species(2).unwrap();
    let coordinates = reference_coordinates(2).unwrap();
    let space = leftover_space(&species).unwrap();
    let described = space
        .describe(
            ndarray::ArrayView1::from(coordinates.as_slice()),
            Some(&species),
        )
        .unwrap();
    assert_eq!(
        described.values().len(),
        leftover_descriptor_dim(&species).unwrap()
    );
}

#[test]
fn leftover_vector_length_matches_declared_dimension() {
    let species = water_species(2).unwrap();
    let coordinates = reference_coordinates(2).unwrap();
    let leftover = leftover_values(&coordinates, &species).unwrap();
    assert_eq!(leftover.len(), leftover_descriptor_dim(&species).unwrap());
    assert!(leftover.iter().all(|value| value.is_finite()));
}

#[test]
fn engine_binary_digest_is_part_of_identity() {
    let left = system_signature(6, engine_digest(0x11)).unwrap();
    let right = system_signature(6, engine_digest(0x22)).unwrap();
    assert_ne!(left.digest(), right.digest());
}

#[test]
fn larger_water_signatures_scale_without_changing_engine_kind() {
    let dimer = system_signature(2, engine_digest(0x11)).unwrap();
    let hexamer = system_signature(6, engine_digest(0x11)).unwrap();
    assert_eq!(dimer.engine.kind, hexamer.engine.kind);
    assert_eq!(dimer.engine.config_digest, hexamer.engine.config_digest);
    assert_eq!(dimer.descriptor.schema, hexamer.descriptor.schema);
    assert_ne!(dimer.digest(), hexamer.digest());
    assert_eq!(dimer.coordinate_dim, 18);
    assert_eq!(hexamer.coordinate_dim, 54);
}

#[test]
fn validator_uses_leftover_dimension() {
    let species = water_species(2).unwrap();
    let coordinates = reference_coordinates(2).unwrap();
    let leftover_dim = leftover_descriptor_dim(&species).unwrap();
    let validator = validator_config(&coordinates, leftover_dim).unwrap();
    assert_eq!(validator.descriptor_dim, leftover_dim);
    assert_eq!(validator.reference_coordinates, coordinates);
}

#[test]
fn fresh_evaluation_does_not_invent_an_energy() {
    let coordinates = reference_coordinates(6).unwrap();
    let result = fresh_evaluation(6, &coordinates);
    assert!(result.is_err());
    assert!(system_signature(0, engine_digest(0x11)).is_err());
    assert!(fresh_evaluation(3, &[0.0; 9]).is_err());
    assert!(validator_config(&[0.0; 8], 8).is_err());
}

#[test]
fn leftover_space_is_the_catalog_descriptor() {
    let species = water_species(WATER_HEXAMER_MOLECULES).unwrap();
    let space = leftover_space(&species).unwrap();
    let signature = system_signature(WATER_HEXAMER_MOLECULES, engine_digest(0x11)).unwrap();
    assert_eq!(space.schema().name(), DESCRIPTOR_SCHEMA);
    assert_eq!(space.schema().name(), signature.descriptor.schema);
    assert_eq!(space.schema().version(), signature.descriptor.version);
    assert_eq!(
        engine_binary_digest(&[0x11; 8]),
        engine_binary_digest(&[0x11; 8])
    );
    assert_ne!(
        engine_binary_digest(&[0x11; 8]),
        engine_binary_digest(&[0x22; 8])
    );
}

#[test]
fn water_catalog_uses_the_universal_schema_without_sharing_system_identity() {
    let species = water_species(2).unwrap();
    let water_coordinates = reference_coordinates(2).unwrap();
    let water_space = descriptor_space(&species).unwrap();
    let water_descriptor = water_space
        .describe(
            ndarray::ArrayView1::from(water_coordinates.as_slice()),
            Some(&species),
        )
        .unwrap();
    let lj_space = anneal_core::catalog::lj::descriptor_space();
    let lj_coordinates = anneal_core::catalog::lj::reference_coordinates(6).unwrap();
    let lj_species = vec![18; 6];
    let lj_descriptor = lj_space
        .describe(
            ndarray::ArrayView1::from(lj_coordinates.as_slice()),
            Some(&lj_species),
        )
        .unwrap();

    assert_eq!(DESCRIPTOR_SCHEMA, UNIVERSAL_DESCRIPTOR_SCHEMA);
    assert_eq!(water_space.schema().name(), UNIVERSAL_DESCRIPTOR_SCHEMA);
    assert_eq!(water_space.schema().version(), UNIVERSAL_DESCRIPTOR_VERSION);
    assert_eq!(
        water_descriptor.values().len(),
        lj_descriptor.values().len()
    );

    let water_signature = system_signature(2, engine_digest(0x11)).unwrap();
    let lj_signature = anneal_core::catalog::lj::system_signature(6).unwrap();
    assert_eq!(
        water_signature.descriptor.schema,
        lj_signature.descriptor.schema
    );
    assert_ne!(water_signature.digest(), lj_signature.digest());
}

#[cfg(feature = "bank-rpc")]
#[test]
fn leftover_space_builds_a_gfn2_water_scientific_config() {
    use anneal_core::catalog_rpc::server::ServerConfig;

    let species = water_species(2).unwrap();
    let reference = reference_coordinates(2).unwrap();
    let leftover_dim = leftover_descriptor_dim(&species).unwrap();
    let signature = system_signature(2, engine_digest(0x11)).unwrap();
    let digest = signature.digest();
    let config = ServerConfig::new("gfn2-water", "hexamer-test", digest, [0, 1, 2, 3])
        .unwrap()
        .with_scientific_state(
            signature,
            leftover_space(&species).unwrap(),
            validator_config(&reference, leftover_dim).unwrap(),
            8,
            0.05,
            400,
            |coordinates| fresh_evaluation(2, coordinates),
        );
    assert!(config.is_ok());
    assert!(fresh_evaluation(2, &reference).is_err());
}
