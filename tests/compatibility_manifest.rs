use anneal_core::compatibility::{
    AbiStamp, CompatibilityError, EngineDescriptor, ProtocolVersion, validate_eindir_objective,
};
use anneal_core::run_manifest::{ArtifactDigest, RunManifest};
use eindir_core::ffi::{EindirEvalFn, eindir_objective_t, eindir_status_t};

unsafe extern "C" fn constant_objective(
    _user_data: *mut std::ffi::c_void,
    _x: *const dlpk::sys::DLManagedTensorVersioned,
    value_out: *mut f64,
) -> eindir_status_t {
    unsafe { *value_out = 0.0 };
    eindir_status_t::EINDIR_SUCCESS
}

fn test_objective(low: *mut f64, high: *mut f64) -> eindir_objective_t {
    eindir_objective_t {
        dim: 2,
        low,
        high,
        eval_fn: constant_objective as EindirEvalFn,
        grad_fn: None,
        user_data: std::ptr::null_mut(),
        free_fn: None,
        descriptor: std::ptr::null(),
    }
}

#[test]
fn accepts_additive_protocols_and_rejects_incompatible_bridges() {
    let expected = AbiStamp::anneal_default();
    assert_eq!(expected.abi_major, 1);
    assert_eq!(expected.abi_minor, 1);
    assert_eq!(expected.layout_revision, 3);
    let compatible = EngineDescriptor::new("rgpot", ProtocolVersion::new(1, 2), expected);
    assert!(
        compatible
            .validate("anneal.objective", ProtocolVersion::new(1, 1), expected)
            .is_ok()
    );

    let wrong_major = EngineDescriptor::new("rgpot", ProtocolVersion::new(2, 0), expected);
    assert!(matches!(
        wrong_major.validate("anneal.objective", ProtocolVersion::new(1, 1), expected),
        Err(CompatibilityError::ProtocolMajor { .. })
    ));

    let wrong_layout = EngineDescriptor::new(
        "rgpot",
        ProtocolVersion::new(1, 2),
        AbiStamp {
            layout_revision: expected.layout_revision + 1,
            ..expected
        },
    );
    assert!(matches!(
        wrong_layout.validate("anneal.objective", ProtocolVersion::new(1, 1), expected),
        Err(CompatibilityError::AbiLayout { .. })
    ));

    let wrong_major = EngineDescriptor::new(
        "rgpot",
        ProtocolVersion::new(1, 2),
        AbiStamp {
            abi_major: expected.abi_major + 1,
            ..expected
        },
    );
    assert!(matches!(
        wrong_major.validate("anneal.objective", ProtocolVersion::new(1, 1), expected),
        Err(CompatibilityError::AbiMajor { .. })
    ));

    let old_minor = EngineDescriptor::new(
        "rgpot",
        ProtocolVersion::new(1, 2),
        AbiStamp {
            abi_minor: expected.abi_minor - 1,
            ..expected
        },
    );
    assert!(matches!(
        old_minor.validate("anneal.objective", ProtocolVersion::new(1, 1), expected),
        Err(CompatibilityError::AbiMinor { .. })
    ));
}

#[test]
fn manifest_serialization_is_deterministic_and_verifies_artifacts() {
    let mut manifest = RunManifest::new("run-1", 42, 1000);
    let mut engine = EngineDescriptor::new(
        "rgpot",
        ProtocolVersion::new(1, 0),
        AbiStamp::anneal_default(),
    );
    engine.build_identity = Some("rgpot-test@source-revision".to_owned());
    manifest.engine = engine;
    manifest.add_artifact("input.xyz", b"H 0 0 0\n");
    let json = manifest.to_json().unwrap();
    assert_eq!(json, manifest.to_json().unwrap());
    assert!(json.contains("rgpot-test@source-revision"));
    assert_eq!(
        manifest.artifacts[0].sha256,
        ArtifactDigest::of(b"H 0 0 0\n").sha256
    );
    assert!(manifest.verify_artifact("input.xyz", b"H 0 0 0\n").is_ok());
    assert!(manifest.verify_artifact("input.xyz", b"changed\n").is_err());
}

#[test]
fn rejects_malformed_eindir_handles_before_evaluation() {
    let mut low = [0.0, -1.0];
    let mut high = [1.0, 1.0];
    let objective = test_objective(low.as_mut_ptr(), high.as_mut_ptr());
    assert!(unsafe { validate_eindir_objective(&objective, 2) }.is_ok());

    let malformed = test_objective(low.as_mut_ptr(), std::ptr::null_mut());
    assert!(matches!(
        unsafe { validate_eindir_objective(&malformed, 2) },
        Err(CompatibilityError::ObjectiveBoundsNull { .. })
    ));

    high[0] = -2.0;
    let inverted = test_objective(low.as_mut_ptr(), high.as_mut_ptr());
    assert!(matches!(
        unsafe { validate_eindir_objective(&inverted, 2) },
        Err(CompatibilityError::ObjectiveBoundsInvalid { index: 0, .. })
    ));
}
