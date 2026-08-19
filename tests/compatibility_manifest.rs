use anneal_core::compatibility::{AbiStamp, CompatibilityError, EngineDescriptor, ProtocolVersion};
use anneal_core::run_manifest::{ArtifactDigest, RunManifest};

#[test]
fn accepts_additive_protocols_and_rejects_incompatible_bridges() {
    let expected = AbiStamp::anneal_default();
    let compatible = EngineDescriptor::new("rgpot", ProtocolVersion::new(1, 2), expected);
    assert!(compatible.validate("anneal.objective", ProtocolVersion::new(1, 1), expected).is_ok());

    let wrong_major = EngineDescriptor::new("rgpot", ProtocolVersion::new(2, 0), expected);
    assert!(matches!(
        wrong_major.validate("anneal.objective", ProtocolVersion::new(1, 1), expected),
        Err(CompatibilityError::ProtocolMajor { .. })
    ));

    let wrong_layout = EngineDescriptor::new(
        "rgpot",
        ProtocolVersion::new(1, 2),
        AbiStamp { layout_revision: expected.layout_revision + 1, ..expected },
    );
    assert!(matches!(
        wrong_layout.validate("anneal.objective", ProtocolVersion::new(1, 1), expected),
        Err(CompatibilityError::AbiLayout { .. })
    ));
}

#[test]
fn manifest_serialization_is_deterministic_and_verifies_artifacts() {
    let mut manifest = RunManifest::new("run-1", 42, 1000);
    manifest.engine = EngineDescriptor::new(
        "rgpot",
        ProtocolVersion::new(1, 0),
        AbiStamp::anneal_default(),
    );
    manifest.add_artifact("input.xyz", b"H 0 0 0\n");
    let json = manifest.to_json().unwrap();
    assert_eq!(json, manifest.to_json().unwrap());
    assert_eq!(manifest.artifacts[0].sha256, ArtifactDigest::of(b"H 0 0 0\n").sha256);
    assert!(manifest.verify_artifact("input.xyz", b"H 0 0 0\n").is_ok());
    assert!(manifest.verify_artifact("input.xyz", b"changed\n").is_err());
}
