use std::fs;
use std::path::PathBuf;

#[test]
fn hard_lj_arrays_default_to_the_partition_limit() {
    let script = PathBuf::from(env!("CARGO_MANIFEST_DIR"))
        .join("scripts")
        .join("elja_submit_jcc_lj.sh");
    let source = fs::read_to_string(&script)
        .unwrap_or_else(|error| panic!("failed to read {}: {error}", script.display()));

    assert!(
        source.contains(r#"--time="${ELJA_TIME:-2-00:00:00}""#),
        "hard-LJ arrays must default to the s-normal two-day limit"
    );
}

#[test]
fn hard_lj_arrays_accept_an_isolated_campaign_name() {
    let script = PathBuf::from(env!("CARGO_MANIFEST_DIR"))
        .join("scripts")
        .join("elja_submit_jcc_lj.sh");
    let source = fs::read_to_string(&script)
        .unwrap_or_else(|error| panic!("failed to read {}: {error}", script.display()));

    assert!(
        source.contains(r#"CAMPAIGN=${JCC_CAMPAIGN:-jcc-2026-${STAGE}}"#),
        "hard-LJ arrays must permit a caller-selected campaign namespace"
    );
    assert!(
        source.contains(r#"CATALOG_CAMPAIGN=${CAMPAIGN}"#),
        "the selected namespace must reach every ensemble array"
    );
    assert!(
        source.contains(r#"JCC_QUALIFIER_PYTHON=${QUALIFIER_PYTHON}"#),
        "the validated launcher Python must reach every ensemble qualifier"
    );
}

#[test]
fn hard_lj_manifests_pin_every_scientific_input() {
    let script = PathBuf::from(env!("CARGO_MANIFEST_DIR"))
        .join("scripts")
        .join("elja_jcc_lj_ensemble.sh");
    let source = fs::read_to_string(&script)
        .unwrap_or_else(|error| panic!("failed to read {}: {error}", script.display()));

    for required in [
        r#"source_commit=%s\n"#,
        r#"sha256sum "$BIN""#,
        r#"sha256sum "$0""#,
        r#"sha256sum "$QUALIFIER""#,
        r#"sha256sum "$CALIBRATION""#,
    ] {
        assert!(
            source.contains(required),
            "hard-LJ manifest is missing provenance field {required}"
        );
    }
}

#[test]
fn molecular_build_produces_every_paired_campaign_executable() {
    let script = PathBuf::from(env!("CARGO_MANIFEST_DIR"))
        .join("scripts")
        .join("elja_build_rgpot_ex.sh");
    let source = fs::read_to_string(&script)
        .unwrap_or_else(|error| panic!("failed to read {}: {error}", script.display()));

    for example in ["molecular_cluster", "slab_adsorption", "bank_server"] {
        assert!(
            source.contains(&format!("--example {example}")),
            "Elja molecular build must compile {example}"
        );
        assert!(
            source.contains(&format!("target/release/examples/{example}")),
            "Elja molecular build must verify {example}"
        );
    }
}

#[test]
fn molecular_driver_keeps_the_measured_quench_defaults() {
    let example = PathBuf::from(env!("CARGO_MANIFEST_DIR"))
        .join("examples")
        .join("molecular_cluster.rs");
    let source = fs::read_to_string(&example)
        .unwrap_or_else(|error| panic!("failed to read {}: {error}", example.display()));

    assert!(
        !source.contains("cfg.screen_steps = 6;"),
        "the molecular driver must not replace the measured screening quench with six steps"
    );
    assert!(
        !source.contains("cfg.relax_steps = 60;"),
        "the molecular driver must not cap a full molecular quench at sixty steps"
    );
    assert!(
        source.contains("screen/full/check"),
        "molecular campaign output must expose where charged quench work went"
    );
}
