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
