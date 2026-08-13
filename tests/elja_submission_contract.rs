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
