#[test]
fn crate_publication_keeps_cargo_verification_enabled() {
    let workflow = include_str!("../.github/workflows/release.yml");

    assert!(workflow.contains("cargo publish"));
    assert!(!workflow.contains("--no-verify"));
}

#[test]
fn scientific_tests_optimize_kernels_without_disabling_debug_checks() {
    let manifest = include_str!("../Cargo.toml");

    assert!(manifest.contains("[profile.test]"));
    assert!(manifest.contains("opt-level = 2"));
    assert!(manifest.contains("debug-assertions = true"));
    assert!(manifest.contains("overflow-checks = true"));
}
