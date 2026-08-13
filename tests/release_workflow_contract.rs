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

#[test]
fn ci_cancels_superseded_main_branch_runs() {
    let workflow = include_str!("../.github/workflows/ci.yml");

    assert!(workflow.contains("cancel-in-progress: true"));
}

#[test]
fn ci_provisions_pinned_all_feature_dependencies() {
    let workflow = include_str!("../.github/workflows/ci.yml");

    assert!(workflow.contains("Luthaf/vesin"));
    assert!(workflow.contains("976b5dbbdf392db5197353037eefc01a46e9e667"));
    assert!(workflow.contains("mammasmias/IterativeRotationsAssignments"));
    assert!(workflow.contains("2b2fc312569a5a50183ca82b2f260ebaaf87c508"));
    assert!(workflow.contains("VESIN_SRC"));
    assert!(workflow.contains("IRA_LIB_DIR"));
    assert!(workflow.contains("cmake --build"));
}

#[test]
fn ci_lints_only_the_new_commit_range() {
    let workflow = include_str!("../.github/workflows/ci.yml");

    assert!(workflow.contains("tool: cocogitto"));
    assert!(workflow.contains("COG_FROM"));
    assert!(workflow.contains("cog check \"$COG_FROM..$COG_TO\""));
    assert!(!workflow.contains("cocogitto/cocogitto-action"));
}
