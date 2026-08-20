#[test]
fn crate_publication_keeps_cargo_verification_enabled() {
    let workflow = include_str!("../.github/workflows/release.yml");

    assert!(workflow.contains("cargo publish"));
    assert!(!workflow.contains("--no-verify"));
}

#[test]
fn crate_publication_has_a_registry_fallback_for_git_dependencies() {
    let manifest = include_str!("../Cargo.toml");
    let workflow = include_str!("../.github/workflows/ci.yml");

    assert!(manifest.contains("rgpot-core = { version = \"=3.0.2\", git = \"https://github.com/OmniPotentRPC/rgpot.git\", rev = \"473033d\""));
    assert!(manifest.contains("eindir-core = { git = \"https://github.com/HaoZeke/eindir.git\", rev = \"f3c4213\""));
    assert!(manifest.contains("xtsci-optimize = { git = \"https://github.com/HaoZeke/xtsci-optimize.git\", rev = \"6b4a7fc\""));
    assert!(workflow.contains("cargo publish --dry-run"));
    assert!(!workflow.contains("cargo publish --dry-run --no-verify"));
}

#[test]
fn release_tags_match_every_public_package_version() {
    let workflow = include_str!("../.github/workflows/release.yml");

    assert!(workflow.contains("Validate release metadata"));
    assert!(workflow.contains("${GITHUB_REF_NAME#v}"));
    assert!(workflow.contains("Cargo.toml"));
    assert!(workflow.contains("pyproject.toml"));
    assert!(workflow.contains("CITATION.cff"));
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

#[test]
fn ci_audits_large_blobs_without_a_deprecated_action_runtime() {
    let workflow = include_str!("../.github/workflows/ci.yml");

    assert!(workflow.contains("git rev-list --objects --all"));
    assert!(workflow.contains("git cat-file --batch-check"));
    assert!(!workflow.contains("HaoZeke/large-file-auditor"));
}
