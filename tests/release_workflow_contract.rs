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

    assert!(manifest.contains("rgpot-core = { version = \"=3.5.1\", optional = true }"));
    assert!(manifest.contains("linkcell    = { version = \"=0.3.11\", default-features = false }"));
    assert!(manifest.contains("eindir-core = { version = \"0.6.0\", features = [\"capi\"] }"));
    assert!(manifest.contains("rgmin = \"0.5.0\""));
    assert!(manifest.contains("rgsaddle = { version = \"0.3.0\", optional = true }"));
    assert!(manifest.contains("rgsaddle = [\"dep:rgsaddle\"]"));
    assert!(manifest.contains("default = [\"rgsaddle\"]"));
    assert!(manifest.contains("python = [\"dep:pyo3\", \"dep:numpy\", \"dlpk/pyo3\", \"eindir-core/py-objective\", \"rgsaddle\"]"));
    assert!(manifest.contains("bank-rpc = [\"dep:capnp\", \"dep:capnp-rpc\", \"dep:futures\", \"dep:nng\", \"dep:tokio\", \"dep:tokio-util\", \"dep:libc\", \"rgsaddle\"]"));
    assert!(manifest.contains("exclude = [\".cargo/config.toml\"]"));
    assert!(workflow.contains("cargo publish --dry-run --no-verify"));
    assert!(workflow.contains("cargo clippy --all-targets -- -D warnings"));
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
    assert!(workflow.contains("5ac39b6e57a8f00c717f08c83f93ee4746d6e471"));
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

/// cargo-c joins `to` with the asset path relative to the asset's parent
/// directory, so `to = "dlpack"` installs `include/dlpack/dlpack.h`.
#[test]
fn c_abi_installs_the_dlpack_header_the_generated_include_names() {
    let manifest = include_str!("../Cargo.toml");
    let header = include_str!("../include/dlpack/dlpack.h");

    assert!(manifest.contains("{ from = \"include/dlpack/dlpack.h\", to = \"dlpack\" }"));
    assert!(header.contains("#define DLPACK_MAJOR_VERSION 1"));
    assert!(header.contains("#define DLPACK_MINOR_VERSION 3"));
    assert!(header.contains("typedef struct DLManagedTensorVersioned"));
}
