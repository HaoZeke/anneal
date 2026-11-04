#[test]
fn crate_publication_keeps_cargo_verification_enabled() {
    let workflow = include_str!("../.github/workflows/release.yml");

    assert!(workflow.contains("cargo publish"));
    assert!(!workflow.contains("--no-verify"));
}

#[test]
fn crate_publication_has_a_registry_fallback_for_git_dependencies() {
    fn strings(value: &toml::Value) -> Vec<&str> {
        value
            .as_array()
            .expect("manifest string array")
            .iter()
            .map(|entry| entry.as_str().expect("manifest string entry"))
            .collect()
    }

    let manifest: toml::Value =
        toml::from_str(include_str!("../Cargo.toml")).expect("Cargo.toml parses");
    let dependencies = manifest
        .get("dependencies")
        .and_then(toml::Value::as_table)
        .expect("[dependencies]");
    let workflow = include_str!("../.github/workflows/ci.yml");

    for (name, version, git) in [
        ("linkcell", "=0.3.11", "https://github.com/d-SEAMS/linkcell"),
        ("rgmin", "0.5.2", "https://github.com/OmniPotentRPC/rgmin"),
        (
            "rgsaddle",
            "0.3.2",
            "https://github.com/OmniPotentRPC/rgsaddle",
        ),
        (
            "readcon-db",
            "0.2.0",
            "https://github.com/lode-org/readcon-db",
        ),
    ] {
        let dependency = dependencies
            .get(name)
            .and_then(toml::Value::as_table)
            .unwrap_or_else(|| panic!("missing dependency table {name}"));
        assert_eq!(
            dependency.get("version").and_then(toml::Value::as_str),
            Some(version),
            "{name} registry fallback"
        );
        assert_eq!(
            dependency.get("git").and_then(toml::Value::as_str),
            Some(git),
            "{name} source"
        );
    }

    for section in ["dependencies", "build-dependencies", "dev-dependencies"] {
        if let Some(dependencies) = manifest.get(section).and_then(toml::Value::as_table) {
            for (name, value) in dependencies {
                let Some(dependency) = value.as_table() else {
                    continue;
                };
                if dependency.contains_key("git") {
                    let version = dependency
                        .get("version")
                        .and_then(toml::Value::as_str)
                        .unwrap_or_else(|| panic!("{section}.{name} lacks a registry fallback"));
                    assert!(
                        !version.trim().is_empty(),
                        "{section}.{name} empty fallback"
                    );
                    let rev = dependency
                        .get("rev")
                        .and_then(toml::Value::as_str)
                        .unwrap_or_else(|| panic!("{section}.{name} lacks a revision"));
                    assert!(
                        rev.len() == 40 && rev.bytes().all(|byte| byte.is_ascii_hexdigit()),
                        "{section}.{name} requires a full Git revision"
                    );
                }
            }
        }
    }

    assert_eq!(
        dependencies["rgpot-core"]["version"].as_str(),
        Some("=3.5.1")
    );
    assert_eq!(dependencies["rgpot-core"]["optional"].as_bool(), Some(true));
    assert_eq!(
        dependencies["linkcell"]["default-features"].as_bool(),
        Some(false)
    );
    assert_eq!(
        dependencies["eindir-core"]["version"].as_str(),
        Some("0.6.0")
    );
    assert_eq!(
        strings(&dependencies["eindir-core"]["features"]),
        vec!["capi"]
    );
    assert_eq!(dependencies["rgsaddle"]["optional"].as_bool(), Some(true));
    assert_eq!(
        strings(&manifest["features"]["rgsaddle"]),
        vec!["dep:rgsaddle"]
    );
    assert!(strings(&manifest["features"]["default"]).is_empty());
    assert_eq!(
        strings(&manifest["features"]["python"]),
        vec![
            "dep:pyo3",
            "dep:numpy",
            "dlpk/pyo3",
            "eindir-core/py-objective",
            "rgsaddle"
        ]
    );
    assert_eq!(
        strings(&manifest["features"]["bank-rpc"]),
        vec![
            "dep:capnp",
            "dep:capnp-rpc",
            "dep:futures",
            "dep:nng",
            "dep:tokio",
            "dep:tokio-util",
            "dep:libc",
            "rgsaddle"
        ]
    );
    assert_eq!(
        strings(&manifest["package"]["exclude"]),
        vec![".cargo/config.toml"]
    );
    // `cargo package` writes anneal-core-${version}.crate. The version
    // comes from cargo metadata.
    assert!(workflow.contains("cargo package"));
    assert!(!workflow.contains("cargo package --no-verify"));
    assert!(workflow.contains("anneal-core-${version}.crate"));
    assert!(workflow.contains("pkg[\"version\"]"));
    assert!(workflow.contains("test -f \"$crate\""));
    assert!(!workflow.contains("anneal-core-*.crate"));
    assert!(!workflow.contains("tail -1"));
    assert!(!workflow.contains("cargo publish --dry-run --no-verify"));
    assert!(workflow.contains("cargo clippy --all-targets -- -D warnings"));
    assert!(workflow.contains("rm -rf \"$pkg/.cargo\""));
    assert!(workflow.contains("cargo check --locked"));
    assert!(workflow.contains("cargo check --locked --features rgsaddle,python"));
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
