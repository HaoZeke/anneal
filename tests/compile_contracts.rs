//! Compile-time checks for the certified component constructor, and for a
//! downstream `match` on `HopStop` that has no wildcard arm.
//!
//! The fixtures must fail, and the failure must be the sealed-trait or
//! non-exhaustive error. Rustc 1.99 shortens the primary bound to
//! `Gaussian: CertifiedMoveFor<f64, _>` and puts the full types on the
//! help line, so the check reads that line.

use std::path::Path;
use std::process::Command;
use std::sync::OnceLock;

#[test]
fn checked_rejects_uncertified_component_combinations() {
    let facts = probe_facts();
    expect_compile_fail(
        &facts,
        "tests/ui/checked_rejects_downstream_self_certification.rs",
        &[
            "error[E0277]",
            "the trait bound `AlwaysReject: variant::sealed::Acceptance` is not satisfied",
        ],
    );
    expect_compile_fail(
        &facts,
        "tests/ui/checked_rejects_unbounded_box_move.rs",
        &[
            "error[E0277]",
            "the trait `CertifiedMoveFor<_, BoxConstrained<f64>>` is not implemented for `anneal_core::movekernel::Gaussian`",
        ],
    );
    expect_compile_fail(
        &facts,
        "tests/ui/checked_rejects_uncertified_accept.rs",
        &[
            "error[E0277]",
            "the trait bound `AlwaysReject: CertifiedAcceptance<f64>` is not satisfied",
        ],
    );
    expect_compile_fail(
        &facts,
        "tests/ui/hop_stop_needs_a_wildcard_arm.rs",
        &["error[E0004]", "non-exhaustive patterns", "HopStop"],
    );
}

struct ProbeFacts {
    edition: String,
    version: String,
    eindir_req: String,
    eindir_features: Vec<String>,
}

/// Edition and the eindir-core requirement come from `cargo metadata`.
/// The probe package version is this package's version.
fn probe_facts() -> ProbeFacts {
    let cargo = std::env::var("CARGO").unwrap_or_else(|_| "cargo".to_string());
    let output = Command::new(cargo)
        .args([
            "metadata",
            "--offline",
            "--format-version",
            "1",
            "--no-deps",
        ])
        .current_dir(env!("CARGO_MANIFEST_DIR"))
        .output()
        .expect("cargo metadata");
    assert!(
        output.status.success(),
        "cargo metadata failed\n{}",
        String::from_utf8_lossy(&output.stderr)
    );
    let metadata = String::from_utf8(output.stdout).expect("metadata utf-8");
    let edition = json_string_field(&metadata, "edition").expect("package edition");
    let (eindir_req, eindir_features) = dependency(&metadata, "eindir-core");
    ProbeFacts {
        edition,
        version: env!("CARGO_PKG_VERSION").to_string(),
        eindir_req,
        eindir_features,
    }
}

fn dependency(metadata: &str, name: &str) -> (String, Vec<String>) {
    let marker = format!("\"name\":\"{name}\"");
    let start = metadata
        .find(&marker)
        .unwrap_or_else(|| panic!("missing {name}"));
    let rest = &metadata[start..];
    let end = rest.find("},{\"name\"").unwrap_or(rest.len());
    let object = &rest[..end];
    let req = json_string_field(object, "req").unwrap_or_else(|| panic!("missing {name} req"));
    let features = json_string_array(object, "features");
    (req, features)
}

fn json_string_field(text: &str, key: &str) -> Option<String> {
    let marker = format!("\"{key}\":\"");
    let start = text.find(&marker)? + marker.len();
    let rest = &text[start..];
    let end = rest.find('"')?;
    Some(rest[..end].to_string())
}

fn json_string_array(text: &str, key: &str) -> Vec<String> {
    let marker = format!("\"{key}\":[");
    let Some(start) = text.find(&marker) else {
        return Vec::new();
    };
    let rest = &text[start + marker.len()..];
    let Some(end) = rest.find(']') else {
        return Vec::new();
    };
    rest[..end]
        .split(',')
        .map(|item| item.trim().trim_matches('"').to_string())
        .filter(|item| !item.is_empty())
        .collect()
}

/// Compile one UI fixture as a downstream bin and require each fragment.
fn expect_compile_fail(facts: &ProbeFacts, relative: &str, fragments: &[&str]) {
    let manifest_dir = Path::new(env!("CARGO_MANIFEST_DIR"));
    let root = tempfile::tempdir().expect("probe directory");
    let src = root.path().join("src");
    std::fs::create_dir_all(&src).expect("probe src");
    std::fs::copy(manifest_dir.join(relative), src.join("main.rs")).expect("copy fixture");
    let features = facts
        .eindir_features
        .iter()
        .map(|feature| format!("\"{feature}\""))
        .collect::<Vec<_>>()
        .join(", ");
    let manifest = format!(
        "[package]\nname = \"anneal-ui-probe\"\nversion = \"{}\"\nedition = \"{}\"\npublish = false\n\n[dependencies]\nanneal-core = {{ path = \"{}\" }}\neindir-core = {{ version = \"{}\", features = [{features}] }}\n",
        facts.version,
        facts.edition,
        manifest_dir.display(),
        facts.eindir_req,
    );
    std::fs::write(root.path().join("Cargo.toml"), manifest).expect("probe manifest");
    let cargo = std::env::var("CARGO").unwrap_or_else(|_| "cargo".to_string());
    let output = Command::new(cargo)
        .arg("check")
        .arg("--offline")
        .arg("--manifest-path")
        .arg(root.path().join("Cargo.toml"))
        .env("CARGO_TARGET_DIR", probe_target())
        .env("CARGO_TERM_COLOR", "never")
        .output()
        .expect("cargo check");
    let text = format!(
        "{}\n{}",
        String::from_utf8_lossy(&output.stdout),
        String::from_utf8_lossy(&output.stderr)
    );
    assert!(
        !output.status.success(),
        "{relative} compiled, but the contract says it must fail\n{text}"
    );
    for fragment in fragments {
        assert!(
            text.contains(fragment),
            "{relative} failed without {fragment}\n{text}"
        );
    }
}

fn probe_target() -> &'static Path {
    static TARGET: OnceLock<tempfile::TempDir> = OnceLock::new();
    TARGET
        .get_or_init(|| tempfile::tempdir().expect("probe target"))
        .path()
}
