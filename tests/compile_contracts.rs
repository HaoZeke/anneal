//! Compile-time checks for the certified component constructor, and for a
//! downstream `match` on `HopStop` that has no wildcard arm.
//!
//! The fixtures must fail, and the failure must be the sealed-trait or
//! non-exhaustive error. Rustc's help text for "other types implement this
//! trait" changed layout after 1.88, so the check reads that error instead
//! of a snapshot of the help text.

use std::path::Path;
use std::process::Command;

#[test]
fn checked_rejects_uncertified_component_combinations() {
    expect_compile_fail(
        "tests/ui/checked_rejects_downstream_self_certification.rs",
        &[
            "error[E0277]",
            "the trait bound `AlwaysReject: variant::sealed::Acceptance` is not satisfied",
        ],
    );
    expect_compile_fail(
        "tests/ui/checked_rejects_unbounded_box_move.rs",
        &[
            "error[E0277]",
            "the trait bound `anneal_core::movekernel::Gaussian: CertifiedMoveFor<f64, BoxConstrained<f64>>` is not satisfied",
        ],
    );
    expect_compile_fail(
        "tests/ui/checked_rejects_uncertified_accept.rs",
        &[
            "error[E0277]",
            "the trait bound `AlwaysReject: CertifiedAcceptance<f64>` is not satisfied",
        ],
    );
    expect_compile_fail(
        "tests/ui/hop_stop_needs_a_wildcard_arm.rs",
        &[
            "error[E0004]",
            "non-exhaustive patterns",
            "HopStop",
        ],
    );
}

/// Compile one UI fixture as a downstream bin and require each fragment.
fn expect_compile_fail(relative: &str, fragments: &[&str]) {
    let manifest_dir = Path::new(env!("CARGO_MANIFEST_DIR"));
    let root = tempfile::tempdir().expect("probe directory");
    let src = root.path().join("src");
    std::fs::create_dir_all(&src).expect("probe src");
    std::fs::copy(manifest_dir.join(relative), src.join("main.rs")).expect("copy fixture");
    let manifest = format!(
        "[package]\nname = \"anneal-ui-probe\"\nversion = \"0.0.0\"\nedition = \"2024\"\npublish = false\n\n[dependencies]\nanneal-core = {{ path = \"{}\" }}\neindir-core = {{ version = \"0.6.0\", features = [\"capi\"] }}\n",
        manifest_dir.display()
    );
    std::fs::write(root.path().join("Cargo.toml"), manifest).expect("probe manifest");
    let cargo = std::env::var("CARGO").unwrap_or_else(|_| "cargo".to_string());
    let target = std::env::temp_dir().join("anneal-compile-contracts-target");
    std::fs::create_dir_all(&target).expect("shared probe target");
    let output = Command::new(cargo)
        .arg("check")
        .arg("--offline")
        .arg("--manifest-path")
        .arg(root.path().join("Cargo.toml"))
        .env("CARGO_TARGET_DIR", &target)
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
