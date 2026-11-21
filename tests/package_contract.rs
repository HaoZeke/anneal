//! Packaging contract: keep the published crate small and free of paper data.

use std::fs;
use std::path::PathBuf;
use std::process::Command;

fn temp_target(tag: &str) -> PathBuf {
    let mut dir = std::env::temp_dir();
    dir.push(format!("anneal-package-contract-{tag}-{}", std::process::id()));
    fs::create_dir_all(&dir).expect("create temp target dir");
    dir
}

fn cargo_bin() -> String {
    std::env::var("CARGO").unwrap_or_else(|_| "cargo".to_string())
}

#[test]
fn package_list_excludes_paper_and_bench() {
    let target = temp_target("list");
    let output = Command::new(cargo_bin())
        .arg("package")
        .arg("--allow-dirty")
        .arg("--list")
        .env("CARGO_TARGET_DIR", &target)
        .current_dir(env!("CARGO_MANIFEST_DIR"))
        .output()
        .expect("cargo package --list");
    assert!(
        output.status.success(),
        "cargo package --list failed:\n{}",
        String::from_utf8_lossy(&output.stderr)
    );
    let listing = String::from_utf8(output.stdout).expect("utf-8 package list");
    let blocked: Vec<&str> = listing
        .lines()
        .filter(|line| line.starts_with("paper/") || line.starts_with("bench/"))
        .collect();
    assert!(blocked.is_empty(), "unexpected packaged paths: {blocked:?}");
}

#[test]
fn packaged_crate_is_under_ten_mebibytes() {
    let target = temp_target("size");
    let status = Command::new(cargo_bin())
        .arg("package")
        .arg("--allow-dirty")
        .arg("--no-verify")
        .env("CARGO_TARGET_DIR", &target)
        .current_dir(env!("CARGO_MANIFEST_DIR"))
        .status()
        .expect("cargo package --no-verify");
    assert!(status.success(), "cargo package --no-verify failed");

    let crate_name = format!("{}-{}.crate", env!("CARGO_PKG_NAME"), env!("CARGO_PKG_VERSION"));
    let crate_path = target.join("package").join(crate_name);
    let size = fs::metadata(&crate_path)
        .unwrap_or_else(|err| panic!("missing {}: {err}", crate_path.display()))
        .len();
    assert!(
        size <= 10 * 1024 * 1024,
        "crate is {size} bytes, above 10 MiB: {}",
        crate_path.display()
    );
}

#[test]
fn cargo_package_with_verification_succeeds() {
    let target = temp_target("verify");
    let status = Command::new(cargo_bin())
        .arg("package")
        .arg("--allow-dirty")
        .env("CARGO_TARGET_DIR", &target)
        .current_dir(env!("CARGO_MANIFEST_DIR"))
        .status()
        .expect("cargo package");
    assert!(
        status.success(),
        "cargo package (with verification) must succeed"
    );
}
