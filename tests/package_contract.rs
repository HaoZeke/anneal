//! Packaging contract: ship sources, headers, data, benches, examples, and
//! tests under 10 MiB without bulk paper or benchmark campaign artifacts.

use std::collections::HashSet;
use std::fs;
use std::io::Write;
use std::path::{Path, PathBuf};
use std::process::Command;
use std::sync::Mutex;

static HEAVY_PACKAGE_LOCK: Mutex<()> = Mutex::new(());

const REQUIRED_EXACT_FILES: &[&str] = &[
    ".github/workflows/ci.yml",
    ".github/workflows/release.yml",
    ".zenodo.json",
    "CHANGELOG.md",
    "CITATION.cff",
    "CMakeLists.txt",
    "Cargo.lock",
    "Cargo.toml",
    "LICENSE.txt",
    "README.md",
    "bench/lj104-lattice/v2/h2h_cap.f90",
    "build.rs",
    "cbindgen.toml",
    "docs/source/conf.py",
    "meson.build",
    "paper/seeds.toml",
    "pixi.toml",
    "pyproject.toml",
    "tests/package_contract.rs",
    "towncrier.toml",
];

const REQUIRED_PREFIXES: &[&str] = &[
    "benches/",
    "data/",
    "examples/",
    "examples/common/",
    "examples/fixtures/",
    "include/",
    "profiles/",
    "scripts/",
    "src/",
    "tests/",
    "tests/fixtures/",
    "tests/ui/",
];

fn cargo_bin() -> String {
    std::env::var("CARGO").unwrap_or_else(|_| "cargo".to_string())
}

fn manifest_dir() -> PathBuf {
    PathBuf::from(env!("CARGO_MANIFEST_DIR"))
}

fn is_unpacked_crate() -> bool {
    manifest_dir().join("Cargo.toml.orig").is_file()
}

fn skip_repackage_in_unpacked_crate(test_name: &str) -> bool {
    if is_unpacked_crate() {
        let msg = format!(
            "warning: skipped package_contract::{test_name}: already running inside unpacked .crate at {} (Cargo.toml.orig present)\n",
            manifest_dir().display()
        );
        let _ = std::io::stderr().write_all(msg.as_bytes());
        return true;
    }
    false
}

fn is_blocked_packaged_path(line: &str) -> bool {
    (line.starts_with("paper/") && line != "paper/seeds.toml")
        || (line.starts_with("bench/") && line != "bench/lj104-lattice/v2/h2h_cap.f90")
        || (line.starts_with("docs/") && line != "docs/source/conf.py")
        || line.starts_with("branding/")
        || line.starts_with("experiments/")
        || line.starts_with("vendor/")
        || line.split('/').any(|component| component == "__pycache__")
        || line.ends_with(".pyc")
}

fn assert_no_omitted_target_warnings(context: &str, stderr: &str) {
    assert!(
        !stderr.contains("not included in the published package"),
        "{context} emitted omitted-target warnings:\n{stderr}"
    );
}

struct TemporaryFile(PathBuf);

impl TemporaryFile {
    fn create(path: PathBuf) -> Self {
        fs::create_dir_all(path.parent().expect("temporary file parent"))
            .expect("create temporary file parent");
        fs::write(&path, b"package contract bytecode probe").expect("write temporary file");
        Self(path)
    }
}

impl Drop for TemporaryFile {
    fn drop(&mut self) {
        let _ = fs::remove_file(&self.0);
        if let Some(parent) = self.0.parent() {
            let _ = fs::remove_dir(parent);
        }
    }
}

fn extract_include_macro_paths(source: &str) -> Vec<&str> {
    let mut paths = Vec::new();
    for macro_prefix in ["include_str!(\"", "include_bytes!(\""] {
        let mut rest = source;
        while let Some(pos) = rest.find(macro_prefix) {
            let after = &rest[pos + macro_prefix.len()..];
            if let Some(end) = after.find('"') {
                paths.push(&after[..end]);
                rest = &after[end + 1..];
            } else {
                break;
            }
        }
    }
    paths
}

fn walk_relative_files(root: &Path, dir: &Path, out: &mut Vec<String>) {
    let Ok(entries) = fs::read_dir(dir) else {
        return;
    };
    for entry in entries.flatten() {
        let path = entry.path();
        if path.file_name().is_some_and(|name| name == "target") {
            continue;
        }
        if path.is_dir() {
            walk_relative_files(root, &path, out);
        } else if let Ok(rel) = path.strip_prefix(root) {
            out.push(rel.to_string_lossy().replace('\\', "/"));
        }
    }
}

fn assert_packaged_file_set(root: &Path, files: &[String]) {
    let blocked: Vec<&str> = files
        .iter()
        .map(String::as_str)
        .filter(|line| is_blocked_packaged_path(line))
        .collect();
    assert!(blocked.is_empty(), "unexpected packaged paths: {blocked:?}");

    let file_set: HashSet<&str> = files.iter().map(String::as_str).collect();
    for &exact in REQUIRED_EXACT_FILES {
        assert!(
            file_set.contains(exact),
            "packaged file set omitted required file {exact}"
        );
    }
    for &prefix in REQUIRED_PREFIXES {
        assert!(
            files.iter().any(|line| line.starts_with(prefix)),
            "packaged file set omitted required prefix {prefix}"
        );
    }

    let canonical_root = root.canonicalize().expect("canonicalize root");
    let mut missing_includes = Vec::new();
    for rel in files.iter().filter(|path| path.ends_with(".rs")) {
        let full = root.join(rel);
        let Ok(source) = fs::read_to_string(&full) else {
            continue;
        };
        let parent = full.parent().expect("parent dir");
        for target in extract_include_macro_paths(&source) {
            let joined = parent.join(target);
            let Ok(canonical) = joined.canonicalize() else {
                missing_includes.push(format!("{rel} -> {target} (missing on disk)"));
                continue;
            };
            let Ok(target_rel) = canonical.strip_prefix(&canonical_root) else {
                missing_includes.push(format!("{rel} -> {target} (escapes crate root)"));
                continue;
            };
            let target_rel = target_rel.to_string_lossy().replace('\\', "/");
            if !file_set.contains(target_rel.as_str()) {
                missing_includes.push(format!("{rel} -> {target_rel}"));
            }
        }
    }
    assert!(
        missing_includes.is_empty(),
        "packaged .rs files embed unshipped paths via include_str!/include_bytes!:\n{}",
        missing_includes.join("\n")
    );
}

#[test]
fn package_list_excludes_paper_and_bench() {
    let root = manifest_dir();
    if is_unpacked_crate() {
        let mut files = Vec::new();
        walk_relative_files(&root, &root, &mut files);
        assert_packaged_file_set(&root, &files);
        return;
    }

    let bytecode_probe = root
        .join("scripts")
        .join("__pycache__")
        .join(format!("package_contract_{}.pyc", std::process::id()));
    let _bytecode_probe = TemporaryFile::create(bytecode_probe);
    let target = tempfile::tempdir().expect("create temp target dir");
    let output = Command::new(cargo_bin())
        .arg("package")
        .arg("--allow-dirty")
        .arg("--list")
        .env("CARGO_TARGET_DIR", target.path())
        .current_dir(&root)
        .output()
        .expect("cargo package --list");
    let stderr = String::from_utf8_lossy(&output.stderr);
    assert!(
        output.status.success(),
        "cargo package --list failed:\n{stderr}"
    );
    assert_no_omitted_target_warnings("cargo package --list", &stderr);

    let listing = String::from_utf8(output.stdout).expect("utf-8 package list");
    let files: Vec<String> = listing.lines().map(str::to_owned).collect();
    assert_packaged_file_set(&root, &files);
}

#[test]
fn packaged_crate_is_under_ten_mebibytes() {
    if skip_repackage_in_unpacked_crate("packaged_crate_is_under_ten_mebibytes") {
        return;
    }

    let target = tempfile::tempdir().expect("create temp target dir");
    let output = Command::new(cargo_bin())
        .arg("package")
        .arg("--allow-dirty")
        .arg("--no-verify")
        .env("CARGO_TARGET_DIR", target.path())
        .current_dir(manifest_dir())
        .output()
        .expect("cargo package --no-verify");
    let stderr = String::from_utf8_lossy(&output.stderr);
    assert!(
        output.status.success(),
        "cargo package --no-verify failed:\n{stderr}"
    );
    assert_no_omitted_target_warnings("cargo package --no-verify", &stderr);

    let crate_name = format!(
        "{}-{}.crate",
        env!("CARGO_PKG_NAME"),
        env!("CARGO_PKG_VERSION")
    );
    let crate_path = target.path().join("package").join(crate_name);
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
    if skip_repackage_in_unpacked_crate("cargo_package_with_verification_succeeds") {
        return;
    }

    let _guard = HEAVY_PACKAGE_LOCK
        .lock()
        .unwrap_or_else(|poisoned| poisoned.into_inner());
    let target = tempfile::tempdir().expect("create temp target dir");
    let output = Command::new(cargo_bin())
        .arg("package")
        .arg("--allow-dirty")
        .env("CARGO_TARGET_DIR", target.path())
        .env("CARGO_PROFILE_TEST_DEBUG", "line-tables-only")
        .env("CARGO_INCREMENTAL", "0")
        .current_dir(manifest_dir())
        .output()
        .expect("cargo package");
    let stderr = String::from_utf8_lossy(&output.stderr);
    assert!(
        output.status.success(),
        "cargo package (with verification) failed:\n{stderr}"
    );
    assert_no_omitted_target_warnings("cargo package", &stderr);
}

#[test]
fn extracted_crate_runs_its_test_suite() {
    if skip_repackage_in_unpacked_crate("extracted_crate_runs_its_test_suite") {
        return;
    }

    let _guard = HEAVY_PACKAGE_LOCK
        .lock()
        .unwrap_or_else(|poisoned| poisoned.into_inner());
    let package_target = tempfile::tempdir().expect("create temp package target dir");
    let pkg_output = Command::new(cargo_bin())
        .arg("package")
        .arg("--allow-dirty")
        .arg("--no-verify")
        .env("CARGO_TARGET_DIR", package_target.path())
        .current_dir(manifest_dir())
        .output()
        .expect("cargo package --no-verify");
    let pkg_stderr = String::from_utf8_lossy(&pkg_output.stderr);
    assert!(
        pkg_output.status.success(),
        "cargo package --no-verify failed:\n{pkg_stderr}"
    );
    assert_no_omitted_target_warnings("cargo package --no-verify", &pkg_stderr);

    let stem = format!("{}-{}", env!("CARGO_PKG_NAME"), env!("CARGO_PKG_VERSION"));
    let crate_path = package_target
        .path()
        .join("package")
        .join(format!("{stem}.crate"));
    assert!(crate_path.is_file(), "missing {}", crate_path.display());

    let extract_dir = tempfile::tempdir().expect("create temp extract dir");
    let tar_status = Command::new("tar")
        .arg("-xzf")
        .arg(&crate_path)
        .arg("-C")
        .arg(extract_dir.path())
        .status()
        .expect("tar -xzf .crate");
    assert!(tar_status.success(), "tar failed to extract .crate");

    let unpacked = extract_dir.path().join(&stem);
    assert!(
        unpacked.join("tests/package_contract.rs").is_file(),
        "extracted .crate omitted tests/package_contract.rs"
    );
    let _ = fs::remove_dir_all(unpacked.join(".cargo"));

    let child_tmp = tempfile::tempdir().expect("create temp child tmpdir");
    let test_target = tempfile::tempdir().expect("create temp unpacked target dir");
    let check_output = Command::new(cargo_bin())
        .arg("check")
        .arg("--locked")
        .arg("--all-targets")
        .env("CARGO_TARGET_DIR", test_target.path())
        .env("CARGO_PROFILE_TEST_DEBUG", "line-tables-only")
        .env("CARGO_INCREMENTAL", "0")
        .env("TMPDIR", child_tmp.path())
        .current_dir(&unpacked)
        .output()
        .expect("cargo check --locked --all-targets in extracted .crate");
    assert!(
        check_output.status.success(),
        "extracted .crate failed `cargo check --locked --all-targets`:\nstdout:\n{}\nstderr:\n{}",
        String::from_utf8_lossy(&check_output.stdout),
        String::from_utf8_lossy(&check_output.stderr)
    );

    let lib_output = Command::new(cargo_bin())
        .arg("test")
        .arg("--locked")
        .arg("--lib")
        .arg("potentials::")
        .env("CARGO_TARGET_DIR", test_target.path())
        .env("CARGO_PROFILE_TEST_DEBUG", "line-tables-only")
        .env("CARGO_INCREMENTAL", "0")
        .current_dir(&unpacked)
        .output()
        .expect("cargo test --locked --lib potentials:: in extracted .crate");
    assert!(
        lib_output.status.success(),
        "extracted .crate lib test harness failed:\nstdout:\n{}\nstderr:\n{}",
        String::from_utf8_lossy(&lib_output.stdout),
        String::from_utf8_lossy(&lib_output.stderr)
    );

    // Linking all ~250 anneal-core test/example binaries at once takes ~20 GiB,
    // which would exhaust CI runner disk alongside the outer test build; check
    // all targets above and execute the contract, UI, and fixture suites here
    // (or the full `--all-targets` suite when ANNEAL_PACKAGE_CONTRACT_FULL=1).
    let mut cmd = Command::new(cargo_bin());
    cmd.arg("test").arg("--locked");
    if std::env::var_os("ANNEAL_PACKAGE_CONTRACT_FULL").is_some() {
        cmd.arg("--all-targets");
    } else {
        for suite in [
            "package_contract",
            "paper_seeds",
            "version_metadata",
            "release_workflow_contract",
            "elja_submission_contract",
            "search_hit_tol",
            "target_tol_lint",
            "lj_causal_runner",
            "compile_contracts",
            "gmin_tip4p_files",
            "catalog_packing",
            "surface_hop",
        ] {
            cmd.arg("--test").arg(suite);
        }
    }
    let test_output = cmd
        .env("CARGO_TARGET_DIR", test_target.path())
        .env("CARGO_PROFILE_TEST_DEBUG", "line-tables-only")
        .env("CARGO_INCREMENTAL", "0")
        .env("TMPDIR", child_tmp.path())
        .current_dir(&unpacked)
        .output()
        .expect("cargo test in extracted .crate");
    assert!(
        test_output.status.success(),
        "extracted .crate test suite failed:\nstdout:\n{}\nstderr:\n{}",
        String::from_utf8_lossy(&test_output.stdout),
        String::from_utf8_lossy(&test_output.stderr)
    );
}
