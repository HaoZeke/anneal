use serde_yaml_ng::Value;
use std::collections::BTreeSet;
use std::process::Command;

/// The release workflow, parsed. Every release contract below reads this
/// tree, so a step that moves, gains an `if:` or swallows its exit status
/// is seen as that and not as a string that happens to remain.
fn release() -> Value {
    serde_yaml_ng::from_str(include_str!("../.github/workflows/release.yml"))
        .expect("release.yml parses as YAML")
}

fn job<'a>(workflow: &'a Value, name: &str) -> &'a Value {
    workflow["jobs"]
        .get(name)
        .unwrap_or_else(|| panic!("release.yml has no job {name}"))
}

fn steps(job: &Value) -> &[Value] {
    job["steps"].as_sequence().expect("job steps")
}

/// The one step called `name` and its position in the job.
fn step<'a>(job: &'a Value, name: &str) -> (usize, &'a Value) {
    let found: Vec<_> = steps(job)
        .iter()
        .enumerate()
        .filter(|(_, step)| step["name"].as_str() == Some(name))
        .collect();
    assert_eq!(found.len(), 1, "exactly one step named {name:?}");
    found[0]
}

/// The position of the first step that runs the action `prefix`.
fn first_use(job: &Value, prefix: &str) -> usize {
    steps(job)
        .iter()
        .position(|step| {
            step["uses"]
                .as_str()
                .is_some_and(|uses| uses.starts_with(prefix))
        })
        .unwrap_or_else(|| panic!("no step uses {prefix}"))
}

/// The non-blank lines of a step's script, trimmed.
fn script(step: &Value) -> Vec<&str> {
    step["run"]
        .as_str()
        .unwrap_or_else(|| panic!("step {:?} has no run", step["name"]))
        .lines()
        .map(str::trim)
        .filter(|line| !line.is_empty())
        .collect()
}

/// A step that cannot be skipped, and whose failure fails the job.
fn assert_binding(step: &Value) {
    let name = &step["name"];
    for key in ["if", "continue-on-error"] {
        assert!(step.get(key).is_none(), "step {name:?} sets {key}");
    }
}

/// A job whose failure fails every job that needs it.
fn assert_blocking(workflow: &Value, name: &str) {
    for key in ["if", "continue-on-error"] {
        assert!(
            job(workflow, name).get(key).is_none(),
            "job {name} sets {key}"
        );
    }
}

fn needs(job: &Value) -> BTreeSet<&str> {
    match &job["needs"] {
        Value::String(one) => BTreeSet::from([one.as_str()]),
        Value::Sequence(many) => many.iter().filter_map(Value::as_str).collect(),
        other => panic!("needs is {other:?}"),
    }
}

/// Every line of every script in the workflow, with its job.
fn every_script_line(workflow: &Value) -> Vec<(String, String)> {
    let mut lines = Vec::new();
    for (name, job) in workflow["jobs"].as_mapping().expect("jobs") {
        for step in steps(job) {
            if step["run"].is_string() {
                for line in script(step) {
                    lines.push((name.as_str().unwrap().to_owned(), line.to_owned()));
                }
            }
        }
    }
    lines
}

#[test]
fn crate_publication_keeps_cargo_verification_enabled() {
    let workflow = release();
    let lines = every_script_line(&workflow);

    assert!(
        lines
            .iter()
            .any(|(_, line)| line.starts_with("cargo publish"))
    );
    for (name, line) in lines {
        assert!(!line.contains("--no-verify"), "job {name}: {line}");
    }
}

#[test]
fn crate_publication_publishes_anneal_core_alone_at_the_canonical_version() {
    let workflow = release();
    let publish = job(&workflow, "publish-crate");
    let (_, upload) = step(publish, "cargo publish");
    let lines = script(upload);

    // One cargo publish in the job, and it is the last command of the step,
    // so nothing before it can stand in for it or add a second package.
    let publishes: Vec<_> = steps(publish)
        .iter()
        .filter(|step| step["run"].is_string())
        .flat_map(script)
        .filter(|line| line.contains("cargo publish"))
        .collect();
    assert_eq!(publishes, ["cargo publish --locked -p anneal-core"]);
    assert_eq!(lines.last().copied(), publishes.first().copied());

    // The probe asks crates.io for the same crate it then publishes.
    let probe = lines
        .iter()
        .position(|line| line.contains("curl"))
        .expect("crates.io probe");
    assert!(
        lines[probe].contains("https://crates.io/api/v1/crates/anneal-core/${ver}"),
        "{}",
        lines[probe]
    );
    assert!(probe < lines.len() - 1, "probe runs before the publish");

    // The version both use is what the repository says, not what the text
    // of the command looks like.
    let ver = lines
        .iter()
        .find(|line| line.starts_with("ver=$("))
        .expect("ver assignment");
    assert!(
        ver.contains("anneal_version.package_version()"),
        "the version comes from the canonical module: {ver}"
    );
    let output = Command::new("bash")
        .arg("-c")
        .arg(format!("set -euo pipefail; {ver}; printf %s \"$ver\""))
        .current_dir(env!("CARGO_MANIFEST_DIR"))
        .output()
        .expect("bash runs the version command");
    assert!(
        output.status.success(),
        "{}",
        String::from_utf8_lossy(&output.stderr)
    );
    assert_eq!(
        String::from_utf8_lossy(&output.stdout),
        env!("CARGO_PKG_VERSION")
    );
}

#[test]
fn release_tags_match_every_public_package_version() {
    let workflow = release();
    let (_, validate) = step(
        job(&workflow, "validate-release"),
        "Validate release metadata",
    );
    let lines = script(validate);

    assert_binding(validate);
    assert!(lines.contains(&"set -euo pipefail"));
    for check in [
        r#"test "$cargo_version" = "$python_version""#,
        r#"test "$cargo_version" = "$citation_version""#,
        r#"test "${GITHUB_REF_NAME#v}" = "$cargo_version""#,
    ] {
        assert!(lines.contains(&check), "validation lacks {check}");
    }
    assert!(lines.iter().any(|line| line.starts_with("cargo_version=")
        && line.contains("anneal_version.package_version()")));
    assert!(
        lines
            .iter()
            .any(|line| line.starts_with("python_version=") && line.contains("pyproject.toml"))
    );
    assert!(
        lines
            .iter()
            .any(|line| line.starts_with("citation_version=") && line.contains("CITATION.cff"))
    );
    assert_blocking(&workflow, "validate-release");
    for dependent in ["native-libs", "sdist", "wheels"] {
        assert!(needs(job(&workflow, dependent)).contains("validate-release"));
    }
}

#[test]
fn crate_publication_has_a_registry_fallback_for_git_dependencies() {
    let manifest = include_str!("../Cargo.toml");
    let workflow = include_str!("../.github/workflows/ci.yml");

    assert!(manifest.contains("rgpot-core = { version = \"=3.5.1\" }"));
    assert!(manifest.contains("linkcell    = { version = \"=0.3.11\", default-features = false }"));
    assert!(manifest.contains("eindir-core = { version = \"0.6.0\", features = [\"capi\"] }"));
    assert!(manifest.contains("rgmin = \"0.5.0\""));
    assert!(manifest.contains("rgsaddle = { version = \"0.3.0\", optional = true }"));
    assert!(manifest.contains("rgsaddle = [\"dep:rgsaddle\"]"));
    assert!(manifest.contains("default = []"));
    assert!(manifest.contains("python = [\"dep:pyo3\", \"dep:numpy\", \"dlpk/pyo3\", \"eindir-core/py-objective\", \"rgsaddle\"]"));
    assert!(manifest.contains("bank-rpc = [\"dep:capnp\", \"dep:capnp-rpc\", \"dep:futures\", \"dep:nng\", \"dep:tokio\", \"dep:tokio-util\", \"dep:libc\", \"rgsaddle\"]"));
    assert!(manifest.contains("exclude = [\".cargo/config.toml\"]"));
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
    assert!(workflow.contains("cargo clippy --all-targets --features rgpot-ex -- -D warnings"));
    assert!(workflow.contains("cargo clippy --all-targets --all-features -- -D warnings"));
    assert!(workflow.contains("rm -rf \"$pkg/.cargo\""));
    assert!(workflow.contains("cargo check --locked"));
    assert!(workflow.contains("cargo check --locked --features rgsaddle,python"));
}

#[test]
fn native_release_builds_where_artifact_packaging_reads() {
    let workflow = release();
    let native = job(&workflow, "native-libs");
    let (build_at, build) = step(native, "Build static + shared library");
    let (package_at, package) = step(native, "Package artifacts");
    let upload_at = first_use(native, "actions/upload-artifact");

    assert_binding(build);
    assert_binding(package);
    assert!(build_at < package_at && package_at < upload_at);
    assert_blocking(&workflow, "native-libs");

    let build = script(build);
    assert_eq!(build.len(), 1);
    let compact = build[0].replace("${{ ", "${{").replace(" }}", "}}");
    let words: Vec<&str> = compact.split_whitespace().collect();
    assert_eq!(&words[..3], ["cargo", "build", "--release"]);
    let value_of = |flag: &str| {
        let at = words
            .iter()
            .position(|word| *word == flag)
            .unwrap_or_else(|| panic!("{flag}"));
        words[at + 1]
    };
    assert_eq!(value_of("--features"), "capi");
    assert_eq!(value_of("--target"), "${{matrix.target}}");
    let target_dir = value_of("--target-dir");

    // The packager copies from the directory the build wrote to.
    let package = script(package);
    for library in ["static_lib", "shared_lib"] {
        let from = format!(
            "cp {target_dir}/${{{{ matrix.target }}}}/release/${{{{ matrix.{library} }}}} staging/lib/"
        );
        assert!(package.contains(&from.as_str()), "packaging lacks {from}");
    }
}

#[test]
fn release_installs_and_imports_the_built_sdist() {
    let workflow = release();
    let sdist = job(&workflow, "sdist");
    let (built_at, _) = step(sdist, "Build sdist");
    let (install_at, install) = step(sdist, "Install and import sdist");
    let upload_at = first_use(sdist, "actions/upload-artifact");

    assert_binding(install);
    assert_blocking(&workflow, "sdist");
    assert!(
        built_at < install_at,
        "the sdist is built before it is installed"
    );
    assert!(
        install_at < upload_at,
        "the sdist is installed before it is uploaded"
    );

    assert_eq!(
        script(install),
        [
            "python -m venv .sdist-venv",
            ".sdist-venv/bin/python -m pip install dist/anneal-*.tar.gz",
            ".sdist-venv/bin/python scripts/check_installed_anneal.py .sdist-venv",
        ]
    );
}

/// The guard the sdist step runs, against stand-ins for an installed package
/// and for the places an import could wrongly come from.
#[test]
fn the_sdist_guard_accepts_only_a_package_installed_in_the_venv() {
    let temp = tempfile::tempdir().expect("tempdir");
    let venv = temp.path().join(".sdist-venv");
    let site = venv.join("lib/python3/site-packages");
    let loose = venv.join("lib/python3/loose");
    let source = temp.path().join("checkout/python");
    for dir in [&site, &loose, &source] {
        let package = dir.join("anneal");
        std::fs::create_dir_all(&package).expect("package dir");
        std::fs::write(package.join("__init__.py"), "__version__ = '9.9.9'\n").expect("init");
        std::fs::write(package.join("_core.py"), "").expect("core");
    }
    let guard = |from: &std::path::Path| {
        Command::new("python3")
            .args(["scripts/check_installed_anneal.py"])
            .arg(&venv)
            .env("PYTHONPATH", from)
            .env_remove("PYTHONHOME")
            .current_dir(env!("CARGO_MANIFEST_DIR"))
            .output()
            .expect("python3 runs the guard")
    };

    let installed = guard(&site);
    assert!(
        installed.status.success(),
        "{}",
        String::from_utf8_lossy(&installed.stderr)
    );
    assert_eq!(String::from_utf8_lossy(&installed.stdout).trim(), "9.9.9");
    for (name, from) in [
        ("a source tree", &source),
        ("a venv path outside site-packages", &loose),
    ] {
        let output = guard(from);
        assert!(!output.status.success(), "the guard accepted {name}");
        assert!(String::from_utf8_lossy(&output.stderr).contains("was imported from"));
    }
}

#[test]
fn release_validates_every_built_wheel() {
    let workflow = release();
    let wheels = job(&workflow, "wheels");
    let (built_at, _) = step(wheels, "Build wheels");
    let (validate_at, validate) = step(wheels, "Validate wheels");
    let upload_at = first_use(wheels, "actions/upload-artifact");

    assert_binding(validate);
    assert_blocking(&workflow, "wheels");
    assert!(built_at < validate_at && validate_at < upload_at);
    assert_eq!(script(validate), ["pipx run twine check --strict dist/*"]);
}

#[test]
fn the_wheel_matrix_covers_every_supported_platform() {
    let workflow = release();
    let cells: BTreeSet<(&str, &str)> = job(&workflow, "wheels")["strategy"]["matrix"]["include"]
        .as_sequence()
        .expect("wheel matrix")
        .iter()
        .map(|cell| {
            (
                cell["os"].as_str().unwrap(),
                cell["target"].as_str().unwrap(),
            )
        })
        .collect();

    assert_eq!(
        cells,
        BTreeSet::from([
            ("macos-15", "aarch64"),
            ("macos-15", "x86_64"),
            ("ubuntu-22.04", "aarch64"),
            ("ubuntu-latest", "x86_64"),
        ])
    );
}

#[test]
fn every_publish_job_waits_for_the_artifacts_it_publishes() {
    let workflow = release();

    assert!(
        needs(job(&workflow, "publish-pypi")).is_superset(&BTreeSet::from(["sdist", "wheels"]))
    );
    // A crates.io upload cannot be undone, so it needs everything PyPI and
    // the GitHub release need and cannot happen while either is impossible.
    for irreversible in ["publish-crate", "github-release"] {
        assert!(
            needs(job(&workflow, irreversible)).is_superset(&BTreeSet::from([
                "native-libs",
                "sdist",
                "wheels"
            ])),
            "{irreversible} does not wait for the native libraries, the sdist and every wheel"
        );
    }
    for publish in ["publish-pypi", "publish-crate", "github-release"] {
        assert_eq!(
            job(&workflow, publish)["if"].as_str(),
            Some("startsWith(github.ref, 'refs/tags/v')"),
            "{publish} runs on tags only"
        );
    }
}

#[test]
fn scientific_tests_optimize_kernels_without_disabling_debug_checks() {
    let manifest = include_str!("../Cargo.toml");

    assert!(manifest.contains("[profile.test]"));
    assert!(manifest.contains("opt-level = 2"));
    assert!(manifest.contains("debug-assertions = true"));
    assert!(manifest.contains("overflow-checks = true"));
}

/// The text of one top-level job in a workflow, up to the next job.
fn workflow_job<'a>(workflow: &'a str, name: &str) -> &'a str {
    let start = workflow
        .find(&format!("\n  {name}:\n"))
        .unwrap_or_else(|| panic!("job {name}"))
        + 1;
    let rest = &workflow[start..];
    let end = rest
        .match_indices("\n  ")
        .map(|(offset, _)| offset)
        .find(|&offset| {
            let line = rest[offset + 3..].lines().next().unwrap_or_default();
            !line.starts_with(' ') && !line.starts_with('#') && line.ends_with(':')
        })
        .unwrap_or(rest.len());
    &rest[..end]
}

// The default features run once per OS because no matrix Python reaches
// them. The satisfiable features run on Linux for every change and on
// macOS when a native/Python build input changes. CI may trim debuginfo,
// never the optimisation or the checks of the test profile.
#[test]
fn ci_runs_default_everywhere_and_gates_macos_python() {
    let workflow = include_str!("../.github/workflows/ci.yml");
    let changes = workflow_job(workflow, "changes");
    let features = workflow_job(workflow, "test");
    let default = workflow_job(workflow, "test-default");
    assert!(default.contains("os: [ubuntu-latest, macos-15, macos-26]"));
    assert!(features.contains("needs: changes"));
    assert!(features.contains("os: ${{ fromJSON(needs.changes.outputs.test-os) }}"));
    for path in [
        "src/**",
        "python/**",
        "Cargo.*",
        "pyproject.toml",
        ".github/workflows/ci.yml",
    ] {
        assert!(changes.contains(path), "macOS path filter omits {path}");
    }
    assert!(changes.contains(r#"test-os=["ubuntu-latest","macos-15","macos-26"]"#));
    assert!(changes.contains(r#"test-os=["ubuntu-latest"]"#));
    assert!(features.contains("python-version: [\"3.10\", \"3.11\", \"3.12\", \"3.13\"]"));
    assert!(features.contains("cargo test --features \"$CI_FEATURES,lammps\""));
    assert!(features.contains("cargo test --features \"$CI_FEATURES\"\n"));
    assert!(default.contains("run: cargo test\n"));
    assert!(workflow.contains("CARGO_PROFILE_TEST_DEBUG: line-tables-only"));
    for knob in [
        "CARGO_PROFILE_TEST_OPT_LEVEL",
        "CARGO_PROFILE_TEST_DEBUG_ASSERTIONS",
        "CARGO_PROFILE_TEST_OVERFLOW_CHECKS",
        "--test-threads",
    ] {
        assert!(!workflow.contains(knob), "ci.yml sets {knob}");
    }
}

#[test]
fn ci_cancels_superseded_main_branch_runs() {
    let workflow = include_str!("../.github/workflows/ci.yml");

    assert!(workflow.contains("cancel-in-progress: true"));
}

#[test]
fn ci_lint_reports_every_check_after_a_red_fmt() {
    let workflow = include_str!("../.github/workflows/ci.yml");
    let start = workflow.find("\n  lint:\n").expect("lint job");
    let end = workflow[start + 1..]
        .find("\n  test:\n")
        .map(|offset| start + 1 + offset)
        .expect("test job after lint");
    let lint = &workflow[start..end];
    let steps: Vec<&str> = lint.split("      - name: ").skip(1).collect();
    let fmt = steps
        .iter()
        .position(|step| step.starts_with("cargo fmt\n"))
        .expect("fmt step");
    for step in &steps[fmt + 1..] {
        let name = step.lines().next().unwrap_or_default();
        assert!(
            step.contains("if: ${{ !cancelled()"),
            "lint step `{name}` is skipped after an earlier failure"
        );
    }
    for set in [
        "cargo clippy --all-targets -- -D warnings",
        "cargo clippy --all-targets --features rgpot-ex -- -D warnings",
        "cargo clippy --all-targets --features rgsaddle -- -D warnings",
        "cargo clippy --features \"$CI_FEATURES,lammps\" --all-targets -- -D warnings",
        "cargo clippy --all-targets --all-features -- -D warnings",
    ] {
        assert!(lint.contains(set), "lint runs {set}");
    }
    assert!(!lint.contains("always()"));
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

    assert!(workflow.contains("git rev-list --objects HEAD"));
    assert!(workflow.contains("git rev-list --objects \"$range\""));
    assert!(workflow.contains("github.event.pull_request.base.sha"));
    assert!(workflow.contains("github.event.pull_request.head.sha"));
    assert!(workflow.contains("$2 > 1048576"));
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

#[test]
fn rgpot_pair_stays_out_of_the_published_dependencies() {
    let manifest = include_str!("../Cargo.toml");
    let dependencies = manifest
        .split("\n[dependencies]\n")
        .nth(1)
        .and_then(|rest| rest.split("\n[").next())
        .expect("[dependencies] table");
    assert!(!dependencies.contains("git ="));
    assert!(manifest.contains("members = [\"crates/anneal-python\"]"));
    let pyproject = include_str!("../pyproject.toml");
    assert!(pyproject.contains("manifest-path = \"crates/anneal-python/Cargo.toml\""));
    let wheel = include_str!("../crates/anneal-python/src/lib.rs");
    assert!(wheel.contains("anneal_core::install_rgpot_kernels!();"));
}

#[test]
fn wheel_crate_depends_on_anneal_core_at_the_workspace_version() {
    let wheel: toml::Table = include_str!("../crates/anneal-python/Cargo.toml")
        .parse()
        .expect("wheel manifest parses");
    let root: toml::Table = include_str!("../Cargo.toml")
        .parse()
        .expect("root manifest parses");
    let workspace_version = root["workspace"]["package"]["version"]
        .as_str()
        .expect("version");
    let core = &wheel["dependencies"]["anneal-core"];

    assert_eq!(workspace_version, env!("CARGO_PKG_VERSION"));
    assert_eq!(core["version"].as_str(), Some(workspace_version));
    assert_eq!(core["path"].as_str(), Some("../.."));
}

// An example that skips the installation runs Lennard-Jones and Morse on the
// pair loop and cannot evaluate Gupta.
#[test]
fn every_example_installs_the_rgpot_kernels() {
    let dir = std::path::Path::new(env!("CARGO_MANIFEST_DIR")).join("examples");
    let mut mains = 0;
    for entry in std::fs::read_dir(dir).expect("examples") {
        let path = entry.expect("entry").path();
        if path.extension().is_none_or(|ext| ext != "rs") {
            continue;
        }
        let source = std::fs::read_to_string(&path).expect("example source");
        let lines: Vec<&str> = source.lines().collect();
        for (i, line) in lines.iter().enumerate() {
            if line.starts_with("fn main(") {
                mains += 1;
                assert_eq!(
                    lines.get(i + 1).map(|next| next.trim()),
                    Some("anneal_core::install_rgpot_kernels!();"),
                    "{} line {}",
                    path.display(),
                    i + 1
                );
            }
        }
    }
    assert!(mains > 0);
}

// pip builds the sdist from its own files. cargo resolves every path the
// root manifest names, including [patch] targets and optional crates that
// the wheel does not enable, and maturin packs neither.
#[test]
fn the_sdist_packs_every_vendored_path_the_manifest_names() {
    let manifest = include_str!("../Cargo.toml");
    let pyproject = include_str!("../pyproject.toml");
    let mut vendored = 0;
    for piece in manifest.split("path = \"vendor/").skip(1) {
        let name = piece.split('"').next().expect("closing quote");
        vendored += 1;
        let entry = format!("{{ path = \"vendor/{name}/**/*\", format = \"sdist\" }}");
        assert!(
            pyproject.contains(&entry),
            "pyproject.toml does not pack vendor/{name}"
        );
    }
    assert!(vendored > 0);
    let workflow = include_str!("../.github/workflows/ci.yml");
    assert!(workflow.contains("python -m maturin sdist --out dist"));
    assert!(workflow.contains("pip install dist/anneal-*.tar.gz pytest"));
}
