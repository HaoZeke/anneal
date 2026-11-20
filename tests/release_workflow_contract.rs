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

/// The CI workflow, parsed, read the same way as the release workflow.
fn ci() -> Value {
    serde_yaml_ng::from_str(include_str!("../.github/workflows/ci.yml"))
        .expect("ci.yml parses as YAML")
}

fn job<'a>(workflow: &'a Value, name: &str) -> &'a Value {
    workflow["jobs"]
        .get(name)
        .unwrap_or_else(|| panic!("the workflow has no job {name}"))
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

/// The step of a job that carries `id`.
fn step_with_id<'a>(job: &'a Value, id: &str) -> (usize, &'a Value) {
    let found: Vec<_> = steps(job)
        .iter()
        .enumerate()
        .filter(|(_, step)| step["id"].as_str() == Some(id))
        .collect();
    assert_eq!(found.len(), 1, "exactly one step with id {id:?}");
    found[0]
}

/// The lines of a step's script as the shell reads them: a line that ends
/// in a backslash is joined to the next.
fn logical_lines(step: &Value) -> Vec<String> {
    let mut lines = Vec::new();
    let mut pending = String::new();
    for line in script(step) {
        match line.strip_suffix('\\') {
            Some(head) => {
                pending.push_str(head.trim_end());
                pending.push(' ');
            }
            None => {
                pending.push_str(line);
                lines.push(std::mem::take(&mut pending));
            }
        }
    }
    assert!(pending.is_empty(), "script ends in a continuation");
    lines
}

/// The strings a key holds in a YAML sequence.
fn strings(value: &Value) -> Vec<&str> {
    value
        .as_sequence()
        .unwrap_or_else(|| panic!("{value:?} is not a sequence"))
        .iter()
        .map(|item| item.as_str().expect("string item"))
        .collect()
}

/// A step that runs after an earlier failure, and whose own failure fails
/// the job.
fn assert_runs_after_failures(step: &Value) {
    let name = &step["name"];
    let condition = step["if"]
        .as_str()
        .unwrap_or_else(|| panic!("step {name:?} has no condition"));
    assert!(
        condition.starts_with("${{ !cancelled()"),
        "step {name:?} is skipped after an earlier failure: {condition}"
    );
    assert!(
        !condition.contains("always()"),
        "step {name:?}: {condition}"
    );
    assert!(step.get("continue-on-error").is_none(), "step {name:?}");
}

/// Every `steps.<id>.outcome` a condition reads names a step that ran
/// before it.
fn assert_outcomes_come_from_earlier_steps(job: &Value) {
    for (at, step) in steps(job).iter().enumerate() {
        let Some(condition) = step["if"].as_str() else {
            continue;
        };
        for piece in condition.split("steps.").skip(1) {
            let id: String = piece
                .chars()
                .take_while(|c| c.is_ascii_alphanumeric() || *c == '-' || *c == '_')
                .collect();
            let earlier = steps(job)[..at]
                .iter()
                .any(|before| before["id"].as_str() == Some(id.as_str()));
            assert!(
                earlier,
                "step {:?} reads steps.{id}, which no earlier step defines",
                step["name"]
            );
        }
    }
}

/// Every `env` mapping in a workflow, at workflow, job and step level, with
/// where it sits.
fn every_env(workflow: &Value) -> Vec<(String, &serde_yaml_ng::Mapping)> {
    let mut found = Vec::new();
    if let Some(env) = workflow["env"].as_mapping() {
        found.push(("workflow".to_owned(), env));
    }
    for (name, job) in workflow["jobs"].as_mapping().expect("jobs") {
        let name = name.as_str().unwrap();
        if let Some(env) = job["env"].as_mapping() {
            found.push((format!("job {name}"), env));
        }
        for step in steps(job) {
            if let Some(env) = step["env"].as_mapping() {
                found.push((format!("job {name}, step {:?}", step["name"]), env));
            }
        }
    }
    found
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

/// A script line with the spaces inside `${{ }}` removed, so it splits into
/// shell words.
fn compact_expressions(line: &str) -> String {
    line.replace("${{ ", "${{").replace(" }}", "}}")
}

/// The archive entries the packaging step creates with `cp`, and the ones it
/// checks for with `tar tzf`.
fn packaged_and_checked(package: &Value) -> (BTreeSet<String>, BTreeSet<String>) {
    let lines: Vec<String> = script(package)
        .into_iter()
        .map(compact_expressions)
        .collect();
    let mut made_dirs = BTreeSet::new();
    let mut copied = BTreeSet::new();
    for line in &lines {
        let words: Vec<&str> = line.split_whitespace().collect();
        if words.starts_with(&["mkdir", "-p"]) {
            for dir in &words[2..] {
                // `mkdir -p` makes every parent as well.
                let dir = dir.trim_end_matches('/');
                for (at, _) in dir.match_indices('/') {
                    made_dirs.insert(&dir[..at]);
                }
                made_dirs.insert(dir);
            }
        } else if words.first() == Some(&"cp") {
            assert_eq!(words.len(), 3, "{line}");
            let to = words[2]
                .strip_prefix("staging/")
                .unwrap_or_else(|| panic!("{line} copies outside staging"));
            let entry = match to.strip_suffix('/') {
                Some(dir) => {
                    assert!(
                        made_dirs.contains(format!("staging/{dir}").as_str()),
                        "{line} copies into a directory no mkdir made"
                    );
                    let file = words[1].rsplit('/').next().unwrap();
                    format!("./{dir}/{file}")
                }
                None => format!("./{to}"),
            };
            copied.insert(entry);
        }
    }
    let prefix = r#"tar tzf ${{matrix.artifact}}.tar.gz | grep -Fq ""#;
    let checked = lines
        .iter()
        .filter_map(|line| line.strip_prefix(prefix)?.strip_suffix('"'))
        .map(str::to_owned)
        .collect();
    (copied, checked)
}

#[test]
fn the_native_archive_holds_every_header_its_headers_include() {
    let workflow = release();
    let (_, package) = step(job(&workflow, "native-libs"), "Package artifacts");
    let (copied, checked) = packaged_and_checked(package);

    // Each file the packager copies is looked up in the finished archive,
    // and nothing is looked up that was not copied.
    assert_eq!(copied, checked);

    // The headers that ship include these files. Whatever one of them
    // names that lives in include/ ships too, and anneal-core.h is the
    // generated header.
    let config: toml::Table = include_str!("../cbindgen.toml")
        .parse()
        .expect("cbindgen.toml");
    let generated_includes = config["after_includes"].as_str().expect("after_includes");
    let sources = [
        generated_includes,
        include_str!("../include/anneal-core.hpp"),
        include_str!("../include/anneal-version.h"),
    ];
    let mut included = BTreeSet::new();
    for source in sources {
        for line in source.lines() {
            if let Some(rest) = line.trim().strip_prefix("#include") {
                included.insert(rest.trim().trim_matches(['<', '>', '"']).to_owned());
            }
        }
    }
    let include_dir = std::path::Path::new(env!("CARGO_MANIFEST_DIR")).join("include");
    let mut shipped = 0;
    for header in &included {
        if include_dir.join(header).is_file() || header == "anneal-core.h" {
            shipped += 1;
            assert!(
                copied.contains(&format!("./include/{header}")),
                "a shipped header includes {header}, and the archive lacks it"
            );
        }
    }
    assert!(
        included.contains("dlpack/dlpack.h") && shipped >= 3,
        "{included:?}"
    );
}

#[test]
fn native_release_compiles_a_consumer_against_the_archive() {
    let workflow = release();
    let native = job(&workflow, "native-libs");
    let (package_at, _) = step(native, "Package artifacts");
    let (consumer_at, consumer) = step(native, "Compile a consumer against the archive");
    let upload_at = first_use(native, "actions/upload-artifact");

    assert_binding(consumer);
    assert_eq!(consumer["shell"].as_str(), Some("bash"));
    assert!(package_at < consumer_at && consumer_at < upload_at);

    let lines = logical_lines(consumer);
    let at = |wanted: &str| {
        lines
            .iter()
            .position(|line| line == wanted)
            .unwrap_or_else(|| panic!("the consumer step lacks {wanted:?}: {lines:?}"))
    };
    let extract = at("tar xzf ${{ matrix.artifact }}.tar.gz -C consumer");
    let compiles: Vec<usize> = (0..lines.len())
        .filter(|&i| lines[i].starts_with("cc "))
        .collect();
    assert_eq!(compiles.len(), 2, "{lines:?}");
    let (syntax, link) = (compiles[0], compiles[1]);
    let linux_only = at(r#"if [[ "${{ matrix.os }}" == linux ]]; then"#);
    let run = at("./consumer/ffi_smoke");
    assert!(extract < syntax && syntax < linux_only && linux_only < link && link < run);
    assert!(lines[syntax].contains("-fsyntax-only"));
    assert!(lines[link].contains("-Lconsumer/lib") && lines[link].contains("-lanneal_core"));
    assert!(lines[link].contains("-o consumer/ffi_smoke"));

    // The consumer reads headers from the extracted archive and nowhere
    // else, so a file the archive lacks is an error.
    for compile in [syntax, link] {
        assert!(lines[compile].contains("tests/ffi_smoke.c"));
        let include_dirs: Vec<&str> = lines[compile]
            .split_whitespace()
            .filter_map(|word| word.strip_prefix("-I"))
            .collect();
        assert_eq!(include_dirs, ["consumer/include"], "{}", lines[compile]);
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

// The default features run once per OS because no matrix Python reaches
// them. The satisfiable features run on Linux for every change and on
// macOS when a native/Python build input changes. CI may trim debuginfo,
// never the optimisation or the checks of the test profile.
#[test]
fn ci_runs_default_everywhere_and_gates_macos_python() {
    let workflow = ci();
    let changes = job(&workflow, "changes");
    let features = job(&workflow, "test");
    let default = job(&workflow, "test-default");

    assert_eq!(
        strings(&default["strategy"]["matrix"]["os"]),
        ["ubuntu-latest", "macos-15", "macos-26"]
    );
    assert_eq!(default["runs-on"].as_str(), Some("${{ matrix.os }}"));
    let (_, default_test) = step(default, "cargo test");
    assert_eq!(script(default_test), ["cargo test"]);
    assert_binding(default_test);

    assert_eq!(needs(features), BTreeSet::from(["changes"]));
    assert_eq!(features["runs-on"].as_str(), Some("${{ matrix.os }}"));
    assert_eq!(
        features["strategy"]["matrix"]["os"].as_str(),
        Some("${{ fromJSON(needs.changes.outputs.test-os) }}")
    );
    assert_eq!(
        strings(&features["strategy"]["matrix"]["python-version"]),
        ["3.10", "3.11", "3.12", "3.13"]
    );
    let (_, linux) = step(features, "cargo test (all satisfiable features)");
    let (_, macos) = step(features, "cargo test (all satisfiable features, macOS)");
    assert_eq!(
        script(linux),
        [r#"cargo test --features "$CI_FEATURES,lammps""#]
    );
    assert_eq!(script(macos), [r#"cargo test --features "$CI_FEATURES""#]);
    assert_eq!(linux["if"].as_str(), Some("runner.os != 'macOS'"));
    assert_eq!(macos["if"].as_str(), Some("runner.os == 'macOS'"));
    for test in [linux, macos] {
        assert!(test.get("continue-on-error").is_none());
    }
    for name in ["test", "test-default"] {
        assert_blocking(&workflow, name);
    }

    // The matrix the feature job reads is the one the changes job wrote.
    assert_eq!(
        changes["outputs"]["test-os"].as_str(),
        Some("${{ steps.matrix.outputs.test-os }}")
    );
    let (paths_at, paths) = step_with_id(changes, "paths");
    let (matrix_at, matrix) = step_with_id(changes, "matrix");
    assert!(paths_at < matrix_at);
    assert!(
        paths["uses"]
            .as_str()
            .is_some_and(|uses| uses.starts_with("dorny/paths-filter@"))
    );
    let filters: Value =
        serde_yaml_ng::from_str(paths["with"]["filters"].as_str().expect("filters"))
            .expect("the path filters parse as YAML");
    let macos_paths: BTreeSet<&str> = strings(&filters["macos-python"]).into_iter().collect();
    for path in [
        "src/**",
        "python/**",
        "Cargo.*",
        "pyproject.toml",
        ".github/workflows/ci.yml",
    ] {
        assert!(macos_paths.contains(path), "macOS path filter omits {path}");
    }

    // Run the step that picks the matrix, once with each answer the path
    // filter can give.
    let picked = |changed: &str| {
        let dir = tempfile::tempdir().expect("tempdir");
        let output = dir.path().join("output");
        std::fs::write(&output, "").expect("output file");
        let text = matrix["run"]
            .as_str()
            .expect("run")
            .replace("${{ steps.paths.outputs['macos-python'] }}", changed);
        let ran = Command::new("bash")
            .args(["--noprofile", "--norc", "-eo", "pipefail", "-c", &text])
            .env("GITHUB_OUTPUT", &output)
            .output()
            .expect("bash runs the matrix step");
        assert!(
            ran.status.success(),
            "{}",
            String::from_utf8_lossy(&ran.stderr)
        );
        std::fs::read_to_string(&output).expect("output")
    };
    let all = picked("true");
    let linux_only = picked("false");
    assert_eq!(
        all.trim(),
        r#"test-os=["ubuntu-latest","macos-15","macos-26"]"#
    );
    assert_eq!(linux_only.trim(), r#"test-os=["ubuntu-latest"]"#);
    let all: Vec<String> = serde_json::from_str(all.trim().strip_prefix("test-os=").unwrap())
        .expect("a JSON list for fromJSON");
    assert_eq!(all, ["ubuntu-latest", "macos-15", "macos-26"]);
}

#[test]
fn ci_may_trim_debuginfo_never_optimisation_or_checks() {
    let workflow = ci();

    assert_eq!(
        workflow["env"]["CARGO_PROFILE_TEST_DEBUG"].as_str(),
        Some("line-tables-only")
    );
    for (place, env) in every_env(&workflow) {
        for (key, value) in env {
            let key = key.as_str().expect("env key");
            let value = value.as_str().unwrap_or_default();
            assert!(
                !matches!(
                    key,
                    "CARGO_PROFILE_TEST_OPT_LEVEL"
                        | "CARGO_PROFILE_TEST_DEBUG_ASSERTIONS"
                        | "CARGO_PROFILE_TEST_OVERFLOW_CHECKS"
                        | "RUST_TEST_THREADS"
                ),
                "{place} sets {key}"
            );
            for knob in ["opt-level", "debug-assertions", "overflow-checks"] {
                assert!(
                    !value.contains(knob),
                    "{place} sets {key} to a value with {knob}"
                );
            }
        }
    }
    for (name, line) in every_script_line(&workflow) {
        assert!(!line.contains("--test-threads"), "job {name}: {line}");
        if line.contains("cargo test") {
            assert!(!line.contains("--release"), "job {name}: {line}");
        }
    }
}

// Linux jobs add `lammps` to this list themselves. `gpr` has no package a
// runner can fetch, and `nng-transport` gates no source and builds nng
// through cmake.
#[test]
fn ci_feature_list_names_only_features_a_runner_can_build() {
    let workflow = ci();
    let features: Vec<&str> = workflow["env"]["CI_FEATURES"]
        .as_str()
        .expect("CI_FEATURES")
        .split(',')
        .collect();

    for wanted in ["ira", "capi", "python", "rgpot-ex"] {
        assert!(features.contains(&wanted), "CI_FEATURES lacks {wanted}");
    }
    for excluded in ["lammps", "gpr", "nng-transport"] {
        assert!(
            !features.contains(&excluded),
            "CI_FEATURES holds {excluded}"
        );
    }
}

#[test]
fn ci_cancels_superseded_runs() {
    let workflow = ci();

    assert_eq!(
        workflow["concurrency"]["group"].as_str(),
        Some("ci-${{ github.ref }}")
    );
    assert_eq!(
        workflow["concurrency"]["cancel-in-progress"].as_bool(),
        Some(true)
    );
}

#[test]
fn ci_lint_reports_every_check_after_a_red_fmt() {
    let workflow = ci();
    let lint = job(&workflow, "lint");
    let (fmt_at, fmt) = step(lint, "cargo fmt");

    assert_eq!(script(fmt), ["cargo fmt -- --check"]);
    assert_binding(fmt);
    assert_blocking(&workflow, "lint");
    for later in &steps(lint)[fmt_at + 1..] {
        assert_runs_after_failures(later);
    }
    assert_outcomes_come_from_earlier_steps(lint);
    assert!(!serde_yaml_ng::to_string(lint).unwrap().contains("always()"));

    // Every feature set is linted, each by the step of that name, and no
    // set can drop out by sharing a step with another.
    for (name, command) in [
        (
            "cargo clippy (default features)",
            "cargo clippy --all-targets -- -D warnings",
        ),
        (
            "cargo clippy (rgpot-ex)",
            "cargo clippy --all-targets --features rgpot-ex -- -D warnings",
        ),
        (
            "cargo clippy (rgsaddle)",
            "cargo clippy --all-targets --features rgsaddle -- -D warnings",
        ),
        (
            "cargo clippy",
            r#"cargo clippy --features "$CI_FEATURES,lammps" --all-targets -- -D warnings"#,
        ),
        (
            "cargo clippy (all features)",
            "cargo clippy --all-targets --all-features -- -D warnings",
        ),
    ] {
        let (at, clippy) = step(lint, name);
        assert!(at > fmt_at, "{name} runs after fmt");
        let lines = logical_lines(clippy);
        assert_eq!(
            lines
                .iter()
                .filter(|line| line.contains("cargo clippy"))
                .count(),
            1,
            "{name} runs one clippy"
        );
        assert!(
            lines.iter().any(|line| line.ends_with(command)),
            "{name} lacks {command}: {lines:?}"
        );
    }
}

#[test]
fn ci_packages_the_crate_and_checks_it_against_crates_io() {
    let workflow = ci();
    let lint = job(&workflow, "lint");
    let (package_at, package) = step(lint, "Package the crate");
    let (check_at, check) = step(lint, "Check packaged default features against crates.io");

    assert_eq!(package["id"].as_str(), Some("package"));
    assert_eq!(script(package), ["cargo package"]);
    assert_runs_after_failures(package);
    assert!(package_at < check_at);
    assert_eq!(
        check["if"].as_str(),
        Some("${{ !cancelled() && steps.package.outcome == 'success' }}")
    );
    assert_eq!(check["shell"].as_str(), Some("bash"));

    // The packaged crate is the one cargo metadata names for anneal-core,
    // taken apart, cut loose from the development config and lockfile, and
    // checked with a fresh lockfile for both feature sets.
    let lines = script(check);
    let at = |wanted: &str| {
        lines
            .iter()
            .position(|line| *line == wanted)
            .unwrap_or_else(|| panic!("the check lacks {wanted:?}"))
    };
    assert_eq!(lines[0], "set -euo pipefail");
    assert!(
        lines[at("meta=$(cargo metadata --format-version 1 --no-deps)")..]
            .iter()
            .any(|line| line.starts_with("read -r target version <")
                && line.contains(r#"p["name"]=="anneal-core""#)
                && line.contains(r#"pkg["version"]"#))
    );
    let crate_file = at(r#"crate="${target}/package/anneal-core-${version}.crate""#);
    assert_eq!(lines[crate_file + 1], r#"test -f "$crate""#);
    let unpack = at(r#"tar -xzf "$crate" -C "$work""#);
    assert!(crate_file < unpack);
    let cut_config = at(r#"rm -rf "$pkg/.cargo""#);
    let cut_lock = at(r#"rm -f "$pkg/Cargo.lock""#);
    let lock = at("cargo generate-lockfile");
    let default = at("cargo check --locked");
    let features = at("cargo check --locked --features rgsaddle,python");
    assert!(unpack < cut_config && cut_config < cut_lock && cut_lock < lock);
    assert!(lock < default && default < features);
    assert!(
        lines[..lock].contains(&"cd \"$pkg\""),
        "the checks run inside the unpacked crate"
    );
    for pattern in ["anneal-core-*.crate", "tail -1", "--no-verify"] {
        for (name, line) in every_script_line(&workflow) {
            assert!(!line.contains(pattern), "job {name} has {pattern}: {line}");
        }
    }
}

#[test]
fn ci_provisions_pinned_all_feature_dependencies() {
    let workflow = ci();

    for job_name in ["lint", "test"] {
        let job = job(&workflow, job_name);
        for (name, repository, rev, path) in [
            (
                "Checkout Vesin",
                "Luthaf/vesin",
                "5ac39b6e57a8f00c717f08c83f93ee4746d6e471",
                ".ci/vesin",
            ),
            (
                "Checkout IRA",
                "mammasmias/IterativeRotationsAssignments",
                "2b2fc312569a5a50183ca82b2f260ebaaf87c508",
                ".ci/ira",
            ),
        ] {
            let (_, checkout) = step(job, name);
            assert!(
                checkout["uses"]
                    .as_str()
                    .is_some_and(|uses| uses.starts_with("actions/checkout@")),
                "{job_name}: {name}"
            );
            assert_eq!(checkout["with"]["repository"].as_str(), Some(repository));
            assert_eq!(
                checkout["with"]["ref"].as_str(),
                Some(rev),
                "{job_name}: {name}"
            );
            assert_eq!(checkout["with"]["path"].as_str(), Some(path));
            assert_eq!(
                checkout["with"]["persist-credentials"].as_bool(),
                Some(false)
            );
        }
        let (build_at, build) = step(job, "Build all-feature dependencies");
        assert!(build_at > step(job, "Checkout IRA").0);
        assert!(build_at > step(job, "Checkout Vesin").0);
        let lines = script(build);
        for wanted in [
            r#"echo "VESIN_SRC=$GITHUB_WORKSPACE/.ci/vesin" >> "$GITHUB_ENV""#,
            r#"echo "IRA_LIB_DIR=$GITHUB_WORKSPACE/.ci/ira/lib" >> "$GITHUB_ENV""#,
            "cmake --build .ci/ira/build --parallel 2",
        ] {
            assert!(lines.contains(&wanted), "{job_name} lacks {wanted}");
        }
        assert!(
            lines
                .iter()
                .position(|line| line.starts_with("cmake -S .ci/ira -B .ci/ira/build"))
                .unwrap()
                < lines
                    .iter()
                    .position(|line| *line == "cmake --build .ci/ira/build --parallel 2")
                    .unwrap(),
            "{job_name} configures IRA before building it"
        );
    }
}

#[test]
fn ci_lints_only_the_new_commit_range() {
    let workflow = ci();
    let lint = job(&workflow, "lint");
    let (install_at, install) = step(lint, "Install Cocogitto");
    let (check_at, check) = step(lint, "Conventional commits check");

    assert_eq!(install["id"].as_str(), Some("cocogitto"));
    assert_eq!(install["with"]["tool"].as_str(), Some("cocogitto"));
    assert!(
        install["uses"]
            .as_str()
            .is_some_and(|uses| uses.starts_with("taiki-e/install-action@"))
    );
    assert!(install_at < check_at);
    assert_runs_after_failures(install);
    assert!(
        check["if"]
            .as_str()
            .is_some_and(|condition| condition.contains("steps.cocogitto.outcome == 'success'"))
    );
    assert!(
        check["if"]
            .as_str()
            .is_some_and(|condition| condition.starts_with("${{ !cancelled()"))
    );
    assert!(check.get("continue-on-error").is_none());

    // Pull requests check base..head, pushes check before..sha.
    let from = check["env"]["COG_FROM"].as_str().expect("COG_FROM");
    let to = check["env"]["COG_TO"].as_str().expect("COG_TO");
    for (value, wanted) in [
        (from, "github.event.pull_request.base.sha"),
        (from, "github.event.before"),
        (to, "github.event.pull_request.head.sha"),
        (to, "github.sha"),
    ] {
        assert!(value.contains(wanted), "{value} lacks {wanted}");
    }
    assert_eq!(
        script(check).last().copied(),
        Some(r#"cog check "$COG_FROM..$COG_TO""#),
        "an unexempted range ends in a plain check of the whole range"
    );
    for (name, step) in steps_of_every_job(&workflow) {
        assert!(
            !step["uses"]
                .as_str()
                .is_some_and(|uses| uses.starts_with("cocogitto/cocogitto-action")),
            "job {name} uses the cocogitto action"
        );
    }
}

/// Every step of every job, with its job.
fn steps_of_every_job(workflow: &Value) -> Vec<(String, &Value)> {
    let mut found = Vec::new();
    for (name, job) in workflow["jobs"].as_mapping().expect("jobs") {
        for step in steps(job) {
            found.push((name.as_str().unwrap().to_owned(), step));
        }
    }
    found
}

#[test]
fn ci_audits_large_blobs_without_a_deprecated_action_runtime() {
    let workflow = ci();
    let lint = job(&workflow, "lint");
    let (_, audit) = step(lint, "Audit for large files");

    assert_runs_after_failures(audit);
    assert_eq!(audit["shell"].as_str(), Some("bash"));
    let base = audit["env"]["AUDIT_BASE"].as_str().expect("AUDIT_BASE");
    let head = audit["env"]["AUDIT_HEAD"].as_str().expect("AUDIT_HEAD");
    assert!(base.contains("github.event.pull_request.base.sha"));
    assert!(head.contains("github.event.pull_request.head.sha"));
    assert!(head.contains("github.sha"));
    let lines = script(audit);
    for wanted in [
        "objects=$(git rev-list --objects HEAD)",
        r#"objects=$(git rev-list --objects "$range")"#,
        "git cat-file --batch-check='%(objecttype) %(objectsize) %(objectname) %(rest)' |",
        "exit 1",
    ] {
        assert!(lines.contains(&wanted), "the audit lacks {wanted}");
    }
    assert!(
        lines
            .iter()
            .any(|line| line.contains("$2 > 1048576") && line.contains("!($3 in known)"))
    );
    for (name, step) in steps_of_every_job(&workflow) {
        assert!(
            !step["uses"]
                .as_str()
                .is_some_and(|uses| uses.contains("large-file-auditor")),
            "job {name} uses a large file auditor action"
        );
    }
}

/// The audit step's own script, run in a throwaway repository.
#[test]
fn the_large_file_audit_flags_every_new_blob_over_one_mebibyte() {
    let workflow = ci();
    let (_, audit) = step(job(&workflow, "lint"), "Audit for large files");
    let text = audit["run"].as_str().expect("run");

    let dir = tempfile::tempdir().expect("tempdir");
    let git = |args: &[&str]| {
        let output = Command::new("git")
            .args(args)
            .current_dir(dir.path())
            .env("GIT_AUTHOR_NAME", "t")
            .env("GIT_AUTHOR_EMAIL", "t@example.org")
            .env("GIT_COMMITTER_NAME", "t")
            .env("GIT_COMMITTER_EMAIL", "t@example.org")
            .output()
            .expect("git runs");
        assert!(
            output.status.success(),
            "git {args:?}: {}",
            String::from_utf8_lossy(&output.stderr)
        );
        String::from_utf8(output.stdout)
            .expect("utf8")
            .trim()
            .to_owned()
    };
    let commit = |file: &str, bytes: usize, message: &str| {
        std::fs::write(dir.path().join(file), vec![b'a'; bytes]).expect("write blob");
        git(&["add", file]);
        git(&["commit", "-q", "-m", message]);
        git(&["rev-parse", "HEAD"])
    };
    git(&["init", "-q"]);
    std::fs::create_dir(dir.path().join(".github")).expect(".github");
    let known = dir.path().join(".github/large-blobs.txt");
    std::fs::write(&known, "# known blobs\n").expect("known blobs");

    let exactly_one_mebibyte = commit("limit.bin", 1_048_576, "at the limit");
    let over = commit("over.bin", 1_048_577, "over the limit");
    let small = commit("small.txt", 10, "small");
    let over_blob = git(&["hash-object", "over.bin"]);

    let run_audit = |base: &str, head: &str| {
        Command::new("bash")
            .args(["--noprofile", "--norc", "-eo", "pipefail", "-c", text])
            .env("AUDIT_BASE", base)
            .env("AUDIT_HEAD", head)
            .current_dir(dir.path())
            .output()
            .expect("bash runs the audit")
    };
    // A push audits HEAD, which is the commit the checkout holds.
    let push_audit = |sha: &str| {
        git(&["checkout", "-q", sha]);
        run_audit("", sha)
    };

    // A push audits the whole history: the 1 MiB blob passes, the next
    // byte fails, and the failure names the blob.
    let at_limit = push_audit(&exactly_one_mebibyte);
    assert!(
        at_limit.status.success(),
        "{}",
        String::from_utf8_lossy(&at_limit.stderr)
    );
    let pushed = push_audit(&small);
    assert!(
        !pushed.status.success(),
        "a push with an oversized blob passed"
    );
    assert!(String::from_utf8_lossy(&pushed.stderr).contains(&over_blob));

    // A pull request audits what its own commits add.
    let adds_it = run_audit(&exactly_one_mebibyte, &over);
    assert!(
        !adds_it.status.success(),
        "a pull request that adds the blob passed"
    );
    let already_there = run_audit(&over, &small);
    assert!(
        already_there.status.success(),
        "{}",
        String::from_utf8_lossy(&already_there.stderr)
    );

    // The list of blobs develop already carries skips those objects and
    // nothing else.
    std::fs::write(&known, format!("# known blobs\n{over_blob} over.bin\n")).expect("known blobs");
    let listed = push_audit(&small);
    assert!(
        listed.status.success(),
        "{}",
        String::from_utf8_lossy(&listed.stderr)
    );
    let another = commit("another.bin", 2_000_000, "another large blob");
    assert!(!push_audit(&another).status.success());
}

#[test]
fn ci_sdist_job_builds_the_sdist_it_then_installs() {
    let workflow = ci();
    let sdist = job(&workflow, "sdist");
    let (build_at, build) = step(sdist, "Build the sdist");
    let (install_at, install) = step(sdist, "Install from the sdist");
    let (test_at, test) = step(sdist, "pytest against the installed sdist");

    assert!(build_at < install_at && install_at < test_at);
    assert_binding(build);
    assert_binding(install);
    assert_binding(test);
    assert_blocking(&workflow, "sdist");
    assert!(script(build).contains(&"python -m maturin sdist --out dist"));
    assert_eq!(
        script(install),
        [
            "python -m venv .sdist-venv",
            ".sdist-venv/bin/python -m pip install --upgrade pip",
            ".sdist-venv/bin/python -m pip install dist/anneal-*.tar.gz pytest",
        ]
    );

    // pytest runs from the checkout, so the guard first proves the package
    // under test is the installed one and not the source tree. It is the
    // command the release workflow runs on the sdist.
    let release = release();
    let (_, release_install) = step(job(&release, "sdist"), "Install and import sdist");
    let guard = script(release_install)[2];
    assert_eq!(
        script(test),
        [
            guard,
            ".sdist-venv/bin/python -m pytest pytest/test_cluster_search.py"
        ]
    );
}

#[test]
fn ci_never_lets_a_job_or_step_fail_without_failing_the_run() {
    let workflow = ci();

    for (name, job) in workflow["jobs"].as_mapping().expect("jobs") {
        assert!(
            job.get("continue-on-error").is_none(),
            "job {} sets continue-on-error",
            name.as_str().unwrap()
        );
    }
    for (name, step) in steps_of_every_job(&workflow) {
        assert!(
            step.get("continue-on-error").is_none(),
            "job {name}, step {:?} sets continue-on-error",
            step["name"]
        );
    }
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

/// The development dependency of the root crate, the wheel crate's
/// dependency and the lockfile entry must name one rgpot revision. Cargo
/// accepts two revisions side by side, which links two copies of the
/// kernels, and a lockfile that disagrees with a manifest fails `--locked`.
#[test]
fn rgpot_pair_is_pinned_to_one_revision() {
    let rev_of = |manifest: &toml::Table, table: &str| -> String {
        let pin = &manifest[table]["rgpot-pair"];
        assert_eq!(
            pin["git"].as_str(),
            Some("https://github.com/OmniPotentRPC/rgpot")
        );
        pin["rev"]
            .as_str()
            .expect("a rev, not a branch or tag")
            .to_owned()
    };
    let root: toml::Table = include_str!("../Cargo.toml")
        .parse()
        .expect("root manifest");
    let wheel: toml::Table = include_str!("../crates/anneal-python/Cargo.toml")
        .parse()
        .expect("wheel manifest");
    let lock: toml::Table = include_str!("../Cargo.lock").parse().expect("Cargo.lock");

    let rev = rev_of(&root, "dev-dependencies");
    assert_eq!(rev.len(), 40);
    assert!(rev.chars().all(|c| c.is_ascii_hexdigit()), "{rev}");
    assert_eq!(rev_of(&wheel, "dependencies"), rev);

    let entries: Vec<&toml::Value> = lock["package"]
        .as_array()
        .expect("packages")
        .iter()
        .filter(|package| package["name"].as_str() == Some("rgpot-pair"))
        .collect();
    assert_eq!(entries.len(), 1, "one rgpot-pair in Cargo.lock");
    assert_eq!(
        entries[0]["source"].as_str(),
        Some(format!("git+https://github.com/OmniPotentRPC/rgpot?rev={rev}#{rev}").as_str())
    );
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
}
