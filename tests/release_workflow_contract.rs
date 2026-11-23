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

/// The line of the packaging step that lists the finished archive once.
const LIST_ARCHIVE: &str = "tar tzf ${{matrix.artifact}}.tar.gz > archive-entries.txt";

/// The entry a packaging line checks for in the archive listing, if it is
/// such a check.
fn checked_entry(line: &str) -> Option<&str> {
    line.strip_prefix(r#"grep -Fxq ""#)?
        .strip_suffix(r#"" archive-entries.txt"#)
}

/// The archive entries the packaging step creates with `cp`, and the ones it
/// checks for in the archive listing.
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
            let file = words[1].rsplit('/').next().unwrap();
            let entry = match to.strip_suffix('/') {
                Some(dir) => {
                    assert!(
                        made_dirs.contains(format!("staging/{dir}").as_str()),
                        "{line} copies into a directory no mkdir made"
                    );
                    format!("./{dir}/{file}")
                }
                None if to.is_empty() => format!("./{file}"),
                None => format!("./{to}"),
            };
            copied.insert(entry);
        } else if words.starts_with(&["python3", "scripts/native_pkgconfig.py", "staging"]) {
            copied.insert("./lib/pkgconfig/anneal-core.pc".to_owned());
        } else if line == LIST_ARCHIVE {
            // The listing is written beside the archive, not into it.
        } else if let Some(at) = words.iter().position(|word| *word == ">") {
            let to = words[at + 1]
                .strip_prefix("staging/")
                .unwrap_or_else(|| panic!("{line} writes outside staging"));
            copied.insert(format!("./{to}"));
        }
    }
    // The step runs under pipefail, and the archive is listed once to a
    // file after it is written. No check reads tar through a pipe, where an
    // early grep exit would kill tar with SIGPIPE, and each check matches
    // a whole entry.
    assert_eq!(package["shell"].as_str(), Some("bash"));
    let archive = "tar czf ${{matrix.artifact}}.tar.gz -C staging .";
    let written = lines
        .iter()
        .position(|line| line == archive)
        .unwrap_or_else(|| panic!("the packaging step lacks {archive:?}: {lines:?}"));
    let listings: Vec<usize> = (0..lines.len())
        .filter(|&i| lines[i].contains("tar tzf"))
        .collect();
    assert_eq!(listings.len(), 1, "{lines:?}");
    let listed = listings[0];
    assert_eq!(lines[listed], LIST_ARCHIVE);
    assert!(written < listed);
    let mut checked = BTreeSet::new();
    for (at, line) in lines.iter().enumerate() {
        assert!(
            !line.contains('|'),
            "{line} pipes inside the packaging step"
        );
        if line.starts_with("grep") || line.contains(" grep ") {
            let entry = checked_entry(line)
                .unwrap_or_else(|| panic!("{line} is not an exact check of the listing"));
            assert!(listed < at, "{line} runs before the archive is listed");
            assert!(checked.insert(entry.to_owned()), "{line} repeats a check");
        }
    }
    (copied, checked)
}

/// The packaging step's own listing and checks, run under the shell the
/// step uses, against an archive of the entries it copies.
#[test]
fn the_archive_checks_survive_pipefail_and_match_whole_entries() {
    let workflow = release();
    let (_, package) = step(job(&workflow, "native-libs"), "Package artifacts");
    let (copied, checked) = packaged_and_checked(package);
    assert_eq!(copied, checked);
    let cell = |line: &str| {
        line.replace("${{matrix.artifact}}", "anneal-core-test")
            .replace("${{matrix.static_lib}}", "libanneal_core.a")
            .replace("${{matrix.shared_lib}}", "libanneal_core.so")
    };
    let lines: Vec<String> = script(package)
        .into_iter()
        .map(compact_expressions)
        .filter(|line| line == LIST_ARCHIVE || checked_entry(line).is_some())
        .map(|line| cell(&line))
        .collect();
    assert_eq!(lines.len(), checked.len() + 1);
    let checks = lines.join("\n");

    let scratch = std::env::temp_dir().join(format!("anneal-archive-{}", std::process::id()));
    let run = |entries: &[String], padding: usize| {
        let _ = std::fs::remove_dir_all(&scratch);
        let staging = scratch.join("staging");
        std::fs::create_dir_all(&staging).unwrap();
        // Enough entries that the listing outgrows a pipe buffer.
        for i in 0..padding {
            let pad = staging.join(format!("pad/{i:05}-{}", "x".repeat(80)));
            std::fs::create_dir_all(pad.parent().unwrap()).unwrap();
            std::fs::write(pad, "").unwrap();
        }
        for entry in entries {
            let path = staging.join(entry.strip_prefix("./").unwrap());
            std::fs::create_dir_all(path.parent().unwrap()).unwrap();
            std::fs::write(path, "").unwrap();
        }
        let tar = Command::new("tar")
            .args(["czf", "anneal-core-test.tar.gz", "-C", "staging", "."])
            .current_dir(&scratch)
            .status()
            .expect("run tar");
        assert!(tar.success());
        Command::new("bash")
            .args(["--noprofile", "--norc", "-eo", "pipefail", "-c", &checks])
            .current_dir(&scratch)
            .env("PATH", std::env::var_os("PATH").unwrap())
            .output()
            .expect("run bash")
    };
    let every: Vec<String> = cell(&copied.iter().cloned().collect::<Vec<_>>().join("\n"))
        .lines()
        .map(str::to_owned)
        .collect();

    let output = run(&every, 2000);
    assert!(output.status.success(), "{output:?}");
    // Without anneal-core.h the archive still holds anneal-core.hpp, whose
    // name begins with it, so a check that is not anchored would pass.
    for missing in &every {
        let rest: Vec<String> = every.iter().filter(|e| *e != missing).cloned().collect();
        let output = run(&rest, 0);
        assert_eq!(
            output.status.code(),
            Some(1),
            "{missing} is missing: {output:?}"
        );
    }
    assert!(every.contains(&"./include/anneal-core.h".to_owned()));
    assert!(every.contains(&"./include/anneal-core.hpp".to_owned()));
    std::fs::remove_dir_all(&scratch).unwrap();
}

/// Every workflow under .github/workflows, parsed, with its file name.
fn every_workflow() -> Vec<(String, Value)> {
    let dir = std::path::Path::new(env!("CARGO_MANIFEST_DIR")).join(".github/workflows");
    let mut found = Vec::new();
    for entry in std::fs::read_dir(dir).expect("read .github/workflows") {
        let path = entry.unwrap().path();
        if path.extension().and_then(|e| e.to_str()) == Some("yml") {
            let text = std::fs::read_to_string(&path).unwrap();
            let name = path.file_name().unwrap().to_string_lossy().into_owned();
            found.push((
                name,
                serde_yaml_ng::from_str(&text).expect("workflow parses as YAML"),
            ));
        }
    }
    found.sort_by(|a, b| a.0.cmp(&b.0));
    found
}

#[test]
fn cbindgen_is_installed_at_one_exact_version_from_its_lockfile() {
    // Whatever cbindgen version a runner gets writes anneal-core.h, the
    // header every native archive ships. So every install names one exact
    // version and builds it from the lockfile the crate was published with.
    let mut versions = BTreeSet::new();
    let mut installs = 0;
    for (file, workflow) in every_workflow() {
        for (name, job) in workflow["jobs"].as_mapping().expect("jobs") {
            for step in steps(job) {
                let at = format!(
                    "{file}, job {}, step {:?}",
                    name.as_str().unwrap(),
                    step["name"]
                );
                // A prebuilt-binary installer takes whatever it is given.
                if step["uses"]
                    .as_str()
                    .is_some_and(|uses| uses.starts_with("taiki-e/install-action"))
                {
                    let tools = step["with"]["tool"].as_str().unwrap_or_default();
                    assert!(
                        !tools.contains("cbindgen"),
                        "{at} installs cbindgen unpinned"
                    );
                }
                let Some(run) = step["run"].as_str() else {
                    continue;
                };
                for line in run.lines().map(str::trim) {
                    let words: Vec<&str> = line.split_whitespace().collect();
                    let installs_it = words.windows(2).any(|w| w == ["install", "cbindgen"])
                        || (words.contains(&"install") && words.contains(&"cbindgen"));
                    if !installs_it {
                        continue;
                    }
                    installs += 1;
                    assert_eq!(
                        &words[..3],
                        ["cargo", "install", "cbindgen"],
                        "{at}: {line}"
                    );
                    assert!(words.contains(&"--locked"), "{at}: {line}");
                    let flag = words.iter().position(|w| *w == "--version");
                    let version = words[flag.unwrap_or_else(|| panic!("{at}: {line}")) + 1];
                    let parts: Vec<&str> = version.split('.').collect();
                    assert!(
                        parts.len() == 3
                            && parts
                                .iter()
                                .all(|p| !p.is_empty() && p.chars().all(|c| c.is_ascii_digit())),
                        "{at}: {version} is not an exact MAJOR.MINOR.PATCH"
                    );
                    assert_eq!(words.len(), 6, "{at}: {line}");
                    assert_binding(step);
                    versions.insert(version.to_owned());
                }
            }
        }
    }
    assert_eq!(installs, 2, "the release job and the CI ffi job install it");
    assert_eq!(versions.len(), 1, "{versions:?}");

    // Meson asks for at least 0.29.0, which the pinned version satisfies.
    let pinned: Vec<u32> = versions
        .iter()
        .next()
        .unwrap()
        .split('.')
        .map(|p| p.parse().unwrap())
        .collect();
    let meson = include_str!("../meson.build");
    let wanted = meson
        .split("find_program('cbindgen', version: '>=")
        .nth(1)
        .and_then(|rest| rest.split('\'').next())
        .expect("meson.build asks for a cbindgen version");
    let wanted: Vec<u32> = wanted.split('.').map(|p| p.parse().unwrap()).collect();
    assert!(
        pinned >= wanted,
        "{pinned:?} is older than meson's {wanted:?}"
    );
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
fn the_native_archive_carries_the_licence_and_its_readme_section() {
    let workflow = release();
    let (_, package) = step(job(&workflow, "native-libs"), "Package artifacts");
    let (copied, checked) = packaged_and_checked(package);
    for entry in ["./LICENSE.txt", "./README-licensing.md"] {
        assert!(copied.contains(entry), "the archive is not given {entry}");
        assert!(
            checked.contains(entry),
            "the archive is not checked for {entry}"
        );
    }

    let lines = script(package);
    assert!(lines.contains(&"cp LICENSE.txt staging/"), "{lines:?}");
    assert!(
        lines.contains(&"test -s staging/README-licensing.md"),
        "an empty excerpt must fail the step"
    );

    // Run the workflow's own extraction against the repository README. The
    // excerpt is the licensing section and nothing else.
    let extract = lines
        .iter()
        .find(|line| line.ends_with("> staging/README-licensing.md"))
        .expect("a line writes the licensing excerpt");
    let scratch = std::env::temp_dir().join(format!("anneal-licensing-{}", std::process::id()));
    let _ = std::fs::remove_dir_all(&scratch);
    std::fs::create_dir_all(scratch.join("staging")).unwrap();
    let output = Command::new("bash")
        .args(["--noprofile", "--norc", "-eo", "pipefail", "-c", extract])
        .current_dir(&scratch)
        .env("PATH", std::env::var_os("PATH").unwrap())
        .output()
        .expect("run bash");
    // The step reads README.md from the repository root.
    assert!(
        !output.status.success(),
        "no README.md in the scratch directory"
    );
    std::fs::copy(
        std::path::Path::new(env!("CARGO_MANIFEST_DIR")).join("README.md"),
        scratch.join("README.md"),
    )
    .unwrap();
    let output = Command::new("bash")
        .args(["--noprofile", "--norc", "-eo", "pipefail", "-c", extract])
        .current_dir(&scratch)
        .env("PATH", std::env::var_os("PATH").unwrap())
        .output()
        .expect("run bash");
    assert!(output.status.success(), "{output:?}");
    let excerpt = std::fs::read_to_string(scratch.join("staging/README-licensing.md")).unwrap();
    std::fs::remove_dir_all(&scratch).unwrap();

    let headings: Vec<&str> = excerpt.lines().filter(|l| l.starts_with("## ")).collect();
    assert_eq!(headings, ["## License and citation"], "{excerpt}");
    assert!(excerpt.contains("MIT") && excerpt.contains("LICENSE.txt"));
    let readme = include_str!("../README.md");
    assert!(readme.contains(excerpt.trim_end()));

    // The licence file the archive ships is the repository's MIT text.
    assert!(include_str!("../LICENSE.txt").starts_with("MIT License"));
}

fn pkg_config(search: &std::path::Path, args: &[&str]) -> String {
    let output = Command::new("pkg-config")
        .args(args)
        .env_remove("PKG_CONFIG_LIBDIR")
        .env_remove("PKG_CONFIG_SYSROOT_DIR")
        .env("PKG_CONFIG_PATH", search)
        .output()
        .expect("pkg-config must be installed to check the archive's .pc file");
    assert!(output.status.success(), "pkg-config {args:?}: {output:?}");
    String::from_utf8(output.stdout).unwrap().trim().to_owned()
}

#[test]
fn the_native_archive_ships_a_prefix_relative_pkg_config_file() {
    let workflow = release();
    let native = job(&workflow, "native-libs");
    let (build_at, build) = step(native, "Build static + shared library");
    let (list_at, list) = step(native, "List the libraries the static library needs");
    let (package_at, package) = step(native, "Package artifacts");
    assert_binding(list);
    assert_eq!(list["shell"].as_str(), Some("bash"));
    assert!(build_at < list_at && list_at < package_at);

    // The list comes from rustc for the same crate, target and features the
    // build used, and an empty list fails the step.
    let list = logical_lines(list);
    let flags = |line: &str| -> Vec<String> {
        compact_expressions(line)
            .split_whitespace()
            .map(str::to_owned)
            .collect()
    };
    let words = flags(&list[0]);
    let build_words = flags(script(build)[0]);
    assert_eq!(words[..3], ["cargo", "rustc", "--release"], "{list:?}");
    for flag in ["--features", "--target", "--target-dir"] {
        let value = |all: &[String]| {
            let at = all.iter().position(|word| word == flag).unwrap();
            all[at + 1].clone()
        };
        assert_eq!(value(&words), value(&build_words), "{flag}");
    }
    assert!(words.iter().any(|word| word == "--lib"), "{list:?}");
    let after = words.iter().position(|w| w == "--").expect("rustc flags");
    assert_eq!(
        words[after + 1..],
        [
            "--print",
            "native-static-libs",
            "2>&1",
            "|",
            "tee",
            "rustc-native-libs.log"
        ]
    );
    assert!(list.contains(&"test -s native-static-libs.txt".to_owned()));
    assert!(
        list.iter().any(|l| l
            .contains("sed -n 's/^note: native-static-libs: //p' rustc-native-libs.log")
            && l.ends_with("| tail -n 1 > native-static-libs.txt")),
        "{list:?}"
    );

    // The .pc file is generated in the staging tree and checked in the
    // archive.
    let (copied, checked) = packaged_and_checked(package);
    let entry = "./lib/pkgconfig/anneal-core.pc";
    assert!(copied.contains(entry) && checked.contains(entry));
    let generate = script(package)
        .into_iter()
        .position(|line| line.starts_with("python3 scripts/native_pkgconfig.py"))
        .expect("the packaging step generates the .pc file");
    assert_eq!(
        script(package)[generate],
        "python3 scripts/native_pkgconfig.py staging --private-libs-file native-static-libs.txt"
    );

    // What the .pc file names is what the archive holds: the library
    // cargo-c names, in lib/, and headers in include/.
    let manifest: toml::Table = include_str!("../Cargo.toml").parse().unwrap();
    let capi = &manifest["package"]["metadata"]["capi"];
    let library = capi["library"]["name"].as_str().unwrap();
    let cells = native["strategy"]["matrix"]["include"]
        .as_sequence()
        .unwrap();
    assert!(cells.len() >= 3);
    for cell in cells {
        assert_eq!(
            cell["static_lib"].as_str().unwrap(),
            format!("lib{library}.a")
        );
        let shared = cell["shared_lib"].as_str().unwrap();
        assert!(shared.starts_with(&format!("lib{library}.")), "{shared}");
    }

    // Run the generator and read what it wrote.
    let scratch = std::env::temp_dir().join(format!("anneal-pc-{}", std::process::id()));
    let _ = std::fs::remove_dir_all(&scratch);
    let root = scratch.join("one");
    std::fs::create_dir_all(root.join("include/dlpack")).unwrap();
    std::fs::create_dir_all(root.join("lib")).unwrap();
    std::fs::write(scratch.join("libs.txt"), "-lgcc_s -lm\n-ldl\n").unwrap();
    let repo = std::path::Path::new(env!("CARGO_MANIFEST_DIR"));
    let run = Command::new("python3")
        .arg(repo.join("scripts/native_pkgconfig.py"))
        .arg(&root)
        .arg("--private-libs-file")
        .arg(scratch.join("libs.txt"))
        .output()
        .expect("run python3");
    assert!(run.status.success(), "{run:?}");
    let pc = std::fs::read_to_string(root.join("lib/pkgconfig/anneal-core.pc")).unwrap();

    let version = manifest["workspace"]["package"]["version"]
        .as_str()
        .unwrap();
    let description = capi["pkg_config"]["description"].as_str().unwrap();
    let field = |key: &str| {
        pc.lines()
            .find_map(|line| line.strip_prefix(key))
            .unwrap_or_else(|| panic!("the .pc file has no {key}: {pc}"))
    };
    assert_eq!(field("prefix="), "${pcfiledir}/../..");
    assert_eq!(field("libdir="), "${prefix}/lib");
    assert_eq!(field("includedir="), "${prefix}/include");
    assert_eq!(field("Name: "), "anneal-core");
    assert_eq!(field("Version: "), version);
    assert_eq!(field("Description: "), description);
    assert_eq!(field("Cflags: "), "-I${includedir}");
    assert_eq!(field("Libs: "), format!("-L${{libdir}} -l{library}"));
    assert_eq!(field("Libs.private: "), "-lgcc_s -lm -ldl");
    for line in pc.lines() {
        assert!(
            !line.contains(scratch.to_str().unwrap()) && !line.contains("/home/"),
            "an absolute path in {line:?}"
        );
    }

    // pkg-config reads it. The same tree answers from where it is moved to.
    let moved = scratch.join("two");
    for tree in [&root, &moved] {
        if tree == &moved {
            std::fs::rename(&root, &moved).unwrap();
        }
        let search = tree.join("lib/pkgconfig");
        let canonical = |text: &str, flag: &str| -> Vec<std::path::PathBuf> {
            text.split_whitespace()
                .filter_map(|w| w.strip_prefix(flag))
                .map(|dir| std::fs::canonicalize(dir).unwrap())
                .collect()
        };
        let cflags = pkg_config(&search, &["--cflags", "anneal-core"]);
        assert_eq!(
            canonical(&cflags, "-I"),
            [std::fs::canonicalize(tree.join("include")).unwrap()]
        );
        let libs = pkg_config(&search, &["--libs", "anneal-core"]);
        assert_eq!(
            canonical(&libs, "-L"),
            [std::fs::canonicalize(tree.join("lib")).unwrap()]
        );
        assert!(
            libs.split_whitespace().any(|w| w == "-lanneal_core"),
            "{libs}"
        );
        assert!(!libs.contains("-lgcc_s"), "{libs}");
        let statics = pkg_config(&search, &["--static", "--libs", "anneal-core"]);
        assert!(
            statics.ends_with("-lanneal_core -lgcc_s -lm -ldl"),
            "{statics}"
        );
        assert_eq!(
            pkg_config(&search, &["--modversion", "anneal-core"]),
            version
        );
    }
    std::fs::remove_dir_all(&scratch).unwrap();

    // Without the list the generator still writes a .pc file, and an empty
    // list file is an error, not a silently incomplete static link line.
    let empty = std::env::temp_dir().join(format!("anneal-pc-empty-{}", std::process::id()));
    std::fs::create_dir_all(empty.join("include")).unwrap();
    std::fs::create_dir_all(empty.join("lib")).unwrap();
    std::fs::write(empty.join("none.txt"), "\n").unwrap();
    let run = Command::new("python3")
        .arg(repo.join("scripts/native_pkgconfig.py"))
        .arg(&empty)
        .arg("--private-libs-file")
        .arg(empty.join("none.txt"))
        .output()
        .unwrap();
    assert!(!run.status.success(), "an empty libs file was accepted");
    std::fs::remove_dir_all(&empty).unwrap();
}

#[test]
fn native_release_cross_builds_aarch64_linux_without_secrets() {
    let workflow = release();
    let native = job(&workflow, "native-libs");
    let cells = native["strategy"]["matrix"]["include"]
        .as_sequence()
        .unwrap();
    let triple = "aarch64-unknown-linux-gnu";
    let aarch64: Vec<&Value> = cells
        .iter()
        .filter(|cell| cell["target"].as_str() == Some(triple))
        .collect();
    assert_eq!(aarch64.len(), 1, "one aarch64 Linux cell");
    let cell = aarch64[0];
    assert_eq!(cell["os"].as_str(), Some("linux"));
    assert_eq!(cell["artifact"].as_str(), Some("anneal-core-linux-aarch64"));
    assert_eq!(cell["static_lib"].as_str(), Some("libanneal_core.a"));
    assert_eq!(cell["shared_lib"].as_str(), Some("libanneal_core.so"));

    // A hosted x86_64 runner, so no runner label that needs a plan, and no
    // secret anywhere in the job.
    assert_eq!(cell["runner"].as_str(), Some("ubuntu-latest"));
    let text = serde_yaml_ng::to_string(native).unwrap();
    assert!(!text.contains("secrets."), "the native job reads a secret");

    // The cell links with the cross compiler and runs under qemu-user,
    // against the cross libc. Every other cell links with the runner's cc
    // and runs directly.
    assert_eq!(cell["consumer_cc"].as_str(), Some("aarch64-linux-gnu-gcc"));
    assert_eq!(
        cell["emulator"].as_str(),
        Some("qemu-aarch64 -L /usr/aarch64-linux-gnu")
    );
    let names: BTreeSet<&str> = cells
        .iter()
        .map(|cell| cell["artifact"].as_str().unwrap())
        .collect();
    assert_eq!(names.len(), cells.len(), "artifact names are unique");
    for other in cells.iter().filter(|other| *other != cell) {
        assert_eq!(other["consumer_cc"].as_str(), Some("cc"), "{other:?}");
        assert_eq!(other["emulator"].as_str(), Some(""), "{other:?}");
    }

    // cargo and the cc crate are told which compiler builds that target.
    let upper = triple.to_uppercase().replace('-', "_");
    let lower = triple.replace('-', "_");
    let env = native["env"].as_mapping().expect("job env");
    assert_eq!(
        env[format!("CARGO_TARGET_{upper}_LINKER").as_str()].as_str(),
        cell["consumer_cc"].as_str()
    );
    assert_eq!(
        env[format!("CC_{lower}").as_str()].as_str(),
        cell["consumer_cc"].as_str()
    );

    // The toolchain is installed for that cell only, before anything builds.
    let (install_at, install) = step(native, "Install the aarch64 Linux cross toolchain");
    let (build_at, _) = step(native, "Build static + shared library");
    assert!(install_at < build_at);
    assert_eq!(
        install["if"].as_str(),
        Some("matrix.target == 'aarch64-unknown-linux-gnu'")
    );
    assert!(install.get("continue-on-error").is_none());
    let lines = script(install);
    assert_eq!(lines[0], "sudo apt-get update");
    let packages: BTreeSet<&str> = lines[1]
        .strip_prefix("sudo apt-get install -y --no-install-recommends ")
        .expect("an apt install")
        .split_whitespace()
        .collect();
    assert_eq!(
        packages,
        BTreeSet::from([
            "gcc-aarch64-linux-gnu",
            "libc6-dev-arm64-cross",
            "qemu-user"
        ])
    );
    // The rust target is added by the toolchain step for every cell.
    let (_, toolchain) = step(native, "Install Rust toolchain");
    assert_eq!(
        toolchain["with"]["targets"].as_str(),
        Some("${{ matrix.target }}")
    );

    // The release attaches every tarball the job uploads.
    let release_job = job(&workflow, "github-release");
    let uses = |prefix: &str| {
        steps(release_job)
            .iter()
            .find(|s| s["uses"].as_str().is_some_and(|u| u.starts_with(prefix)))
            .unwrap_or_else(|| panic!("github-release has no {prefix}"))
    };
    let download = uses("actions/download-artifact");
    assert_eq!(download["with"]["merge-multiple"].as_bool(), Some(true));
    let create = uses("softprops/action-gh-release");
    assert_eq!(
        create["with"]["files"].as_str().unwrap(),
        format!("{}*.tar.gz", download["with"]["path"].as_str().unwrap())
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
    // The syntax checks use the runner's own compiler, which parses the
    // header whatever the target. The links use the cell's compiler.
    let target_cc = "${{ matrix.consumer_cc }} ";
    let compiles: Vec<usize> = (0..lines.len())
        .filter(|&i| lines[i].starts_with("cc ") || lines[i].starts_with(target_cc))
        .collect();
    assert_eq!(compiles.len(), 4, "{lines:?}");
    let [syntax, pc_syntax, link, pc_link] = compiles[..] else {
        unreachable!()
    };
    let linux_only = at(r#"if [[ "${{ matrix.os }}" == linux ]]; then"#);
    let path = at(r#"export PKG_CONFIG_PATH="$PWD/consumer/lib/pkgconfig""#);
    let exists = at("pkg-config --exists anneal-core");
    let run = at("${{ matrix.emulator }} ./consumer/ffi_smoke");
    let version = at(
        r#"test "$(${{ matrix.emulator }} ./consumer/ffi_pkgconfig)" = "$(pkg-config --modversion anneal-core)""#,
    );
    assert!(extract < syntax && syntax < path && path < exists && exists < pc_syntax);
    assert!(pc_syntax < linux_only && linux_only < link && link < run && run < pc_link);
    assert!(pc_link < version);
    assert!(lines[syntax].starts_with("cc ") && lines[syntax].contains("-fsyntax-only"));
    assert!(lines[pc_syntax].starts_with("cc ") && lines[pc_syntax].contains("-fsyntax-only"));
    assert!(lines[link].starts_with(target_cc) && lines[pc_link].starts_with(target_cc));
    assert!(lines[link].contains("-Lconsumer/lib") && lines[link].contains("-lanneal_core"));
    assert!(lines[link].contains("-o consumer/ffi_smoke"));
    assert!(lines[pc_link].contains("-o consumer/ffi_pkgconfig"));
    assert!(!lines[pc_link].contains("-fsyntax-only"));

    // Where the cell needs pkg-config installed, that comes before it is used.
    let install = at("command -v pkg-config >/dev/null || brew install pkgconf");
    assert!(extract < install && install < path);
    assert_eq!(
        lines[install - 1],
        r#"if [[ "${{ matrix.os }}" == macos ]]; then"#
    );
    assert_eq!(lines[install + 1], "fi");

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

    // The pkg-config builds take every flag from the archive's .pc file:
    // no -I, -L or -l of their own, and the search path is the extracted
    // archive's alone.
    // The flags reach the compiler as quoted arrays that read fills from
    // pkg-config's output, never as an unquoted $(pkg-config ...), which
    // the shell would also glob. Each assignment stands alone, so a failing
    // pkg-config fails the step.
    let cflags_query = at("pc_cflags=$(pkg-config --cflags anneal-core)");
    let libs_query = at("pc_libs=$(pkg-config --libs anneal-core)");
    let cflags = at(r#"read -ra cflags <<< "$pc_cflags""#);
    let libs = at(r#"read -ra libs <<< "$pc_libs""#);
    assert!(exists < cflags_query && cflags_query < cflags && cflags < pc_syntax);
    assert!(exists < libs_query && libs_query < libs && libs < pc_link);
    for line in &lines {
        if line.contains("cflags=") || line.contains("libs=") || line.contains("-ra ") {
            assert!(
                [cflags_query, libs_query, cflags, libs]
                    .iter()
                    .any(|&i| lines[i] == *line),
                "{line} sets the flags again"
            );
        }
    }
    let cflags_words = r#""${cflags[@]}""#;
    let libs_words = r#""${libs[@]}""#;
    for compile in [pc_syntax, pc_link] {
        let words: Vec<&str> = lines[compile].split_whitespace().collect();
        assert!(words.contains(&"tests/ffi_smoke.c"), "{}", lines[compile]);
        assert!(words.contains(&cflags_words), "{}", lines[compile]);
        assert!(
            !lines[compile].contains("$(pkg-config"),
            "{}",
            lines[compile]
        );
        for word in &words {
            assert!(
                !word.starts_with("-I") && !word.starts_with("-L") && !word.starts_with("-l"),
                "{} adds {word} by hand",
                lines[compile]
            );
        }
    }
    assert!(!lines[pc_syntax].contains(libs_words));
    assert!(
        lines[pc_link]
            .split_whitespace()
            .any(|word| word == libs_words)
    );
    for line in &lines {
        for name in [
            "PKG_CONFIG_LIBDIR",
            "PKG_CONFIG_SYSROOT_DIR",
            "PKG_CONFIG_ALLOW",
        ] {
            assert!(!line.contains(name), "{line}");
        }
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

// The token every job gets reads the repository and nothing more. The
// GitHub release job alone may write to it, and PyPI's trusted publishing
// alone gets an OIDC token.
#[test]
fn only_the_github_release_job_may_write_to_the_repository() {
    let workflow = release();
    let permissions = |value: &Value| -> Vec<(String, String)> {
        value
            .as_mapping()
            .unwrap_or_else(|| panic!("permissions {value:?} is not a mapping"))
            .iter()
            .map(|(scope, access)| {
                (
                    scope.as_str().unwrap().to_owned(),
                    access.as_str().unwrap().to_owned(),
                )
            })
            .collect()
    };
    let pair = |scope: &str, access: &str| vec![(scope.to_owned(), access.to_owned())];

    assert_eq!(
        permissions(&workflow["permissions"]),
        pair("contents", "read")
    );
    for (name, job) in workflow["jobs"].as_mapping().expect("jobs") {
        let name = name.as_str().unwrap();
        let granted = job.get("permissions").map(permissions);
        let expected = match name {
            "github-release" => Some(pair("contents", "write")),
            "publish-pypi" => Some(pair("id-token", "write")),
            _ => None,
        };
        assert_eq!(granted, expected, "job {name} permissions");
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

// The feature suite runs these contract tests under every matrix Python,
// and some of them run the TOML-reading scripts with it. Python 3.10 has
// no tomllib, so the cell installs tomli, under the dev group's own
// marker, before either cargo test step.
#[test]
fn ci_installs_tomli_for_python_3_10_before_the_feature_tests() {
    let workflow = ci();
    let features = job(&workflow, "test");
    assert!(strings(&features["strategy"]["matrix"]["python-version"]).contains(&"3.10"));

    let python_at = first_use(features, "actions/setup-python");
    let (install_at, install) = step(features, "Install tomli (Python 3.10)");
    assert_binding(install);
    let pyproject: toml::Table = include_str!("../pyproject.toml").parse().unwrap();
    let dev = pyproject["dependency-groups"]["dev"]
        .as_array()
        .expect("the dev dependency group");
    let tomli = dev
        .iter()
        .filter_map(|item| item.as_str())
        .find(|item| item.starts_with("tomli"))
        .expect("the dev group names tomli");
    assert_eq!(tomli, "tomli>=2; python_version < '3.11'");
    assert_eq!(
        script(install),
        [format!(r#"python -m pip install "{tomli}""#)]
    );

    assert!(python_at < install_at);
    for name in [
        "cargo test (all satisfiable features)",
        "cargo test (all satisfiable features, macOS)",
    ] {
        let (test_at, _) = step(features, name);
        assert!(
            install_at < test_at,
            "{name} runs before tomli is installed"
        );
    }
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
