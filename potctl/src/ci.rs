//! CI helpers: repo preflight, macOS/pixi host-SDK env, meson/cmake/torch legs.
//!
//! GHA / Nickel own the job graph and matrices; thin steps call `potctl ci …`.

use std::env;
use std::fs;
use std::io::Write;
use std::path::{Path, PathBuf};
use std::process::{Command, Stdio};

use crate::lockstep;

/// PyTorch CPU wheel constraint used when forcing a non-CUDA install in CI.
pub const TORCH_CPU_SPEC: &str = "torch>=2.10,<2.11";
pub const TORCH_CPU_INDEX: &str = "https://download.pytorch.org/whl/cpu";
pub const METATENSOR_TORCH_SPEC: &str = "metatensor-torch>=0.8.4,<0.9";
pub const METATOMIC_TORCH_SPEC: &str = "metatomic-torch>=0.1.9,<0.2";
pub const VESIN_SPEC: &str = "vesin>=0.5.2,<0.6";
pub const VESIN_TORCH_SPEC: &str = "vesin-torch>=0.5.2,<0.6";

pub type Result<T> = std::result::Result<T, String>;

/// Which C++ build system the matrix leg uses (GHA `matrix.sys`).
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum BuildSys {
    Meson,
    Cmake,
}

impl BuildSys {
    pub fn parse(s: &str) -> Result<Self> {
        match s.to_ascii_lowercase().as_str() {
            "meson" => Ok(Self::Meson),
            "cmake" => Ok(Self::Cmake),
            other => Err(format!(
                "unknown build sys {other:?}; expected meson or cmake"
            )),
        }
    }
}

fn parse_bool_flag(s: &str, name: &str) -> Result<bool> {
    match s.to_ascii_lowercase().as_str() {
        "true" | "1" | "yes" | "on" => Ok(true),
        "false" | "0" | "no" | "off" => Ok(false),
        other => Err(format!(
            "invalid {name}={other:?}; expected true/false (or 1/0)"
        )),
    }
}

/// Apply darwin-env plan into this process (macOS only; no-op elsewhere).
pub fn apply_darwin_env_to_process(force: bool) -> Result<()> {
    let os = detect_runner_os();
    let prefix = env::var("CONDA_PREFIX").ok();
    let mut sdk = resolve_sdkroot_hint();
    if sdk.is_none() && (force || os == "macOS" || os == "Darwin") {
        sdk = try_xcrun_sdkroot();
    }
    let plan = if force {
        plan_darwin_env("macOS", prefix.as_deref(), sdk.as_deref())
    } else {
        plan_darwin_env(&os, prefix.as_deref(), sdk.as_deref())
    };
    let Some(p) = plan else {
        return Ok(());
    };
    for k in &p.unsets {
        env::remove_var(k);
    }
    for (k, v) in &p.exports {
        env::set_var(k, v);
    }
    env::set_var("POTCTL_MESON_EXTRA", &p.meson_extra);
    Ok(())
}

fn try_xcrun_sdkroot() -> Option<String> {
    let out = Command::new("xcrun")
        .args(["--show-sdk-path"])
        .output()
        .ok()?;
    if !out.status.success() {
        return None;
    }
    let s = String::from_utf8_lossy(&out.stdout).trim().to_string();
    if s.is_empty() {
        None
    } else {
        Some(s)
    }
}

/// Pure: meson configure argv (after `meson setup <builddir>`).
pub fn meson_configure_args(
    rpc: bool,
    cache: bool,
    with_mpi: bool,
    meson_extra: &str,
) -> Vec<String> {
    let mut args = vec![
        "-Dwith_tests=True".into(),
        "-Dwith_examples=False".into(),
        "-Dwith_xtensor=False".into(),
        "-Dwith_eigen=False".into(),
        format!("-Dwith_rpc={}", if rpc { "true" } else { "false" }),
        format!("-Dwith_cache={}", if cache { "true" } else { "false" }),
        "-Dpure_lib=False".into(),
    ];
    if with_mpi {
        args.push("-Dwith_mpi=enabled".into());
    }
    for tok in meson_extra.split_whitespace() {
        if !tok.is_empty() {
            args.push(tok.to_string());
        }
    }
    args
}

/// Pure: extra cmake -D / compiler flags for macOS (after darwin-env applied).
pub fn cmake_macos_extra_args(
    is_macos: bool,
    sdkroot: Option<&str>,
    conda_prefix: Option<&str>,
    fc: Option<&str>,
) -> Vec<String> {
    if !is_macos {
        return Vec::new();
    }
    let mut args = vec![
        "-DCMAKE_C_COMPILER=/usr/bin/clang".into(),
        "-DCMAKE_CXX_COMPILER=/usr/bin/clang++".into(),
    ];
    if let Some(sdk) = sdkroot.filter(|s| !s.is_empty()) {
        args.push(format!("-DCMAKE_OSX_SYSROOT={sdk}"));
    }
    if let Some(prefix) = conda_prefix.filter(|s| !s.is_empty()) {
        args.push(format!("-DCMAKE_PREFIX_PATH={prefix}"));
        args.push(format!("-DCMAKE_BUILD_RPATH={prefix}/lib"));
        args.push(format!("-DCMAKE_INSTALL_RPATH={prefix}/lib"));
    }
    if let Some(fc) = fc.filter(|s| !s.is_empty()) {
        args.push(format!("-DCMAKE_Fortran_COMPILER={fc}"));
    }
    args
}

/// Pure: core cmake configure flags (RGPOT_*), not including macOS extras.
pub fn cmake_configure_core_args(rpc: bool, cache: bool) -> Vec<String> {
    vec![
        "-DCMAKE_CXX_COMPILER_LAUNCHER=ccache".into(),
        "-DCMAKE_C_COMPILER_LAUNCHER=ccache".into(),
        "-DCMAKE_BUILD_TYPE=Release".into(),
        "-DRGPOT_BUILD_TESTS=True".into(),
        "-DRGPOT_BUILD_EXAMPLES=False".into(),
        "-DRGPOT_WITH_XTENSOR=OFF".into(),
        "-DRGPOT_WITH_EIGEN=OFF".into(),
        format!(
            "-DRGPOT_WITH_RPC={}",
            if rpc { "ON" } else { "OFF" }
        ),
        format!(
            "-DRGPOT_WITH_CACHE={}",
            if cache { "ON" } else { "OFF" }
        ),
        "-DRGPOT_PURE_LIB=OFF".into(),
    ]
}

/// Pure: `cmake --build` argv with an explicit job count. A bare `-j` lets
/// make start every ready compile at once, more than a 7 GB macOS runner
/// holds, and ignores CMAKE_BUILD_PARALLEL_LEVEL.
pub fn cmake_build_args(build_dir: &str, jobs: usize) -> Vec<String> {
    vec![
        "--build".into(),
        build_dir.into(),
        "--parallel".into(),
        jobs.max(1).to_string(),
    ]
}

/// Pure: the `cmake --build` job count, CMAKE_BUILD_PARALLEL_LEVEL when it
/// holds a positive count and the usable CPUs otherwise.
pub fn cmake_build_jobs(level: Option<&str>, cpus: usize) -> usize {
    level
        .and_then(|s| s.trim().parse::<usize>().ok())
        .filter(|&n| n > 0)
        .unwrap_or(cpus.max(1))
}

fn run_cmake_build(root: &Path, build_dir: &str) -> Result<()> {
    let level = env::var("CMAKE_BUILD_PARALLEL_LEVEL").ok();
    let cpus = std::thread::available_parallelism().map_or(1, |n| n.get());
    let args = cmake_build_args(build_dir, cmake_build_jobs(level.as_deref(), cpus));
    let refs: Vec<&str> = args.iter().map(String::as_str).collect();
    run_cmd("cmake", &refs, root)
}

fn run_cmd(bin: &str, args: &[&str], cwd: &Path) -> Result<()> {
    let status = Command::new(bin)
        .args(args)
        .current_dir(cwd)
        .stdin(Stdio::null())
        .status()
        .map_err(|e| format!("failed to spawn {bin}: {e}"))?;
    if status.success() {
        Ok(())
    } else {
        Err(format!(
            "{bin} {} failed with {status}",
            args.join(" ")
        ))
    }
}

fn maybe_ccache_stats() {
    let _ = Command::new("ccache").arg("-s").status();
}

/// Meson setup + compile + test (orchestrator `Build & Test (Meson)` leg).
pub fn run_meson_test(
    root: &Path,
    rpc: bool,
    cache: bool,
    build_dir: &str,
    apply_darwin: bool,
) -> Result<()> {
    if apply_darwin {
        apply_darwin_env_to_process(false)?;
    }
    let meson_extra = env::var("POTCTL_MESON_EXTRA").unwrap_or_default();
    let with_mpi = env::var("POTCTL_WITH_MPI").ok().as_deref() == Some("enabled");
    let mut setup_args: Vec<String> = vec!["setup".into(), build_dir.into()];
    setup_args.extend(meson_configure_args(rpc, cache, with_mpi, &meson_extra));
    let setup_refs: Vec<&str> = setup_args.iter().map(String::as_str).collect();
    run_cmd("meson", &setup_refs, root)?;
    run_cmd("meson", &["compile", "-C", build_dir], root)?;
    run_cmd(
        "meson",
        &["test", "-C", build_dir, "--print-errorlogs"],
        root,
    )?;
    maybe_ccache_stats();
    Ok(())
}

/// CMake configure + build + ctest (orchestrator `Build & Test (CMake)` leg).
pub fn run_cmake_test(
    root: &Path,
    rpc: bool,
    cache: bool,
    build_dir: &str,
    apply_darwin: bool,
) -> Result<()> {
    if apply_darwin {
        apply_darwin_env_to_process(false)?;
    }
    let os = detect_runner_os();
    let is_macos = os == "macOS" || os == "Darwin";
    let sdk = env::var("SDKROOT").ok();
    let conda = env::var("CONDA_PREFIX").ok();
    let fc = env::var("FC").ok();

    let mut args: Vec<String> = vec!["-S".into(), ".".into(), "-B".into(), build_dir.into()];
    args.extend(cmake_configure_core_args(rpc, cache));
    args.extend(cmake_macos_extra_args(
        is_macos,
        sdk.as_deref(),
        conda.as_deref(),
        fc.as_deref(),
    ));
    let cmake_refs: Vec<&str> = args.iter().map(String::as_str).collect();
    run_cmd("cmake", &cmake_refs, root)?;
    run_cmake_build(root, build_dir)?;
    run_cmd(
        "ctest",
        &["--test-dir", build_dir, "--output-on-failure"],
        root,
    )?;
    maybe_ccache_stats();
    Ok(())
}

/// `potctl ci build-test --sys meson|cmake --rpc … --cache …`
pub fn run_build_test(
    root: &Path,
    sys: &str,
    rpc: &str,
    cache: &str,
    build_dir: Option<&str>,
    no_darwin: bool,
) -> Result<()> {
    let sys = BuildSys::parse(sys)?;
    let rpc = parse_bool_flag(rpc, "rpc")?;
    let cache = parse_bool_flag(cache, "cache")?;
    let apply_darwin = !no_darwin;
    match sys {
        BuildSys::Meson => {
            let dir = build_dir.unwrap_or("bbdir");
            run_meson_test(root, rpc, cache, dir, apply_darwin)
        }
        BuildSys::Cmake => {
            let dir = build_dir.unwrap_or("build");
            run_cmake_test(root, rpc, cache, dir, apply_darwin)
        }
    }
}

/// Paths that must exist for a healthy rgpot checkout (build/release CI).
pub fn required_paths(root: &Path) -> Vec<PathBuf> {
    vec![
        root.join("meson.build"),
        root.join("CMakeLists.txt"),
        root.join("pixi.toml"),
        root.join("rgpot-core").join("Cargo.toml"),
        root.join("potctl").join("Cargo.toml"),
        root.join("towncrier.toml"),
        root.join("pyproject.toml"),
        root.join("cog.toml"),
    ]
}

pub fn preflight(root: &Path, lockstep_check: bool) -> Result<()> {
    let mut missing = Vec::new();
    for p in required_paths(root) {
        if !p.is_file() {
            missing.push(p.display().to_string());
        }
    }
    if !missing.is_empty() {
        return Err(format!("missing required files:\n  {}", missing.join("\n  ")));
    }

    if lockstep_check {
        lockstep::assert_lockstep(root, None, false)?;
    } else {
        // Still surface versions for logs without failing on drift (rare).
        let version = lockstep::read_workspace_version(root).unwrap_or_else(|_| "?".into());
        println!("preflight: release version = {version} (lockstep not enforced)");
    }

    println!("preflight ok ({})", root.display());
    Ok(())
}

/// `potctl ci release-assert` — lockstep + cog.toml parse + cog --version (release-prepare leg).
pub fn run_release_assert(root: &Path) -> Result<()> {
    lockstep::assert_lockstep(root, None, false)?;
    let cog_toml = root.join("cog.toml");
    let text = std::fs::read_to_string(&cog_toml)
        .map_err(|e| format!("read cog.toml: {e}"))?;
    // Minimal TOML presence check (full schema is cog's concern).
    if !text.contains('[') {
        return Err("cog.toml looks empty or invalid".into());
    }
    println!("cog.toml parse ok (read {} bytes)", text.len());
    let status = Command::new("cog")
        .arg("--version")
        .current_dir(root)
        .status()
        .map_err(|e| format!("spawn cog: {e}"))?;
    if !status.success() {
        return Err(format!("cog --version failed with {status}"));
    }
    Ok(())
}

/// How we invoke towncrier once installed (direct binary vs `python -m towncrier`).
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum TowncrierInvoke {
    Bin,
    PythonModule,
}

/// Pure: prefer direct `towncrier` on PATH, else python -m if pip installed it.
pub fn select_towncrier_invoke(has_towncrier_bin: bool, has_python: bool) -> Option<TowncrierInvoke> {
    if has_towncrier_bin {
        Some(TowncrierInvoke::Bin)
    } else if has_python {
        Some(TowncrierInvoke::PythonModule)
    } else {
        None
    }
}

/// Install/locate towncrier for CI runners that only have potctl (no uvx/pipx preinstalled).
/// Order: existing `towncrier` → `pipx install` → `python3 -m pip install --user` → `uvx` last.
fn ensure_towncrier_available() -> Result<TowncrierInvoke> {
    if which_bin("towncrier").is_some() {
        return Ok(TowncrierInvoke::Bin);
    }
    if which_bin("pipx").is_some() {
        println!("potctl: installing towncrier via pipx");
        let st = Command::new("pipx")
            .args(["install", "towncrier"])
            .status()
            .map_err(|e| format!("pipx install towncrier: {e}"))?;
        if st.success() && which_bin("towncrier").is_some() {
            return Ok(TowncrierInvoke::Bin);
        }
        // pipx may install but PATH not refreshed in same process; try common pipx bin dir.
        if let Ok(home) = env::var("HOME") {
            let cand = PathBuf::from(home).join(".local/bin/towncrier");
            if cand.is_file() {
                return Ok(TowncrierInvoke::Bin);
            }
        }
    }
    let py = which_bin("python3").or_else(|| which_bin("python"));
    if let Some(ref py) = py {
        println!("potctl: installing towncrier via {} -m pip --user", py.display());
        let _ = Command::new(py)
            .args(["-m", "ensurepip", "--upgrade"])
            .status();
        let st = Command::new(py)
            .args(["-m", "pip", "install", "--user", "--quiet", "towncrier"])
            .status()
            .map_err(|e| format!("pip install towncrier: {e}"))?;
        if st.success() {
            if which_bin("towncrier").is_some() {
                return Ok(TowncrierInvoke::Bin);
            }
            // scripts may land in ~/.local/bin not on PATH
            if let Ok(home) = env::var("HOME") {
                let cand = PathBuf::from(&home).join(".local/bin/towncrier");
                if cand.is_file() {
                    // Prepend to PATH for subsequent Command::new("towncrier")
                    let local = PathBuf::from(&home).join(".local/bin");
                    let mut paths = vec![local];
                    if let Ok(cur) = env::var("PATH") {
                        for p in env::split_paths(&cur) {
                            paths.push(p);
                        }
                    }
                    if let Ok(joined) = env::join_paths(paths) {
                        env::set_var("PATH", joined);
                    }
                    if which_bin("towncrier").is_some() {
                        return Ok(TowncrierInvoke::Bin);
                    }
                }
            }
            return Ok(TowncrierInvoke::PythonModule);
        }
    }
    if which_bin("uvx").is_some() {
        // uvx runs without install; treated as module-like at call site
        return Ok(TowncrierInvoke::PythonModule);
    }
    Err(
        "towncrier not available: install with `pipx install towncrier` or `pip install --user towncrier` (or provide uvx)"
            .into(),
    )
}

fn run_towncrier_argv(root: &Path, invoke: TowncrierInvoke, args: &[&str]) -> Result<()> {
    let status = match invoke {
        TowncrierInvoke::Bin => {
            let mut cmd = Command::new("towncrier");
            cmd.args(args).current_dir(root);
            cmd.status()
                .map_err(|e| format!("spawn towncrier: {e}"))?
        }
        TowncrierInvoke::PythonModule => {
            // Prefer python -m towncrier; if only uvx works, use that as last resort.
            if let Some(py) = which_bin("python3").or_else(|| which_bin("python")) {
                let mut cmd = Command::new(py);
                cmd.arg("-m").arg("towncrier").args(args).current_dir(root);
                let st = cmd
                    .status()
                    .map_err(|e| format!("spawn python -m towncrier: {e}"))?;
                if st.success() || which_bin("uvx").is_none() {
                    if st.success() {
                        return Ok(());
                    }
                    return Err(format!("python -m towncrier {} failed with {st}", args.join(" ")));
                }
            }
            if which_bin("uvx").is_some() {
                let mut cmd = Command::new("uvx");
                cmd.arg("towncrier").args(args).current_dir(root);
                let st = cmd
                    .status()
                    .map_err(|e| format!("spawn uvx towncrier: {e}"))?;
                if st.success() {
                    return Ok(());
                }
                return Err(format!("uvx towncrier {} failed with {st}", args.join(" ")));
            }
            return Err("no python/uvx for towncrier module invoke".into());
        }
    };
    if status.success() {
        Ok(())
    } else {
        Err(format!("towncrier {} failed with {status}", args.join(" ")))
    }
}

/// `potctl ci towncrier-draft` — towncrier build --draft if any newsfragments exist.
pub fn run_towncrier_draft(root: &Path) -> Result<()> {
    let frag_dir = root.join("docs/newsfragments");
    let has_frags = frag_dir.is_dir()
        && std::fs::read_dir(&frag_dir)
            .map(|rd| {
                rd.filter_map(|e| e.ok()).any(|e| {
                    let n = e.file_name();
                    let s = n.to_string_lossy();
                    s.ends_with(".md")
                })
            })
            .unwrap_or(false);
    if !has_frags {
        println!("No newsfragments; skip draft (ok if this PR only touches tooling)");
        return Ok(());
    }
    let invoke = ensure_towncrier_available()?;
    run_towncrier_argv(root, invoke, &["build", "--draft"])
}

/// `potctl ci cog-bump-dry-run` — PR tip (workspace version) or workflow_dispatch bump kind.
pub fn run_cog_bump_dry_run(root: &Path, bump: Option<&str>) -> Result<()> {
    let mut cmd = Command::new("cog");
    cmd.current_dir(root);
    match bump {
        None | Some("") => {
            let ver = lockstep::read_workspace_version(root)?;
            cmd.args(["bump", "--version", &ver, "--dry-run"]);
            let output = cmd
                .output()
                .map_err(|e| format!("spawn cog bump: {e}"))?;
            let _ = std::io::Write::write_all(&mut std::io::stdout(), &output.stdout);
            let _ = std::io::Write::write_all(&mut std::io::stderr(), &output.stderr);
            if !output.status.success() {
                println!(
                    "note: cog dry-run non-zero (often dirty/unpushed state on PR); lockstep + cog.toml ok above"
                );
            }
            Ok(())
        }
        Some("auto") => {
            run_cmd_status_ok("cog", &["bump", "--auto", "--dry-run"], root)
        }
        Some("patch") => {
            run_cmd_status_ok("cog", &["bump", "--patch", "--dry-run"], root)
        }
        Some("minor") => {
            run_cmd_status_ok("cog", &["bump", "--minor", "--dry-run"], root)
        }
        Some("major") => {
            run_cmd_status_ok("cog", &["bump", "--major", "--dry-run"], root)
        }
        Some(other) => Err(format!(
            "unknown cog bump kind {other:?}; expected auto|patch|minor|major or omit for the workspace version"
        )),
    }
}

fn run_cmd_status_ok(bin: &str, args: &[&str], cwd: &Path) -> Result<()> {
    let status = Command::new(bin)
        .args(args)
        .current_dir(cwd)
        .status()
        .map_err(|e| format!("failed to spawn {bin}: {e}"))?;
    if status.success() {
        Ok(())
    } else {
        Err(format!("{bin} {} failed with {status}", args.join(" ")))
    }
}

/// True when `help` lists `flag` as its own token.
///
/// Cog 7's `cog check --help` documents `--from-latest-tag`, which is not
/// a `--from` option. A substring search treats that boolean as `--from`
/// and then invokes a flag the binary rejects.
pub fn cog_help_has_flag(help: &str, flag: &str) -> bool {
    help.split(|c: char| {
        c.is_whitespace()
            || matches!(
                c,
                ',' | '[' | ']' | '(' | ')' | '|' | '=' | '{' | '}'
            )
    })
    .any(|tok| tok == flag)
}

/// Arguments for `cog check` over `base..HEAD` on the installed cog.
///
/// Cog 7 takes a positional range and `--ignore-merge-commits`.
/// `--from-latest-tag` is a boolean in the same clap group as the range,
/// so it is not passed together with `base..HEAD`.
pub fn cog_check_argv(help: &str, base: &str) -> Vec<String> {
    let mut args = vec!["check".to_string()];
    if cog_help_has_flag(help, "--ignore-merge-commits") {
        args.push("--ignore-merge-commits".to_string());
    }
    if cog_help_has_flag(help, "--from") {
        args.push("--from".to_string());
        args.push(base.to_string());
    } else {
        args.push(format!("{base}..HEAD"));
    }
    args
}

/// Git merge subjects, plus this repo's single-parent `merge:` sync commits.
///
/// `ignore_merge_commits` in cog 7 skips commits with more than one parent.
/// Develop syncs such as `merge: take the 3.5.1 release from main` have one
/// parent, so cog still reports `Commit type merge not allowed`.
pub fn is_ignored_merge_commit_subject(subj: &str) -> bool {
    let s = subj.trim();
    let lower = s.to_ascii_lowercase();
    lower.starts_with("merge ")
        || lower.starts_with("merge:")
        || lower.starts_with("merge(")
        || lower.starts_with("pull request")
}

/// Cog's failure text lists only `Commit type merge not allowed` errors.
///
/// Cog prefixes the report with a bare `Error:` line. That header is not a
/// commit-type failure.
pub fn cog_failure_is_only_merge_types(report: &str) -> bool {
    let mut saw_error = false;
    for line in report.lines() {
        let Some(rest) = line.trim().strip_prefix("Error:") else {
            continue;
        };
        let rest = rest.trim();
        if rest.is_empty() {
            continue;
        }
        saw_error = true;
        if !rest.starts_with("Commit type `merge` not allowed") {
            return false;
        }
    }
    saw_error
}

/// Conventional-commit subject prefixes accepted by release-prepare cog-check fallback.
pub fn is_conventional_commit_subject(subj: &str) -> bool {
    let s = subj.trim();
    if s.is_empty() {
        return false;
    }
    if is_ignored_merge_commit_subject(s) {
        return true;
    }
    // type(scope): or type:
    let types = [
        "feat", "fix", "perf", "bench", "refactor", "docs", "test", "build", "ci", "chore",
        "style", "revert", "enh", "bug", "maint", "doc", "tst", "bld", "gen", "simd", "pot",
        "python", "nlist",
    ];
    for t in types {
        if s.starts_with(&format!("{t}(")) || s.starts_with(&format!("{t}:")) {
            return true;
        }
    }
    false
}

/// Resolve active Python interpreter (PATH `python` / `python3`).
fn python_bin() -> Result<PathBuf> {
    which_bin("python")
        .or_else(|| which_bin("python3"))
        .ok_or_else(|| "python not on PATH (run under pixi shell)".into())
}

fn which_bin(name: &str) -> Option<PathBuf> {
    let path = env::var_os("PATH")?;
    for dir in env::split_paths(&path) {
        let cand = dir.join(name);
        if cand.is_file() {
            return Some(cand);
        }
    }
    None
}

fn py_eval(py: &Path, code: &str) -> Result<String> {
    let out = Command::new(py)
        .args(["-c", code])
        .output()
        .map_err(|e| format!("python -c: {e}"))?;
    if !out.status.success() {
        return Err(format!(
            "python -c failed: {}",
            String::from_utf8_lossy(&out.stderr)
        ));
    }
    Ok(String::from_utf8_lossy(&out.stdout).trim().to_string())
}

/// Pure: torch X.Y major.minor from a full version string (strip `+cu*` suffix first).
pub fn torch_mm_from_version(ver: &str) -> Option<String> {
    let base = ver.split('+').next().unwrap_or(ver);
    let parts: Vec<&str> = base.split('.').collect();
    if parts.len() >= 2 {
        Some(format!("{}.{}", parts[0], parts[1]))
    } else {
        None
    }
}

/// Pure: required purelib paths for metatomic C++ CI layout.
pub fn metatomic_layout_paths(purelib: &Path, torch_mm: &str) -> Vec<PathBuf> {
    vec![
        purelib.join("torch/lib/libtorch_global_deps.so"),
        purelib.join("torch/share/cmake"),
        purelib.join("metatensor/include"),
        purelib.join(format!("metatensor/torch/torch-{torch_mm}/include")),
        purelib.join(format!("metatomic/torch/torch-{torch_mm}/include")),
        purelib.join("vesin/include/vesin.h"),
        purelib.join("vesin/lib"),
    ]
}

fn pip_install(py: &Path, args: &[&str]) -> Result<()> {
    if which_bin("uv").is_some() {
        let mut cmd = Command::new("uv");
        cmd.args(["pip", "install", "--python"])
            .arg(py)
            .arg("--no-cache");
        for a in args {
            cmd.arg(a);
        }
        let st = cmd.status().map_err(|e| format!("uv pip: {e}"))?;
        if st.success() {
            return Ok(());
        }
        return Err(format!("uv pip install failed with {st}"));
    }
    let _ = Command::new(py)
        .args(["-m", "ensurepip", "--upgrade"])
        .status();
    let mut cmd = Command::new(py);
    cmd.args(["-m", "pip", "install", "--no-cache-dir"]);
    for a in args {
        cmd.arg(a);
    }
    let st = cmd.status().map_err(|e| format!("pip: {e}"))?;
    if st.success() {
        Ok(())
    } else {
        Err(format!("pip install failed with {st}"))
    }
}

fn reinstall_torch_cpu(py: &Path) -> Result<()> {
    println!("Reinstalling CPU torch wheel (libtorch missing or CUDA build detected)");
    if which_bin("uv").is_some() {
        let st = Command::new("uv")
            .args([
                "pip",
                "install",
                "--python",
            ])
            .arg(py)
            .args([
                "--reinstall",
                "--no-cache",
                "--index-url",
                TORCH_CPU_INDEX,
                TORCH_CPU_SPEC,
            ])
            .status()
            .map_err(|e| format!("uv pip torch: {e}"))?;
        if st.success() {
            return Ok(());
        }
        return Err(format!("uv pip reinstall torch failed with {st}"));
    }
    let _ = Command::new(py)
        .args(["-m", "ensurepip", "--upgrade"])
        .status();
    let st = Command::new(py)
        .args([
            "-m",
            "pip",
            "install",
            "--force-reinstall",
            "--no-cache-dir",
            "--index-url",
            TORCH_CPU_INDEX,
            TORCH_CPU_SPEC,
        ])
        .status()
        .map_err(|e| format!("pip torch: {e}"))?;
    if st.success() {
        Ok(())
    } else {
        Err(format!("pip reinstall torch failed with {st}"))
    }
}

fn read_torch_version_file(purelib: &Path) -> Option<String> {
    let vf = purelib.join("torch/version.py");
    let text = fs::read_to_string(vf).ok()?;
    for line in text.lines() {
        let t = line.trim();
        if let Some(rest) = t.strip_prefix("__version__") {
            let rest = rest.trim().trim_start_matches('=').trim();
            let q = rest.trim_matches(|c| c == '"' || c == '\'');
            if !q.is_empty() && q != rest.trim() || rest.starts_with('"') || rest.starts_with('\'') {
                let s = rest.trim_matches(|c| c == '"' || c == '\'');
                if !s.is_empty() {
                    return Some(s.to_string());
                }
            }
        }
    }
    // simpler: regex-free scan
    if let Some(i) = text.find("__version__") {
        let slice = &text[i..];
        if let Some(eq) = slice.find('=') {
            let v = slice[eq + 1..].trim();
            let end = v.find('\n').unwrap_or(v.len());
            let tok = v[..end].trim().trim_matches(|c| c == '"' || c == '\'');
            if !tok.is_empty() {
                return Some(tok.to_string());
            }
        }
    }
    None
}

fn append_github_env(lines: &[(&str, String)]) -> Result<()> {
    let Some(path) = env::var_os("GITHUB_ENV") else {
        for (k, v) in lines {
            println!("export {k}={v}");
        }
        return Ok(());
    };
    let mut f = fs::OpenOptions::new()
        .create(true)
        .append(true)
        .open(&path)
        .map_err(|e| format!("GITHUB_ENV open: {e}"))?;
    for (k, v) in lines {
        writeln!(f, "{k}={v}").map_err(|e| format!("GITHUB_ENV write: {e}"))?;
    }
    Ok(())
}

/// `potctl ci ensure-torch-metatomic` — CPU torch + metatensor/metatomic/vesin purelib layout.
/// Writes TORCH_LIB / TORCH_CMAKE / PURELIB / LD_LIBRARY_PATH / CMAKE_PREFIX_PATH to GITHUB_ENV when set.
pub fn run_ensure_torch_metatomic(_root: &Path) -> Result<()> {
    let py = python_bin()?;
    let purelib_s = py_eval(&py, "import sysconfig; print(sysconfig.get_paths()['purelib'])")?;
    let mut purelib = PathBuf::from(&purelib_s);

    let torch_ver = read_torch_version_file(&purelib).unwrap_or_default();
    let libtorch = purelib.join("torch/lib/libtorch_global_deps.so");
    let need_cpu = !libtorch.is_file()
        || torch_ver.contains("+cu")
        || torch_ver.contains("+cuda");
    if need_cpu {
        println!(
            "Forcing CPU torch (current version: '{}')",
            if torch_ver.is_empty() {
                "missing"
            } else {
                &torch_ver
            }
        );
        reinstall_torch_cpu(&py)?;
        purelib = PathBuf::from(py_eval(
            &py,
            "import sysconfig; print(sysconfig.get_paths()['purelib'])",
        )?);
    }

    let mut need: Vec<&str> = Vec::new();
    if !purelib.join("metatensor/include").is_dir() {
        need.push(METATENSOR_TORCH_SPEC);
    }
    if !purelib.join("metatomic/torch").is_dir() {
        need.push(METATOMIC_TORCH_SPEC);
    }
    if !purelib.join("vesin/include/vesin.h").is_file() {
        need.push(VESIN_SPEC);
        need.push(VESIN_TORCH_SPEC);
    }
    if need.is_empty() {
        println!("metatomic/metatensor/vesin purelib trees already present");
    } else {
        println!("Installing missing wheels: {}", need.join(" "));
        let refs: Vec<&str> = need.to_vec();
        pip_install(&py, &refs)?;
        purelib = PathBuf::from(py_eval(
            &py,
            "import sysconfig; print(sysconfig.get_paths()['purelib'])",
        )?);
    }

    let torch_ver = read_torch_version_file(&purelib)
        .ok_or_else(|| format!("torch version not found under {}", purelib.display()))?;
    let tver = torch_mm_from_version(&torch_ver)
        .ok_or_else(|| format!("could not parse torch major.minor from {torch_ver:?}"))?;
    let required = metatomic_layout_paths(&purelib, &tver);
    let missing: Vec<String> = required
        .iter()
        .filter(|p| !p.exists())
        .map(|p| p.display().to_string())
        .collect();
    if !missing.is_empty() {
        eprintln!("missing C++ layout paths:");
        for m in &missing {
            eprintln!("  {m}");
        }
        return Err("metatomic C++ purelib layout incomplete".into());
    }
    let torch_share = purelib.join("torch/share/cmake");
    println!(
        "C++ metatomic layout ok; torch {tver} python {}",
        py_eval(&py, "import sys; print(sys.version.split()[0])").unwrap_or_default()
    );
    println!("torch cmake {}", torch_share.display());
    println!("vesin.h {}", purelib.join("vesin/include/vesin.h").display());

    let torch_lib = purelib.join("torch/lib");
    let ld = match env::var("LD_LIBRARY_PATH") {
        Ok(cur) if !cur.is_empty() => format!("{}:{cur}", torch_lib.display()),
        _ => torch_lib.display().to_string(),
    };
    let cmake_pp = match env::var("CMAKE_PREFIX_PATH") {
        Ok(cur) if !cur.is_empty() => format!("{}:{cur}", torch_share.display()),
        _ => torch_share.display().to_string(),
    };
    append_github_env(&[
        ("TORCH_LIB", torch_lib.display().to_string()),
        ("TORCH_CMAKE", torch_share.display().to_string()),
        ("PURELIB", purelib.display().to_string()),
        ("LD_LIBRARY_PATH", ld),
        ("CMAKE_PREFIX_PATH", cmake_pp),
    ])?;
    Ok(())
}

fn apply_torch_env_from_github_or_purelib() -> Result<()> {
    if env::var_os("TORCH_LIB").is_some() && env::var_os("CMAKE_PREFIX_PATH").is_some() {
        return Ok(());
    }
    let py = python_bin()?;
    let purelib = PathBuf::from(py_eval(
        &py,
        "import sysconfig; print(sysconfig.get_paths()['purelib'])",
    )?);
    let torch_lib = purelib.join("torch/lib");
    let torch_cmake = purelib.join("torch/share/cmake");
    if env::var_os("TORCH_LIB").is_none() {
        env::set_var("TORCH_LIB", &torch_lib);
    }
    if env::var_os("TORCH_CMAKE").is_none() {
        env::set_var("TORCH_CMAKE", &torch_cmake);
    }
    let ld = match env::var("LD_LIBRARY_PATH") {
        Ok(cur) if !cur.is_empty() => format!("{}:{cur}", torch_lib.display()),
        _ => torch_lib.display().to_string(),
    };
    env::set_var("LD_LIBRARY_PATH", &ld);
    let cmake_pp = match env::var("CMAKE_PREFIX_PATH") {
        Ok(cur) if !cur.is_empty() => format!("{}:{cur}", torch_cmake.display()),
        _ => torch_cmake.display().to_string(),
    };
    env::set_var("CMAKE_PREFIX_PATH", &cmake_pp);
    println!("CMAKE_PREFIX_PATH={cmake_pp}");
    println!("LD_LIBRARY_PATH={ld}");
    Ok(())
}

/// Meson setup argv of the metatomic leg. With `rpc` the rpc stack is on, which
/// builds every engine plugin (UMA, Metatomic, XTB and the classical, MOPAC and
/// NWChem ones) and their tests.
pub fn metatomic_meson_setup_args(build_dir: &str, rpc: bool) -> Vec<String> {
    let mut args: Vec<String> = vec![
        "setup".into(),
        build_dir.into(),
        "-Dwith_tests=true".into(),
        "-Dwith_examples=false".into(),
        "-Dwith_metatomic=true".into(),
        format!("-Dwith_rpc={rpc}"),
        "-Dwith_cache=false".into(),
        "--buildtype=debug".into(),
    ];
    if rpc {
        // The engine leg builds every engine plugin, xtb's and Skala's
        // included, and the Skala tests.
        args.push("-Dwith_xtb=true".into());
        args.push("-Dwith_skala=true".into());
    }
    args
}

/// `potctl ci metatomic-test` — meson setup/compile/test with metatomic+vesin (orchestrator leg).
pub fn run_metatomic_test(root: &Path, build_dir: &str, rpc: bool) -> Result<()> {
    apply_torch_env_from_github_or_purelib()?;
    let setup = metatomic_meson_setup_args(build_dir, rpc);
    let setup: Vec<&str> = setup.iter().map(String::as_str).collect();
    run_cmd("meson", &setup, root)?;
    run_cmd("meson", &["compile", "-C", build_dir], root)?;
    run_cmd(
        "meson",
        &["test", "-C", build_dir, "--print-errorlogs"],
        root,
    )?;
    maybe_ccache_stats();
    Ok(())
}

/// Plugin ABI tests that must exist and pass in the plugin leg.
pub const PLUGIN_TESTS: [&str; 3] = ["PluginTest", "PluginContractTest", "ExternalPluginTest"];

/// Meson setup argv for the plugin leg: tests on, plugin host built (the
/// default; only `with_rpc_client_only` drops it), no optional engines.
pub fn plugin_meson_setup_args(build_dir: &str) -> Vec<String> {
    vec![
        "setup".into(),
        build_dir.into(),
        "-Dwith_tests=true".into(),
        "-Dwith_examples=false".into(),
        "-Dwith_rpc=false".into(),
        "-Dwith_rpc_client_only=false".into(),
        "-Dwith_cache=false".into(),
        "--buildtype=debug".into(),
    ]
}

/// Meson test argv naming the plugin tests; meson errors on an unknown name,
/// so a test dropped from the build fails the leg instead of vanishing.
pub fn plugin_meson_test_args(build_dir: &str) -> Vec<String> {
    let mut args: Vec<String> = vec![
        "test".into(),
        "-C".into(),
        build_dir.into(),
        "--print-errorlogs".into(),
    ];
    args.extend(PLUGIN_TESTS.iter().map(|t| t.to_string()));
    args
}

/// `potctl ci plugin-test` — meson build with the plugin host, then the plugin suite.
pub fn run_plugin_test(root: &Path, build_dir: &str, apply_darwin: bool) -> Result<()> {
    if apply_darwin {
        apply_darwin_env_to_process(false)?;
    }
    let setup = plugin_meson_setup_args(build_dir);
    let setup_refs: Vec<&str> = setup.iter().map(String::as_str).collect();
    run_cmd("meson", &setup_refs, root)?;
    run_cmd("meson", &["compile", "-C", build_dir], root)?;
    let named = plugin_meson_test_args(build_dir);
    let named_refs: Vec<&str> = named.iter().map(String::as_str).collect();
    run_cmd("meson", &named_refs, root)?;
    run_cmd(
        "meson",
        &["test", "-C", build_dir, "--print-errorlogs", "--suite", "plugin"],
        root,
    )?;
    maybe_ccache_stats();
    Ok(())
}

/// `potctl ci xtb-tblite-test` — meson setup/compile/test with xtb+tblite.
pub fn run_xtb_tblite_test(root: &Path, build_dir: &str) -> Result<()> {
    let setup = [
        "setup",
        build_dir,
        "-Dwith_tests=true",
        "-Dwith_examples=false",
        "-Dwith_xtb=true",
        "-Dwith_tblite=true",
        "-Dwith_rpc=false",
        "-Dwith_cache=false",
        "--buildtype=debug",
    ];
    run_cmd("meson", &setup, root)?;
    run_cmd("meson", &["compile", "-C", build_dir], root)?;
    run_cmd(
        "meson",
        &["test", "-C", build_dir, "--print-errorlogs"],
        root,
    )?;
    maybe_ccache_stats();
    Ok(())
}

/// Pure: meson argv for the XcKernel golden leg (`-Dwith_xckernel=true`).
pub fn xckernel_meson_setup_args(build_dir: &str) -> Vec<String> {
    vec![
        "setup".into(),
        build_dir.into(),
        "-Dwith_tests=true".into(),
        "-Dwith_examples=false".into(),
        "-Dwith_xckernel=true".into(),
        "-Dwith_rpc=false".into(),
        "-Dwith_cache=false".into(),
        "-Dwith_xtensor=false".into(),
        "-Dwith_eigen=false".into(),
        "--buildtype=debug".into(),
    ]
}

fn command_stdout(bin: &str, args: &[&str], cwd: &Path) -> Result<String> {
    let out = Command::new(bin)
        .args(args)
        .current_dir(cwd)
        .stdin(Stdio::null())
        .output()
        .map_err(|e| format!("failed to spawn {bin}: {e}"))?;
    if !out.status.success() {
        return Err(format!(
            "{bin} {} failed with {}",
            args.join(" "),
            out.status
        ));
    }
    Ok(String::from_utf8_lossy(&out.stdout).into_owned())
}

/// `potctl ci xckernel-test` — meson setup/compile and the xckernel suite.
///
/// The default matrix leaves `with_xckernel` off, so `XcKernelGoldens` never
/// runs there. This leg fails if that test is not registered.
pub fn run_xckernel_test(root: &Path, build_dir: &str) -> Result<()> {
    let setup = xckernel_meson_setup_args(build_dir);
    let refs: Vec<&str> = setup.iter().map(String::as_str).collect();
    run_cmd("meson", &refs, root)?;
    run_cmd("meson", &["compile", "-C", build_dir], root)?;
    run_cmd(
        "meson",
        &[
            "test",
            "-C",
            build_dir,
            "--suite",
            "xckernel",
            "--print-errorlogs",
        ],
        root,
    )?;
    let listed = command_stdout(
        "meson",
        &["introspect", "--tests", build_dir],
        root,
    )?;
    if !listed.contains("XcKernelGoldens") {
        return Err(
            "meson tests are missing XcKernelGoldens; -Dwith_xckernel did not register the golden suite"
                .into(),
        );
    }
    maybe_ccache_stats();
    Ok(())
}

/// `potctl ci towncrier-check --compare-with SHA`
pub fn run_towncrier_check(root: &Path, compare_with: &str) -> Result<()> {
    if compare_with.is_empty() {
        return Err("towncrier-check requires --compare-with <sha>".into());
    }
    let invoke = ensure_towncrier_available()?;
    run_towncrier_argv(root, invoke, &["check", "--compare-with", compare_with])
}

/// Pure: meson argv for RPC server build (client_bridge_stress).
pub fn bridge_server_meson_setup_args(build_dir: &str) -> Vec<String> {
    vec![
        "setup".into(),
        build_dir.into(),
        "-Dwith_rpc=true".into(),
        "-Dwith_tests=false".into(),
        "-Dwith_examples=false".into(),
    ]
}

/// Pure: cmake argv for RPC client-only configure.
pub fn bridge_client_cmake_configure_args(build_dir: &str) -> Vec<String> {
    vec![
        "-S".into(),
        ".".into(),
        "-B".into(),
        build_dir.into(),
        "-DCMAKE_CXX_COMPILER_LAUNCHER=ccache".into(),
        "-DCMAKE_C_COMPILER_LAUNCHER=ccache".into(),
        "-DCMAKE_BUILD_TYPE=Release".into(),
        "-DRGPOT_RPC_CLIENT_ONLY=ON".into(),
        "-DRGPOT_BUILD_TESTS=ON".into(),
    ]
}

/// Pure: default potserv path under meson server build dir.
pub fn bridge_potserv_path(server_build_dir: &str) -> PathBuf {
    PathBuf::from(server_build_dir)
        .join("CppCore/potserv")
}

/// `potctl ci bridge-server-build` — meson RPC server (no tests).
pub fn run_bridge_server_build(root: &Path, build_dir: &str) -> Result<()> {
    let setup = bridge_server_meson_setup_args(build_dir);
    let refs: Vec<&str> = setup.iter().map(String::as_str).collect();
    run_cmd("meson", &refs, root)?;
    run_cmd("meson", &["compile", "-C", build_dir], root)?;
    Ok(())
}

/// `potctl ci bridge-client-build` — cmake RPC client-only + tests target.
pub fn run_bridge_client_build(root: &Path, build_dir: &str) -> Result<()> {
    let cfg = bridge_client_cmake_configure_args(build_dir);
    let refs: Vec<&str> = cfg.iter().map(String::as_str).collect();
    run_cmd("cmake", &refs, root)?;
    run_cmake_build(root, build_dir)?;
    Ok(())
}

/// `potctl ci bridge-stress` — start potserv, ctest client, kill potserv.
pub fn run_bridge_stress(
    root: &Path,
    server_build_dir: &str,
    client_build_dir: &str,
    port: u16,
    potential: &str,
    sleep_secs: u64,
) -> Result<()> {
    let potserv = root.join(bridge_potserv_path(server_build_dir));
    if !potserv.is_file() {
        return Err(format!(
            "missing potserv at {} (run bridge-server-build first)",
            potserv.display()
        ));
    }
    let mut child = Command::new(&potserv)
        .args([port.to_string(), potential.to_string()])
        .current_dir(root)
        .stdin(Stdio::null())
        .stdout(Stdio::null())
        .stderr(Stdio::inherit())
        .spawn()
        .map_err(|e| format!("spawn potserv: {e}"))?;
    std::thread::sleep(std::time::Duration::from_secs(sleep_secs));
    let ctest = Command::new("ctest")
        .args(["--test-dir", client_build_dir, "--output-on-failure"])
        .current_dir(root)
        .status();
    let _ = child.kill();
    let _ = child.wait();
    match ctest {
        Ok(s) if s.success() => Ok(()),
        Ok(s) => Err(format!("ctest failed with {s}")),
        Err(e) => Err(format!("spawn ctest: {e}")),
    }
}

/// Full client_bridge_stress leg (server + client + stress) in one verb for maximum thinning.
pub fn run_bridge_stress_full(
    root: &Path,
    server_build_dir: &str,
    client_build_dir: &str,
    port: u16,
    potential: &str,
) -> Result<()> {
    run_bridge_server_build(root, server_build_dir)?;
    run_bridge_client_build(root, client_build_dir)?;
    run_bridge_stress(root, server_build_dir, client_build_dir, port, potential, 2)
}

/// `potctl ci rpc-integ` — run tests/rpc_integ.py against potserv in bbdir.
pub fn run_rpc_integ(root: &Path, server_bin: &str) -> Result<()> {
    let script = root.join("tests/rpc_integ.py");
    if !script.is_file() {
        return Err(format!("missing {}", script.display()));
    }
    let st = Command::new("python")
        .arg(&script)
        .args(["--server-bin", server_bin])
        .current_dir(root)
        .status()
        .or_else(|_| {
            Command::new("python3")
                .arg(&script)
                .args(["--server-bin", server_bin])
                .current_dir(root)
                .status()
        })
        .map_err(|e| format!("rpc_integ: {e}"))?;
    if st.success() {
        Ok(())
    } else {
        Err(format!("rpc_integ failed with {st}"))
    }
}

fn enforce_pr_commit_subjects(root: &Path, base: &str) -> Result<()> {
    let log = Command::new("git")
        .args(["log", "--format=%s", &format!("{base}..HEAD")])
        .current_dir(root)
        .output()
        .map_err(|e| format!("git log: {e}"))?;
    let text = String::from_utf8_lossy(&log.stdout);
    let mut bad = false;
    for subj in text.lines() {
        if is_ignored_merge_commit_subject(subj) {
            println!("ignored merge commit: {subj}");
            continue;
        }
        if !is_conventional_commit_subject(subj) {
            eprintln!("non-conventional subject: {subj}");
            bad = true;
        }
    }
    if bad {
        Err("PR introduces non-conventional commit subjects".into())
    } else {
        Ok(())
    }
}

/// `potctl ci cog-check [--from SHA]` — conventional commits (PR base..HEAD or full cog check).
///
/// The `--from` flag is potctl's. Cog 7 does not have `cog check --from`;
/// this passes a positional range and `--ignore-merge-commits` when the
/// installed help text lists that option.
pub fn run_cog_check(root: &Path, from_sha: Option<&str>) -> Result<()> {
    let from = from_sha.filter(|s| !s.is_empty());
    if let Some(base) = from {
        let help_out = Command::new("cog").args(["check", "--help"]).output().ok();
        let help = help_out
            .as_ref()
            .map(|o| {
                format!(
                    "{}{}",
                    String::from_utf8_lossy(&o.stdout),
                    String::from_utf8_lossy(&o.stderr)
                )
            })
            .unwrap_or_default();
        let args = cog_check_argv(&help, base);
        let argv: Vec<&str> = args.iter().map(String::as_str).collect();
        match Command::new("cog").args(&argv).current_dir(root).output() {
            Ok(out) if out.status.success() => return Ok(()),
            Ok(out) => {
                let text = format!(
                    "{}{}",
                    String::from_utf8_lossy(&out.stdout),
                    String::from_utf8_lossy(&out.stderr)
                );
                // Multi-parent merges are skipped by cog when
                // ignore_merge_commits is set. Single-parent `merge:`
                // subjects are ignored here.
                if !cog_failure_is_only_merge_types(&text) {
                    eprint!("{text}");
                    if !text.ends_with('\n') {
                        eprintln!();
                    }
                    enforce_pr_commit_subjects(root, base)?;
                    return Err("cog check failed".into());
                }
            }
            Err(e) => {
                eprintln!("::warning::spawn cog check: {e}");
            }
        }
        enforce_pr_commit_subjects(root, base)
    } else {
        let status = Command::new("cog")
            .arg("check")
            .current_dir(root)
            .status()
            .map_err(|e| format!("spawn cog check: {e}"))?;
        if !status.success() {
            println!("advisory cog check (dispatch run)");
        }
        Ok(())
    }
}

/// Environment variables for macOS + pixi: use host Apple clang/SDK, keep conda
/// dep search paths (LIBRARY_PATH/CPATH/DYLD_FALLBACK), prefer conda gfortran.
///
/// Emits `export KEY=value` / `unset KEY` lines suitable for `eval "$(potctl ci darwin-env)"`.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct DarwinEnvPlan {
    pub exports: Vec<(String, String)>,
    pub unsets: Vec<String>,
    pub meson_extra: String,
}

pub fn plan_darwin_env(
    os_name: &str,
    conda_prefix: Option<&str>,
    sdkroot: Option<&str>,
) -> Option<DarwinEnvPlan> {
    if os_name != "macOS" && os_name != "Darwin" {
        return None;
    }

    let mut exports: Vec<(String, String)> = Vec::new();
    let unsets: Vec<String> = vec![
        "CONDA_BUILD_SYSROOT".into(),
        "CC_FOR_BUILD".into(),
        "CXX_FOR_BUILD".into(),
        "CPP_FOR_BUILD".into(),
        "CMAKE_ARGS".into(),
        "host_alias".into(),
        "build_alias".into(),
        "CFLAGS".into(),
        "CXXFLAGS".into(),
        "CPPFLAGS".into(),
        "LDFLAGS".into(),
    ];

    if let Some(sdk) = sdkroot.filter(|s| !s.is_empty()) {
        exports.push(("SDKROOT".into(), sdk.to_string()));
    }

    for (k, v) in [
        ("CC", "/usr/bin/clang"),
        ("CXX", "/usr/bin/clang++"),
        ("OBJC", "/usr/bin/clang"),
        ("OBJCXX", "/usr/bin/clang++"),
        ("LD", "/usr/bin/clang++"),
    ] {
        exports.push((k.into(), v.into()));
    }

    if let Some(prefix) = conda_prefix.filter(|s| !s.is_empty()) {
        let p = Path::new(prefix);
        let lib = p.join("lib");
        let inc = p.join("include");
        let pc = p.join("lib").join("pkgconfig");

        exports.push((
            "CMAKE_PREFIX_PATH".into(),
            prepend_path_var(env::var_os("CMAKE_PREFIX_PATH"), prefix),
        ));
        exports.push((
            "PKG_CONFIG_PATH".into(),
            prepend_path_var(
                env::var_os("PKG_CONFIG_PATH"),
                &pc.to_string_lossy(),
            ),
        ));
        exports.push((
            "LIBRARY_PATH".into(),
            prepend_path_var(env::var_os("LIBRARY_PATH"), &lib.to_string_lossy()),
        ));
        exports.push((
            "CPATH".into(),
            prepend_path_var(env::var_os("CPATH"), &inc.to_string_lossy()),
        ));
        exports.push((
            "DYLD_FALLBACK_LIBRARY_PATH".into(),
            prepend_path_var(
                env::var_os("DYLD_FALLBACK_LIBRARY_PATH"),
                &lib.to_string_lossy(),
            ),
        ));

        let gfortran = p.join("bin").join("gfortran");
        if gfortran.is_file() || env::var_os("POTCTL_TEST_GFORTRAN").is_some() {
            // In tests we may force gfortran presence without the file.
            let fc = if gfortran.is_file() {
                gfortran.to_string_lossy().into_owned()
            } else {
                format!("{prefix}/bin/gfortran")
            };
            for k in ["FC", "F77", "F90", "F95", "GFORTRAN"] {
                exports.push((k.into(), fc.clone()));
            }
        }
    }

    Some(DarwinEnvPlan {
        exports,
        unsets,
        meson_extra: "-Ddefault_library=static".into(),
    })
}

fn prepend_path_var(existing: Option<std::ffi::OsString>, first: &str) -> String {
    match existing {
        Some(e) if !e.is_empty() => format!("{first}:{}", e.to_string_lossy()),
        _ => first.to_string(),
    }
}

/// Shell-escape a value for use inside double quotes in `export KEY="..."`.
pub fn shell_double_quote(s: &str) -> String {
    let mut out = String::with_capacity(s.len() + 2);
    out.push('"');
    for c in s.chars() {
        match c {
            '"' | '\\' | '$' | '`' => {
                out.push('\\');
                out.push(c);
            }
            _ => out.push(c),
        }
    }
    out.push('"');
    out
}

pub fn render_darwin_env_shell(plan: &DarwinEnvPlan) -> String {
    let mut lines = Vec::new();
    lines.push("# potctl ci darwin-env — macOS host clang/SDK + pixi dep paths".to_string());
    for k in &plan.unsets {
        lines.push(format!("unset {k} || true"));
    }
    for (k, v) in &plan.exports {
        lines.push(format!("export {k}={}", shell_double_quote(v)));
    }
    lines.push(format!(
        "export POTCTL_MESON_EXTRA={}",
        shell_double_quote(&plan.meson_extra)
    ));
    lines.push("".to_string());
    lines.join("\n")
}

/// Detect runner OS from env (GITHUB_ACTIONS sets RUNNER_OS; else uname-ish heuristic).
pub fn detect_runner_os() -> String {
    if let Ok(os) = env::var("RUNNER_OS") {
        return os;
    }
    if let Ok(os) = env::var("OSTYPE") {
        if os.contains("darwin") {
            return "macOS".into();
        }
    }
    #[cfg(target_os = "macos")]
    {
        return "macOS".into();
    }
    #[cfg(not(target_os = "macos"))]
    {
        "Linux".into()
    }
}

pub fn resolve_sdkroot_hint() -> Option<String> {
    if let Ok(s) = env::var("SDKROOT") {
        if !s.is_empty() {
            return Some(s);
        }
    }
    // Do not shell out in unit tests; real CI sets SDKROOT via xcrun in workflow
    // before eval, or pass --sdkroot. Optional: try xcrun if present.
    None
}

pub fn print_darwin_env(
    force: bool,
    conda_prefix: Option<&str>,
    sdkroot: Option<&str>,
) -> Result<()> {
    let os = detect_runner_os();
    let prefix = conda_prefix
        .map(str::to_string)
        .or_else(|| env::var("CONDA_PREFIX").ok())
        .or_else(|| env::var("PIXI_PROJECT_ROOT").ok().and_then(|_| env::var("CONDA_PREFIX").ok()));

    let sdk = sdkroot
        .map(str::to_string)
        .or_else(resolve_sdkroot_hint);

    let plan = if force {
        plan_darwin_env("macOS", prefix.as_deref(), sdk.as_deref())
    } else {
        plan_darwin_env(&os, prefix.as_deref(), sdk.as_deref())
    };

    match plan {
        Some(p) => {
            print!("{}", render_darwin_env_shell(&p));
            Ok(())
        }
        None => {
            // Non-macOS: emit a no-op so `eval "$(…)"` is safe on all matrix legs.
            println!("# potctl ci darwin-env: no-op (runner os = {os})");
            println!("export POTCTL_MESON_EXTRA=\"\"");
            Ok(())
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::path::Path;

    #[test]
    fn required_paths_are_under_root() {
        let root = Path::new("/tmp/rgpot-fake");
        let paths = required_paths(root);
        assert!(paths.iter().all(|p| p.starts_with(root)));
        assert!(paths.iter().any(|p| p.ends_with("potctl/Cargo.toml")));
    }

    #[test]
    fn darwin_plan_none_on_linux() {
        assert!(plan_darwin_env("Linux", Some("/conda"), Some("/sdk")).is_none());
    }

    #[test]
    fn darwin_plan_sets_clang_and_unsets_conda_cross() {
        let p = plan_darwin_env("macOS", Some("/opt/conda"), Some("/SDKs/MacOSX.sdk")).unwrap();
        assert!(p.exports.iter().any(|(k, v)| k == "CC" && v == "/usr/bin/clang"));
        assert!(p.exports.iter().any(|(k, v)| k == "SDKROOT" && v.contains("MacOSX")));
        assert!(p.unsets.contains(&"LDFLAGS".to_string()));
        assert!(p.unsets.contains(&"CONDA_BUILD_SYSROOT".to_string()));
        assert_eq!(p.meson_extra, "-Ddefault_library=static");
        assert!(p.exports.iter().any(|(k, v)| k == "LIBRARY_PATH" && v.starts_with("/opt/conda/lib")));
        assert!(p.exports.iter().any(|(k, v)| k == "CPATH" && v.starts_with("/opt/conda/include")));
    }

    #[test]
    fn darwin_plan_gfortran_when_forced_in_tests() {
        env::set_var("POTCTL_TEST_GFORTRAN", "1");
        let p = plan_darwin_env("macOS", Some("/opt/conda"), None).unwrap();
        env::remove_var("POTCTL_TEST_GFORTRAN");
        assert!(p.exports.iter().any(|(k, v)| k == "FC" && v.ends_with("gfortran")));
    }

    #[test]
    fn shell_quote_escapes_meta() {
        assert_eq!(shell_double_quote(r#"a"b"#), r#""a\"b""#);
        assert_eq!(shell_double_quote("x$y"), r#""x\$y""#);
    }

    #[test]
    fn render_includes_exports_and_unsets() {
        let p = plan_darwin_env("macOS", Some("/c"), Some("/sdk")).unwrap();
        let s = render_darwin_env_shell(&p);
        assert!(s.contains("unset LDFLAGS || true"));
        assert!(s.contains("export CC="));
        assert!(s.contains("POTCTL_MESON_EXTRA="));
        assert!(s.contains("/sdk"));
    }

    #[test]
    fn semver_still_ok_via_lockstep_module() {
        assert!(lockstep::semver_ok("1.2.0-rc.1"));
    }

    #[test]
    fn build_sys_parse() {
        assert_eq!(BuildSys::parse("meson").unwrap(), BuildSys::Meson);
        assert_eq!(BuildSys::parse("CMAKE").unwrap(), BuildSys::Cmake);
        assert!(BuildSys::parse("ninja").is_err());
    }

    #[test]
    fn parse_bool_flag_accepts_matrix_strings() {
        assert!(parse_bool_flag("true", "rpc").unwrap());
        assert!(!parse_bool_flag("false", "cache").unwrap());
        assert!(parse_bool_flag("maybe", "rpc").is_err());
    }

    #[test]
    fn meson_configure_args_include_rpc_cache_and_extra() {
        let a = meson_configure_args(true, false, true, "-Ddefault_library=static");
        assert!(a.iter().any(|x| x == "-Dwith_rpc=true"));
        assert!(a.iter().any(|x| x == "-Dwith_cache=false"));
        assert!(a.iter().any(|x| x == "-Dwith_tests=True"));
        assert!(a.iter().any(|x| x == "-Dwith_mpi=enabled"));
        assert!(a.iter().any(|x| x == "-Ddefault_library=static"));
        let off = meson_configure_args(true, false, false, "");
        assert!(!off.iter().any(|x| x.contains("with_mpi")));
    }

    #[test]
    fn cmake_configure_core_rpc_cache_on_off() {
        let on = cmake_configure_core_args(true, true);
        assert!(on.iter().any(|x| x == "-DRGPOT_WITH_RPC=ON"));
        assert!(on.iter().any(|x| x == "-DRGPOT_WITH_CACHE=ON"));
        let off = cmake_configure_core_args(false, false);
        assert!(off.iter().any(|x| x == "-DRGPOT_WITH_RPC=OFF"));
        assert!(off.iter().any(|x| x == "-DRGPOT_WITH_CACHE=OFF"));
    }

    #[test]
    fn cmake_build_names_its_job_count() {
        assert_eq!(
            cmake_build_args("build", 3),
            vec![
                "--build".to_string(),
                "build".to_string(),
                "--parallel".to_string(),
                "3".to_string(),
            ]
        );
        assert_eq!(cmake_build_args("build", 0)[3], "1");
        assert_eq!(cmake_build_jobs(None, 3), 3);
        assert_eq!(cmake_build_jobs(Some(" 8 "), 3), 8);
        assert_eq!(cmake_build_jobs(Some("0"), 3), 3);
        assert_eq!(cmake_build_jobs(Some("all"), 3), 3);
        assert_eq!(cmake_build_jobs(None, 0), 1);
    }

    #[test]
    fn cmake_macos_extra_empty_on_linux() {
        assert!(cmake_macos_extra_args(false, Some("/sdk"), Some("/c"), Some("/fc")).is_empty());
    }

    #[test]
    fn cmake_macos_extra_has_clang_and_prefix() {
        let a = cmake_macos_extra_args(true, Some("/SDKs/X.sdk"), Some("/opt/conda"), Some("/opt/conda/bin/gfortran"));
        assert!(a.iter().any(|x| x == "-DCMAKE_C_COMPILER=/usr/bin/clang"));
        assert!(a.iter().any(|x| x.contains("CMAKE_OSX_SYSROOT=/SDKs/X.sdk")));
        assert!(a.iter().any(|x| x.contains("CMAKE_PREFIX_PATH=/opt/conda")));
        assert!(a.iter().any(|x| x.contains("CMAKE_Fortran_COMPILER=/opt/conda/bin/gfortran")));
    }

    #[test]
    fn conventional_commit_subjects() {
        assert!(is_conventional_commit_subject("feat(ci): thin steps"));
        assert!(is_conventional_commit_subject("fix: waitid stub"));
        assert!(is_conventional_commit_subject("chore: bump"));
        assert!(is_conventional_commit_subject("bench: time one force call"));
        assert!(is_conventional_commit_subject("pot(sw): add the pair kernel"));
        assert!(is_conventional_commit_subject("python(sw): bind the energy"));
        assert!(is_conventional_commit_subject(
            "nlist: bin atoms on a cutoff-sized cell grid"
        ));
        assert!(is_conventional_commit_subject("Merge branch 'main'"));
        assert!(is_conventional_commit_subject(
            "merge: take the wheel ISA audit from main"
        ));
        assert!(is_ignored_merge_commit_subject(
            "merge: take the 3.5.1 release from main"
        ));
        assert!(is_conventional_commit_subject(
            "simd: keep vfmadd behind the runtime ISA check"
        ));
        assert!(!is_conventional_commit_subject("wip stuff"));
        assert!(!is_conventional_commit_subject(""));
    }

    #[test]
    fn cog7_check_uses_positional_range_not_from() {
        let cog7 = "\
Usage: cog check [OPTIONS] [RANGE]

Options:
  -l, --from-latest-tag
          Check commit history, starting from the latest tag to HEAD
  -i, --ignore-merge-commits
          Ignore merge commits messages
";
        assert!(!cog_help_has_flag(cog7, "--from"));
        assert!(cog_help_has_flag(cog7, "--from-latest-tag"));
        assert!(cog_help_has_flag(cog7, "--ignore-merge-commits"));
        assert_eq!(
            cog_check_argv(cog7, "67ba594"),
            vec![
                "check".to_string(),
                "--ignore-merge-commits".to_string(),
                "67ba594..HEAD".to_string(),
            ]
        );
        let legacy = "Options:\n      --from <REV>\n";
        assert!(cog_help_has_flag(legacy, "--from"));
        assert_eq!(
            cog_check_argv(legacy, "abc"),
            vec![
                "check".to_string(),
                "--from".to_string(),
                "abc".to_string(),
            ]
        );
        assert!(cog_failure_is_only_merge_types(
            "Error: \nFound 4 non compliant commits in abc..HEAD:\n\tError:Commit type `merge` not allowed\n"
        ));
        assert!(cog_failure_is_only_merge_types(
            "\tError:Commit type `merge` not allowed\n"
        ));
        assert!(!cog_failure_is_only_merge_types(
            "error: unexpected argument '--from' found\n"
        ));
        assert!(!cog_failure_is_only_merge_types(
            "\tError:Commit type `merge` not allowed\n\tError:Commit type `wip` not allowed\n"
        ));
    }

    #[test]
    fn torch_mm_and_layout_paths() {
        assert_eq!(
            torch_mm_from_version("2.10.1+cpu").as_deref(),
            Some("2.10")
        );
        assert_eq!(torch_mm_from_version("2.9.0").as_deref(), Some("2.9"));
        let paths = metatomic_layout_paths(Path::new("/p"), "2.10");
        assert!(paths.iter().any(|p| p.ends_with("vesin/include/vesin.h")));
        assert!(paths
            .iter()
            .any(|p| p.ends_with("metatomic/torch/torch-2.10/include")));
    }

    #[test]
    fn towncrier_invoke_selection() {
        assert_eq!(
            select_towncrier_invoke(true, false),
            Some(TowncrierInvoke::Bin)
        );
        assert_eq!(
            select_towncrier_invoke(false, true),
            Some(TowncrierInvoke::PythonModule)
        );
        assert_eq!(select_towncrier_invoke(false, false), None);
        assert_eq!(
            select_towncrier_invoke(true, true),
            Some(TowncrierInvoke::Bin)
        );
    }

    #[test]
    fn xckernel_setup_enables_the_golden_suite() {
        let a = xckernel_meson_setup_args("bbdir-xck");
        assert_eq!(a[0], "setup");
        assert_eq!(a[1], "bbdir-xck");
        assert!(a.iter().any(|x| x == "-Dwith_xckernel=true"));
        assert!(a.iter().any(|x| x == "-Dwith_tests=true"));
        assert!(a.iter().any(|x| x == "-Dwith_rpc=false"));
    }

    #[test]
    fn metatomic_argv_follows_the_rpc_flag() {
        let off = metatomic_meson_setup_args("bbdir-mta", false);
        assert!(off.iter().any(|x| x == "-Dwith_rpc=false"));
        let on = metatomic_meson_setup_args("bbdir-uma-engine", true);
        assert_eq!(on[1], "bbdir-uma-engine");
        assert!(on.iter().any(|x| x == "-Dwith_rpc=true"));
        assert!(on.iter().any(|x| x == "-Dwith_metatomic=true"));
        assert!(on.iter().any(|x| x == "-Dwith_xtb=true"));
        assert!(on.iter().any(|x| x == "-Dwith_skala=true"));
        assert!(!off.iter().any(|x| x == "-Dwith_xtb=true"));
        assert!(!off.iter().any(|x| x == "-Dwith_skala=true"));
    }

    #[test]
    fn plugin_argv_builders() {
        let s = plugin_meson_setup_args("bbdir-plugin");
        assert_eq!(s[0], "setup");
        assert_eq!(s[1], "bbdir-plugin");
        assert!(s.iter().any(|x| x == "-Dwith_tests=true"));
        assert!(s.iter().any(|x| x == "-Dwith_rpc_client_only=false"));
        let t = plugin_meson_test_args("bbdir-plugin");
        for name in PLUGIN_TESTS {
            assert!(t.iter().any(|x| x == name), "missing {name}");
        }
    }

    #[test]
    fn bridge_argv_builders() {
        let s = bridge_server_meson_setup_args("bbdir_server");
        assert!(s.iter().any(|x| x == "-Dwith_rpc=true"));
        assert_eq!(s[1], "bbdir_server");
        let c = bridge_client_cmake_configure_args("build_client");
        assert!(c.iter().any(|x| x == "-DRGPOT_RPC_CLIENT_ONLY=ON"));
        assert!(c.iter().any(|x| x == "build_client"));
        let p = bridge_potserv_path("bbdir_server");
        assert!(p.ends_with("potserv"));
        assert!(p.to_string_lossy().contains("bbdir_server"));
    }

    #[test]
    fn run_build_test_rejects_bad_sys() {
        let err = run_build_test(
            Path::new("/tmp/nope"),
            "make",
            "true",
            "false",
            None,
            true,
        )
        .unwrap_err();
        assert!(err.contains("unknown build sys") || err.contains("make"));
    }
}
