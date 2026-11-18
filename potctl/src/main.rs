//! potctl — rgpot project control plane (release lockstep, CI helpers, cosmo APE).

mod ci;
#[cfg(feature = "cosmo-host")]
mod cosmo;
mod lockstep;

use clap::{Parser, Subcommand};
use std::path::PathBuf;
use std::process::ExitCode;

#[derive(Parser, Debug)]
#[command(
    name = "potctl",
    version,
    about = "rgpot project control plane",
    long_about = "Repo-local CLI for release lockstep, CI helpers, and Cosmopolitan APE build. \
Run from the rgpot checkout (walks up for meson.build + pixi.toml + rgpot-core/)."
)]
struct Cli {
    #[command(subcommand)]
    command: Commands,
}

#[derive(Subcommand, Debug)]
enum Commands {
    /// Release version: write it in Cargo.toml, check the generated copies, print notes.
    Release {
        #[command(subcommand)]
        action: ReleaseCmd,
    },
    /// Continuous integration helpers (preflight, darwin/pixi env, …).
    Ci {
        #[command(subcommand)]
        action: CiCmd,
    },
    /// Cosmopolitan APE (`potctl.com`) build orchestration (Rust only; no project bash).
    /// Host-only (`cosmo-host` feature); not compiled into the APE binary itself.
    #[cfg(feature = "cosmo-host")]
    Cosmo {
        #[command(subcommand)]
        action: CosmoCmd,
    },
}

#[cfg(feature = "cosmo-host")]
#[derive(Subcommand, Debug)]
enum CosmoCmd {
    /// Build potctl as Cosmopolitan APE via nightly rustc + cosmocc linker shim.
    ///
    /// Requires COSMO (cosmo monorepo after `make toolchain`), rustup nightly, and
    /// host `potctl-cosmo-ld` (built automatically if missing). Env: POTCTL_COSMO_OUT,
    /// POTCTL_COSMO_ARCHS, COSMO_CC, COSMOCC_ARCHES (see potctl/cosmo/README.md).
    Build,
}

#[derive(Subcommand, Debug)]
enum ReleaseCmd {
    /// Write the release version into Cargo.toml [workspace.package] (then run cogapp).
    Sync {
        /// Semver (optional leading v), e.g. 1.2.3 or v1.2.3-rc.1
        version: String,
    },
    /// Assert every version copy matches Cargo.toml; optionally require CHANGELOG section.
    Assert {
        /// Expected semver (optional; often the git tag without v).
        version: Option<String>,
        /// Require CHANGELOG.md to contain ## [version] (or the workspace version if omitted).
        #[arg(long, env = "REQUIRE_CHANGELOG")]
        require_changelog: bool,
    },
    /// Print or write one CHANGELOG.md section for GH Release body.
    Notes {
        version: String,
        /// Write section to this path instead of stdout.
        #[arg(short = 'o', long = "out")]
        out: Option<PathBuf>,
    },
}

#[derive(Subcommand, Debug)]
enum CiCmd {
    /// Check required files (+ optional lockstep assert). Safe first step in GHA jobs.
    Preflight {
        /// Skip lockstep version agreement (only check files exist).
        #[arg(long)]
        no_lockstep: bool,
    },
    /// Print shell exports for macOS + pixi host SDK/clang fix; no-op on Linux.
    /// Usage in GHA: `eval "$(potctl ci darwin-env)"` then meson/cmake with `$POTCTL_MESON_EXTRA`.
    /// Prefer `meson-test` / `cmake-test` which apply darwin-env internally.
    DarwinEnv {
        /// Emit macOS plan even if RUNNER_OS is not macOS (local testing).
        #[arg(long)]
        force: bool,
        /// Override CONDA_PREFIX (default: env CONDA_PREFIX from pixi shell).
        #[arg(long)]
        conda_prefix: Option<String>,
        /// Override SDKROOT (default: env SDKROOT; set via `xcrun --show-sdk-path` in workflow first).
        #[arg(long)]
        sdkroot: Option<String>,
    },
    /// Meson setup + compile + test (orchestrator matrix leg). Applies darwin-env internally.
    /// Thin GHA: `potctl ci meson-test --rpc ${{ matrix.rpc }} --cache ${{ matrix.cache }}`
    MesonTest {
        /// RPC feature (`true`/`false`; GHA matrix.rpc).
        #[arg(long)]
        rpc: String,
        /// Cache feature (`true`/`false`; GHA matrix.cache).
        #[arg(long)]
        cache: String,
        /// Build directory (default: bbdir).
        #[arg(long, default_value = "bbdir")]
        build_dir: String,
        /// Skip applying darwin-env inside this process.
        #[arg(long)]
        no_darwin: bool,
    },
    /// CMake configure + build + ctest (orchestrator matrix leg). Applies darwin-env internally.
    /// Thin GHA: `potctl ci cmake-test --rpc ${{ matrix.rpc }} --cache ${{ matrix.cache }}`
    CmakeTest {
        #[arg(long)]
        rpc: String,
        #[arg(long)]
        cache: String,
        #[arg(long, default_value = "build")]
        build_dir: String,
        #[arg(long)]
        no_darwin: bool,
    },
    /// Either meson or cmake via `--sys` (same as matrix.sys).
    BuildTest {
        /// `meson` or `cmake` (GHA matrix.sys).
        #[arg(long)]
        sys: String,
        #[arg(long)]
        rpc: String,
        #[arg(long)]
        cache: String,
        /// Override build dir (default: bbdir for meson, build for cmake).
        #[arg(long)]
        build_dir: Option<String>,
        #[arg(long)]
        no_darwin: bool,
    },
    /// Release-prepare: lockstep assert + cog.toml readable + cog --version.
    /// Thin GHA: `potctl ci release-assert`
    ReleaseAssert,
    /// Release-prepare: `uvx towncrier build --draft` if newsfragments exist.
    /// Thin GHA: `potctl ci towncrier-draft`
    TowncrierDraft,
    /// Release-prepare: cog bump dry-run (omit `--bump` for meson tip version; advisory on fail).
    /// Thin GHA: `potctl ci cog-bump-dry-run` or `… --bump auto|patch|minor|major`
    CogBumpDryRun {
        /// `auto`/`patch`/`minor`/`major` for workflow_dispatch; omit to use meson.build version.
        #[arg(long)]
        bump: Option<String>,
    },
    /// Release-prepare: cog check + conventional subject fallback for PR range.
    /// Thin GHA: `potctl ci cog-check --from ${{ github.event.pull_request.base.sha }}`
    /// (`--from` is potctl's; cog 7 receives a positional `<base>..HEAD` range).
    CogCheck {
        /// PR base SHA (`github.event.pull_request.base.sha`); omit for advisory full-repo check.
        #[arg(long)]
        from: Option<String>,
    },
    /// Orchestrator towncrier job: `towncrier check --compare-with <sha>`.
    /// Thin GHA: `potctl ci towncrier-check --compare-with ${{ github.event.pull_request.base.sha }}`
    TowncrierCheck {
        #[arg(long)]
        compare_with: String,
    },
    /// Potentials metatomic leg: CPU torch + metatensor/metatomic/vesin purelib layout.
    /// Writes GITHUB_ENV torch/cmake paths. Thin GHA: `potctl ci ensure-torch-metatomic`
    EnsureTorchMetatomic,
    /// Potentials metatomic leg: meson metatomic+vesin build+test.
    /// Thin GHA: `potctl ci metatomic-test` (after ensure-torch-metatomic / pixi metatomicbld)
    MetatomicTest {
        #[arg(long, default_value = "bbdir-mta")]
        build_dir: String,
        /// Build with the rpc stack, which turns on the libuma_engine plugin.
        #[arg(long)]
        rpc: bool,
    },
    /// Plugin leg: meson build with the plugin host, then PluginTest,
    /// PluginContractTest, ExternalPluginTest and the rest of the plugin suite.
    /// Thin GHA: `potctl ci plugin-test`
    PluginTest {
        #[arg(long, default_value = "bbdir-plugin")]
        build_dir: String,
        /// Skip darwin toolchain env setup.
        #[arg(long)]
        no_darwin: bool,
    },
    /// Potentials xtb+tblite leg.
    /// Thin GHA: `potctl ci xtb-tblite-test`
    XtbTbliteTest {
        #[arg(long, default_value = "bbdir-tb")]
        build_dir: String,
    },
    /// XcKernel golden leg. The default meson matrix leaves with_xckernel off.
    /// Thin GHA: `potctl ci xckernel-test`
    XckernelTest {
        #[arg(long, default_value = "bbdir-xck")]
        build_dir: String,
    },
    /// RPC integration (`tests/rpc_integ.py`).
    /// Thin GHA: `potctl ci rpc-integ --server-bin bbdir/CppCore/potserv`
    RpcInteg {
        #[arg(long, default_value = "bbdir/CppCore/potserv")]
        server_bin: String,
    },
    /// Client bridge stress: meson RPC server build only.
    BridgeServerBuild {
        #[arg(long, default_value = "bbdir_server")]
        build_dir: String,
    },
    /// Client bridge stress: cmake RPC client-only build.
    BridgeClientBuild {
        #[arg(long, default_value = "build_client")]
        build_dir: String,
    },
    /// Client bridge stress: potserv + ctest (after server/client builds).
    BridgeStress {
        #[arg(long, default_value = "bbdir_server")]
        server_build_dir: String,
        #[arg(long, default_value = "build_client")]
        client_build_dir: String,
        #[arg(long, default_value = "12345")]
        port: u16,
        #[arg(long, default_value = "LJ")]
        potential: String,
    },
    /// Full client_bridge_stress leg (server + client + potserv/ctest) — one GHA line.
    /// Thin GHA: `potctl ci bridge-stress-full`
    BridgeStressFull {
        #[arg(long, default_value = "bbdir_server")]
        server_build_dir: String,
        #[arg(long, default_value = "build_client")]
        client_build_dir: String,
        #[arg(long, default_value = "12345")]
        port: u16,
        #[arg(long, default_value = "LJ")]
        potential: String,
    },
}

fn run() -> Result<(), String> {
    let cli = Cli::parse();
    let root = lockstep::repo_root()?;

    match cli.command {
        Commands::Release { action } => match action {
            ReleaseCmd::Sync { version } => lockstep::sync_versions(&root, &version),
            ReleaseCmd::Assert {
                version,
                require_changelog,
            } => lockstep::assert_lockstep(&root, version.as_deref(), require_changelog),
            ReleaseCmd::Notes { version, out } => {
                lockstep::extract_changelog(&root, &version, out.as_deref())
            }
        },
        Commands::Ci { action } => match action {
            CiCmd::Preflight { no_lockstep } => ci::preflight(&root, !no_lockstep),
            CiCmd::DarwinEnv {
                force,
                conda_prefix,
                sdkroot,
            } => ci::print_darwin_env(force, conda_prefix.as_deref(), sdkroot.as_deref()),
            CiCmd::MesonTest {
                rpc,
                cache,
                build_dir,
                no_darwin,
            } => ci::run_build_test(
                &root,
                "meson",
                &rpc,
                &cache,
                Some(&build_dir),
                no_darwin,
            ),
            CiCmd::CmakeTest {
                rpc,
                cache,
                build_dir,
                no_darwin,
            } => ci::run_build_test(
                &root,
                "cmake",
                &rpc,
                &cache,
                Some(&build_dir),
                no_darwin,
            ),
            CiCmd::BuildTest {
                sys,
                rpc,
                cache,
                build_dir,
                no_darwin,
            } => ci::run_build_test(
                &root,
                &sys,
                &rpc,
                &cache,
                build_dir.as_deref(),
                no_darwin,
            ),
            CiCmd::ReleaseAssert => ci::run_release_assert(&root),
            CiCmd::TowncrierDraft => ci::run_towncrier_draft(&root),
            CiCmd::CogBumpDryRun { bump } => {
                ci::run_cog_bump_dry_run(&root, bump.as_deref())
            }
            CiCmd::CogCheck { from } => ci::run_cog_check(&root, from.as_deref()),
            CiCmd::TowncrierCheck { compare_with } => {
                ci::run_towncrier_check(&root, &compare_with)
            }
            CiCmd::EnsureTorchMetatomic => ci::run_ensure_torch_metatomic(&root),
            CiCmd::MetatomicTest { build_dir, rpc } => {
                ci::run_metatomic_test(&root, &build_dir, rpc)
            }
            CiCmd::PluginTest { build_dir, no_darwin } => {
                ci::run_plugin_test(&root, &build_dir, !no_darwin)
            }
            CiCmd::XtbTbliteTest { build_dir } => {
                ci::run_xtb_tblite_test(&root, &build_dir)
            }
            CiCmd::XckernelTest { build_dir } => {
                ci::run_xckernel_test(&root, &build_dir)
            }
            CiCmd::RpcInteg { server_bin } => ci::run_rpc_integ(&root, &server_bin),
            CiCmd::BridgeServerBuild { build_dir } => {
                ci::run_bridge_server_build(&root, &build_dir)
            }
            CiCmd::BridgeClientBuild { build_dir } => {
                ci::run_bridge_client_build(&root, &build_dir)
            }
            CiCmd::BridgeStress {
                server_build_dir,
                client_build_dir,
                port,
                potential,
            } => ci::run_bridge_stress(
                &root,
                &server_build_dir,
                &client_build_dir,
                port,
                &potential,
                2,
            ),
            CiCmd::BridgeStressFull {
                server_build_dir,
                client_build_dir,
                port,
                potential,
            } => ci::run_bridge_stress_full(
                &root,
                &server_build_dir,
                &client_build_dir,
                port,
                &potential,
            ),
        },
        #[cfg(feature = "cosmo-host")]
        Commands::Cosmo { action } => match action {
            CosmoCmd::Build => cosmo::build_ape(&root),
        },
    }
}

fn main() -> ExitCode {
    match run() {
        Ok(()) => ExitCode::SUCCESS,
        Err(e) => {
            eprintln!("error: potctl: {e}");
            ExitCode::FAILURE
        }
    }
}
