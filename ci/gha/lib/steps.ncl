let P = import "pins.ncl" in
{
  pins = P,

  run_pixi = fun script => { shell = "pixi run bash -e {0}", run = script },

  run_pixi_env = fun env script =>
    { shell = "pixi run -e %{env} bash -e {0}", run = script },

  run_bash = fun script => { shell = "bash -e {0}", run = script },

  checkout_minimal = { uses = P.checkout },

  # cl.exe is not on PATH on a bare Windows runner; meson reports
  # "Unknown compiler(s): [['cl.exe']]" without this.
  activate_msvc = {
    name = "Activate MSVC (Windows)",
    uses = P.msvc_dev_cmd,
    with = { arch = "x64", vsversion = "2022" },
  },

  checkout_full = {
    uses = P.checkout,
    with = { submodules = "recursive", "fetch-depth" = 0 },
  },

  checkout_depth = fun depth =>
    { uses = P.checkout, with = { "fetch-depth" = depth } },

  setup_pixi = fun envs activate =>
    {
      uses = P.setup_pixi,
      with = {
        "pixi-version" = P.pixi_version,
        cache = true,
        environments = envs,
        "activate-environment" = activate,
      },
    },

  setup_pixi_default = fun envs =>
    {
      uses = P.setup_pixi,
      with = {
        "pixi-version" = P.pixi_version,
        cache = true,
        environments = envs,
        "activate-environment" = true,
      },
    },

  setup_pixi_cache_write_main = fun envs =>
    {
      uses = P.setup_pixi,
      with = {
        "pixi-version" = P.pixi_version,
        cache = true,
        "cache-write" = "${{ github.event_name == 'push' && github.ref_name == 'main' }}",
        environments = envs,
        "activate-environment" = true,
      },
    },

  ensure_potctl = {
    name = "Ensure potctl (OS-matched portable/fat artifact)",
    uses = P.ensure_potctl,
    with = { "artifact-prefix" = "ci-tools-${{ github.sha }}" },
  },

  restore_ci_tools_linux = {
    name = "Restore potctl from tools job",
    uses = P.restore_ci_tools,
    with = { "artifact-name" = "${{ env.CI_TOOLS_ARTIFACT }}" },
  },

  setup_ci_tools_upload = fun install_cog artifact_name step_name =>
    {
      name = step_name,
      uses = P.setup_ci_tools,
      with = {
        "install-cog" = install_cog,
        "upload-artifact" = "true",
        "artifact-name" = artifact_name,
      },
    },

  job_ci_tools_os_matrix = fun os_list =>
    {
      name = "ci-tools (potctl / ${{ matrix.os }})",
      "runs-on" = "${{ matrix.os }}",
      "timeout-minutes" = 10,
      strategy = {
        "fail-fast" = false,
        matrix = { os = os_list },
      },
      steps = [
        { uses = P.checkout },
        {
          name = "Stage portable/fat potctl",
          uses = P.setup_ci_tools,
          with = {
            "upload-artifact" = "true",
            "artifact-name" = "ci-tools-${{ github.sha }}-${{ runner.os }}",
          },
        },
      ],
    },

  ccache_env = {
    CCACHE_DIR = "${{ github.workspace }}/.ccache",
    CCACHE_COMPILERCHECK = "content",
    CCACHE_MAXSIZE = "500M",
  },

  ccache_step = fun key_prefix =>
    {
      name = "Restore Compiler Cache",
      uses = P.cache,
      with = {
        path = ".ccache",
        key = "%{key_prefix}-${{ runner.os }}-${{ github.sha }}",
        "restore-keys" = "%{key_prefix}-${{ runner.os }}-",
      },
    },

  ccache_step_matrix_sys = {
    name = "Restore Compiler Cache",
    uses = P.cache,
    with = {
      path = ".ccache",
      key = "ccache-${{ runner.os }}-${{ matrix.sys }}-${{ github.sha }}",
      "restore-keys" = "ccache-${{ runner.os }}-${{ matrix.sys }}-",
    },
  },

  darwin_env_prelude = m%"
  # macOS/pixi host clang/SDK via potctl (unit-tested); Linux is a no-op eval.
  if [ "${{ runner.os }}" = "macOS" ]; then
    export SDKROOT="$(xcrun --show-sdk-path)"
  fi
  eval "$(potctl ci darwin-env)"
"%,

  concurrency_ref = fun group_prefix =>
    {
      group = "%{group_prefix}-${{ github.ref }}",
      "cancel-in-progress" = true,
    },

  # Push stays on main. Pull requests also target develop, the integration branch.
  on_main_pr = {
    push = { branches = ["main"] },
    pull_request = { branches = ["main", "develop"] },
  },

  on_main_pr_any = {
    push = { branches = ["main"] },
    pull_request = {},
  },

  # Single-runner Linux potctl (+ optional cog) for release/release-prepare style workflows.
  job_tools_linux = fun install_cog step_name =>
    {
      "runs-on" = P.ubuntu,
      "timeout-minutes" = 10,
      steps = [
        { uses = P.checkout },
        {
          name = step_name,
          uses = P.setup_ci_tools,
          with = {
            "install-cog" = install_cog,
            "upload-artifact" = "true",
            "artifact-name" = "${{ env.CI_TOOLS_ARTIFACT }}",
          },
        },
      ],
    },
}
