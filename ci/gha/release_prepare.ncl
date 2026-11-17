# Generated target: .github/workflows/release-prepare.yml
let S = import "lib/steps.ncl" in
let P = S.pins in
let release_paths = [
  "cog.toml",
  "towncrier.toml",
  "potctl/**",
  "Cargo.toml",
  "Cargo.lock",
  "docs/newsfragments/**",
  "rgpot-core/Cargo.toml",
  "meson.build",
  "CMakeLists.txt",
  "pixi.toml",
  "CHANGELOG.md",
  ".github/workflows/release.yml",
  ".github/workflows/release-prepare.yml",
  ".github/actions/**",
  "ci/gha/**",
]
in
{
  name = "Release prepare (cog dry-run)",
  on = {
    workflow_dispatch = {
      inputs = {
        bump = {
          description = "SemVer component for cog dry-run",
          required = true,
          default = "auto",
          type = "choice",
          options = ["auto", "patch", "minor", "major"],
        },
      },
    },
    pull_request = { paths = release_paths },
  },
  concurrency = S.concurrency_ref "release-prepare",
  permissions = { contents = "read" },
  env = { CI_TOOLS_ARTIFACT = "ci-tools-${{ github.sha }}-Linux" },
  jobs = {
    tools = S.job_tools_linux "true" "Stage portable potctl + cog (Linux musl + upload)",
    prepare = {
      needs = ["tools"],
      "runs-on" = P.ubuntu,
      "timeout-minutes" = 10,
      steps = [
        S.checkout_depth 0,
        {
          name = "Restore potctl + cog from tools job",
          uses = P.restore_ci_tools,
          with = { "artifact-name" = "${{ env.CI_TOOLS_ARTIFACT }}" },
        },
        S.setup_pixi_default "devbld",
        # potctl owns step bodies; nickel owns job/matrix/needs only
        { name = "Assert lockstep surfaces agree", run = "potctl ci release-assert" },
        { name = "Towncrier draft (fragments must build cleanly)", run = "potctl ci towncrier-draft" },
        # rgpot-core uses git deps (dlpk, eindir-core) not on crates.io; publish is
        # off (publish = false). Validate the locked graph builds instead.
        S.run_pixi m%"
          cargo check -p rgpot-core --locked
        "%
        & { name = "Cargo check rgpot-core (locked)" },
        { name = "Cog bump dry-run (PR tip version)", run = "potctl ci cog-bump-dry-run" },
        {
          name = "Cog bump dry-run (workflow_dispatch)",
          "if" = "github.event_name == 'workflow_dispatch'",
          run = "potctl ci cog-bump-dry-run --bump ${{ inputs.bump }}",
        },
        {
          name = "Cog check (conventional commits since last tag)",
          run = "potctl ci cog-check --from ${{ github.event.pull_request.base.sha || '' }}",
        },
      ],
    },
  },
}
