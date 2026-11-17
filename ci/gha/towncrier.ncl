# Audit/reference mirror of orchestrator towncrier job (not exported as live workflow).
# Live: .github/workflows/ci-orchestrator.yml
let S = import "lib/steps.ncl" in
let P = S.pins in
{
  name = "towncrier",
  on = {
    pull_request = {
      paths = [
        "CppCore/**",
        "rgpot-core/**",
        "include/**",
        "meson.build",
        "meson_options.txt",
        "CMakeLists.txt",
        "pixi.toml",
        "CHANGELOG.md",
        "docs/newsfragments/**",
        "towncrier.toml",
        "cog.toml",
        "potctl/**",
        "Cargo.toml",
      ],
    },
  },
  concurrency = S.concurrency_ref "towncrier",
  permissions = { contents = "read", "pull-requests" = "read" },
  jobs = {
    fragments = {
      "runs-on" = P.ubuntu,
      "timeout-minutes" = 10,
      steps = [
        S.checkout_depth 0,
        {
          name = "Towncrier check vs base branch",
          run = m%"
            pipx install towncrier
            BASE="${{ github.event.pull_request.base.sha }}"
            towncrier check --compare-with "$BASE"
          "%,
        },
      ],
    },
  },
}
