# Generated target: .github/workflows/ci_docs.yml
let S = import "lib/steps.ncl" in
let P = S.pins in
{
  name = "Build and Deploy Documentation",
  on = {
    push = { branches = ["main"], tags = ["*"] },
    pull_request = {},
    schedule = [{ cron = "0 8 * * 1" }],
  },
  # Workflow stays read-only. The deploy job opts into contents: write.
  permissions = { contents = "read" },
  jobs = {
    build_docs = {
      name = "Build documentation",
      "runs-on" = P.ubuntu,
      "timeout-minutes" = 10,
      # One group per ref, so parallel pull requests do not cancel each other
      # or a pages deploy. A new push to the same ref still cancels its build.
      concurrency = {
        group = "docs-${{ github.workflow }}-${{ github.ref }}",
        "cancel-in-progress" = true,
      },
      steps = [
        {
          name = "Checkout repository",
          uses = P.checkout,
          with = { "fetch-depth" = 0 },
        },
        S.setup_pixi_cache_write_main "docs",
        S.run_pixi m%"
          pixi r -e docs doccheck
          pixi r -e docs python -m unittest discover -s tests/docs -p 'test_docs_*.py' -v
        "%
        & { name = "Check docs against the source" },
        # sphinxbld runs sphinx-build -W, so a warning fails this step.
        S.run_pixi m%"
          pixi r docbld
        "%
        & { name = "Generate Docs" },
        {
          name = "Upload artifact",
          uses = P.upload_artifact,
          with = { name = "documentation", path = "docs/build" },
        },
      ],
    },
    deploy = {
      name = "Deploy documentation",
      needs = ["build_docs"],
      "if" = "github.event_name == 'push' && (github.ref == 'refs/heads/main' || startsWith(github.ref, 'refs/tags/'))",
      "runs-on" = P.ubuntu,
      "timeout-minutes" = 10,
      permissions = { contents = "write" },
      # Shared pages group, and a running deploy is left to finish.
      concurrency = {
        group = "pages",
        "cancel-in-progress" = false,
      },
      steps = [
        { uses = P.checkout, with = { "persist-credentials" = false } },
        {
          name = "Download documentation",
          uses = P.download_artifact,
          with = { name = "documentation", path = "docs/build" },
        },
        {
          name = "Deploy",
          uses = P.gh_pages,
          with = {
            github_token = "${{ secrets.GITHUB_TOKEN }}",
            publish_dir = "docs/build",
            cname = "rgpot.rgoswami.me",
          },
        },
      ],
    },
  },
}
