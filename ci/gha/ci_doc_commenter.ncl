# Generated target: .github/workflows/ci_doc_commenter.yml
let S = import "lib/steps.ncl" in
let P = S.pins in
{
  name = "Comment on pull request",
  on = {
    workflow_run = {
      workflows = ["Build and Deploy Documentation"],
      types = ["completed"],
    },
  },
  concurrency = S.concurrency_cancel,
  jobs = {
    pr_comment = {
      "if" = "github.event.workflow_run.event == 'pull_request' && github.event.workflow_run.conclusion == 'success'",
      "runs-on" = P.ubuntu,
      "timeout-minutes" = 10,
      permissions = {
        "pull-requests" = "write",
        issues = "write",
        actions = "read",
      },
      steps = [
        {
          uses = P.doc_previewer,
          with = {
            workflow_run_id = "${{ github.event.workflow_run.id }}",
            head_sha = "${{ github.event.workflow_run.head_sha }}",
            artifact_name = "documentation",
            comment_header = "### rgPot Documentation Preview",
          },
        },
      ],
    },
  },
}
