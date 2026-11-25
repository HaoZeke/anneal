The CI workflow contract tests parse ``ci.yml`` as YAML and check steps,
jobs and conditions, the way the release workflow tests do. The audit of
large files and the step that picks the macOS test matrix also run from
the contract tests, in a throwaway repository and shell.
