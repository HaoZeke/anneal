Every CI job that runs on a runner sets `timeout-minutes`: its measured p95
plus a margin that covers the longest per-test timeout it runs, so meson or
CTest still kills and names a hung test before the job times out.
`ci/gha/README.md` gives the rule and the numbers, and shows how to tell a job
no runner took, which a timeout cannot catch, from one that hung.
The reusable `isa-audit.yml` takes a `timeout-minutes` input, 10 by default.
