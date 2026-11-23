# Audit/reference mirror of orchestrator docs_snapper/docs_lychee (not exported as live workflow).
# Live: .github/workflows/ci-orchestrator.yml
let S = import "lib/steps.ncl" in
let P = S.pins in
{
  name = "Docs quality (snapper + lychee)",
  on = {
    push = { branches = ["main"] },
    pull_request = {},
    schedule = [{ cron = "15 7 * * 1" }],
  },
  concurrency = S.concurrency_cancel,
  permissions = { contents = "read" },
  jobs = {
    snapper = {
      name = "snapper semantic line breaks",
      "runs-on" = P.ubuntu,
      "timeout-minutes" = 10,
      steps = [
        S.checkout_minimal,
        {
          name = "Install snapper-fmt",
          run = m%"
            curl --proto '=https' --tlsv1.2 -LsSf \
              https://github.com/TurtleTech-ehf/snapper/releases/download/v0.10.0/snapper-fmt-installer.sh | sh
            echo "$HOME/.cargo/bin" >> "$GITHUB_PATH"
            if command -v snapper-fmt >/dev/null 2>&1 && ! command -v snapper >/dev/null 2>&1; then
              ln -sf "$(command -v snapper-fmt)" "$HOME/.cargo/bin/snapper"
            fi
            snapper --version || snapper-fmt --version
          "%,
        },
        {
          name = "Check prose formatting",
          run = m%"
            set -euo pipefail
            SNAPPER_BIN="$(command -v snapper || command -v snapper-fmt)"
            mapfile -t files < <(find docs/orgmode -type f -name '*.org' ! -path '*/.pixi/*' | sort)
            if [ "${#files[@]}" -eq 0 ]; then
              echo "No org prose files found"
              exit 0
            fi
            printf 'Checking %s org file(s)\n' "${#files[@]}"
            if ! out=$("$SNAPPER_BIN" --check --config .snapper.toml --format org "${files[@]}" 2>&1); then
              printf '%s\n' "$out"
              exit 1
            fi
            printf '%s\n' "$out"
            if printf '%s\n' "$out" | grep -q 'would reformat:'; then
              echo "::error::snapper would reformat prose; run \`pixi r snapper-fix\` and commit"
              exit 1
            fi
          "%,
        },
      ],
    },
    lychee = {
      name = "lychee link check",
      "runs-on" = P.ubuntu,
      "timeout-minutes" = 10,
      steps = [
        S.checkout_minimal,
        {
          name = "Restore lychee cache",
          uses = P.cache,
          with = {
            path = ".lycheecache",
            key = "lychee-${{ runner.os }}-${{ hashFiles('.lychee.toml', 'README.md', 'docs/**/*.org', 'docs/**/*.md') }}",
            "restore-keys" = "lychee-${{ runner.os }}-",
          },
        },
        {
          name = "Link check",
          uses = P.lychee,
          with = {
            args = m%"
              --config .lychee.toml
              --cache
              --max-cache-age 2d
              --cache-exclude-status 429,500,502,503,504
              --no-progress
              README.md
              CHANGELOG.md
              CODE_OF_CONDUCT.md
              readme_src.org
              docs/orgmode/**/*.org
              rgpot-core/README.md
              rgpot-core/README.org
            "%,
            fail = true,
          },
          env = { GITHUB_TOKEN = "${{ secrets.GITHUB_TOKEN }}" },
        },
      ],
    },
  },
}
