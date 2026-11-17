# Audit/reference mirror of orchestrator prek job (not exported as live workflow).
# Live: .github/workflows/ci-orchestrator.yml
let S = import "lib/steps.ncl" in
let P = S.pins in
{
  name = "prek",
  on = S.on_main_pr_any,
  concurrency = S.concurrency_ref "prek",
  jobs = {
    prek = {
      "runs-on" = P.ubuntu,
      "timeout-minutes" = 10,
      steps = [
        S.checkout_depth 0,
        {
          name = "Install environment",
          uses = P.setup_pixi,
          with = {
            "pixi-version" = P.pixi_version,
            cache = true,
            "cache-write" = "${{ github.event_name == 'push' && github.ref_name == 'main' }}",
            "activate-environment" = true,
          },
        },
        {
          name = "Install prek",
          run = m%"
            if ! command -v prek >/dev/null 2>&1; then
              curl --proto '=https' --tlsv1.2 -LsSf \
                https://github.com/j178/prek/releases/latest/download/prek-installer.sh | sh
              echo "$HOME/.cargo/bin" >> "$GITHUB_PATH"
            fi
            prek --version
          "%,
        },
        {
          name = "Install system hook deps",
          run = m%"
            if ! bash ci/gha/apt-install.sh cppcheck clang-format-16; then
              bash ci/gha/apt-install.sh cppcheck clang-format
            fi
            if ! command -v snapper >/dev/null 2>&1 && ! command -v snapper-fmt >/dev/null 2>&1; then
              curl --proto '=https' --tlsv1.2 -LsSf \
                https://github.com/TurtleTech-ehf/snapper/releases/download/v0.10.0/snapper-fmt-installer.sh | sh || true
              echo "$HOME/.cargo/bin" >> "$GITHUB_PATH"
            fi
            if command -v snapper-fmt >/dev/null 2>&1 && ! command -v snapper >/dev/null 2>&1; then
              ln -sf "$(command -v snapper-fmt)" "$HOME/.cargo/bin/snapper"
              echo "$HOME/.cargo/bin" >> "$GITHUB_PATH"
            fi
          "%,
        },
        S.run_pixi m%"
          if command -v snapper-fmt >/dev/null 2>&1 && ! command -v snapper >/dev/null 2>&1; then
            mkdir -p "$HOME/.local/bin"
            ln -sf "$(command -v snapper-fmt)" "$HOME/.local/bin/snapper"
            export PATH="$HOME/.local/bin:$HOME/.cargo/bin:$PATH"
          fi
          prek validate-config prek.toml
          prek run -a --config prek.toml
        "%
        & { name = "Run prek on all files" },
        {
          name = "Audit for large files",
          run = "bash ci/gha/audit-large-files.sh",
        },
      ],
    },
  },
}
