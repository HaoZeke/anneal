# Generated target: .github/workflows/release.yml
let S = import "lib/steps.ncl" in
let P = S.pins in
{
  name = "Release",
  on = { push = { tags = ["v*"] } },
  permissions = { contents = "write" },
  env = { CI_TOOLS_ARTIFACT = "ci-tools-${{ github.sha }}-Linux" },
  jobs = {
    tools = S.job_tools_linux "false" "Stage portable potctl (Linux musl + upload)",
    gate = {
      needs = ["tools"],
      "runs-on" = P.ubuntu,
      outputs = {
        is_prerelease = "${{ steps.prerelease.outputs.is_prerelease }}",
        version = "${{ steps.meta.outputs.version }}",
        tag = "${{ steps.meta.outputs.tag }}",
      },
      steps = [
        S.checkout_depth 0,
        S.restore_ci_tools_linux,
        {
          name = "Tag / version meta",
          id = "meta",
          run = m%"
            TAG="${GITHUB_REF#refs/tags/}"
            VERSION="${TAG#v}"
            echo "tag=${TAG}" >> "$GITHUB_OUTPUT"
            echo "version=${VERSION}" >> "$GITHUB_OUTPUT"
            echo "Release gate for ${TAG} (${VERSION})"
          "%,
        },
        {
          name = "Assert lockstep versions + CHANGELOG section (fail fast)",
          run = m%"
            VERSION="${{ steps.meta.outputs.version }}"
            potctl release assert --require-changelog "$VERSION"
            potctl release notes "$VERSION" -o "${RUNNER_TEMP}/release_notes.md"
            echo "--- release notes (head) ---"
            head -25 "${RUNNER_TEMP}/release_notes.md"
          "%,
        },
        {
          name = "Upload release notes for gh-release job",
          uses = P.upload_artifact,
          with = {
            name = "release-notes-${{ github.sha }}",
            path = "${{ runner.temp }}/release_notes.md",
            "retention-days" = 3,
          },
        },
        {
          name = "Detect pre-release (RC) tag",
          id = "prerelease",
          run = m%"
            VERSION="${{ steps.meta.outputs.version }}"
            if [[ "$VERSION" == *-* ]]; then
              echo "is_prerelease=true" >> "$GITHUB_OUTPUT"
              echo "Pre-release: will skip cargo publish (crates.io stable only)."
            else
              echo "is_prerelease=false" >> "$GITHUB_OUTPUT"
            fi
          "%,
        },
      ],
    },
    "cpp-smoke" = {
      needs = ["gate"],
      "runs-on" = P.ubuntu,
      steps = [
        S.checkout_minimal,
        S.setup_pixi_default "devbld",
        S.run_pixi m%"
          meson setup build-release-smoke \
            -Dwith_tests=false \
            -Dwith_examples=false \
            -Dwith_rpc=false \
            -Dwith_cache=false \
            -Dpure_lib=true \
            --prefix="${RUNNER_TEMP}/rgpot-prefix"
          meson compile -C build-release-smoke
          meson install -C build-release-smoke
          test -d "${RUNNER_TEMP}/rgpot-prefix/include" || test -d "${RUNNER_TEMP}/rgpot-prefix/lib" \
            || ls -la "${RUNNER_TEMP}/rgpot-prefix"
        "%
        & { name = "C++ install smoke (meson, default features off)" },
      ],
    },
    "rust-test" = {
      needs = ["gate"],
      "runs-on" = P.ubuntu,
      steps = [
        S.checkout_minimal,
        S.setup_pixi_default "devbld",
        { uses = P.rust_toolchain, with = { toolchain = "stable" } },
        {
          uses = P.rust_cache,
          with = {
            workspaces = ". -> target",
            "shared-key" = "rgpot-core-release",
            "cache-targets" = true,
          },
        },
        {
          name = "Install cargo-nextest",
          uses = P.install_action,
          with = { tool = "cargo-nextest" },
        },
        S.run_pixi m%"
          cargo nextest run -p rgpot-core --all-features
          cargo test --doc -p rgpot-core
        "%
        & { name = "Run Rust tests" },
      ],
    },
    publish = {
      needs = ["gate", "cpp-smoke", "rust-test"],
      "if" = "needs.gate.outputs.is_prerelease != 'true'",
      "runs-on" = P.ubuntu,
      steps = [
        S.checkout_minimal,
        { uses = P.rust_toolchain, with = { toolchain = "stable" } },
        {
          uses = P.rust_cache,
          with = {
            workspaces = ". -> target",
            "shared-key" = "rgpot-core-release",
            "cache-targets" = true,
          },
        },
        {
          name = "Cargo publish (rgpot-core, locked only)",
          # Skip when package.publish = false (git deps not on crates.io yet).
          run = m%"
            set -euo pipefail
            if grep -qE '^publish\s*=\s*false' rgpot-core/Cargo.toml; then
              echo "rgpot-core has publish = false; skipping crates.io upload."
              echo "Re-enable publish and switch git deps to registry versions first."
              exit 0
            fi
            # publish.yml (OIDC) uploads the same tag; whichever finishes
            # second must not fail the release and skip gh-release.
            version=$(cargo metadata --no-deps --format-version 1 \
              | jq -r '.packages[] | select(.name == "rgpot-core") | .version')
            if curl -fsS -A "rgpot-release (github.com/OmniPotentRPC/rgpot)" \
                "https://crates.io/api/v1/crates/rgpot-core/${version}" >/dev/null; then
              echo "rgpot-core ${version} is already on crates.io; skipping cargo publish."
              exit 0
            fi
            cargo publish -p rgpot-core --locked
          "%,
          env = { CARGO_REGISTRY_TOKEN = "${{ secrets.CARGO_REGISTRY_TOKEN }}" },
        },
      ],
    },
    "publish-skip-rc" = {
      needs = ["gate", "cpp-smoke", "rust-test"],
      "if" = "needs.gate.outputs.is_prerelease == 'true'",
      "runs-on" = P.ubuntu,
      steps = [
        {
          run = m%"
            echo "Skipped cargo publish for pre-release ${{ needs.gate.outputs.tag }}."
            echo "GitHub Release will still be created as prerelease."
          "%,
        },
      ],
    },
    "gh-release" = {
      needs = ["gate", "cpp-smoke", "rust-test", "publish", "publish-skip-rc"],
      "if" = m%"
        always() &&
        needs.gate.result == 'success' &&
        needs.cpp-smoke.result == 'success' &&
        needs.rust-test.result == 'success' &&
        (needs.publish.result == 'success' || needs.publish.result == 'skipped') &&
        (needs.publish-skip-rc.result == 'success' || needs.publish-skip-rc.result == 'skipped')
      "%,
      "runs-on" = P.ubuntu,
      steps = [
        S.checkout_minimal,
        {
          name = "Download release notes",
          uses = P.download_artifact,
          with = {
            name = "release-notes-${{ github.sha }}",
            path = "${{ runner.temp }}/release-notes",
          },
        },
        {
          name = "Create GitHub Release",
          uses = P.gh_release,
          with = {
            body_path = "${{ runner.temp }}/release-notes/release_notes.md",
            generate_release_notes = false,
            fail_on_unmatched_files = false,
            prerelease = "${{ needs.gate.outputs.is_prerelease == 'true' }}",
          },
        },
      ],
    },
  },
}
