# Generated target: .github/workflows/cosmo-potctl.yml
let S = import "lib/steps.ncl" in
let P = S.pins in
let cosmo_paths = [
  "potctl/**",
  ".github/workflows/cosmo-potctl.yml",
  ".github/workflows/ci-orchestrator.yml",
  ".github/workflows/release.yml",
  ".github/workflows/release-prepare.yml",
  ".github/actions/setup-ci-tools/**",
  ".github/actions/ensure-potctl/**",
  ".github/actions/restore-ci-tools/**",
  "ci/gha/**",
]
in
let install_cosmo = m%"
set -euo pipefail
need_build=0
if [ ! -d .cosmo-build ]; then
  need_build=1
fi
COSMO_ROOT="${GITHUB_WORKSPACE}/.cosmo-build"
if [ "$need_build" -eq 0 ]; then
  has_cc=0
  for cand in \
    "${COSMO_ROOT}/bin/x86_64-unknown-cosmo-cc" \
    "${COSMO_ROOT}/o/third_party/gcc/bin/x86_64-linux-cosmo-gcc"
  do
    [ -x "$cand" ] && has_cc=1 && break
  done
  if [ "$has_cc" -eq 0 ] && [ -d "${COSMO_ROOT}/.cosmocc" ]; then
    if find "${COSMO_ROOT}/.cosmocc" -type f -name 'x86_64-linux-cosmo-gcc' -executable 2>/dev/null | head -1 | grep -q .; then
      has_cc=1
    fi
  fi
  [ "$has_cc" -eq 0 ] && need_build=1
fi
if [ "$need_build" -eq 1 ]; then
  rm -rf .cosmo-build
  git clone --depth 1 --branch "${COSMO_REF}" https://github.com/jart/cosmopolitan .cosmo-build
  cd .cosmo-build
  make -j"$(nproc)" MODE= toolchain || make -j"$(nproc)" toolchain
  cd "${GITHUB_WORKSPACE}"
else
  echo "Using cached .cosmo-build (cosmo CC present)"
  ls -la .cosmo-build | head -8
fi
echo "COSMO=${COSMO_ROOT}" >> "$GITHUB_ENV"
CC_FOUND=""
for cand in \
  "${COSMO_ROOT}/bin/x86_64-unknown-cosmo-cc" \
  "${COSMO_ROOT}/o/third_party/gcc/bin/x86_64-linux-cosmo-gcc"
do
  if [ -x "$cand" ]; then CC_FOUND="$cand"; break; fi
done
if [ -z "$CC_FOUND" ] && [ -d "${COSMO_ROOT}/.cosmocc" ]; then
  CC_FOUND="$(find "${COSMO_ROOT}/.cosmocc" -type f -name 'x86_64-linux-cosmo-gcc' -executable 2>/dev/null | head -1 || true)"
fi
if [ -z "$CC_FOUND" ] && [ -d "${COSMO_ROOT}/.cosmocc" ]; then
  CC_FOUND="$(find "${COSMO_ROOT}/.cosmocc" -type f -name cosmocc -executable 2>/dev/null | head -1 || true)"
  if [ -n "$CC_FOUND" ]; then
    echo "COSMOCC_ARCHES=x86_64" >> "$GITHUB_ENV"
  fi
fi
if [ -n "$CC_FOUND" ]; then
  echo "COSMO_CC=${CC_FOUND}" >> "$GITHUB_ENV"
  echo "COSMOCC_ARCHES=x86_64" >> "$GITHUB_ENV"
  echo "Found cosmo CC: ${CC_FOUND}"
  BINDIR="$(dirname "$CC_FOUND")"
  echo "${BINDIR}" >> "$GITHUB_PATH"
  # Do NOT set LIBRARY_PATH globally — breaks host cc for build scripts.
  ls -la "${BINDIR}" | head -15 || true
  find "${COSMO_ROOT}/.cosmocc" \( -name 'crt1.o' -o -name 'crti.o' -o -name 'libc.a' -o -name 'crt.o' \) 2>/dev/null | head -15 || true
else
  echo "ERROR: no cosmo CC after toolchain install"
  find "${COSMO_ROOT}" -maxdepth 6 \( -name 'x86_64-linux-cosmo-gcc' -o -name cosmocc -o -name '*cosmo*cc' \) 2>/dev/null | head -40 || true
  exit 1
fi
"%
in
let install_rustup = m%"
set -euo pipefail
if ! command -v rustup >/dev/null; then
  curl --proto '=https' --tlsv1.2 -sSf https://sh.rustup.rs | sh -s -- -y --default-toolchain none
  echo "$HOME/.cargo/bin" >> "$GITHUB_PATH"
fi
# shellcheck disable=SC1091
source "$HOME/.cargo/env" 2>/dev/null || true
rustup toolchain install nightly --profile minimal
rustup component add rust-src --toolchain nightly
"%
in
let build_ape = m%"
set -euo pipefail
# shellcheck disable=SC1091
source "$HOME/.cargo/env" 2>/dev/null || true
# Host potctl + potctl-cosmo-ld (Rust linker shim for rustc cosmo target).
cargo build --release -p potctl --bins
export PATH="${GITHUB_WORKSPACE}/target/release:${PATH}"
command -v potctl
command -v potctl-cosmo-ld
export COSMO="${GITHUB_WORKSPACE}/.cosmo-build"
export POTCTL_COSMO_ARCHS=x86_64
export POTCTL_COSMO_OUT="${GITHUB_WORKSPACE}/potctl/cosmo/out"
export POTCTL_COSMO_SMOKE=1
export PATH="${COSMO}:${COSMO}/bin:${PATH}"
if [ -d "${COSMO}/.cosmocc" ]; then
  BINDIR="$(find "${COSMO}/.cosmocc" -type d -name bin 2>/dev/null | head -1 || true)"
  [ -n "$BINDIR" ] && export PATH="${BINDIR}:${PATH}"
fi
# cosmocc owns APE CRT/specs; potctl-cosmo-ld plants aarch64 companions + extracts rlibs.
export COSMOCC_ARCHES="${COSMOCC_ARCHES:-x86_64}"
export COSMO_CC=""
if [ -d "${COSMO}/.cosmocc" ]; then
  export COSMO_CC="$(find "${COSMO}/.cosmocc" -type f -name cosmocc -executable 2>/dev/null | head -1 || true)"
fi
if [ -z "${COSMO_CC:-}" ] && [ -x "${COSMO}/bin/cosmocc" ]; then
  export COSMO_CC="${COSMO}/bin/cosmocc"
fi
if [ -z "${COSMO_CC:-}" ]; then
  export COSMO_CC="$(find "${COSMO}/.cosmocc" -type f -name 'x86_64-linux-cosmo-gcc' -executable 2>/dev/null | head -1 || true)"
fi
echo "COSMO_CC=${COSMO_CC:-<unset>} COSMOCC_ARCHES=${COSMOCC_ARCHES}"
test -n "${COSMO_CC:-}" && test -x "${COSMO_CC}"
export POTCTL_COSMO_LINKER_TRACE="${POTCTL_COSMO_LINKER_TRACE:-0}"
potctl cosmo build
test -f potctl/cosmo/out/potctl.com
test -x potctl/cosmo/out/potctl.com || chmod +x potctl/cosmo/out/potctl.com
ls -la potctl/cosmo/out/
./potctl/cosmo/out/potctl.com --help | head -8
./potctl/cosmo/out/potctl.com ci preflight
./potctl/cosmo/out/potctl.com release assert
"%
in
let smoke_ape = m%"
set -euo pipefail
BIN="${RUNNER_TEMP}/cosmo-potctl/potctl.com"
chmod +x "$BIN" || true
file "$BIN" 2>/dev/null || true
head -c 120 "$BIN" | xxd | head -3 || true
"$BIN" --version 2>&1 || "$BIN" --help 2>&1 | head -8
"$BIN" ci preflight
"$BIN" release assert
echo "cosmo potctl smoke ok on ${{ matrix.os }}"
"%
in
{
  name = "potctl Cosmopolitan APE",
  on = {
    pull_request = { paths = cosmo_paths },
    workflow_dispatch = {},
    push = { branches = ["main"], paths = cosmo_paths },
  },
  concurrency = S.concurrency_ref "cosmo-potctl",
  permissions = { contents = "read" },
  env = { COSMO_REF = "master" },
  jobs = {
    "build-ape" = {
      name = "build potctl.com (cosmo)",
      "runs-on" = P.ubuntu,
      "timeout-minutes" = 120,
      steps = [
        S.checkout_minimal,
        {
          name = "Cache Cosmopolitan tree",
          uses = P.cache,
          with = {
            path = ".cosmo-build",
            key = "cosmo-${{ env.COSMO_REF }}-${{ runner.os }}-v3",
            "restore-keys" = m%"
              cosmo-${{ env.COSMO_REF }}-${{ runner.os }}-v3
              cosmo-${{ env.COSMO_REF }}-${{ runner.os }}-
            "%,
          },
        },
        { name = "Install Cosmopolitan toolchain (or restore cache)", run = install_cosmo },
        { name = "Install rustup nightly + rust-src", run = install_rustup },
        {
          name = "Cache cargo (cosmo target)",
          uses = P.cache,
          with = {
            path = m%"
              ~/.cargo/registry
              ~/.cargo/git
              target/cosmo-x86_64
            "%,
            key = "cosmo-cargo-${{ runner.os }}-${{ hashFiles('potctl/Cargo.toml', 'Cargo.lock') }}-v1",
            "restore-keys" = "cosmo-cargo-${{ runner.os }}-",
          },
        },
        { name = "Build potctl APE (potctl cosmo build)", id = "ape", run = build_ape },
        {
          name = "Upload potctl.com APE artifact",
          "if" = "success()",
          uses = P.upload_artifact,
          with = {
            name = "potctl-cosmo-ape",
            path = "potctl/cosmo/out/potctl.com",
            "retention-days" = 14,
            "if-no-files-found" = "error",
          },
        },
      ],
    },
    "smoke-ape" = {
      name = "smoke potctl.com (${{ matrix.os }})",
      needs = ["build-ape"],
      "if" = "needs.build-ape.result == 'success'",
      "runs-on" = "${{ matrix.os }}",
      strategy = {
        "fail-fast" = false,
        matrix = { os = [P.ubuntu, P.macos] },
      },
      steps = [
        S.checkout_minimal,
        {
          name = "Download potctl.com (same artifact on every OS)",
          uses = P.download_artifact,
          with = {
            name = "potctl-cosmo-ape",
            path = "${{ runner.temp }}/cosmo-potctl",
          },
        },
        {
          name = "Run APE on ${{ matrix.os }}",
          run = smoke_ape,
          "working-directory" = "${{ github.workspace }}",
        },
      ],
    },
  },
}
