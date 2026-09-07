#!/usr/bin/env bash
# Run cooperative-search verification inside a bounded Terra allocation.
set -euo pipefail
if [[ -z "${SLURM_JOB_ID:-}" && -z "${INVOCATION_ID:-}" ]]; then
  printf 'verification requires Slurm or a bounded systemd user service\n' >&2
  exit 2
fi
cd "${ANNEAL_VERIFY_ROOT:?set ANNEAL_VERIFY_ROOT to the source snapshot}"
export PATH="$HOME/.cargo/bin:$PATH"
export CARGO_BUILD_JOBS="${ANNEAL_VERIFY_JOBS:-${SLURM_CPUS_PER_TASK:-2}}"
unset CARGO_TARGET_DIR
isolation_toolchain="$PWD/scripts/cargo_target_isolation.cmake"
if [[ -n "${CMAKE_TOOLCHAIN_FILE:-}" && "$CMAKE_TOOLCHAIN_FILE" != "$isolation_toolchain" ]]; then
  export ANNEAL_CMAKE_BASE_TOOLCHAIN="$CMAKE_TOOLCHAIN_FILE"
fi
export CMAKE_TOOLCHAIN_FILE="$isolation_toolchain"
: "${ANNEAL_VERIFY_TARGET:?set ANNEAL_VERIFY_TARGET}"
# Native dependency builds launch their own Cargo process. Its target
# directory must not share the outer process's locked build directory.
verify_cargo() {
  local subcommand="$1"
  shift
  cargo "$subcommand" --target-dir "$ANNEAL_VERIFY_TARGET" "$@"
}
printf 'source=%s allocation=%s rustc=%s\n' "${ANNEAL_SOURCE_REV:?set ANNEAL_SOURCE_REV}" "${SLURM_JOB_ID:-$INVOCATION_ID}" "$(rustc --version)"
case "${1:-full}" in
  regressions)
    status=0
    verify_cargo test --locked --test catalog_policy published_reference_does_not_change_a_stalled_chains_work || status=1
    verify_cargo test --locked --example lj_cluster_search unknown_mechanism_is_rejected_before_search || status=1
    verify_cargo test --locked --test transition_graph unresolved_probes_do_not_certify_shared_return_dynamics || status=1
    exit "$status"
    ;;
  markov)
    verify_cargo test --locked --lib superbasin::tests
    verify_cargo test --locked --test transition_graph
    ;;
  probes)
    status=0
    verify_cargo test --locked --features bank-rpc --test persistent_chain || status=1
    verify_cargo test --locked --features bank-rpc --test cooperative_search || status=1
    verify_cargo test --locked --features bank-rpc --example lj_cluster_search || status=1
    exit "$status"
    ;;
  surfaces)
    verify_cargo test --locked --test surface_evidence
    ;;
  full)
    cargo fmt --all -- --check
    status=0
    verify_cargo test --locked --no-fail-fast || status=1
    verify_cargo test --locked --no-fail-fast --features bank-rpc,ira,featomic || status=1
    verify_cargo clippy --locked --features bank-rpc,ira,featomic --all-targets -- -D warnings || status=1
    exit "$status"
    ;;
  *)
    printf 'unknown verification mode: %s\n' "$1" >&2
    exit 2
    ;;
esac
