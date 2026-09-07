#!/usr/bin/env bash
# Run cooperative-search verification inside a bounded Terra allocation.
set -euo pipefail
if [[ -z "${SLURM_JOB_ID:-}" && -z "${INVOCATION_ID:-}" ]]; then
  printf 'verification requires Slurm or a bounded systemd user service\n' >&2
  exit 2
fi
cd "${ANNEAL_VERIFY_ROOT:?set ANNEAL_VERIFY_ROOT to the source snapshot}"
export PATH="$HOME/.cargo/bin:$PATH"
export CARGO_BUILD_JOBS="${SLURM_CPUS_PER_TASK:-4}"
export CARGO_TARGET_DIR="${ANNEAL_VERIFY_TARGET:?set ANNEAL_VERIFY_TARGET}"
printf 'source=%s allocation=%s rustc=%s\n' "${ANNEAL_SOURCE_REV:?set ANNEAL_SOURCE_REV}" "${SLURM_JOB_ID:-$INVOCATION_ID}" "$(rustc --version)"
case "${1:-full}" in
  regressions)
    status=0
    cargo test --locked --test catalog_policy published_reference_does_not_change_a_stalled_chains_work || status=1
    cargo test --locked --example lj_cluster_search unknown_mechanism_is_rejected_before_search || status=1
    cargo test --locked --test transition_graph unresolved_probes_do_not_certify_shared_return_dynamics || status=1
    exit "$status"
    ;;
  markov)
    cargo test --locked --lib superbasin::tests
    cargo test --locked --test transition_graph
    ;;
  full)
    cargo fmt --all -- --check
    cargo test --locked
    cargo test --locked --features bank-rpc,ira,featomic
    cargo clippy --locked --features bank-rpc,ira,featomic --all-targets -- -D warnings
    ;;
  *)
    printf 'unknown verification mode: %s\n' "$1" >&2
    exit 2
    ;;
esac
