#![cfg(unix)]

use anneal_core::campaign::CampaignConfig;
use std::fs;
use std::os::unix::fs::PermissionsExt;
use std::path::{Path, PathBuf};
use std::process::Command;
use std::time::{SystemTime, UNIX_EPOCH};

struct TempTree(PathBuf);

impl TempTree {
    fn new() -> Self {
        let nonce = SystemTime::now()
            .duration_since(UNIX_EPOCH)
            .expect("system clock must follow the Unix epoch")
            .as_nanos();
        let path = std::env::temp_dir().join(format!(
            "anneal-lj-causal-runner-{}-{nonce}",
            std::process::id()
        ));
        fs::create_dir_all(&path)
            .unwrap_or_else(|error| panic!("failed to create {}: {error}", path.display()));
        Self(path)
    }

    fn path(&self) -> &Path {
        &self.0
    }
}

impl Drop for TempTree {
    fn drop(&mut self) {
        let _ = fs::remove_dir_all(&self.0);
    }
}

fn write_executable(path: &Path, source: &str) {
    fs::write(path, source)
        .unwrap_or_else(|error| panic!("failed to write {}: {error}", path.display()));
    let mut permissions = fs::metadata(path)
        .unwrap_or_else(|error| panic!("failed to stat {}: {error}", path.display()))
        .permissions();
    permissions.set_mode(0o755);
    fs::set_permissions(path, permissions)
        .unwrap_or_else(|error| panic!("failed to chmod {}: {error}", path.display()));
}

fn value<'a>(text: &'a str, name: &str) -> &'a str {
    text.lines()
        .find_map(|line| line.strip_prefix(&format!("{name}=")))
        .unwrap_or_else(|| panic!("missing {name} in:\n{text}"))
}

#[test]
fn causal_smoke_campaign_names_every_channel_and_one_paired_ensemble() {
    let path = PathBuf::from(env!("CARGO_MANIFEST_DIR"))
        .join("scripts/campaigns/lj-catalog-causal-smoke.toml");
    let text = fs::read_to_string(&path)
        .unwrap_or_else(|error| panic!("failed to read {}: {error}", path.display()));
    let config = CampaignConfig::parse(&text)
        .unwrap_or_else(|error| panic!("failed to parse {}: {error}", path.display()));

    assert_eq!(config.campaign, "lj-catalog-causal-smoke");
    assert_eq!(config.ensemble.replicas, 4);
    assert_eq!(config.ensemble.wave, 4);
    assert!(config.channels.shared_screen);
    assert!(config.channels.seam_ladder);
    assert!(config.channels.frontier_exchange);
    assert!(!config.channels.shared_bias);
    assert!(!config.channels.entropic_bias);
    assert!(!config.channels.histo_screen);
    assert_eq!(
        config.extra.get("SEED_OFFSET_BASE").map(String::as_str),
        Some("12000000")
    );
}

#[test]
fn causal_runner_pairs_shared_and_private_catalog_topologies() {
    let repository = PathBuf::from(env!("CARGO_MANIFEST_DIR"));
    let runner = repository.join("scripts/elja_jcc_lj_causal_pair.sh");
    assert!(
        runner.is_file(),
        "missing causal runner {}",
        runner.display()
    );

    let temp = TempTree::new();
    let root = temp.path().join("root");
    let bin_dir = root.join("target/release/examples");
    let calibration_dir = root.join("repro/results_jcc/calibration");
    fs::create_dir_all(&bin_dir).unwrap();
    fs::create_dir_all(&calibration_dir).unwrap();
    let source_commit = Command::new("git")
        .args(["-C", repository.to_str().unwrap(), "rev-parse", "HEAD"])
        .output()
        .expect("git must resolve the source checkout");
    assert!(source_commit.status.success());
    fs::write(root.join("SOURCE_COMMIT"), &source_commit.stdout).unwrap();
    fs::write(calibration_dir.join("lj13.json"), "{}\n").unwrap();

    let server = bin_dir.join("catalog_server");
    write_executable(
        &server,
        r#"#!/usr/bin/env bash
set -euo pipefail
replicas=$8
state=$9
printf '%s|%s|%s|%s\n' "$6" "$7" "$replicas" "$state" >>"$FAKE_SERVER_LOG"
case $replicas in
  *,*) port=30999 ;;
  *) port=$((31000 + replicas)) ;;
esac
printf '{"addr":"127.0.0.1:%s"}\n' "$port"
trap 'exit 0' TERM INT
while :; do sleep 1; done
"#,
    );

    let search = bin_dir.join("lj_cluster_search");
    write_executable(
        &search,
        r#"#!/usr/bin/env bash
set -euo pipefail
printf 'catalog_rpc=%s\n' "$CATALOG_RPC"
printf 'brain_listen=%s\n' "${CATALOG_BRAIN_LISTEN:-}"
printf 'brain_peers=%s\n' "${CATALOG_BRAIN_PEERS:-}"
printf 'ensemble=%s\n' "$CATALOG_ENSEMBLE"
printf 'catalog_sharing=%s\n' "$CATALOG_SHARING"
printf 'seed_offset=%s\n' "$SEED_OFFSET"
if [[ ${FAKE_OMIT_CONFIG:-0} != 1 ]]; then
  printf '{"schema":"anneal-cluster-config-v1","soap_mode":"flexible"}\n' \
    >"$ANNEAL_RESOLVED_CONFIG"
fi
case $CATALOG_SHARING in
  shared) sharing=true ;;
  private) sharing=false ;;
  *) exit 3 ;;
esac
printf '{"kind":"manifest_header","campaign":"%s","ensemble":"%s","sharing":%s}\n' \
  "$CATALOG_CAMPAIGN" "$CATALOG_ENSEMBLE" "$sharing" >"$CATALOG_TRACE"
if [[ ${FAKE_OMIT_SLICE:-0} != 1 ]]; then
  printf '{"kind":"slice","replica":%s,"slice":1,"slice_charged_work":1,"slice_energy":-44.326801}\n' \
    "$CATALOG_REPLICA" >>"$CATALOG_TRACE"
fi
if [[ ${FAKE_RPC_FALLBACK:-0} == 1 ]]; then
  printf '{"kind":"rpc_fallback","replica":%s,"sequence":2,"aggregate_charged":1}\n' \
    "$CATALOG_REPLICA" >>"$CATALOG_TRACE"
fi
printf 'LJ%s, budget %s charged evaluations, 1 seeds\n' "$1" "$2"
printf 'seed %s: best -44.326801\n' "$SEED_OFFSET"
printf '1/1 solved\n'
"#,
    );
    let build_seal = Command::new("sha256sum")
        .current_dir(&root)
        .args([
            "target/release/examples/lj_cluster_search",
            "target/release/examples/catalog_server",
        ])
        .output()
        .expect("sha256sum must seal the staged executables");
    assert!(build_seal.status.success());
    fs::write(root.join("BUILD_SHA256SUMS"), build_seal.stdout).unwrap();
    let qualifier = root.join("qualifier.sh");
    write_executable(
        &qualifier,
        r#"#!/usr/bin/env bash
set -euo pipefail
output=
while (($#)); do
  case $1 in
    --output)
      output=$2
      shift 2
      ;;
    *) shift ;;
  esac
done
test -n "$output"
printf '{"qualified":true}\n' >"$output"
"#,
    );

    let out_root = root.join("output");
    let mut seeds_by_arm = Vec::new();
    let mut resolved_hashes = Vec::new();
    for arm in ["shared", "control"] {
        let server_log = root.join(format!("servers-{arm}.log"));
        let output = Command::new("bash")
            .arg(&runner)
            .args(["13", "10", "7", "0.1", arm])
            .env("SLURM_JOB_ID", "4242")
            .env("LJ_ROOT", &root)
            .env("LJ_BIN", &search)
            .env("CATALOG_SERVER_BIN", &server)
            .env("JCC_SOURCE_ROOT", &repository)
            .env("JCC_SOURCE_COMMIT_FILE", root.join("SOURCE_COMMIT"))
            .env("ANNEAL_REPRO_ROOT", root.join("repro"))
            .env("JCC_QUALIFIER", &qualifier)
            .env("JCC_QUALIFIER_PYTHON", "/bin/bash")
            .env("LJ_OUT", &out_root)
            .env("CATALOG_CAMPAIGN", "causal-test")
            .env("CATALOG_REPLICAS", "2")
            .env("CATALOG_WAVE", "2")
            .env("CATALOG_SLICE", "1")
            .env("CATALOG_MAX_HOPS", "3")
            .env("CATALOG_POPULATION_INTERVAL", "10")
            .env("CATALOG_BRAIN_PORT_BASE", "29000")
            .env("SEED_OFFSET_BASE", "1200")
            .env("FAKE_SERVER_LOG", &server_log)
            .env_remove("CATALOG_CONFIG")
            .output()
            .unwrap_or_else(|error| panic!("failed to run {}: {error}", runner.display()));
        assert!(
            output.status.success(),
            "{arm} runner failed\nstdout:\n{}\nstderr:\n{}",
            String::from_utf8_lossy(&output.stdout),
            String::from_utf8_lossy(&output.stderr)
        );

        let ensemble = format!("lj13-{arm}-0007");
        let run = out_root
            .join("causal-test")
            .join("lj13")
            .join(arm)
            .join(ensemble);
        assert!(run.join("TERMINAL_OK").is_file());
        assert!(run.join("qualification.json").is_file());
        let manifest = fs::read_to_string(run.join("run.manifest")).unwrap();
        assert_eq!(value(&manifest, "arm"), arm);
        assert_eq!(value(&manifest, "replicas"), "2");
        assert_eq!(value(&manifest, "per_replica_budget"), "10");
        assert_eq!(value(&manifest, "aggregate_budget"), "20");
        assert_eq!(value(&manifest, "seed_base"), "1214");
        assert_eq!(value(&manifest, "catalog_capacity"), "30");
        assert_eq!(value(&manifest, "transport_noise"), "0.05");
        assert!(run.join("resolved-config.json").is_file());
        assert_eq!(value(&manifest, "resolved_config_sha256").len(), 64);
        assert_eq!(value(&manifest, "qualifier_sha256").len(), 64);
        resolved_hashes.push(value(&manifest, "resolved_config_sha256").to_owned());

        let worker0 = fs::read_to_string(run.join("workers/replica-0.out")).unwrap();
        let worker1 = fs::read_to_string(run.join("workers/replica-1.out")).unwrap();
        seeds_by_arm.push((
            value(&worker0, "seed_offset").to_owned(),
            value(&worker1, "seed_offset").to_owned(),
        ));

        let servers = fs::read_to_string(&server_log).unwrap();
        let server_lines = servers.lines().collect::<Vec<_>>();
        if arm == "shared" {
            assert_eq!(value(&manifest, "catalog_topology"), "shared");
            assert_eq!(value(&worker0, "catalog_sharing"), "shared");
            assert_eq!(value(&worker1, "catalog_sharing"), "shared");
            assert_eq!(server_lines.len(), 1, "{servers}");
            assert!(server_lines[0].contains("|0,1|"), "{servers}");
            assert_eq!(
                value(&worker0, "catalog_rpc"),
                value(&worker1, "catalog_rpc")
            );
            assert!(!value(&worker0, "brain_peers").is_empty());
            assert!(!value(&worker1, "brain_peers").is_empty());
            assert_eq!(value(&worker0, "ensemble"), value(&worker1, "ensemble"));
        } else {
            assert_eq!(value(&manifest, "catalog_topology"), "private_per_replica");
            assert_eq!(value(&worker0, "catalog_sharing"), "private");
            assert_eq!(value(&worker1, "catalog_sharing"), "private");
            assert_eq!(server_lines.len(), 2, "{servers}");
            assert!(server_lines.iter().any(|line| line.contains("|0|")));
            assert!(server_lines.iter().any(|line| line.contains("|1|")));
            assert_ne!(
                value(&worker0, "catalog_rpc"),
                value(&worker1, "catalog_rpc")
            );
            assert_eq!(value(&worker0, "brain_peers"), "");
            assert_eq!(value(&worker1, "brain_peers"), "");
            assert_eq!(value(&worker0, "ensemble"), value(&worker1, "ensemble"));
        }
    }

    assert_eq!(seeds_by_arm[0], seeds_by_arm[1]);
    assert_eq!(resolved_hashes[0], resolved_hashes[1]);

    let incomplete_server_log = root.join("servers-incomplete.log");
    let incomplete = Command::new("bash")
        .arg(&runner)
        .args(["13", "10", "8", "0.1", "shared"])
        .env("SLURM_JOB_ID", "4243")
        .env("LJ_ROOT", &root)
        .env("LJ_BIN", &search)
        .env("CATALOG_SERVER_BIN", &server)
        .env("JCC_SOURCE_COMMIT_FILE", root.join("SOURCE_COMMIT"))
        .env("ANNEAL_REPRO_ROOT", root.join("repro"))
        .env("JCC_QUALIFIER", &qualifier)
        .env("JCC_QUALIFIER_PYTHON", "/bin/bash")
        .env("LJ_OUT", root.join("incomplete-output"))
        .env("CATALOG_CAMPAIGN", "causal-incomplete-test")
        .env("CATALOG_REPLICAS", "2")
        .env("CATALOG_WAVE", "2")
        .env("CATALOG_SLICE", "1")
        .env("CATALOG_MAX_HOPS", "3")
        .env("CATALOG_POPULATION_INTERVAL", "10")
        .env("CATALOG_BRAIN_PORT_BASE", "29100")
        .env("SEED_OFFSET_BASE", "1200")
        .env("FAKE_SERVER_LOG", &incomplete_server_log)
        .env("FAKE_OMIT_SLICE", "1")
        .env_remove("CATALOG_CONFIG")
        .output()
        .unwrap_or_else(|error| panic!("failed to run {}: {error}", runner.display()));
    assert!(
        !incomplete.status.success(),
        "runner accepted traces without analyzer slices\nstdout:\n{}\nstderr:\n{}",
        String::from_utf8_lossy(&incomplete.stdout),
        String::from_utf8_lossy(&incomplete.stderr)
    );

    let missing_config_server_log = root.join("servers-missing-config.log");
    let missing_config = Command::new("bash")
        .arg(&runner)
        .args(["13", "10", "9", "0.1", "shared"])
        .env("SLURM_JOB_ID", "4244")
        .env("LJ_ROOT", &root)
        .env("LJ_BIN", &search)
        .env("CATALOG_SERVER_BIN", &server)
        .env("JCC_SOURCE_COMMIT_FILE", root.join("SOURCE_COMMIT"))
        .env("ANNEAL_REPRO_ROOT", root.join("repro"))
        .env("JCC_QUALIFIER", &qualifier)
        .env("JCC_QUALIFIER_PYTHON", "/bin/bash")
        .env("LJ_OUT", root.join("missing-config-output"))
        .env("CATALOG_CAMPAIGN", "causal-missing-config-test")
        .env("CATALOG_REPLICAS", "2")
        .env("CATALOG_WAVE", "2")
        .env("CATALOG_SLICE", "1")
        .env("CATALOG_MAX_HOPS", "3")
        .env("CATALOG_POPULATION_INTERVAL", "10")
        .env("CATALOG_BRAIN_PORT_BASE", "29200")
        .env("SEED_OFFSET_BASE", "1200")
        .env("FAKE_SERVER_LOG", &missing_config_server_log)
        .env("FAKE_OMIT_CONFIG", "1")
        .env_remove("CATALOG_CONFIG")
        .output()
        .unwrap_or_else(|error| panic!("failed to run {}: {error}", runner.display()));
    assert!(
        !missing_config.status.success(),
        "runner accepted workers without resolved configurations\nstdout:\n{}\nstderr:\n{}",
        String::from_utf8_lossy(&missing_config.stdout),
        String::from_utf8_lossy(&missing_config.stderr)
    );

    let fallback_server_log = root.join("servers-fallback.log");
    let fallback = Command::new("bash")
        .arg(&runner)
        .args(["13", "10", "10", "0.1", "shared"])
        .env("SLURM_JOB_ID", "4245")
        .env("LJ_ROOT", &root)
        .env("LJ_BIN", &search)
        .env("CATALOG_SERVER_BIN", &server)
        .env("JCC_SOURCE_COMMIT_FILE", root.join("SOURCE_COMMIT"))
        .env("ANNEAL_REPRO_ROOT", root.join("repro"))
        .env("JCC_QUALIFIER", &qualifier)
        .env("JCC_QUALIFIER_PYTHON", "/bin/bash")
        .env("LJ_OUT", root.join("fallback-output"))
        .env("CATALOG_CAMPAIGN", "causal-fallback-test")
        .env("CATALOG_REPLICAS", "2")
        .env("CATALOG_WAVE", "2")
        .env("CATALOG_SLICE", "1")
        .env("CATALOG_MAX_HOPS", "3")
        .env("CATALOG_POPULATION_INTERVAL", "10")
        .env("CATALOG_BRAIN_PORT_BASE", "29300")
        .env("SEED_OFFSET_BASE", "1200")
        .env("FAKE_SERVER_LOG", &fallback_server_log)
        .env("FAKE_RPC_FALLBACK", "1")
        .env_remove("CATALOG_CONFIG")
        .output()
        .unwrap_or_else(|error| panic!("failed to run {}: {error}", runner.display()));
    assert!(
        !fallback.status.success(),
        "runner sealed traces containing RPC fallback evidence\nstdout:\n{}\nstderr:\n{}",
        String::from_utf8_lossy(&fallback.stdout),
        String::from_utf8_lossy(&fallback.stderr)
    );
}
