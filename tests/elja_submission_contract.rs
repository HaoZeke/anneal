use std::fs;
#[cfg(unix)]
use std::os::unix::fs::PermissionsExt;
use std::path::PathBuf;

#[test]
fn hard_lj_arrays_default_to_the_partition_limit() {
    let script = PathBuf::from(env!("CARGO_MANIFEST_DIR"))
        .join("scripts")
        .join("elja_submit_jcc_lj.sh");
    let source = fs::read_to_string(&script)
        .unwrap_or_else(|error| panic!("failed to read {}: {error}", script.display()));

    assert!(
        source.contains(r#"--time="${ELJA_TIME:-2-00:00:00}""#),
        "hard-LJ arrays must default to the s-normal two-day limit"
    );
}

#[test]
fn hard_lj_arrays_submit_causal_pairs_at_paper_budgets() {
    let script = PathBuf::from(env!("CARGO_MANIFEST_DIR"))
        .join("scripts")
        .join("elja_submit_jcc_lj.sh");
    let source = fs::read_to_string(&script)
        .unwrap_or_else(|error| panic!("failed to read {}: {error}", script.display()));

    assert!(
        source.contains("RUNNER=$ROOT/scripts/elja_jcc_lj_causal_pair.sh"),
        "hard-LJ arrays must use the paired shared/private causal runner"
    );
    assert!(
        source.contains("for n in 38 55 75 98 102 104; do"),
        "hard-LJ arrays must cover every analyzed cluster"
    );
    for (system, budget) in [
        (38, 400_000),
        (55, 1_000_000),
        (75, 4_000_000),
        (98, 4_000_000),
        (102, 4_000_000),
        (104, 4_000_000),
    ] {
        assert!(
            source.contains(&format!("[{system}]={budget}")),
            "LJ{system} must use a per-replica budget of {budget}"
        );
    }
    assert!(
        source.contains(r#""$RUNNER" "$n" "$budget" slurm-array "$radius" "$arm""#),
        "the causal runner must receive system, budget, index, radius, and arm"
    );
    assert!(
        source.contains(r#"--cpus-per-task="${ELJA_CPUS_PER_TASK:-8}""#),
        "both arms must reserve capacity for four workers and four coordinators"
    );
}

#[test]
fn hard_lj_arrays_accept_an_isolated_campaign_name() {
    let script = PathBuf::from(env!("CARGO_MANIFEST_DIR"))
        .join("scripts")
        .join("elja_submit_jcc_lj.sh");
    let source = fs::read_to_string(&script)
        .unwrap_or_else(|error| panic!("failed to read {}: {error}", script.display()));

    assert!(
        source.contains(r#"CAMPAIGN=${JCC_CAMPAIGN:-jcc-2026-${STAGE}}"#),
        "hard-LJ arrays must permit a caller-selected campaign namespace"
    );
    assert!(
        source.contains(r#"CATALOG_CAMPAIGN=${CAMPAIGN}"#),
        "the selected namespace must reach every ensemble array"
    );
    assert!(
        source.contains(r#"JCC_QUALIFIER_PYTHON=${QUALIFIER_PYTHON}"#),
        "the validated launcher Python must reach every ensemble qualifier"
    );
}

#[test]
fn hard_lj_manifests_pin_every_scientific_input() {
    let script = PathBuf::from(env!("CARGO_MANIFEST_DIR"))
        .join("scripts")
        .join("elja_jcc_lj_causal_pair.sh");
    let source = fs::read_to_string(&script)
        .unwrap_or_else(|error| panic!("failed to read {}: {error}", script.display()));

    for required in [
        r#"source_commit=%s\n"#,
        r#"catalog_topology=%s\n"#,
        r#"brain_topology=%s\n"#,
        r#"wave=%s\n"#,
        r#"resolved_config_sha256=%s\n"#,
        r#"binary_sha256=%s\n"#,
        r#"server_sha256=%s\n"#,
        r#"runner_sha256=%s\n"#,
        r#"qualifier_sha256=%s\n"#,
        r#"calibration_sha256=%s\n"#,
        r#"ANNEAL_RESOLVED_CONFIG=$worker/resolved-config.json"#,
        r#""$QUALIFIER_PYTHON" "$QUALIFIER""#,
    ] {
        assert!(
            source.contains(required),
            "hard-LJ manifest is missing provenance field {required}"
        );
    }
}

#[test]
fn hard_lj_campaigns_bind_every_coordination_channel() {
    let root = PathBuf::from(env!("CARGO_MANIFEST_DIR"));
    let submitter = fs::read_to_string(root.join("scripts/elja_submit_jcc_lj.sh"))
        .unwrap_or_else(|error| panic!("failed to read LJ submitter: {error}"));
    let runner = fs::read_to_string(root.join("scripts/elja_jcc_lj_causal_pair.sh"))
        .unwrap_or_else(|error| panic!("failed to read LJ runner: {error}"));

    for (name, value) in [
        ("CATALOG_SHARED_SCREEN", "1"),
        ("CATALOG_SHARED_BIAS", "0"),
        ("CATALOG_ENTROPIC_BIAS", "0"),
        ("CATALOG_HISTO_SCREEN", "0"),
        ("CATALOG_SEAM_LADDER", "1"),
        ("CATALOG_FRONTIER_EXCHANGE", "1"),
        ("CATALOG_COOP_WELLS", "1"),
        ("CATALOG_BRIDGE", "0"),
        ("CATALOG_DIFFICULTY", "0"),
        ("CATALOG_PACKING_PAVE", "0"),
    ] {
        assert!(
            submitter.contains(&format!("{name}={value}")),
            "the LJ submitter must bind {name}={value}"
        );
        assert!(
            runner.contains(&format!("require_protocol_value {name} {value}")),
            "the LJ runner must reject a conflicting {name}"
        );
        let manifest_name = name
            .strip_prefix("CATALOG_")
            .unwrap()
            .to_ascii_lowercase();
        assert!(
            runner.contains(&format!("{manifest_name}=%s\\n")),
            "the LJ manifest must record {name}"
        );
    }

    for forbidden in ["CATALOG_TEMP_LADDER", "CATALOG_MD_ENGINE"] {
        assert!(
            runner.contains(&format!("reject_protocol_variable {forbidden}")),
            "the production LJ runner must reject inherited {forbidden}"
        );
    }
}

#[test]
fn molecular_build_produces_every_paired_campaign_executable() {
    let script = PathBuf::from(env!("CARGO_MANIFEST_DIR"))
        .join("scripts")
        .join("elja_build_rgpot_ex.sh");
    let source = fs::read_to_string(&script)
        .unwrap_or_else(|error| panic!("failed to read {}: {error}", script.display()));

    for example in [
        "molecular_cluster",
        "slab_adsorption",
        "bank_server",
        "bank_peek",
    ] {
        assert!(
            source.contains(&format!("--example {example}")),
            "Elja molecular build must compile {example}"
        );
        assert!(
            source.contains(&format!("target/release/examples/{example}")),
            "Elja molecular build must verify {example}"
        );
    }
}

#[test]
fn causal_campaigns_bind_source_to_built_artifacts() {
    let root = PathBuf::from(env!("CARGO_MANIFEST_DIR"));
    let lj_build = fs::read_to_string(root.join("scripts/elja_build_lj.sh"))
        .unwrap_or_else(|error| panic!("failed to read LJ build: {error}"));
    assert!(
        lj_build.contains("cargo build --offline --locked --release"),
        "the LJ build must use the committed dependency graph"
    );
    let lj_submit = fs::read_to_string(root.join("scripts/elja_submit_jcc_lj.sh"))
        .unwrap_or_else(|error| panic!("failed to read LJ submitter: {error}"));
    let lj_runner = fs::read_to_string(root.join("scripts/elja_jcc_lj_causal_pair.sh"))
        .unwrap_or_else(|error| panic!("failed to read LJ runner: {error}"));
    for (name, source) in [("LJ submitter", lj_submit), ("LJ runner", lj_runner)] {
        assert!(
            source.contains("SOURCE_COMMIT=$SOURCE_COMMIT does not match HEAD=$HEAD"),
            "{name} must compare the recorded source with Git HEAD"
        );
        assert!(
            source.contains("sha256sum -c BUILD_SHA256SUMS"),
            "{name} must verify the LJ build artifact seal"
        );
        assert!(
            source.contains("git diff --quiet HEAD --"),
            "{name} must reject tracked source changes"
        );
    }

    let molecular_build = fs::read_to_string(root.join("scripts/elja_build_rgpot_ex.sh"))
        .unwrap_or_else(|error| panic!("failed to read molecular build: {error}"));
    assert!(
        molecular_build.contains("git diff --quiet HEAD --")
            && molecular_build.contains("git rev-parse HEAD >SOURCE_COMMIT"),
        "the molecular build must bind its artifacts to a clean source commit"
    );
    assert!(
        molecular_build.contains("MOLSLAB_BUILD_SHA256SUMS"),
        "the molecular build must seal engines, drivers, bank, and inspector"
    );
    assert!(
        molecular_build.contains("cargo build --offline --locked --release"),
        "the molecular build must use the committed dependency graph"
    );
    for script in ["elja_submit_jcc_molslab.sh", "elja_jcc_molslab_ensemble.sh"] {
        let source = fs::read_to_string(root.join("scripts").join(script))
            .unwrap_or_else(|error| panic!("failed to read {script}: {error}"));
        assert!(
            source.contains("SOURCE_COMMIT=$SOURCE_COMMIT does not match HEAD=$HEAD"),
            "{script} must compare the recorded source with Git HEAD"
        );
        assert!(
            source.contains("sha256sum -c MOLSLAB_BUILD_SHA256SUMS"),
            "{script} must verify the molecular build artifact seal"
        );
        assert!(
            source.contains("git diff --quiet HEAD --"),
            "{script} must reject tracked source changes"
        );
    }
}

#[test]
fn production_builders_use_the_committed_dependency_graph() {
    let root = PathBuf::from(env!("CARGO_MANIFEST_DIR"));
    for script in [
        "elja_build_brains.sh",
        "elja_build_lj.sh",
        "elja_build_rgpot_ex.sh",
        "terra_build_brains.sh",
        "terra_build_lj.sh",
    ] {
        let source = fs::read_to_string(root.join("scripts").join(script))
            .unwrap_or_else(|error| panic!("failed to read {script}: {error}"));
        let logical_lines = source.replace("\\\n", " ");
        for command in logical_lines
            .lines()
            .filter(|line| line.contains("cargo build ") || line.contains("cargo test "))
        {
            assert!(
                command.contains("--locked"),
                "{script} must use the committed dependency graph: {command}"
            );
        }
    }
}

#[test]
fn terra_molecular_builder_seals_exact_sources_engines_and_drivers() {
    let root = PathBuf::from(env!("CARGO_MANIFEST_DIR"));
    let script = root.join("scripts/terra_build_rgpot_ex.sh");
    let source = fs::read_to_string(&script)
        .unwrap_or_else(|error| panic!("failed to read {}: {error}", script.display()));

    for required in [
        "SLURM_JOB_ID",
        "EXPECTED_COMMIT",
        "RGPOT_EXPECTED_COMMIT",
        "git diff --quiet HEAD --",
        "git -C \"$RGPOT\" diff --quiet HEAD --",
        "install --locked -e xtbbld",
        "run --locked -e xtbbld meson setup",
        "run --locked -e xtbbld meson compile",
        "-Dwith_xtb=true",
        "-Dwith_fortran_pots=enabled",
        "cargo build --locked --release",
        "rgpot_xtb_force",
        "rgpot_cuh2_force",
        "engines/libxtb_engine.so",
        "engines/librgpot_cuh2.so",
        "git rev-parse HEAD >SOURCE_COMMIT",
        "RGPOT_SOURCE_COMMIT",
        "MOLSLAB_BUILD_SHA256SUMS",
        "molecular_cluster\" 2 20 1",
        "slab_adsorption\"",
        "cuh2_fcc_slab.con\" 15 1",
    ] {
        assert!(
            source.contains(required),
            "Terra molecular builder is missing {required}"
        );
    }
}

#[test]
fn molecular_campaigns_bind_rgpot_source_and_lockfile() {
    let root = PathBuf::from(env!("CARGO_MANIFEST_DIR"));
    for builder in ["elja_build_rgpot_ex.sh", "terra_build_rgpot_ex.sh"] {
        let source = fs::read_to_string(root.join("scripts").join(builder))
            .unwrap_or_else(|error| panic!("failed to read {builder}: {error}"));
        for required in [
            "RGPOT_EXPECTED_COMMIT",
            "git -C \"$RGPOT\" diff --quiet HEAD --",
            "install --locked -e xtbbld",
            "run --locked -e xtbbld meson",
            "RGPOT_SOURCE_COMMIT",
            "RGPOT_PIXI_LOCK_SHA256",
            "MOLSLAB_BUILD_SHA256SUMS",
        ] {
            assert!(source.contains(required), "{builder} is missing {required}");
        }
    }

    for script in [
        "elja_submit_jcc_molslab.sh",
        "elja_jcc_molslab_ensemble.sh",
    ] {
        let source = fs::read_to_string(root.join("scripts").join(script))
            .unwrap_or_else(|error| panic!("failed to read {script}: {error}"));
        for required in [
            "RGPOT_SOURCE_COMMIT",
            "RGPOT_PIXI_LOCK_SHA256",
            "git -C \"$RGPOT\" diff --quiet HEAD --",
            "sha256sum \"$RGPOT/pixi.lock\"",
        ] {
            assert!(source.contains(required), "{script} is missing {required}");
        }
    }

    let runner = fs::read_to_string(root.join("scripts/elja_jcc_molslab_ensemble.sh"))
        .unwrap_or_else(|error| panic!("failed to read molecular runner: {error}"));
    assert!(runner.contains(r#"rgpot_source_commit=%s\n"#));
    assert!(runner.contains(r#"rgpot_pixi_lock_sha256=%s\n"#));
}

#[test]
fn molecular_ensembles_are_isolated_paired_slurm_runs() {
    let runner = PathBuf::from(env!("CARGO_MANIFEST_DIR"))
        .join("scripts")
        .join("elja_jcc_molslab_ensemble.sh");
    let source = fs::read_to_string(&runner)
        .unwrap_or_else(|error| panic!("failed to read {}: {error}", runner.display()));

    for required in [
        r#"REPLICAS=4"#,
        r#"127.0.0.1:0"#,
        r#"private_endpoints"#,
        r#"BANK_SHARING=$ARM"#,
        r#"BANK_RPC=${private_endpoints[$replica]}"#,
        r#"export OMP_NUM_THREADS=1"#,
        r#"export OPENBLAS_NUM_THREADS=1"#,
        r#"export MKL_NUM_THREADS=1"#,
        r#"grep -F -q "capnp bank ""#,
        r#"grep -F -q " own walk""#,
        r#"source_commit=%s\n"#,
        r#"soap_mode=%s\n"#,
        r#"requested_options=%s\n"#,
        r#"resolved_config_sha256=%s\n"#,
        r#"binary_sha256=%s\n"#,
        r#"engine_sha256=%s\n"#,
        r#"ANNEAL_SOAP_MODE=$SOAP_MODE"#,
        r#"ANNEAL_RESOLVED_CONFIG=$worker/resolved-config.json"#,
        r#"TARGET_TOLERANCE=1e-3"#,
        r#"export TARGET_TOL=$TARGET_TOLERANCE"#,
        r#"target_tolerance=%s\n"#,
        r#"bank_sync=charged_slices\n"#,
        r#"touch "$OUT/TERMINAL_OK""#,
        r#"mapfile -d '' artifacts"#,
        r#"find . -type f ! -name SHA256SUMS -print0 | sort -z"#,
        r#"sha256sum "${artifacts[@]}" >SHA256SUMS"#,
    ] {
        assert!(
            source.contains(required),
            "molecule/slab ensemble runner is missing contract {required}"
        );
    }
}

#[test]
fn molecular_submission_pairs_every_system_and_arm() {
    let script = PathBuf::from(env!("CARGO_MANIFEST_DIR"))
        .join("scripts")
        .join("elja_submit_jcc_molslab.sh");
    let source = fs::read_to_string(&script)
        .unwrap_or_else(|error| panic!("failed to read {}: {error}", script.display()));

    for system in ["h2o2", "h2o4", "h2o6", "cuh2"] {
        assert!(
            source.contains(&format!("{system}:")),
            "molecule/slab submission omits {system}"
        );
    }
    assert!(source.contains("for arm in shared control"));
    assert!(source.contains("for soap_mode in flexible off"));
    assert!(source.contains(r#"CAMPAIGN=${JCC_CAMPAIGN:-jcc-2026-${STAGE}}"#));
    assert!(
        source.contains(r#"--cpus-per-task="${ELJA_CPUS_PER_TASK:-8}""#),
        "both arms must reserve capacity for four workers and four banks"
    );
}

#[test]
fn physical_drivers_accept_and_record_every_soap_mode() {
    for driver in ["molecular_cluster.rs", "slab_adsorption.rs"] {
        let path = PathBuf::from(env!("CARGO_MANIFEST_DIR"))
            .join("examples")
            .join(driver);
        let source = fs::read_to_string(&path)
            .unwrap_or_else(|error| panic!("failed to read {}: {error}", path.display()));
        for required in [
            "ANNEAL_SOAP_MODE",
            "ANNEAL_RESOLVED_CONFIG",
            "SoapProposalMode::Flexible",
            "SoapProposalMode::Rigid",
            "SoapProposalMode::Off",
            "resolved_json()",
        ] {
            assert!(source.contains(required), "{driver} omits {required}");
        }
    }
}

#[test]
fn molecular_driver_keeps_the_measured_quench_defaults() {
    let example = PathBuf::from(env!("CARGO_MANIFEST_DIR"))
        .join("examples")
        .join("molecular_cluster.rs");
    let source = fs::read_to_string(&example)
        .unwrap_or_else(|error| panic!("failed to read {}: {error}", example.display()));

    assert!(
        !source.contains("cfg.screen_steps = 6;"),
        "the molecular driver must not replace the measured screening quench with six steps"
    );
    assert!(
        !source.contains("cfg.relax_steps = 60;"),
        "the molecular driver must not cap a full molecular quench at sixty steps"
    );
    assert!(
        source.contains("screen/full/check"),
        "molecular campaign output must expose where charged quench work went"
    );
}

#[test]
fn slab_driver_reports_quench_accounting() {
    let driver = PathBuf::from(env!("CARGO_MANIFEST_DIR"))
        .join("examples")
        .join("slab_adsorption.rs");
    let source = fs::read_to_string(&driver)
        .unwrap_or_else(|error| panic!("failed to read {}: {error}", driver.display()));

    for field in [
        "stats.screen_charged",
        "stats.full_charged",
        "stats.check_charged",
        "stats.screens",
        "stats.capped",
    ] {
        assert!(
            source.contains(field),
            "slab driver must report quench accounting field {field}"
        );
    }
}

#[test]
fn cooperative_driver_honours_a_hop_cap() {
    let driver = PathBuf::from(env!("CARGO_MANIFEST_DIR"))
        .join("examples")
        .join("lj_cluster_search.rs");
    let source = fs::read_to_string(&driver)
        .unwrap_or_else(|error| panic!("failed to read {}: {error}", driver.display()));
    assert!(
        source.contains("CATALOG_MAX_HOPS"),
        "cooperative driver must honour a hop cap so many-chain explore is hops, not a short ledger"
    );
    assert!(
        source.contains("run_cfg.max_hops = Some(hops)"),
        "CATALOG_MAX_HOPS must set Config::max_hops"
    );
}

#[test]
fn occupancy_many_chains_starts_one_brain_per_replica() {
    let script = PathBuf::from(env!("CARGO_MANIFEST_DIR"))
        .join("scripts")
        .join("elja_jcc_lj_many_chains.sh");
    let source = fs::read_to_string(&script)
        .unwrap_or_else(|error| panic!("failed to read {}: {error}", script.display()));

    for required in [
        "CATALOG_BRAIN_LISTEN",
        "CATALOG_BRAIN_PEERS",
        "CATALOG_BRAIN_PORT_BASE",
    ] {
        assert!(
            source.contains(required),
            "many-chains occupancy talking must export {required} on every replica"
        );
    }
    assert!(
        source.contains(r#"${r}=tcp://127.0.0.1:$((BRAIN_PORT_BASE + r))"#),
        "brain peers must be replica=tcp://host:port pairs the worker already parses"
    );
    assert!(
        source.contains(r#"brain_peers "$replica" "$wave_start" "$wave_end""#),
        "brain peers must be the live wave, not all 48 replicas"
    );
}

#[test]
fn occupancy_driver_compiles_brains_into_the_bank_rpc_build() {
    let driver = PathBuf::from(env!("CARGO_MANIFEST_DIR"))
        .join("examples")
        .join("lj_cluster_search.rs");
    let source = fs::read_to_string(&driver)
        .unwrap_or_else(|error| panic!("failed to read {}: {error}", driver.display()));

    let listen = source
        .find(r#"std::env::var("CATALOG_BRAIN_LISTEN")"#)
        .expect("worker must read CATALOG_BRAIN_LISTEN");
    let prelude = &source[listen.saturating_sub(250)..listen];
    assert!(
        prelude.contains(r#"#[cfg(feature = "bank-rpc")]"#),
        "occupancy is featomic,ira,bank-rpc; brains gated on nng-transport never start"
    );
    assert!(
        !prelude.contains("nng-transport"),
        "nng-transport is not an occupancy feature; brains must not require it"
    );
}

#[test]
fn cooperative_share_optimization_is_one_quench_not_ten() {
    let driver = PathBuf::from(env!("CARGO_MANIFEST_DIR"))
        .join("examples")
        .join("lj_cluster_search.rs");
    let source = fs::read_to_string(&driver)
        .unwrap_or_else(|error| panic!("failed to read {}: {error}", driver.display()));

    assert!(
        source.contains("run_cfg.polish_records = run_cfg.relax_steps;"),
        "cooperative share-tolerance optimization must be one quench, not ten"
    );
    assert!(
        !source.contains("polish_records = run_cfg.relax_steps.saturating_mul(10)"),
        "ten full quenches per record spend a short replica on one improvement"
    );
}

#[test]
fn occupancy_brains_sbatch_require_family_floor_and_leftover_well_stop() {
    let scripts = PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("scripts");
    let cases = [
        ("terra_lj38_occ_brains.sbatch", "38 400000 13"),
        ("terra_lj75_occ_brains.sbatch", "75 4000000 1"),
        ("terra_lj98_occ_brains.sbatch", "98 4000000 0"),
        ("elja_lj38_occ_brains.sbatch", "38 400000 0"),
        ("elja_lj75_occ_brains.sbatch", "75 4000000 0"),
        ("elja_lj98_occ_brains.sbatch", "98 4000000 0"),
    ];
    for (name, launch) in cases {
        let source = fs::read_to_string(scripts.join(name))
            .unwrap_or_else(|error| panic!("failed to read {name}: {error}"));
        assert!(
            !source.contains("CATALOG_MIN_FAMILIES"),
            "{name} must not override the Fiedler-and-DECAF family floor"
        );
        assert!(
            source.contains("export CATALOG_WAVE=24"),
            "{name} must set CATALOG_WAVE=24"
        );
        assert!(
            source.contains(&format!("elja_jcc_lj_many_chains.sh {launch}")),
            "{name} must launch paper-budget ensemble {launch}"
        );
        assert!(
            source.contains(r#"grep -a -F -q "$symbol""#)
                && source.contains("occupancy min families")
                && source.contains("occupancy leave archive hole"),
            "{name} must grep the binary for occupancy min families and occupancy leave archive hole"
        );
        assert!(
            source.contains(r#"grep -a -F -q "Refuse""#)
                && source.contains(r#"grep -a -F -q "gt stop packing""#),
            "{name} must grep the binary for Refuse or gt stop packing"
        );
    }
}

#[test]
fn marks_submit_pins_the_crossing_floor_commit() {
    let root = PathBuf::from(env!("CARGO_MANIFEST_DIR"));
    let submit = fs::read_to_string(root.join("scripts/elja_submit_occ_marks.sh"))
        .unwrap_or_else(|error| panic!("failed to read marks submit: {error}"));
    assert!(
        submit.contains("NEED=9dde60d5130623c7862e29181697508af4dc2ba8"),
        "marks submit must require a descendant of leftover-dwell 9dde60d"
    );
    assert!(
        submit.contains("merge-base --is-ancestor"),
        "marks submit must accept later walk-fix commits on that line"
    );
    assert!(
        submit.contains("anneal-occ-9dde60d"),
        "marks submit must use an isolated tree, not anneal-stop"
    );
    assert!(
        !submit.contains("lj75-shared-0002"),
        "marks submit must not write sealed ensemble 0002"
    );
    assert!(
        !submit.contains("lj75-shared-0003"),
        "marks submit must not write sealed ensemble 0003"
    );
    assert!(
        submit.contains("lj75-occ-dwell"),
        "marks submit must write the leftover-dwell campaign, not anneal-stop"
    );
    let hops = fs::read_to_string(root.join("scripts/elja_lj75_occ_marks.sbatch"))
        .unwrap_or_else(|error| panic!("failed to read marks hops: {error}"));
    assert!(
        !hops.contains("--mem="),
        "f2zw: omit --mem on the Marks hops allocation"
    );
    assert!(
        hops.contains("elja_jcc_lj_many_chains.sh 75 4000000"),
        "Marks hops must use the paper 4e6 budget"
    );
    assert!(
        hops.contains("CATALOG_WAVE=24"),
        "Marks hops must use WAVE 24"
    );
}

#[test]
fn marks_build_and_submit_bind_source_to_binaries() {
    let root = PathBuf::from(env!("CARGO_MANIFEST_DIR"));
    let build = fs::read_to_string(root.join("scripts/elja_build_lj.sh"))
        .unwrap_or_else(|error| panic!("failed to read LJ build: {error}"));
    assert!(
        build.contains("git diff --quiet HEAD --"),
        "the build must reject tracked source changes"
    );
    assert!(
        build.contains("git rev-parse HEAD >SOURCE_COMMIT"),
        "the build must derive SOURCE_COMMIT from the checkout it compiles"
    );
    assert!(
        build.contains("sha256sum") && build.contains("BUILD_SHA256SUMS"),
        "the build must seal the campaign executables in a checksum manifest"
    );

    let submit = fs::read_to_string(root.join("scripts/elja_submit_occ_marks.sh"))
        .unwrap_or_else(|error| panic!("failed to read Marks submit: {error}"));
    assert!(
        submit.contains("git -C \"$ROOT\" rev-parse HEAD")
            && submit.contains("SOURCE_COMMIT=$SRC does not match HEAD=$HEAD"),
        "submission must compare the recorded source to the checkout HEAD"
    );
    assert!(
        submit.contains("sha256sum -c BUILD_SHA256SUMS"),
        "submission must verify the exact executables emitted by the build"
    );
}

#[test]
fn terra_audit_build_binds_the_census_calibrator_to_source() {
    let root = PathBuf::from(env!("CARGO_MANIFEST_DIR"));
    let build = fs::read_to_string(root.join("scripts/terra_build_brains.sh"))
        .unwrap_or_else(|error| panic!("failed to read Terra build: {error}"));

    assert!(
        build.contains("--example lj_census_calibration"),
        "the Terra audit build must compile the production census calibrator"
    );
    assert!(
        build.contains("target/release/examples/lj_census_calibration"),
        "the Terra audit build must hash the production census calibrator"
    );
    assert!(
        build.contains("seeded-random-cluster-quench-v1"),
        "the Terra audit build must reject a calibrator without target-blind provenance"
    );
}

#[cfg(unix)]
#[test]
fn census_calibration_runner_is_executable() {
    let script =
        PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("scripts/elja_jcc_lj_calibration.sh");
    let mode = fs::metadata(&script)
        .unwrap_or_else(|error| panic!("failed to stat {}: {error}", script.display()))
        .permissions()
        .mode();
    assert_ne!(
        mode & 0o111,
        0,
        "the calibration submitter invokes the runner as an executable"
    );
}

#[test]
fn census_calibration_campaign_covers_every_analyzed_lj_system() {
    let root = PathBuf::from(env!("CARGO_MANIFEST_DIR"));
    let runner = fs::read_to_string(root.join("scripts/elja_jcc_lj_calibration.sh"))
        .unwrap_or_else(|error| panic!("failed to read calibration runner: {error}"));
    let submitter = fs::read_to_string(root.join("scripts/elja_submit_jcc_calibration.sh"))
        .unwrap_or_else(|error| panic!("failed to read calibration submitter: {error}"));
    let finalizer = fs::read_to_string(root.join("scripts/elja_jcc_finalize_calibration.sh"))
        .unwrap_or_else(|error| panic!("failed to read calibration finalizer: {error}"));

    assert!(
        runner.contains("38|55|75|98|102|104)"),
        "the calibration runner must accept every hard-LJ analysis system"
    );
    for system in [38, 55, 75, 98, 102, 104] {
        assert!(
            submitter.contains(&format!("[{system}]={system}00000")),
            "the calibration submitter must assign a deterministic LJ{system} seed"
        );
    }
    assert!(
        submitter.contains("for n in 38 55 75 98 102 104; do"),
        "the calibration submitter must launch every hard-LJ analysis system"
    );
    assert!(
        finalizer.contains("for n in 38 55 75 98 102 104; do"),
        "the calibration finalizer must validate every hard-LJ analysis system"
    );
}
