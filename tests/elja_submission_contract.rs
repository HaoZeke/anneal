use std::fs;
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
        .join("elja_jcc_lj_ensemble.sh");
    let source = fs::read_to_string(&script)
        .unwrap_or_else(|error| panic!("failed to read {}: {error}", script.display()));

    for required in [
        r#"source_commit=%s\n"#,
        r#"requested_options=%s\n"#,
        r#"resolved_config_sha256=%s\n"#,
        r#"binary_sha256=%s\n"#,
        r#"runner_sha256=%s\n"#,
        r#"qualifier_sha256=%s\n"#,
        r#"calibration_sha256=%s\n"#,
        r#"ANNEAL_RESOLVED_CONFIG=$worker/resolved-config.json"#,
    ] {
        assert!(
            source.contains(required),
            "hard-LJ manifest is missing provenance field {required}"
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
        r#"source_commit=%s\n"#,
        r#"soap_mode=%s\n"#,
        r#"requested_options=%s\n"#,
        r#"resolved_config_sha256=%s\n"#,
        r#"binary_sha256=%s\n"#,
        r#"engine_sha256=%s\n"#,
        r#"ANNEAL_SOAP_MODE=$SOAP_MODE"#,
        r#"ANNEAL_RESOLVED_CONFIG=$worker/resolved-config.json"#,
        r#"bank_sync=charged_slices\n"#,
        r#"touch "$OUT/TERMINAL_OK""#,
        r#"xargs -0 sha256sum >SHA256SUMS"#,
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
        submit.contains("NEED=7fe8d5a431a4ce227f8e3d424bd299f3eae51103"),
        "marks submit must require a descendant of packing-gt-stop 7fe8d5a"
    );
    assert!(
        submit.contains("merge-base --is-ancestor"),
        "marks submit must accept later walk-fix commits on that line"
    );
    assert!(
        submit.contains("anneal-occ-7fe8d5a"),
        "marks submit must use an isolated tree, not anneal-stop"
    );
    assert!(
        !submit.contains("lj75-shared-0002"),
        "marks submit must not write sealed ensemble 0002"
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
