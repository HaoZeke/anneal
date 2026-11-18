use anneal_core::methods::cluster_hopping::{ArmSelection, Config, SoapProposalMode};
use anneal_core::potentials::PairKind;

#[test]
fn resolved_configuration_is_stable_and_mode_sensitive() {
    let flexible = Config::recommended_molecular(vec![8, 1, 1], vec![vec![0, 1, 2]], 1.0);
    let json_a = flexible
        .resolved_json()
        .expect("serialize resolved configuration");
    let json_b = flexible
        .resolved_json()
        .expect("repeat resolved configuration");
    assert_eq!(json_a, json_b);
    assert!(json_a.starts_with(r#"{"schema":"anneal-cluster-config-v1""#));
    assert!(json_a.contains(r#""soap_mode":"flexible""#));
    assert_eq!(flexible.resolved_sha256().unwrap().len(), 64);

    let mut rigid = flexible.clone();
    rigid.soap_mode = SoapProposalMode::Rigid;
    assert_ne!(
        flexible.resolved_sha256().unwrap(),
        rigid.resolved_sha256().unwrap()
    );

    let mut off = flexible;
    off.soap_mode = SoapProposalMode::Off;
    assert!(
        off.resolved_json()
            .unwrap()
            .contains(r#""soap_mode":"off""#)
    );
    assert_ne!(
        rigid.resolved_sha256().unwrap(),
        off.resolved_sha256().unwrap()
    );
}

#[test]
fn default_witness_ceiling_is_absent_from_the_resolved_json() {
    let cfg = Config::recommended(38);
    assert!(!cfg.witness_tripleton_ceiling);
    let json = cfg.resolved_json().unwrap();
    assert!(!json.contains("witness_tripleton"));
    assert_eq!(cfg.arm_selection, ArmSelection::Posterior);
    assert!(!cfg.resolved_json().unwrap().contains("arm_selection"));
    assert!(!cfg.resolved_json().unwrap().contains("target_energy"));
    let mut aimed = cfg.clone();
    aimed.target_energy = Some(-1.0);
    assert!(
        aimed
            .resolved_json()
            .unwrap()
            .contains("\"target_energy\":-1.0")
    );
    assert_ne!(
        aimed.resolved_sha256().unwrap(),
        cfg.resolved_sha256().unwrap()
    );
    assert!(!json.contains("reseed_hops"));
    assert_eq!(
        cfg.resolved_sha256().unwrap(),
        "093919b1e5b858235e24062ff5b382ca1dfaa960d0ad5dc6427434246a49e62d"
    );
    let mut uniform = cfg;
    uniform.arm_selection = ArmSelection::Uniform;
    assert!(uniform.depth_reward);
    assert!(uniform.allocate_moves);
    assert!(
        uniform
            .resolved_json()
            .unwrap()
            .contains(r#""arm_selection":"uniform""#)
    );
    assert_ne!(
        uniform.resolved_sha256().unwrap(),
        "a1bd030f28c13e57f6dbaa50302b8f1312139fc1684f2fa24801a361664cf5d2"
    );
}

#[test]
fn a_long_ranged_pair_is_resolved_only_when_set() {
    let cfg = Config::recommended(38);
    assert!(!cfg.long_ranged);
    assert!(!cfg.resolved_json().unwrap().contains("long_ranged"));
    let mut bound = cfg.clone();
    bound.long_ranged = true;
    assert!(
        bound
            .resolved_json()
            .unwrap()
            .contains("\"long_ranged\":true")
    );
    assert_ne!(
        bound.resolved_sha256().unwrap(),
        cfg.resolved_sha256().unwrap()
    );
}

#[test]
fn a_basin_stride_is_resolved_only_off_its_default() {
    let cfg = Config::recommended(38);
    assert_eq!(cfg.basin_stride, 1);
    assert!(!cfg.resolved_json().unwrap().contains("basin_stride"));
    let mut strided = cfg.clone();
    strided.basin_stride = 4;
    assert!(
        strided
            .resolved_json()
            .unwrap()
            .contains("\"basin_stride\":4")
    );
    assert_ne!(
        strided.resolved_sha256().unwrap(),
        cfg.resolved_sha256().unwrap()
    );
}

#[test]
fn paper_preset_priors_stay_out_of_the_resolved_hash() {
    let cfg = Config::recommended(38);
    let json = cfg.resolved_json().unwrap();
    for name in ["prior_soap", "homotop_reorder", "general_collective"] {
        assert!(
            !json.contains(name),
            "{name} at the paper preset stays out of the resolved JSON"
        );
    }
    let mut dropped = cfg.clone();
    dropped.prior_soap = false;
    assert!(
        dropped
            .resolved_json()
            .unwrap()
            .contains("\"prior_soap\":false")
    );
    assert_ne!(
        dropped.resolved_sha256().unwrap(),
        cfg.resolved_sha256().unwrap()
    );
    let mut plane = cfg;
    plane.general_collective = true;
    assert!(
        plane
            .resolved_json()
            .unwrap()
            .contains("\"general_collective\":true")
    );
    assert_ne!(
        plane.resolved_sha256().unwrap(),
        "a1bd030f28c13e57f6dbaa50302b8f1312139fc1684f2fa24801a361664cf5d2"
    );
}

#[test]
fn removed_prior_keys_are_rejected_in_json_and_toml() {
    use anneal_core::methods::cluster_hopping::REMOVED_PRIOR_KEYS;

    let valid_json = Config::recommended(38).resolved_json().unwrap();
    Config::validate_json_keys(&valid_json).expect("current resolved JSON is valid");
    Config::validate_toml_keys(
        "prior_soap = false\n[continuous_symmetry]\ngroup = \"measure\"\ninterval = 10\n",
    )
    .expect("supported TOML keys are valid");

    for &removed in REMOVED_PRIOR_KEYS {
        if removed == "reseed_hops" {
            let bad_json =
                r#"{"continuous_symmetry":{"group":"measure","interval":10,"reseed_hops":5}}"#;
            let err = Config::validate_json_keys(bad_json).unwrap_err();
            assert_eq!(err.key, "continuous_symmetry.reseed_hops");
            assert!(err.to_string().contains("was removed"));

            let bad_toml =
                "[continuous_symmetry]\ngroup = \"measure\"\ninterval = 10\nreseed_hops = 5\n";
            let toml_err = Config::validate_toml_keys(bad_toml).unwrap_err();
            assert_eq!(toml_err.key, "continuous_symmetry.reseed_hops");
            assert!(toml_err.to_string().contains("was removed"));
        } else {
            let bad_json = format!(r#"{{"{removed}":false}}"#);
            let err = Config::validate_json_keys(&bad_json).unwrap_err();
            assert_eq!(err.key, removed);
            assert!(err.to_string().contains("was removed"));

            let bad_toml = format!("{removed} = false\n");
            let toml_err = Config::validate_toml_keys(&bad_toml).unwrap_err();
            assert_eq!(toml_err.key, removed);
            assert!(toml_err.to_string().contains("was removed"));
        }
    }
}

#[test]
fn continuous_symmetry_serde_rejects_removed_reseed_hops() {
    use anneal_core::methods::cluster_hopping::ContinuousSymmetry;

    let ok: ContinuousSymmetry =
        serde_json::from_str(r#"{"group":"measure","interval":10}"#).expect("valid measure");
    assert_eq!(ok, ContinuousSymmetry::Measure { interval: 10 });

    let err = serde_json::from_str::<ContinuousSymmetry>(
        r#"{"group":"measure","interval":10,"reseed_hops":5}"#,
    )
    .unwrap_err();
    assert!(
        err.to_string().contains("reseed_hops") && err.to_string().contains("was removed"),
        "unexpected error: {err}"
    );

    let toml_err = toml::from_str::<ContinuousSymmetry>(
        "group = \"measure\"\ninterval = 10\nreseed_hops = 5\n",
    )
    .unwrap_err();
    assert!(
        toml_err.to_string().contains("reseed_hops")
            && toml_err.to_string().contains("was removed"),
        "unexpected toml error: {toml_err}"
    );
}

#[test]
fn prior_switch_and_env_parsers_reject_removed_switches() {
    use anneal_core::methods::cluster_hopping::PriorSwitches;

    assert_eq!(
        Config::parse_prior_switches("1").unwrap(),
        PriorSwitches {
            soap: true,
            all_off: false,
        }
    );
    assert_eq!(
        Config::parse_prior_switches("0").unwrap(),
        PriorSwitches {
            soap: false,
            all_off: true,
        }
    );
    assert_eq!(
        Config::parse_prior_switches("soap").unwrap(),
        PriorSwitches {
            soap: false,
            all_off: false,
        }
    );

    for removed in [
        "reseed",
        "prior_reseed",
        "growth",
        "prior_growth",
        "core",
        "prior_core",
        "handover",
        "prior_handover",
        "motif",
        "prior_motif",
        "soap,reseed",
    ] {
        let err = Config::parse_prior_switches(removed).unwrap_err();
        assert!(
            err.to_string().contains("was removed"),
            "expected removal error for {removed:?}, got {err}"
        );
    }
    assert!(
        Config::parse_prior_switches("nonexistent_prior")
            .unwrap_err()
            .to_string()
            .contains("unknown prior switch")
    );

    for (var, val) in [
        ("H2H_HANDOVER", "0"),
        ("H2H_HANDOVER", "1"),
        ("PRIOR_RESEED", "0"),
        ("PRIOR_GROWTH", "0"),
        ("PRIOR_CORE", "0"),
        ("PRIOR_HANDOVER", "0"),
        ("PRIOR_MOTIF", "0"),
        ("H2H_PRIORS", "core"),
    ] {
        let err = Config::check_removed_env_vars([(var, val)]).unwrap_err();
        assert!(
            err.to_string().contains("was removed"),
            "expected removal error for {var}={val}, got {err}"
        );
    }
    Config::check_removed_env_vars([("H2H_PRIORS", "0"), ("H2H_PRIORS", "soap")])
        .expect("supported H2H_PRIORS values are accepted");
}

#[test]
fn a_removed_key_nested_in_a_list_is_rejected() {
    let json = r#"{"passes":[{"prior_reseed":false}]}"#;
    let err = Config::validate_json_keys(json).unwrap_err();
    assert_eq!(err.key, "passes.prior_reseed");
    assert!(
        err.to_string().contains("was removed"),
        "unexpected json error: {err}"
    );

    let toml = "passes = [{ prior_motif = true }]\n";
    let err = Config::validate_toml_keys(toml).unwrap_err();
    assert_eq!(err.key, "passes.prior_motif");
    assert!(
        err.to_string().contains("was removed"),
        "unexpected toml error: {err}"
    );
}

#[test]
fn reseed_hops_names_the_interval_and_other_keys_name_prior_soap() {
    let reseed = Config::validate_json_keys(
        r#"{"continuous_symmetry":{"group":"measure","interval":10,"reseed_hops":5}}"#,
    )
    .unwrap_err();
    assert_eq!(reseed.key, "continuous_symmetry.reseed_hops");
    let text = reseed.to_string();
    assert!(
        text.contains("takes only") && text.contains("interval"),
        "reseed_hops should say the measure group takes only interval, got: {text}"
    );

    let prior = Config::validate_json_keys(r#"{"prior_reseed":false}"#).unwrap_err();
    let text = prior.to_string();
    assert!(
        text.contains("prior_soap"),
        "a removed prior key should name prior_soap, got: {text}"
    );
}

#[test]
fn campaign_toml_rejects_removed_prior_keys_and_extra_switches() {
    use anneal_core::campaign::CampaignConfig;

    let base = "\
campaign = \"smoke\"\n\
[channels]\n\
shared_screen = true\n\
shared_bias = false\n\
entropic_bias = false\n\
histo_screen = false\n\
seam_ladder = false\n\
frontier_exchange = false\n\
[ensemble]\n\
replicas = 2\n\
wave = 2\n\
slice = 10\n\
max_hops = 100\n\
population_interval = 50\n";

    CampaignConfig::parse(base).expect("valid campaign config");
    CampaignConfig::parse(&format!("{base}[extra]\nH2H_PRIORS = \"soap\"\n"))
        .expect("supported H2H_PRIORS in extra");

    for bad in [
        format!("prior_reseed = false\n{base}"),
        format!("{base}[extra]\nH2H_HANDOVER = \"1\"\n"),
        format!("{base}[extra]\nPRIOR_CORE = \"0\"\n"),
        format!("{base}[extra]\nH2H_PRIORS = \"motif\"\n"),
    ] {
        let err = CampaignConfig::parse(&bad).unwrap_err();
        assert!(
            err.to_string().contains("was removed"),
            "expected clear removal error, got: {err}"
        );
    }
}

#[test]
fn pair_scaling_is_general_and_lennard_jones_is_unchanged() {
    let stock = Config::recommended(80);
    let lennard_jones = Config::recommended_for_pair(80, PairKind::LennardJones);
    assert_eq!(
        stock.resolved_json().unwrap(),
        lennard_jones.resolved_json().unwrap()
    );

    let morse = Config::recommended_for_pair(80, PairKind::Morse { rho: 6.0 });
    assert_eq!(morse.energy_scale, 1.0);
    assert!((morse.length_scale - 2.0_f64.powf(-1.0 / 6.0)).abs() < 1e-15);
    assert!(morse.start_radius() < stock.start_radius());
}
