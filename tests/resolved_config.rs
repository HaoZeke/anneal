use anneal_core::methods::cluster_hopping::{ArmSelection, Config, SoapProposalMode};

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
    assert_eq!(
        cfg.resolved_sha256().unwrap(),
        "a1bd030f28c13e57f6dbaa50302b8f1312139fc1684f2fa24801a361664cf5d2"
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
