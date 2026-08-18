use anneal_core::methods::cluster_hopping::{Config, SoapProposalMode};

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
