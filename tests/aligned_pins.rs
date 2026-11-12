//! Pins shared with rgsaddle develop.
const MANIFEST: &str = include_str!("../Cargo.toml");

#[test]
fn pins_match_rgsaddle() {
    assert!(MANIFEST.contains("rgmin = \"0.5.2\""));
    assert!(MANIFEST.contains("tag = \"v0.3.5\""));
    assert!(MANIFEST.contains("readcon-core = \"=0.17.1\""));
    assert!(MANIFEST.contains("readcon-db = \"=0.2.0\""));
    assert!(MANIFEST.contains("rgsaddle = { version = \"0.3.2\""));
}
