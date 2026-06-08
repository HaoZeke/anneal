#[test]
fn version_constant_matches_manifest() {
    let manifest = include_str!("../Cargo.toml");
    let package_version = manifest
        .lines()
        .skip_while(|line| line.trim() != "[package]")
        .skip(1)
        .find_map(|line| {
            let trimmed = line.trim();
            trimmed
                .strip_prefix("version")
                .and_then(|rest| rest.split_once('='))
                .map(|(_, value)| value.trim().trim_matches('"'))
        })
        .expect("Cargo.toml [package] version");

    assert_eq!(anneal_core::ANNEAL_VERSION, package_version);
}
