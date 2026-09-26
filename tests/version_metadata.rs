fn toml_value<'a>(source: &'a str, section: &str, key: &str) -> &'a str {
    source
        .lines()
        .skip_while(|line| line.trim() != section)
        .skip(1)
        .find_map(|line| {
            let trimmed = line.trim();
            trimmed
                .strip_prefix(key)
                .and_then(|rest| rest.split_once('='))
                .map(|(_, value)| value.trim().trim_matches('"'))
        })
        .unwrap_or_else(|| panic!("{section} {key}"))
}

fn cff_value<'a>(source: &'a str, key: &str) -> &'a str {
    source
        .lines()
        .find_map(|line| {
            line.trim()
                .strip_prefix(key)
                .and_then(|rest| rest.strip_prefix(':'))
                .map(|value| value.trim().trim_matches('"'))
        })
        .unwrap_or_else(|| panic!("CITATION.cff {key}"))
}

#[test]
fn all_public_version_metadata_matches_manifest() {
    let package_version = toml_value(include_str!("../Cargo.toml"), "[package]", "version");

    assert_eq!(anneal_core::ANNEAL_VERSION, package_version);
    assert_eq!(
        toml_value(include_str!("../pyproject.toml"), "[project]", "version"),
        package_version
    );
    assert_eq!(
        toml_value(include_str!("../pixi.toml"), "[workspace]", "version"),
        package_version
    );
    assert_eq!(
        toml_value(
            include_str!("../towncrier.toml"),
            "[tool.towncrier]",
            "version"
        ),
        package_version
    );
    assert_eq!(
        cff_value(include_str!("../CITATION.cff"), "version"),
        package_version
    );
}

#[test]
fn citation_exposes_the_software_concept_doi() {
    let citation = include_str!("../CITATION.cff");

    assert!(
        citation.contains("identifiers:\n  - type: doi\n    value: \"10.5281/zenodo.10672746\"")
    );
}

#[test]
fn changelog_has_release_notes_for_the_public_version() {
    let package_version = toml_value(include_str!("../Cargo.toml"), "[package]", "version");
    let changelog = include_str!("../CHANGELOG.md");
    let towncrier = include_str!("../towncrier.toml");

    assert!(
        changelog.contains(&format!(
            "## [{package_version}](https://github.com/HaoZeke/anneal/tree/v{package_version})"
        )),
        "CHANGELOG.md has no tag-linked release section for package version {package_version}"
    );
    assert!(towncrier.contains("/tree/v{version}"));
}
