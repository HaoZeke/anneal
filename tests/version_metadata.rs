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
        cff_value(include_str!("../CITATION.cff"), "version"),
        package_version
    );

    let meson = include_str!("../meson.build");
    assert!(meson.contains(&format!("version: '{package_version}'")));

    let cmake = include_str!("../CMakeLists.txt");
    assert!(cmake.contains("Cargo.toml"));
    assert!(cmake.contains("VERSION ${ANNEAL_VERSION}"));
    assert!(cmake.contains("CMAKE_CONFIGURE_DEPENDS"));

    let header = include_str!("../include/anneal-version.h");
    assert!(header.contains(&format!(
        "#define ANNEAL_CORE_VERSION \"{package_version}\""
    )));

    let zenodo = include_str!("../.zenodo.json");
    assert!(zenodo.contains(&format!("\"version\": \"{package_version}\"")));

    let lock = include_str!("../Cargo.lock");
    assert_eq!(lock_package_version(lock, "anneal-core"), package_version);
    assert_eq!(
        column_zero_version(include_str!("../Cargo.toml")),
        package_version
    );

    let readme = include_str!("../README.md");
    assert!(readme.contains(&format!("badge/version-{package_version}-")));

    let version_rs = include_str!("../src/version.rs");
    assert!(version_rs.contains("env!(\"CARGO_PKG_VERSION\")"));
    assert!(
        !version_rs
            .lines()
            .any(|line| line.split('"').any(is_plain_triple)),
        "src/version.rs still has a handwritten version"
    );

    let conf = include_str!("../docs/source/conf.py");
    assert!(conf.contains("anneal_version"));
    assert!(conf.contains("package_version"));

    let towncrier = include_str!("../towncrier.toml");
    assert!(towncrier.contains("package = \"anneal_version\""));
    assert!(towncrier.contains("package_dir = \"scripts\""));
    assert!(
        !towncrier
            .lines()
            .any(|line| line.trim_start().starts_with("version"))
    );
    let reader = include_str!("../scripts/anneal_version.py");
    assert!(reader.contains("Cargo.toml"));
    assert!(reader.contains("anneal-core"));
    assert!(reader.contains("[\"package\"][\"version\"]"));
}

/// The first column-0 `version = "` line is `[package].version`.
fn column_zero_version(source: &str) -> &str {
    source
        .lines()
        .find_map(|line| {
            line.strip_prefix("version = \"")
                .and_then(|rest| rest.strip_suffix('"'))
        })
        .unwrap_or_else(|| panic!("Cargo.toml has no column-0 version"))
}

fn lock_package_version<'a>(source: &'a str, name: &str) -> &'a str {
    let lines: Vec<&str> = source.lines().collect();
    let mut index = 0;
    while index < lines.len() {
        if lines[index].trim() != "[[package]]" {
            index += 1;
            continue;
        }
        index += 1;
        let start = index;
        while index < lines.len() && lines[index].trim() != "[[package]]" {
            index += 1;
        }
        let block = &lines[start..index];
        let package_name = block.iter().find_map(|line| {
            line.trim()
                .strip_prefix("name")
                .and_then(|rest| rest.split_once('='))
                .map(|(_, value)| value.trim().trim_matches('"'))
        });
        if package_name == Some(name) {
            return block
                .iter()
                .find_map(|line| {
                    line.trim()
                        .strip_prefix("version")
                        .and_then(|rest| rest.split_once('='))
                        .map(|(_, value)| value.trim().trim_matches('"'))
                })
                .unwrap_or_else(|| panic!("Cargo.lock {name} has no version"));
        }
    }
    panic!("Cargo.lock has no {name} package");
}

fn is_plain_triple(text: &str) -> bool {
    let mut parts = text.split('.');
    let Some(major) = parts.next() else {
        return false;
    };
    let Some(minor) = parts.next() else {
        return false;
    };
    let Some(patch) = parts.next() else {
        return false;
    };
    parts.next().is_none()
        && !major.is_empty()
        && !minor.is_empty()
        && !patch.is_empty()
        && major.chars().all(|ch| ch.is_ascii_digit())
        && minor.chars().all(|ch| ch.is_ascii_digit())
        && patch.chars().all(|ch| ch.is_ascii_digit())
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
