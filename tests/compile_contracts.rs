//! Compile-time checks for the certified component constructor.

use std::collections::BTreeMap;
use std::path::{Path, PathBuf};
use std::time::SystemTime;

#[test]
fn checked_rejects_uncertified_component_combinations() {
    // trybuild checks from `{target}/tests/trybuild`, and this workspace keeps
    // the target directory outside the tree. That cargo does not walk up to
    // `.cargo/config.toml`, so it builds crates.io metatensor-sys. That build
    // runs `cargo rustc` in the target directory trybuild already locked.
    // Point the same patch at the directory trybuild does walk, which is not
    // a parent of metatensor's own `--locked` build.
    // `{target}/{profile}/deps/<bin>-<hash>`
    let binary = PathBuf::from(std::env::args_os().next().expect("test binary"));
    let deps_dir = binary.parent().expect("deps directory");
    let target = deps_dir
        .parent()
        .and_then(Path::parent)
        .expect("target directory")
        .to_path_buf();
    // trybuild's own `cargo metadata` does not see a `--config
    // build.target-dir` passed to the parent. It uses `CARGO_TARGET_DIR`
    // or the workspace config. Write the patch into each of those trees.
    let mut targets = vec![target];
    if let Some(configured) = configured_target_dir()
        && !targets.contains(&configured)
    {
        targets.push(configured);
    }
    let vendor = Path::new(env!("CARGO_MANIFEST_DIR")).join("vendor/metatensor-sys");
    let mut patch = format!(
        "[patch.crates-io]\nmetatensor-sys = {{ path = \"{}\" }}\n",
        vendor.display()
    );
    // trybuild does not walk up to the workspace config. A development
    // registry patch has to be repeated here or the UI crate cannot
    // resolve it. The committed config has no patch.
    let workspace = Path::new(env!("CARGO_MANIFEST_DIR")).join(".cargo/config.toml");
    if let Ok(text) = std::fs::read_to_string(workspace)
        && let Some((_, body)) = text.split_once("[patch.crates-io]")
    {
        for line in body.lines() {
            let trimmed = line.trim();
            if trimmed.is_empty() || trimmed.starts_with('#') {
                continue;
            }
            if trimmed.starts_with('[') {
                break;
            }
            patch.push_str(trimmed);
            patch.push('\n');
        }
    }
    // trybuild copies the workspace lock and passes `--offline`. A lock
    // entry with no `source` was resolved through a path patch, and that
    // patch is not inherited from the parent cargo's command line. Point
    // the same crates at the roots the parent build already compiled.
    append_unsourced_registry_patches(&mut patch, deps_dir);
    for target in &targets {
        // The project directory is `{target}/tests/trybuild/<package>`. Its
        // cargo build of metatensor lives beside that directory, not under
        // it, so a config here is visible to trybuild and not to the nested
        // `cargo rustc --locked`.
        let config = target.join("tests/trybuild/anneal-core/.cargo");
        std::fs::create_dir_all(&config).unwrap();
        std::fs::write(config.join("config.toml"), &patch).unwrap();
    }

    let tests = trybuild::TestCases::new();
    tests.compile_fail("tests/ui/checked_rejects_*.rs");
}

/// Registry dependencies whose lock entries have no `source`.
fn unsourced_registry_deps() -> BTreeMap<String, String> {
    let manifest_dir = Path::new(env!("CARGO_MANIFEST_DIR"));
    let Ok(lock) = std::fs::read_to_string(manifest_dir.join("Cargo.lock")) else {
        return BTreeMap::new();
    };
    let Ok(manifest) = std::fs::read_to_string(manifest_dir.join("Cargo.toml")) else {
        return BTreeMap::new();
    };
    let mut wanted = BTreeMap::new();
    for block in lock.split("[[package]]").skip(1) {
        let header = block.split("dependencies").next().unwrap_or(block);
        if header.contains("\nsource = ") {
            continue;
        }
        let Some(name) = lock_field(header, "name") else {
            continue;
        };
        let Some(version) = lock_field(header, "version") else {
            continue;
        };
        if registry_dependency(&manifest, &name) {
            wanted.insert(name, version);
        }
    }
    wanted
}

fn configured_target_dir() -> Option<PathBuf> {
    let manifest_dir = Path::new(env!("CARGO_MANIFEST_DIR"));
    let text = std::fs::read_to_string(manifest_dir.join(".cargo/config.toml")).ok()?;
    for line in text.lines() {
        let line = line.split('#').next().unwrap_or(line).trim();
        let Some(rest) = line.strip_prefix("target-dir") else {
            continue;
        };
        let Some(rest) = rest.trim_start().strip_prefix('=') else {
            continue;
        };
        let rest = rest.trim().trim_matches('"');
        if rest.is_empty() {
            return None;
        }
        return Some(manifest_dir.join(rest));
    }
    None
}

fn lock_field(header: &str, key: &str) -> Option<String> {
    let prefix = format!("\n{key} = \"");
    let start = header.find(&prefix)? + prefix.len();
    let rest = &header[start..];
    let end = rest.find('"')?;
    Some(rest[..end].to_string())
}

/// A manifest dependency requested from a registry, not a path or git pin.
fn registry_dependency(manifest: &str, name: &str) -> bool {
    for line in manifest.lines() {
        let code = line.split('#').next().unwrap_or(line).trim();
        let Some(rest) = code.strip_prefix(name) else {
            continue;
        };
        let rest = rest.trim_start();
        let Some(rest) = rest.strip_prefix('=') else {
            continue;
        };
        let rest = rest.trim();
        return !rest.contains("path") && !rest.contains("git");
    }
    false
}

fn append_unsourced_registry_patches(patch: &mut String, deps_dir: &Path) {
    let wanted = unsourced_registry_deps();
    if wanted.is_empty() {
        return;
    }
    let mut found: BTreeMap<String, (SystemTime, PathBuf)> = BTreeMap::new();
    let Ok(entries) = std::fs::read_dir(deps_dir) else {
        return;
    };
    for entry in entries.flatten() {
        let file_name = entry.file_name();
        let file_name = file_name.to_string_lossy();
        if !file_name.ends_with(".d") {
            continue;
        }
        let Some(package) = wanted
            .keys()
            .find(|name| dep_info_names(name, &file_name))
            .cloned()
        else {
            continue;
        };
        let Ok(text) = std::fs::read_to_string(entry.path()) else {
            continue;
        };
        let Some(root) = crate_root_matching(&text, &package, &wanted[&package]) else {
            continue;
        };
        let modified = entry
            .metadata()
            .and_then(|meta| meta.modified())
            .unwrap_or(SystemTime::UNIX_EPOCH);
        match found.get(&package) {
            Some((previous, _)) if *previous >= modified => {}
            _ => {
                found.insert(package.clone(), (modified, root));
            }
        }
    }
    for (name, (_, root)) in found {
        if patch.contains(&format!("{name} = ")) {
            continue;
        }
        patch.push_str(&format!("{name} = {{ path = \"{}\" }}\n", root.display()));
    }
}

fn dep_info_names(package: &str, file_name: &str) -> bool {
    let crate_name = package.replace('-', "_");
    let stem = file_name.trim_end_matches(".d");
    stem == crate_name
        || stem == format!("lib{crate_name}")
        || file_name.starts_with(&format!("{crate_name}-"))
        || file_name.starts_with(&format!("lib{crate_name}-"))
}

fn crate_root_matching(dep_info: &str, name: &str, version: &str) -> Option<PathBuf> {
    for token in dep_info.split_whitespace() {
        let token = token.trim_end_matches(':');
        if !token.ends_with(".rs") {
            continue;
        }
        let mut dir = PathBuf::from(token);
        for _ in 0..12 {
            if !dir.pop() {
                break;
            }
            let manifest = dir.join("Cargo.toml");
            if manifest.is_file() && manifest_matches(&manifest, name, version) {
                return Some(dir);
            }
        }
    }
    None
}

fn manifest_matches(path: &Path, name: &str, version: &str) -> bool {
    let Ok(text) = std::fs::read_to_string(path) else {
        return false;
    };
    let mut in_package = false;
    let mut found_name = false;
    let mut found_version = false;
    for line in text.lines() {
        let trimmed = line.trim();
        if trimmed.starts_with('[') {
            if in_package {
                break;
            }
            in_package = trimmed == "[package]";
            continue;
        }
        if !in_package {
            continue;
        }
        if field_equals(trimmed, "name", name) {
            found_name = true;
        }
        if field_equals(trimmed, "version", version) {
            found_version = true;
        }
    }
    found_name && found_version
}

fn field_equals(line: &str, key: &str, value: &str) -> bool {
    let Some(rest) = line.strip_prefix(key) else {
        return false;
    };
    let Some(rest) = rest.trim_start().strip_prefix('=') else {
        return false;
    };
    rest.trim().trim_matches('"') == value
}
