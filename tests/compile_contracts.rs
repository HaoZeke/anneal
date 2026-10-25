//! Compile-time checks for the certified component constructor.

use std::path::{Path, PathBuf};

#[test]
fn checked_rejects_uncertified_component_combinations() {
    // trybuild checks from `{target}/tests/trybuild`, and this workspace keeps
    // the target directory outside the tree. That cargo does not walk up to
    // `.cargo/config.toml`, so it builds crates.io metatensor-sys. That build
    // runs `cargo rustc` in the target directory trybuild already locked.
    // Point the same patch at the directory trybuild does walk, which is not
    // a parent of metatensor's own `--locked` build.
    // `{target}/{profile}/deps/<bin>-<hash>`
    let target = PathBuf::from(std::env::args_os().next().expect("test binary"))
        .parent()
        .and_then(Path::parent)
        .and_then(Path::parent)
        .expect("target directory")
        .to_path_buf();
    // The project directory is `{target}/tests/trybuild/<package>`. Its cargo
    // build of metatensor lives beside that directory, not under it, so a
    // config here is visible to trybuild and not to `cargo rustc --locked`.
    let config = target.join("tests/trybuild/anneal-core/.cargo");
    std::fs::create_dir_all(&config).unwrap();
    let vendor = Path::new(env!("CARGO_MANIFEST_DIR")).join("vendor/metatensor-sys");
    let mut patch = format!(
        "[patch.crates-io]\nmetatensor-sys = {{ path = \"{}\" }}\n",
        vendor.display()
    );
    // trybuild does not walk up to the workspace config. A development
    // registry patch has to be repeated here or the UI crate cannot
    // resolve it. The committed config has no patch.
    let workspace = Path::new(env!("CARGO_MANIFEST_DIR")).join(".cargo/config.toml");
    if let Ok(text) = std::fs::read_to_string(workspace) {
        if let Some((_, body)) = text.split_once("[patch.crates-io]") {
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
    }
    std::fs::write(config.join("config.toml"), patch).unwrap();

    let tests = trybuild::TestCases::new();
    tests.compile_fail("tests/ui/checked_rejects_*.rs");
}
