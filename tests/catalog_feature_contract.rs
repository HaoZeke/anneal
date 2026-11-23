#[cfg(not(feature = "catalog"))]
mod without_catalog {
    use std::collections::{BTreeMap, BTreeSet, VecDeque};
    use std::path::Path;
    use std::process::Command;

    /// Crates that must not appear once `catalog` is left off.
    const READCON_CRATES: &[&str] = &["readcon-core", "readcon-db", "heed", "lmdb-master-sys"];

    /// `--offline` cargo tree fails on a cold registry cache: the
    /// no-default job never fetches the python feature set, so crates
    /// such as bytemuck are not on disk. `cargo metadata` resolves that
    /// set and downloads what it needs. The walk then reads the resolved
    /// graph instead of a text tree.
    fn cargo_metadata(extra_args: &[&str]) -> serde_json::Value {
        let cargo = std::env::var("CARGO").unwrap_or_else(|_| "cargo".to_owned());
        let output = Command::new(cargo)
            .current_dir(env!("CARGO_MANIFEST_DIR"))
            .env("CARGO_TERM_COLOR", "never")
            .args([
                "metadata",
                "--format-version",
                "1",
                "--locked",
                "--no-default-features",
            ])
            .args(extra_args)
            .output()
            .expect("invoke cargo metadata");
        assert!(
            output.status.success(),
            "cargo metadata failed: {}",
            String::from_utf8_lossy(&output.stderr)
        );
        serde_json::from_slice(&output.stdout).expect("cargo metadata json")
    }

    fn json_array<'a>(value: &'a serde_json::Value, key: &str) -> &'a [serde_json::Value] {
        value
            .get(key)
            .and_then(serde_json::Value::as_array)
            .map(Vec::as_slice)
            .unwrap_or(&[])
    }

    /// Normal edges of one workspace member, including transitive and
    /// target-specific edges. Dev and build edges are not the link graph.
    /// A null `kind` is a normal edge in the format-version 1 resolve.
    fn normal_dependencies_of(metadata: &serde_json::Value, root: &str) -> BTreeSet<String> {
        let mut name_of: BTreeMap<&str, &str> = BTreeMap::new();
        for package in metadata["packages"].as_array().expect("packages") {
            let id = package["id"].as_str().expect("package id");
            let name = package["name"].as_str().expect("package name");
            name_of.insert(id, name);
        }
        let root_id = metadata["workspace_members"]
            .as_array()
            .expect("workspace_members")
            .iter()
            .filter_map(|member| member.as_str())
            .find(|id| name_of.get(id) == Some(&root))
            .unwrap_or_else(|| panic!("workspace member {root} is missing from the resolve"));

        let mut edges: BTreeMap<&str, Vec<&str>> = BTreeMap::new();
        for node in metadata["resolve"]["nodes"]
            .as_array()
            .expect("resolve nodes")
        {
            let id = node["id"].as_str().expect("node id");
            let mut deps = Vec::new();
            for dep in json_array(node, "deps") {
                let kinds = json_array(dep, "dep_kinds");
                let normal = kinds.is_empty()
                    || kinds.iter().any(|kind| {
                        matches!(
                            kind.get("kind").and_then(|value| value.as_str()),
                            None | Some("normal")
                        )
                    });
                if normal && let Some(pkg) = dep["pkg"].as_str() {
                    deps.push(pkg);
                }
            }
            edges.insert(id, deps);
        }

        let mut names = BTreeSet::new();
        let mut seen = BTreeSet::new();
        let mut queue = VecDeque::from([root_id]);
        while let Some(id) = queue.pop_front() {
            if !seen.insert(id) {
                continue;
            }
            let name = name_of
                .get(id)
                .copied()
                .unwrap_or_else(|| panic!("resolve id {id} has no package"));
            names.insert(name.to_string());
            queue.extend(edges.get(id).into_iter().flatten().copied());
        }
        names
    }

    fn assert_readcon_absent(root: &str, names: &BTreeSet<String>, required: &[&str]) {
        assert!(
            names.contains(root),
            "{root} was not walked; the resolve check would pass vacuously"
        );
        for crate_name in required {
            assert!(
                names.contains(*crate_name),
                "{root} resolve did not reach {crate_name}, so the readcon check is not looking at the dependency tree"
            );
        }
        for crate_name in READCON_CRATES {
            assert!(
                !names.contains(*crate_name),
                "{root} with --no-default-features still depends on {crate_name}"
            );
        }
    }

    #[test]
    fn no_default_features_drops_readcon_and_lmdb_from_dependency_tree() {
        // `--features python` is the feature set whose cold-cache fetch
        // failed. Both workspace roots are walked in that resolve.
        let metadata = cargo_metadata(&["--features", "python"]);
        assert_readcon_absent(
            "anneal-core",
            &normal_dependencies_of(&metadata, "anneal-core"),
            &["ndarray", "eindir-core", "digest"],
        );
        assert_readcon_absent(
            "anneal-python",
            &normal_dependencies_of(&metadata, "anneal-python"),
            &["anneal-core", "pyo3", "rgsaddle"],
        );
    }

    #[cfg(unix)]
    #[test]
    fn no_default_features_artifacts_export_no_readcon_symbols() {
        let nm = ["nm", "llvm-nm"]
            .into_iter()
            .find(|tool| {
                Command::new(tool)
                    .arg("--version")
                    .output()
                    .is_ok_and(|out| out.status.success())
            })
            .expect("nm or llvm-nm is required to inspect symbols");

        let inspect = |path: &Path| {
            let out = Command::new(nm)
                .args(["-g", "--defined-only"])
                .arg(path)
                .output()
                .unwrap_or_else(|err| panic!("invoke {nm} on {}: {err}", path.display()));
            assert!(
                out.status.success(),
                "{nm} failed on {}: {}",
                path.display(),
                String::from_utf8_lossy(&out.stderr)
            );
            let stdout = String::from_utf8_lossy(&out.stdout);
            let leaked: Vec<&str> = stdout
                .lines()
                .filter(|line| {
                    let sym = line.split_whitespace().last().unwrap_or_default();
                    let stripped = sym.strip_prefix('_').unwrap_or(sym);
                    stripped.starts_with("rkr_")
                        || stripped.starts_with("rkrdb_")
                        || stripped.starts_with("mdb_")
                        || sym.contains("readcon_core")
                        || sym.contains("readcon_db")
                })
                .collect();
            assert!(
                leaked.is_empty(),
                "readcon/lmdb symbols leaked into {}:\n{}",
                path.display(),
                leaked.join("\n")
            );
        };

        let exe = std::env::current_exe().expect("current test executable");
        inspect(&exe);

        // Build an isolated staticlib + rlib under --no-default-features so
        // stale artifacts in a shared target directory cannot mask or pollute
        // the check.
        let cargo = std::env::var("CARGO").unwrap_or_else(|_| "cargo".to_owned());
        let tmp = tempfile::tempdir().expect("tempdir for isolated target");
        let build = Command::new(cargo)
            .current_dir(env!("CARGO_MANIFEST_DIR"))
            .env("CARGO_TARGET_DIR", tmp.path())
            .args([
                "rustc",
                "--offline",
                "--locked",
                "--lib",
                "--no-default-features",
                "--",
                "--crate-type=rlib",
                "--crate-type=staticlib",
            ])
            .output()
            .expect("invoke cargo rustc --no-default-features");
        assert!(
            build.status.success(),
            "isolated --no-default-features build failed: {}",
            String::from_utf8_lossy(&build.stderr)
        );
        let debug_dir = tmp.path().join("debug");
        inspect(&debug_dir.join("libanneal_core.rlib"));
        inspect(&debug_dir.join("libanneal_core.a"));
    }
}

#[cfg(feature = "catalog")]
mod with_catalog {
    #[test]
    fn default_catalog_feature_exposes_corpus_and_network_db_apis() {
        let _corpus_ty = std::any::type_name::<anneal_core::minima_db::MinimaCorpus>();
        let _pes_ty = std::any::type_name::<anneal_core::pes_db::PesNetworkDatabase>();
    }
}
