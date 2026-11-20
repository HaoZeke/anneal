#[cfg(not(feature = "catalog"))]
mod without_catalog {
    use std::path::Path;
    use std::process::Command;

    fn run_cargo_tree(package: &str, extra_args: &[&str]) -> String {
        let cargo = std::env::var("CARGO").unwrap_or_else(|_| "cargo".to_owned());
        let output = Command::new(cargo)
            .current_dir(env!("CARGO_MANIFEST_DIR"))
            .args([
                "tree",
                "--offline",
                "--no-default-features",
                "--prefix",
                "none",
                "-p",
                package,
            ])
            .args(extra_args)
            .output()
            .expect("invoke cargo tree");
        assert!(
            output.status.success(),
            "cargo tree failed for {package}: {}",
            String::from_utf8_lossy(&output.stderr)
        );
        String::from_utf8(output.stdout).expect("utf-8 cargo tree output")
    }

    #[test]
    fn no_default_features_drops_readcon_and_lmdb_from_dependency_tree() {
        for (package, extra) in [
            ("anneal-core", &[][..]),
            ("anneal-python", &["--features", "python"][..]),
        ] {
            let tree = run_cargo_tree(package, extra);
            for line in tree.lines() {
                let crate_name = line.split_whitespace().next().unwrap_or_default();
                assert!(
                    !matches!(
                        crate_name,
                        "readcon-core" | "readcon-db" | "heed" | "lmdb-master-sys"
                    ),
                    "{package} with --no-default-features still depends on {crate_name}:\n{tree}"
                );
            }
        }
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
