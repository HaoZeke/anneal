use std::env;

#[cfg(any(feature = "gen-header", feature = "schema"))]
use std::path::PathBuf;

#[cfg(feature = "gen-header")]
#[path = "version_macros.rs"]
mod version_macros;

/// Generate C header via cbindgen (only when `gen-header` feature is active).
/// Run `cargo build --features gen-header` or `pixi r gen-header` to regenerate.
#[cfg(feature = "gen-header")]
fn generate_c_header(crate_dir: &str) {
    // Once any rerun-if line is printed, cargo stops rescanning the
    // package, so name every input of the header. Without these a
    // changed doc comment would not regenerate it.
    for input in ["src", "cbindgen.toml", "version_macros.rs", "Cargo.toml"] {
        println!("cargo::rerun-if-changed={input}");
    }

    let output_dir = PathBuf::from(crate_dir).join("include");
    std::fs::create_dir_all(&output_dir).unwrap();

    let mut config =
        cbindgen::Config::from_file("cbindgen.toml").expect("Unable to find cbindgen.toml");

    let version = env::var("CARGO_PKG_VERSION").unwrap();
    let cargo_core = ["MAJOR", "MINOR", "PATCH"]
        .map(|part| env::var(format!("CARGO_PKG_VERSION_{part}")).unwrap());
    let ours = version_macros::numeric_core(&version)
        .map(|(major, minor, patch)| [major, minor, patch].map(|n| n.to_string()));
    assert_eq!(
        ours,
        Some(cargo_core),
        "version_macros.rs parses {version:?} differently from cargo"
    );
    // The cog markers let the cogapp check see the same block
    // that ci/version/rgpot_version.py writes, so neither copy can drift.
    let version_block = format!(
        "/* [[[cog import rgpot_version as v; cog.outl(v.header_macros())]]] */\n\
         {macros}\n\
         /* [[[end]]] */",
        macros = version_macros::header_macros(&version),
    );
    config.after_includes = Some(match config.after_includes {
        Some(existing) => format!("{existing}\n\n{version_block}"),
        None => version_block,
    });

    cbindgen::Builder::new()
        .with_crate(crate_dir)
        .with_config(config)
        .generate()
        .expect("Unable to generate C bindings")
        .write_to_file(output_dir.join("rgpot.h"));

    let path = output_dir.join("rgpot.h");
    format_c_header(crate_dir, &path);

    // Ensure the file ends with exactly one newline (keeps end-of-file-fixer happy).
    let content = std::fs::read_to_string(&path).unwrap();
    std::fs::write(&path, format!("{}\n", content.trim_end())).unwrap();
}

#[cfg(feature = "gen-header")]
fn format_c_header(crate_dir: &str, header: &std::path::Path) {
    let repo_root = PathBuf::from(crate_dir).join("..");
    let style_file = repo_root.join("CppCore").join(".clang-format");
    let style_arg = if style_file.is_file() {
        println!("cargo::rerun-if-changed={}", style_file.display());
        format!("--style=file:{}", style_file.display())
    } else {
        "--style=LLVM".to_owned()
    };

    let pixi_cf = repo_root
        .join(".pixi")
        .join("envs")
        .join("devbld")
        .join("bin")
        .join("clang-format");
    let mut candidates: Vec<PathBuf> = vec![PathBuf::from("clang-format")];
    if pixi_cf.is_file() {
        candidates.push(pixi_cf);
    }
    candidates.push(PathBuf::from("clang-format-19"));
    candidates.push(PathBuf::from("clang-format-16"));

    for bin in candidates {
        match std::process::Command::new(&bin)
            .args([&style_arg, "-i"])
            .arg(header)
            .status()
        {
            Ok(status) if status.success() => return,
            Ok(status) => panic!(
                "{} failed while formatting {} with status {status}",
                bin.display(),
                header.display()
            ),
            Err(err) if err.kind() == std::io::ErrorKind::NotFound => continue,
            Err(err) => panic!(
                "failed to invoke {} on {}: {err}",
                bin.display(),
                header.display()
            ),
        }
    }
    println!(
        "cargo::warning=clang-format not found; {} left unformatted",
        header.display()
    );
}

/// Source revision of the build: `RGPOT_SOURCE_REVISION`, else the git
/// short hash of the checkout, else empty.
fn source_revision(crate_dir: &str) -> String {
    if let Ok(rev) = env::var("RGPOT_SOURCE_REVISION") {
        return rev.trim().to_owned();
    }
    std::process::Command::new("git")
        .args(["-C", crate_dir, "rev-parse", "--short=12", "HEAD"])
        .output()
        .ok()
        .filter(|out| out.status.success())
        .map(|out| String::from_utf8_lossy(&out.stdout).trim().to_owned())
        .unwrap_or_default()
}

fn main() {
    #[allow(unused_variables)]
    let crate_dir = env::var("CARGO_MANIFEST_DIR").unwrap();

    println!("cargo::rerun-if-env-changed=RGPOT_SOURCE_REVISION");
    println!(
        "cargo::rustc-env=RGPOT_SOURCE_REVISION={}",
        source_revision(&crate_dir)
    );
    if let Ok(target) = env::var("TARGET") {
        println!("cargo::rustc-env=RGPOT_BUILD_TARGET={target}");
    }

    // eindir-core is a normal Cargo dependency now; its capi #[no_mangle] symbols
    // resolve through the shared crate, so no prebuilt static lib link is needed.

    #[cfg(feature = "gen-header")]
    generate_c_header(&crate_dir);

    #[cfg(feature = "schema")]
    compile_capnp_schema(&crate_dir);
}

#[cfg(feature = "schema")]
fn compile_capnp_schema(crate_dir: &str) {
    // Prefer the bundled schema (works on crates.io).
    // Fall back to the monorepo location for local development.
    let bundled = PathBuf::from(crate_dir)
        .join("schema")
        .join("Potentials.capnp");
    let monorepo = PathBuf::from(crate_dir)
        .join("..")
        .join("CppCore")
        .join("rgpot")
        .join("rpc")
        .join("Potentials.capnp");

    let (capnp_dir, capnp_schema) = if bundled.exists() {
        (PathBuf::from(crate_dir).join("schema"), bundled)
    } else if monorepo.exists() {
        (
            PathBuf::from(crate_dir)
                .join("..")
                .join("CppCore")
                .join("rgpot")
                .join("rpc"),
            monorepo,
        )
    } else {
        // Neither found — generate stub so rpc/mod.rs include! doesn't fail.
        let out_dir = env::var("OUT_DIR").unwrap();
        let stub = PathBuf::from(&out_dir).join("Potentials_capnp.rs");
        std::fs::write(
            &stub,
            "// Auto-generated stub: Potentials.capnp not found at build time.\n",
        )
        .expect("Failed to write Cap'n Proto stub");
        return;
    };

    // The build-identity fields exist only in schema revisions that carry
    // them; the producer code is compiled in when they are present.
    println!("cargo::rustc-check-cfg=cfg(rgpot_schema_build_identity)");
    if std::fs::read_to_string(&capnp_schema)
        .map(|text| text.contains("buildVersion"))
        .unwrap_or(false)
    {
        println!("cargo::rustc-cfg=rgpot_schema_build_identity");
    }
    println!("cargo::rerun-if-changed={}", capnp_schema.display());

    // The vendored copy is the pinned potentials-schema file; the compiled
    // schema adds the rgpot-local series next to it, as the meson build does.
    let local = capnp_dir.join("local");
    println!("cargo::rerun-if-changed={}", local.display());
    let vendored = std::fs::read_to_string(&capnp_schema)
        .expect("Failed to read Cap'n Proto schema")
        .replace("\r\n", "\n");
    let effective_dir = PathBuf::from(env::var("OUT_DIR").unwrap()).join("schema");
    std::fs::create_dir_all(&effective_dir).expect("Failed to create the schema directory");
    let effective = effective_dir.join("Potentials.capnp");
    std::fs::write(&effective, apply_series(&vendored, &local))
        .expect("Failed to write the effective schema");

    capnpc::CompilerCommand::new()
        .src_prefix(&effective_dir)
        .file(&effective)
        .run()
        .expect("Failed to compile Cap'n Proto schema");
}

/// The vendored schema with every diff in `local/series` applied in order.
#[cfg(feature = "schema")]
fn apply_series(schema: &str, local: &std::path::Path) -> String {
    let Ok(series) = std::fs::read_to_string(local.join("series")) else {
        return schema.to_owned();
    };
    let mut text = schema.to_owned();
    for name in series
        .lines()
        .map(str::trim)
        .filter(|line| !line.is_empty() && !line.starts_with('#'))
    {
        let diff = std::fs::read_to_string(local.join(name))
            .unwrap_or_else(|err| panic!("local series diff {name}: {err}"));
        text = apply_diff(&text, &diff, name);
    }
    text
}

/// Hunks of a unified diff as (old start line, body lines), read the way
/// `schema_ledger.parse_diff` reads them.
#[cfg(feature = "schema")]
fn parse_hunks(diff: &str) -> Vec<(usize, Vec<String>)> {
    let mut hunks: Vec<(usize, Vec<String>)> = Vec::new();
    let mut in_hunk = false;
    for raw in diff.lines() {
        if raw.starts_with("+++ ") {
            in_hunk = false;
        } else if let Some(rest) = raw.strip_prefix("@@ -") {
            let start = rest
                .split([',', ' '])
                .next()
                .and_then(|n| n.parse().ok())
                .unwrap_or_else(|| panic!("bad hunk header {raw:?}"));
            hunks.push((start, Vec::new()));
            in_hunk = true;
        } else if in_hunk && (raw.starts_with([' ', '+', '-'])) {
            hunks.last_mut().unwrap().1.push(raw.to_owned());
        } else if in_hunk && raw.is_empty() {
            hunks.last_mut().unwrap().1.push(" ".to_owned());
        } else if !raw.starts_with('\\') {
            in_hunk = false;
        }
    }
    hunks
}

/// Applies a diff like `schema_ledger.apply_hunks`: exact context, at the
/// matching offset nearest the hunk header, never before the previous hunk.
#[cfg(feature = "schema")]
fn apply_diff(text: &str, diff: &str, name: &str) -> String {
    let lines: Vec<&str> = text.lines().collect();
    let mut out: Vec<&str> = Vec::new();
    let mut pos = 0;
    let hunks = parse_hunks(diff);
    for (number, (old_start, body)) in hunks.iter().enumerate() {
        let old: Vec<&str> = body
            .iter()
            .filter(|l| l.starts_with([' ', '-']))
            .map(|l| &l[1..])
            .collect();
        let want = if old.is_empty() {
            *old_start
        } else {
            old_start.saturating_sub(1)
        };
        let end = (lines.len() + 1).saturating_sub(old.len());
        let at = (pos..end.max(pos))
            .filter(|&i| lines[i..i + old.len()] == old[..])
            .min_by_key(|&i| i.abs_diff(want))
            .unwrap_or_else(|| panic!("{name}: hunk {} does not apply", number + 1));
        out.extend(&lines[pos..at]);
        out.extend(
            body.iter()
                .filter(|l| l.starts_with([' ', '+']))
                .map(|l| &l[1..]),
        );
        pos = at + old.len();
    }
    out.extend(&lines[pos..]);
    let mut joined = out.join("\n");
    if text.ends_with('\n') || text.is_empty() {
        joined.push('\n');
    }
    joined
}
