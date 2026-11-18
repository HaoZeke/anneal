//! Repository lint: energy hit comparisons must go through `energy_hits`.

use std::fs;
use std::path::{Path, PathBuf};

const SKIP_DIRS: &[&str] = &["vendor", "target", ".git", "crates/anneal-python/target"];

const ALLOWED_FILES: &[&str] = &[
    "src/target_tol.rs",
    "src/python.rs",
    "scripts/target_tol.py",
    "pytest/test_target_tol.py",
    "scripts/growpal_h2h/run_campaign.py",
    "scripts/growpal_h2h_v2/run_campaign.py",
    "scripts/gad_published/run_campaign.py",
    "scripts/pimc_lj38/run_campaign.py",
    "scripts/ase_nonlj/summarize.py",
];

fn root() -> PathBuf {
    PathBuf::from(env!("CARGO_MANIFEST_DIR"))
}

fn skip_dir(path: &Path) -> bool {
    SKIP_DIRS.iter().any(|name| {
        path.components()
            .any(|component| component.as_os_str() == std::ffi::OsStr::new(name))
    })
}

fn strip_rust_line(line: &str) -> &str {
    line.split("//").next().unwrap_or(line).trim()
}

fn strip_py_line(line: &str) -> String {
    let mut out = String::new();
    let mut in_string = false;
    let mut quote = '\0';
    let mut escaped = false;
    for ch in line.chars() {
        if in_string {
            out.push(ch);
            if escaped {
                escaped = false;
            } else if ch == '\\' {
                escaped = true;
            } else if ch == quote {
                in_string = false;
            }
            continue;
        }
        if ch == '"' || ch == '\'' {
            in_string = true;
            quote = ch;
            out.push(ch);
            continue;
        }
        if ch == '#' {
            break;
        }
        out.push(ch);
    }
    out.trim().to_string()
}

fn allowed(rel: &str) -> bool {
    ALLOWED_FILES.contains(&rel)
}

fn raw_target_leq(line: &str) -> bool {
    let lower = line.to_ascii_lowercase();
    let mut rest = lower.as_str();
    while let Some(pos) = rest.find("<= target") {
        let after = pos + "<= target".len();
        match rest.as_bytes().get(after) {
            None | Some(b' ') | Some(b'+') | Some(b')') | Some(b',') | Some(b';') => return true,
            _ => {}
        }
        rest = &rest[after..];
    }
    false
}

fn is_hit_violation_rust(line: &str) -> bool {
    let trimmed = line.trim_start();
    if trimmed.starts_with("///") || trimmed.starts_with("//!") || trimmed.starts_with('*') {
        return false;
    }
    if line.contains("energy_hits") || line.contains("stop_below") {
        return false;
    }
    raw_target_leq(line) || line.to_ascii_lowercase().contains("<= self.target")
}

fn is_hit_violation_py(line: &str) -> bool {
    if line.contains("energy_hits") {
        return false;
    }
    let stripped = strip_py_line(line);
    let lower = stripped.to_ascii_lowercase();
    raw_target_leq(&stripped)
        || lower.contains("<= self.target")
        || lower.contains("<= reference")
        || lower.contains("best <= target")
        || lower.contains("\"h2h_tol\": \"0.001\"")
        || lower.contains("'h2h_tol': '0.001'")
        || lower.contains("h2h_tol=\"0.001\"")
        || lower.contains("h2h_tol='0.001'")
}

fn is_hit_violation_sh(line: &str) -> bool {
    let trimmed = line.split('#').next().unwrap_or(line).trim();
    trimmed.contains("H2H_TOL=0.001") || trimmed.contains("H2H_TOL=\"0.001\"")
}

fn walk_files(dir: &Path, exts: &[&str], sink: &mut dyn FnMut(PathBuf)) {
    let entries =
        fs::read_dir(dir).unwrap_or_else(|error| panic!("read {}: {error}", dir.display()));
    for entry in entries {
        let entry = entry.expect("dir entry");
        let path = entry.path();
        if path.is_dir() {
            if !skip_dir(&path) {
                walk_files(&path, exts, sink);
            }
            continue;
        }
        if exts
            .iter()
            .any(|ext| path.extension().is_some_and(|e| e == *ext))
        {
            sink(path);
        }
    }
}

#[test]
fn no_raw_energy_hit_comparisons_outside_energy_hits() {
    let root = root();
    let mut violations = Vec::new();

    walk_files(&root.join("src"), &["rs"], &mut |path| {
        let rel = path
            .strip_prefix(&root)
            .expect("under root")
            .display()
            .to_string();
        if allowed(&rel) {
            return;
        }
        let text = fs::read_to_string(&path).expect("read source");
        for (index, line) in text.lines().enumerate() {
            let stripped = strip_rust_line(line);
            if !stripped.is_empty() && is_hit_violation_rust(stripped) {
                violations.push(format!("{}:{}: {}", rel, index + 1, stripped));
            }
        }
    });

    walk_files(&root.join("examples"), &["rs"], &mut |path| {
        let rel = path
            .strip_prefix(&root)
            .expect("under root")
            .display()
            .to_string();
        if allowed(&rel) {
            return;
        }
        let text = fs::read_to_string(&path).expect("read example");
        for (index, line) in text.lines().enumerate() {
            let stripped = strip_rust_line(line);
            if !stripped.is_empty() && is_hit_violation_rust(stripped) {
                violations.push(format!("{}:{}: {}", rel, index + 1, stripped));
            }
        }
    });

    walk_files(&root.join("scripts"), &["py"], &mut |path| {
        let rel = path
            .strip_prefix(&root)
            .expect("under root")
            .display()
            .to_string();
        if allowed(&rel) {
            return;
        }
        let text = fs::read_to_string(&path).expect("read script");
        for (index, line) in text.lines().enumerate() {
            let stripped = strip_py_line(line);
            if !stripped.is_empty() && is_hit_violation_py(&stripped) {
                violations.push(format!("{}:{}: {}", rel, index + 1, stripped));
            }
        }
    });

    walk_files(&root.join("scripts"), &["sh"], &mut |path| {
        let rel = path
            .strip_prefix(&root)
            .expect("under root")
            .display()
            .to_string();
        let text = fs::read_to_string(&path).expect("read shell");
        for (index, line) in text.lines().enumerate() {
            if is_hit_violation_sh(line) {
                violations.push(format!("{}:{}: {}", rel, index + 1, line.trim()));
            }
        }
    });

    assert!(
        violations.is_empty(),
        "raw target-energy hit comparisons must use energy_hits:\n{}",
        violations.join("\n")
    );
}
