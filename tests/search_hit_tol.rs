//! Solved flags and first encounters share `SEARCH_HIT_TOL` (`1e-4`).
//! The ensemble SOLVED marker, `tally.solved` and `first_target_calls` moved
//! from reference + 1e-3 to reference + 1e-4.
//! `lj_ensemble_splice`, `ttt_lj38`, `lj_core_census` and `lj_lattice_probe`
//! score hits with `DEFAULT_HIT_TOL`.

use std::fs;
use std::path::{Path, PathBuf};

fn root() -> PathBuf {
    PathBuf::from(env!("CARGO_MANIFEST_DIR"))
}

fn read(rel: &str) -> String {
    fs::read_to_string(root().join(rel)).unwrap_or_else(|error| panic!("read {rel}: {error}"))
}

fn compact(text: &str) -> String {
    text.split_whitespace().collect::<Vec<_>>().join("")
}

fn is_search_tol(arg: &str) -> bool {
    matches!(
        compact(arg).as_str(),
        "anneal_core::SEARCH_HIT_TOL" | "crate::SEARCH_HIT_TOL" | "SEARCH_HIT_TOL"
    )
}

fn is_bare_search_literal(arg: &str) -> bool {
    matches!(compact(arg).as_str(), "1e-4" | "1.0e-4" | "0.0001")
}

fn is_default_hit_tol(arg: &str) -> bool {
    matches!(
        compact(arg).as_str(),
        "anneal_core::DEFAULT_HIT_TOL" | "crate::DEFAULT_HIT_TOL" | "DEFAULT_HIT_TOL"
    )
}

fn is_hard_coded_hit_tol(arg: &str) -> bool {
    matches!(
        compact(arg).as_str(),
        "1e-3" | "1.0e-3" | "0.001" | "1e-4" | "1.0e-4" | "0.0001"
    )
}

/// Arguments of each `name(...)` call. Nested parentheses stay inside an argument.
fn call_args<'a>(text: &'a str, name: &str) -> Vec<Vec<&'a str>> {
    let needle = format!("{name}(");
    let mut out = Vec::new();
    let mut from = 0;
    while let Some(rel) = text[from..].find(&needle) {
        let open = from + rel + needle.len();
        let (args, len) = balanced(&text[open..]);
        out.push(split_args(args));
        from = open + len;
    }
    out
}

fn balanced(text: &str) -> (&str, usize) {
    let mut depth = 1usize;
    for (index, ch) in text.char_indices() {
        match ch {
            '(' => depth += 1,
            ')' => {
                depth -= 1;
                if depth == 0 {
                    return (&text[..index], index + 1);
                }
            }
            _ => {}
        }
    }
    (text, text.len())
}

fn split_args(args: &str) -> Vec<&str> {
    let mut out = Vec::new();
    let mut depth = 0isize;
    let mut start = 0usize;
    for (index, ch) in args.char_indices() {
        match ch {
            '(' | '[' | '{' => depth += 1,
            ')' | ']' | '}' => depth -= 1,
            ',' if depth == 0 => {
                out.push(args[start..index].trim());
                start = index + 1;
            }
            _ => {}
        }
    }
    let tail = args[start..].trim();
    if !tail.is_empty() {
        out.push(tail);
    }
    out
}

fn replace_nth(text: &str, from: &str, to: &str, n: usize) -> String {
    let mut start = 0usize;
    for index in 0..=n {
        let Some(rel) = text[start..].find(from) else {
            panic!("occurrence {index} of {from:?} missing");
        };
        if index == n {
            let at = start + rel;
            return format!("{}{to}{}", &text[..at], &text[at + from.len()..]);
        }
        start += rel + from.len();
    }
    unreachable!()
}

fn cluster_bench_problems(text: &str) -> Vec<String> {
    let mut problems = Vec::new();
    let hits = call_args(text, "energy_hits");
    let encounters = call_args(text, "first_encounter");
    if hits.len() != 1 || encounters.len() != 1 {
        problems.push(format!(
            "expected one hit flag and one encounter, found {} and {}",
            hits.len(),
            encounters.len()
        ));
    }
    for args in &hits {
        let tol = args.last().copied().unwrap_or("");
        if !is_search_tol(tol) {
            problems.push(format!("hit flag tolerance is {tol:?}"));
        }
    }
    for args in &encounters {
        let tol = args.get(2).copied().unwrap_or("");
        if !is_search_tol(tol) {
            problems.push(format!("first_encounter tolerance is {tol:?}"));
        }
    }
    if let (Some(hit), Some(encounter)) = (
        hits.first().and_then(|args| args.last().copied()),
        encounters.first().and_then(|args| args.get(2).copied()),
    ) && compact(hit) != compact(encounter)
    {
        problems.push(format!(
            "hit flag {hit:?} and first_encounter {encounter:?} differ"
        ));
    }
    problems
}

fn discovery_addends(text: &str) -> Vec<String> {
    let compact_text = text.split_whitespace().collect::<Vec<_>>().join(" ");
    let marker = "reference.map(|r| r + ";
    let mut out = Vec::new();
    let mut rest = compact_text.as_str();
    while let Some(index) = rest.find(marker) {
        let after = &rest[index + marker.len()..];
        let end = after.find(')').unwrap_or(after.len());
        out.push(after[..end].to_string());
        rest = &after[end..];
    }
    out
}

fn ensemble_problems(text: &str) -> Vec<String> {
    let mut problems = Vec::new();
    let hits = call_args(text, "energy_hits");
    if hits.is_empty() {
        problems.push("lj_cluster_search has no hit flags".into());
    }
    for args in &hits {
        let tol = args.last().copied().unwrap_or("");
        if !is_search_tol(tol) {
            problems.push(format!("hit flag tolerance is {tol:?}"));
        }
    }
    let addends = discovery_addends(text);
    if addends.len() != 1 {
        problems.push(format!("expected one discovery level, found {addends:?}"));
    }
    for addend in &addends {
        if !is_search_tol(addend) {
            problems.push(format!("discovery level adds {addend:?}"));
        }
    }
    if let Some(hit) = hits.first().and_then(|args| args.last().copied()) {
        for addend in &addends {
            if compact(hit) != compact(addend) {
                problems.push(format!("hit flag {hit:?} and discovery {addend:?} differ"));
            }
        }
    }
    problems
}

fn literal_after(text: &str, marker: &str) -> f64 {
    let index = text
        .find(marker)
        .unwrap_or_else(|| panic!("missing {marker}"));
    let rest = text[index + marker.len()..].trim_start();
    let end = rest.find([';', '\n']).unwrap_or(rest.len());
    rest[..end]
        .trim()
        .trim_end_matches(';')
        .parse()
        .unwrap_or_else(|error| panic!("parse {marker}: {error}"))
}

fn walk_sources(dir: &Path, out: &mut Vec<PathBuf>) {
    let skip = ["target", "vendor", ".git", ".venv"];
    let entries = match fs::read_dir(dir) {
        Ok(entries) => entries,
        Err(_) => return,
    };
    for entry in entries.flatten() {
        let path = entry.path();
        let name = entry.file_name();
        let name = name.to_string_lossy();
        if skip.contains(&name.as_ref()) {
            continue;
        }
        if path.is_dir() {
            walk_sources(&path, out);
            continue;
        }
        let ext = path.extension().and_then(|ext| ext.to_str());
        if matches!(ext, Some("rs" | "py")) {
            out.push(path);
        }
    }
}

#[test]
fn search_hit_tol_is_one_value_in_rust_and_python() {
    let rust = literal_after(
        &read("src/target_tol.rs"),
        "pub const SEARCH_HIT_TOL: f64 = ",
    );
    let python = literal_after(&read("scripts/target_tol.py"), "SEARCH_HIT_TOL: float = ");
    assert_eq!(rust.to_bits(), python.to_bits());
    assert_eq!(rust.to_bits(), anneal_core::SEARCH_HIT_TOL.to_bits());
    assert_eq!(anneal_core::SEARCH_HIT_TOL.to_bits(), 1.0e-4_f64.to_bits());
    assert!(
        std::hint::black_box(anneal_core::SEARCH_HIT_TOL)
            < std::hint::black_box(anneal_core::DEFAULT_HIT_TOL)
    );
}

#[test]
fn cluster_bench_hit_and_encounter_share_one_tolerance() {
    let text = read("examples/cluster_bench.rs");
    assert_eq!(text.matches("anneal_core::SEARCH_HIT_TOL").count(), 2);
    assert!(
        cluster_bench_problems(&text).is_empty(),
        "{:?}",
        cluster_bench_problems(&text)
    );
    let flag = replace_nth(&text, "anneal_core::SEARCH_HIT_TOL", "1e-4", 0);
    assert!(
        !cluster_bench_problems(&flag).is_empty(),
        "a literal hit flag must fail"
    );
    let encounter = replace_nth(
        &text,
        "anneal_core::SEARCH_HIT_TOL",
        "anneal_core::DEFAULT_HIT_TOL",
        1,
    );
    assert!(
        !cluster_bench_problems(&encounter).is_empty(),
        "an encounter on DEFAULT_HIT_TOL must fail"
    );
}

#[test]
fn ensemble_discovery_uses_the_hit_flag_tolerance() {
    let text = read("examples/lj_cluster_search.rs");
    assert!(
        ensemble_problems(&text).is_empty(),
        "{:?}",
        ensemble_problems(&text)
    );
    let stop = text.replacen(
        "r + anneal_core::SEARCH_HIT_TOL",
        "r + anneal_core::DEFAULT_HIT_TOL",
        1,
    );
    assert_ne!(text, stop);
    assert!(
        !ensemble_problems(&stop).is_empty(),
        "a discovery level of DEFAULT_HIT_TOL must fail"
    );
    let flag = replace_nth(&text, "anneal_core::SEARCH_HIT_TOL", "1e-4", 0);
    assert!(
        !ensemble_problems(&flag).is_empty(),
        "a literal hit flag must fail"
    );
}

#[test]
fn no_energy_hits_call_passes_a_bare_search_literal() {
    let mut files = Vec::new();
    walk_sources(&root(), &mut files);
    let mut offenders = Vec::new();
    for path in files {
        let text = fs::read_to_string(&path).unwrap_or_default();
        for args in call_args(&text, "energy_hits") {
            let tol = args.last().copied().unwrap_or("");
            if is_bare_search_literal(tol) {
                offenders.push(format!("{}: {tol}", path.display()));
            }
        }
    }
    assert!(
        offenders.is_empty(),
        "a bare 1e-4 is not a named hit tolerance:\n{}",
        offenders.join("\n")
    );
}

const ENSEMBLE_MOVE: &str = "The ensemble SOLVED marker, `tally.solved` and `first_target_calls` moved from reference + 1e-3 to reference + 1e-4.";

fn flat(text: &str) -> String {
    text.lines()
        .map(|line| {
            let trimmed = line.trim();
            let trimmed = trimmed
                .strip_prefix("//!")
                .or_else(|| trimmed.strip_prefix("///"))
                .or_else(|| trimmed.strip_prefix("//"))
                .or_else(|| trimmed.strip_prefix('#'))
                .unwrap_or(trimmed);
            trimmed.trim()
        })
        .collect::<Vec<_>>()
        .join(" ")
        .split_whitespace()
        .collect::<Vec<_>>()
        .join(" ")
}

/// Drivers whose hit window is the paper default, not the search window.
const DEFAULT_WINDOW_DRIVERS: &[&str] = &[
    "examples/lj_core_census.rs",
    "examples/lj_ensemble_splice.rs",
    "examples/lj_lattice_probe.rs",
    "examples/ttt_lj38.rs",
];

fn default_window_problems(text: &str) -> Vec<String> {
    let hits = call_args(text, "energy_hits");
    let mut problems = Vec::new();
    if hits.is_empty() {
        problems.push("no energy_hits call".into());
    }
    for args in &hits {
        let tol = args.last().copied().unwrap_or("");
        if is_hard_coded_hit_tol(tol) {
            problems.push(format!("hard-coded tolerance {tol:?}"));
        } else if !is_default_hit_tol(tol) {
            problems.push(format!("tolerance {tol:?} is not DEFAULT_HIT_TOL"));
        }
    }
    problems
}

#[test]
fn the_ensemble_move_is_stated_plainly() {
    for rel in [
        "src/target_tol.rs",
        "scripts/target_tol.py",
        "examples/lj_cluster_search.rs",
        "examples/common/ensemble_report.rs",
        "src/methods/ensemble.rs",
        "pytest/test_target_tol.py",
    ] {
        let text = flat(&read(rel));
        assert!(
            text.contains(ENSEMBLE_MOVE),
            "{rel} does not state the ensemble move"
        );
        assert!(
            !text.contains("those drivers were scored"),
            "{rel} still claims every driver was already scored at 1e-4"
        );
        assert!(
            !text.contains("already at 1e-4"),
            "{rel} still says something was already at 1e-4"
        );
    }
    let own = flat(&read("tests/search_hit_tol.rs"));
    assert!(own.contains(ENSEMBLE_MOVE));
}

#[test]
fn default_window_drivers_use_default_hit_tol() {
    for rel in DEFAULT_WINDOW_DRIVERS {
        let text = read(rel);
        let problems = default_window_problems(&text);
        assert!(problems.is_empty(), "{rel}: {problems:?}");
        let hard = text.replacen("anneal_core::DEFAULT_HIT_TOL", "1e-3", 1);
        assert_ne!(text, hard, "{rel} has no DEFAULT_HIT_TOL to hard-code");
        assert!(
            !default_window_problems(&hard).is_empty(),
            "{rel}: a hard-coded tolerance must fail"
        );
        let other = text.replacen(
            "anneal_core::DEFAULT_HIT_TOL",
            "anneal_core::SEARCH_HIT_TOL",
            1,
        );
        assert_ne!(text, other);
        assert!(
            !default_window_problems(&other).is_empty(),
            "{rel}: a different tolerance must fail"
        );
    }
}
