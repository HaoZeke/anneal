// Copyright (c) anneal developers
//! Source guard against reintroducing size-tuned or named-motif search branches.

use std::fs;
use std::path::{Path, PathBuf};

struct Allow {
    path: &'static str,
    needle: &'static str,
    reason: &'static str,
}

fn expected_matches(entry: &Allow) -> usize {
    match (entry.path, entry.needle) {
        ("src/methods/dls_published.rs", "mackay") => 4,
        ("scripts/morse_h2h/run_campaign.py", "n < 6") => 2,
        ("scripts/pimc_lj38/run_campaign.py", "n == 38") => 3,
        ("src/twin.rs", "if n < 8") => 2,
        ("src/structure.rs", "Template::Icosahedral" | "Template::FaceCentredCubic") => 4,
        ("src/structure.rs", "FaceCentredCubic," | "Icosahedral,") => 2,
        ("src/landfold.rs", "marks") => 9,
        ("src/methods/cluster_hopping.rs", "fcc") => 6,
        (
            "src/methods/cluster_hopping.rs",
            "Template::FaceCentredCubic" | "Template::Icosahedral",
        ) => 2,
        ("src/methods/dls_published.rs", "icosahedron") => 2,
        _ => 1,
    }
}

const ALLOW: &[Allow] = &[
    // Existing behavioural findings recorded in AUDIT.md.
    Allow {
        path: "src/methods/dls_published.rs",
        needle: "mackay",
        reason: "published DLS baseline reproduces its named primary-site template",
    },
    Allow {
        path: "src/methods/dls_published.rs",
        needle: "n <= *limit",
        reason: "explicit published-reproduction API; not used by Config::recommended",
    },
    Allow {
        path: "scripts/morse_h2h/run_campaign.py",
        needle: "n < 6",
        reason: "paper campaign input validation, not production search behavior",
    },
    Allow {
        path: "scripts/pimc_lj38/run_campaign.py",
        needle: "n == 38",
        reason: "named paper-system runner, not production search behavior",
    },
    Allow {
        path: "scripts/tip4p_h2h/run_shell.py",
        needle: "n == 21",
        reason: "published TIP4P database runner, not production search behavior",
    },
    Allow {
        path: "src/lattice.rs",
        needle: "FaceCentredCubic",
        reason: "existing named lattice constructor; replacement is audited",
    },
    Allow {
        path: "src/lattice.rs",
        needle: "Icosahedral",
        reason: "existing named lattice constructor; replacement is audited",
    },
    Allow {
        path: "src/lattice.rs",
        needle: "decahedral",
        reason: "existing named lattice constructor; replacement is audited",
    },
    // General geometric cardinality guards, not per-system tuning.
    Allow {
        path: "src/twin.rs",
        needle: "if n < 8",
        reason: "a four-atom sheet plus an exterior needs at least eight points",
    },
    Allow {
        path: "src/methods/cluster_hopping/general.rs",
        needle: "if n < 6",
        reason: "generic tetrahedral-hole regrowth needs a seed simplex and later points",
    },
    // Counters and sample counts named `n`, not atom counts.
    Allow {
        path: "src/history.rs",
        needle: "if n < 6",
        reason: "trajectory sample-count guard, not a system-size branch",
    },
    Allow {
        path: "src/methods/portfolio.rs",
        needle: "if n < 6",
        reason: "archive sample-count guard, not a system-size branch",
    },
    Allow {
        path: "src/methods/cluster_search.rs",
        needle: "n >= 12",
        reason: "Good-Turing observation-count guard, not an atom count",
    },
    // Diagnostic-only morphology labels.
    Allow {
        path: "src/structure.rs",
        needle: "Template::Icosahedral",
        reason: "PTM label is diagnostic, but its point template is also an audited builder input",
    },
    Allow {
        path: "src/structure.rs",
        needle: "Template::FaceCentredCubic",
        reason: "PTM label is diagnostic, but its point template is also an audited builder input",
    },
    Allow {
        path: "src/structure.rs",
        needle: "FaceCentredCubic,",
        reason: "PTM enum label is diagnostic, but its point template is an audited builder input",
    },
    Allow {
        path: "src/structure.rs",
        needle: "Icosahedral,",
        reason: "PTM enum label is diagnostic, but its point template is an audited builder input",
    },
    Allow {
        path: "src/morphology.rs",
        needle: "ICOSAHEDRAL",
        reason: "published descriptor reference used only by diagnostic morphology code",
    },
    Allow {
        path: "src/morphology.rs",
        needle: "FCC",
        reason: "published descriptor reference used only by diagnostic morphology code",
    },
    Allow {
        path: "src/landfold.rs",
        needle: "marks",
        reason: "local variable abbreviates landmarks and is unrelated to Marks motifs",
    },
    Allow {
        path: "src/methods/cluster_hopping.rs",
        needle: "fcc",
        reason: "trace-only PTM counters; comments explicitly prohibit downstream steering",
    },
    Allow {
        path: "src/methods/cluster_hopping.rs",
        needle: "Template::FaceCentredCubic",
        reason: "trace-only PTM classification; it does not select a move",
    },
    Allow {
        path: "src/methods/cluster_hopping.rs",
        needle: "Template::Icosahedral",
        reason: "trace-only PTM classification; it does not select a move",
    },
    Allow {
        path: "src/methods/dls_published.rs",
        needle: "icosahedron",
        reason: "published DLS baseline builds a named primary-site template",
    },
    Allow {
        path: "src/catalog/molecular.rs",
        needle: "WATER_HEXAMER_MOLECULES",
        reason: "paper-system preset constant; removal from search code is audited",
    },
];

fn production_files(root: &Path) -> Vec<PathBuf> {
    fn visit(path: &Path, out: &mut Vec<PathBuf>) {
        for entry in fs::read_dir(path).expect("read source directory") {
            let path = entry.expect("source entry").path();
            if path.is_dir() {
                visit(&path, out);
            } else if path
                .extension()
                .is_some_and(|extension| extension == "rs" || extension == "py")
            {
                out.push(path);
            }
        }
    }
    let mut out = Vec::new();
    visit(&root.join("src"), &mut out);
    visit(&root.join("scripts"), &mut out);
    out.sort();
    out
}

fn source_code(line: &str) -> &str {
    let rust = line.split_once("//").map_or(line, |(code, _)| code);
    rust.split_once('#').map_or(rust, |(code, _)| code)
}

fn production_lines(text: &str) -> Vec<(usize, &str)> {
    let mut lines = Vec::new();
    let mut cfg_test = false;
    let mut skipped_test_module = false;
    for (index, line) in text.lines().enumerate() {
        if skipped_test_module {
            if line == "}" {
                skipped_test_module = false;
            }
            continue;
        }
        if line.trim().starts_with("#[cfg(") && line.contains("test") {
            cfg_test = true;
            continue;
        }
        if cfg_test && line.trim_start().starts_with("#[") {
            continue;
        }
        if cfg_test && line.trim_start().starts_with("mod ") {
            skipped_test_module = line.contains('{');
            cfg_test = false;
            continue;
        }
        cfg_test = false;
        lines.push((index + 1, line));
    }
    lines
}

fn python_lines(text: &str) -> Vec<(usize, &str)> {
    let mut quoted = false;
    let mut out = Vec::new();
    for (index, line) in text.lines().enumerate() {
        let delimiters = line.matches("\"\"\"").count() + line.matches("'''").count();
        if !quoted {
            let code = line
                .split("\"\"\"")
                .next()
                .unwrap_or(line)
                .split("'''")
                .next()
                .unwrap_or(line);
            if !code.trim().is_empty() {
                out.push((index + 1, code));
            }
        }
        if delimiters % 2 == 1 {
            quoted = !quoted;
        }
    }
    out
}

fn motif_hit(code: &str) -> bool {
    let lower = code.to_ascii_lowercase();
    let terms: Vec<&str> = lower
        .split(|character: char| !character.is_ascii_alphanumeric())
        .filter(|word| !word.is_empty())
        .collect();
    terms.iter().any(|word| {
        matches!(
            *word,
            "icosahedral"
                | "icosahedron"
                | "decahedral"
                | "decahedron"
                | "mackay"
                | "fivefold"
                | "leary"
                | "marks"
                | "fcc"
                | "pentagon"
                | "pentagons"
        )
    }) || lower.contains("facecentredcubic")
        || lower.contains("five_fold")
        || lower.contains("motifclass")
        || lower.contains("motif_class")
        || lower.contains("d5h")
        || lower.contains("d_5h")
        || lower.contains("cyclic_about_z(5)")
}

fn size_hit(code: &str) -> bool {
    [
        "n",
        "n_points",
        "n_atoms",
        "n_bodies",
        "atom_count",
        "natoms",
    ]
    .iter()
    .any(|variable| {
        code.match_indices(variable).any(|(start, _)| {
            let end = start + variable.len();
            let left_boundary = start == 0
                || !code.as_bytes()[start - 1].is_ascii_alphanumeric()
                    && code.as_bytes()[start - 1] != b'_';
            let right_boundary = end == code.len()
                || !code.as_bytes()[end].is_ascii_alphanumeric() && code.as_bytes()[end] != b'_';
            left_boundary
                && right_boundary
                && ["<=", "==", "!=", ">=", "<", ">"].iter().any(|comparison| {
                    let rest = code[end..].trim_start();
                    let Some(rest) = rest.strip_prefix(comparison) else {
                        return false;
                    };
                    let operand = rest.trim_start();
                    let digits = operand
                        .chars()
                        .take_while(char::is_ascii_digit)
                        .collect::<String>();
                    digits.parse::<usize>().is_ok_and(|number| number >= 5)
                        || operand.starts_with("*limit")
                })
        })
    })
}

fn numeric_match_arm_hit(code: &str) -> bool {
    let pattern = code
        .split_once("=>")
        .map(|(pattern, _)| pattern)
        .or_else(|| {
            code.trim_start()
                .strip_prefix("case ")
                .and_then(|case| case.split_once(':').map(|(pattern, _)| pattern))
        });
    let Some(pattern) = pattern else {
        return false;
    };
    let pattern = pattern
        .trim()
        .strip_prefix("case ")
        .unwrap_or(pattern.trim())
        .trim_start_matches('|');
    let arms = pattern.split('|').map(str::trim).collect::<Vec<_>>();
    arms.len() > 1
        && arms.iter().all(|arm| arm.parse::<usize>().is_ok())
        && arms
            .iter()
            .any(|arm| arm.parse::<usize>().is_ok_and(|value| value >= 5))
}

fn forbidden_identifier_hit(code: &str) -> bool {
    let lower = code.to_ascii_lowercase();
    let words = lower
        .split(|character: char| !character.is_ascii_alphanumeric() && character != '_')
        .filter(|word| !word.is_empty())
        .collect::<Vec<_>>();
    words
        .iter()
        .any(|word| matches!(*word, "is_water_groups" | "prior_motif" | "lj_shells"))
        || code
            .split(|character: char| !character.is_ascii_alphanumeric() && character != '_')
            .any(|word| word == "HOT_CAP")
}

fn atomic_number_chart_hit(code: &str) -> bool {
    let compact = code
        .chars()
        .filter(|character| !character.is_whitespace())
        .collect::<String>()
        .to_ascii_lowercase();
    compact.contains("matchz")
        || compact.contains("matchatomic_number")
        || (compact.contains("atomic_number") && compact.contains("radius"))
        || (compact.contains("covalent") && compact.contains("[f64;"))
}

fn fixed_geometry_threshold_hit(code: &str) -> bool {
    let compact = code
        .chars()
        .filter(|character| !character.is_whitespace())
        .collect::<String>()
        .to_ascii_lowercase();
    (compact.contains("diameter") || compact.contains("largest/smallest"))
        && [">=1.2", ">1.2", "<=1.2", "<1.2"]
            .iter()
            .any(|threshold| compact.contains(threshold))
        || compact.contains("orders:vec![2,3,4,5,6]")
}

fn named_size_constant_hit(code: &str) -> bool {
    let compact: String = code
        .chars()
        .filter(|character| !character.is_whitespace())
        .collect();
    let Some(rest) = compact.strip_prefix("const").or_else(|| {
        compact
            .strip_prefix("pubconst")
            .or_else(|| compact.strip_prefix("pub(crate)const"))
    }) else {
        return false;
    };
    let Some((declaration, value)) = rest.split_once('=') else {
        return false;
    };
    let Some((name, ty)) = declaration.split_once(':') else {
        return false;
    };
    if !matches!(ty, "usize" | "u32" | "u64") {
        return false;
    }
    let name = name.to_ascii_lowercase();
    let digits: String = value.chars().take_while(char::is_ascii_digit).collect();
    let size_name = name == "active_volume_min"
        || name
            .split('_')
            .any(|part| matches!(part, "atoms" | "points" | "bodies" | "molecules"));
    size_name && digits.parse::<usize>().is_ok_and(|number| number >= 5)
}

fn group_size_hit(code: &str) -> bool {
    let compact: String = code
        .chars()
        .filter(|character| !character.is_whitespace())
        .collect();
    ["groups[0].len()", "groups.len()"]
        .iter()
        .filter_map(|subject| compact.split(subject).nth(1).map(|rest| (*subject, rest)))
        .any(|(subject, rest)| {
            ["<=", "==", "!=", ">=", "<", ">"]
                .iter()
                .filter_map(|comparison| rest.strip_prefix(comparison))
                .any(|rest| {
                    let operand: String = rest
                        .chars()
                        .take_while(|character| {
                            character.is_ascii_alphanumeric()
                                || matches!(character, '_' | ':' | '(' | ')')
                        })
                        .collect();
                    let numeric = operand.parse::<usize>().is_ok()
                        && (subject == "groups[0].len()"
                            || compact.contains(&format!("{subject}=="))
                            || compact.contains(&format!("{subject}!=")));
                    let named_count = subject == "groups.len()"
                        && operand.rsplit("::").next().is_some_and(|name| {
                            name.chars().any(|character| character.is_ascii_uppercase())
                        });
                    numeric || named_count
                })
        })
}

#[test]
fn detectors_cover_size_motif_and_named_count_examples() {
    assert!(size_hit("if n_points == 38 { choose_special_move(); }"));
    assert!(motif_hit("if fivefold_score > cutoff { choose_arm(); }"));
    assert!(motif_hit("let group = d5h_ops();"));
    assert!(named_size_constant_hit(
        "const MINIMUM_PACKING_ATOMS: usize = 13;"
    ));
    assert!(group_size_hit("groups[0].len() >= 4"));
    assert!(group_size_hit(
        "groups.len() > crate::catalog::molecular::WATER_HEXAMER_MOLECULES"
    ));
    assert!(!size_hit("if samples < limit { collect_more(); }"));
}

#[test]
fn new_rules_reject_prefix_and_accept_postfix_fixtures() {
    type Fixture = (&'static str, &'static str, fn(&str) -> bool);
    let fixtures: &[Fixture] = &[
        (
            "if n_bodies > 10 { cool(); }",
            "if acceptance > target { cool(); }",
            size_hit,
        ),
        (
            "38 | 55 | 75 => tuned()",
            "_ => general()",
            numeric_match_arm_hit,
        ),
        (
            "case 38 | 55 | 75: tune()",
            "case _: general()",
            numeric_match_arm_hit,
        ),
        (
            "if is_water_groups(groups) { tune(); }",
            "if groups_are_rigid(groups) { tune(); }",
            forbidden_identifier_hit,
        ),
        (
            "let prior_motif = score();",
            "let descriptor_residual = score();",
            forbidden_identifier_hit,
        ),
        (
            "const HOT_CAP: usize = 10;",
            "let cap = observations.len();",
            forbidden_identifier_hit,
        ),
        (
            "match z { 1 => 0.31, _ => 0.0 }",
            "let radius = pair_minimum(species);",
            atomic_number_chart_hit,
        ),
        (
            "if groups.len() == 1 { special(); }",
            "if groups.len() == expected.len() { validate(); }",
            group_size_hit,
        ),
        (
            "const LJ_SHELLS: &[usize] = &[13, 55];",
            "let shells = observed_shells(points);",
            forbidden_identifier_hit,
        ),
        (
            "if n <= *limit { return row; }",
            "if residual <= *limit { return row; }",
            size_hit,
        ),
        (
            "largest / smallest >= 1.2",
            "largest > smallest",
            fixed_geometry_threshold_hit,
        ),
        (
            "orders: vec![2, 3, 4, 5, 6]",
            "orders: (2..=n).collect()",
            fixed_geometry_threshold_hit,
        ),
    ];
    for (prefix, postfix, detector) in fixtures {
        assert!(detector(prefix), "prefix fixture escaped: {prefix}");
        assert!(
            !detector(postfix),
            "postfix fixture was rejected: {postfix}"
        );
    }
    assert!(motif_hit("prior_motif = fivefold_score"));
    assert!(!motif_hit("descriptor_residual = runtime_score"));

    let prefix =
        "#[cfg(test)]\nmod tests {\n fn hidden() {}\n}\nfn live() { if n_bodies > 10 { tune(); } }";
    let postfix = "#[cfg(test)]\nmod tests {\n fn hidden() { if n_bodies > 10 { tune(); } }\n}\nfn live() { if acceptance > target { tune(); } }";
    assert!(
        production_lines(prefix)
            .iter()
            .any(|(_, line)| size_hit(source_code(line)))
    );
    assert!(
        production_lines(postfix)
            .iter()
            .all(|(_, line)| !size_hit(source_code(line)))
    );
    let python_prefix = "def choose(n_bodies):\n    if n_bodies > 10:\n        return tuned()";
    let python_postfix = "def choose(n_bodies, acceptance, target):\n    if acceptance > target:\n        return tuned()";
    assert!(
        python_lines(python_prefix)
            .iter()
            .any(|(_, line)| size_hit(source_code(line)))
    );
    assert!(
        python_lines(python_postfix)
            .iter()
            .all(|(_, line)| !size_hit(source_code(line)))
    );
    assert!(
        Path::new("search.py")
            .extension()
            .is_some_and(|ext| ext == "py")
    );
    assert!(
        !Path::new("search.pyc")
            .extension()
            .is_some_and(|ext| ext == "py")
    );
}

#[test]
fn search_source_has_no_unreviewed_size_or_motif_branches() {
    let root = Path::new(env!("CARGO_MANIFEST_DIR"));
    let mut failures = Vec::new();
    let mut used = vec![0usize; ALLOW.len()];
    for file in production_files(root) {
        let relative = file
            .strip_prefix(root)
            .expect("source below root")
            .to_string_lossy();
        let text = fs::read_to_string(&file).expect("read source");
        let python = file.extension().is_some_and(|extension| extension == "py");
        let lines = if python {
            python_lines(&text)
        } else {
            production_lines(&text)
        };
        for (index, line) in lines {
            let code = source_code(line).trim();
            let hit = size_hit(code)
                || named_size_constant_hit(code)
                || group_size_hit(code)
                || numeric_match_arm_hit(code)
                || forbidden_identifier_hit(code)
                || atomic_number_chart_hit(code)
                || fixed_geometry_threshold_hit(code)
                || (!python && motif_hit(code));
            if code.is_empty() || !hit {
                continue;
            }
            let mut allowed = false;
            for (slot, entry) in ALLOW.iter().enumerate() {
                if relative == entry.path
                    && line
                        .to_ascii_lowercase()
                        .contains(&entry.needle.to_ascii_lowercase())
                {
                    assert!(
                        !entry.reason.trim().is_empty(),
                        "allowlist reasons are mandatory"
                    );
                    used[slot] += 1;
                    allowed = true;
                }
            }
            if !allowed {
                failures.push(format!("{relative}:{index}: {}", line.trim()));
            }
        }
    }
    let stale: Vec<String> = ALLOW
        .iter()
        .zip(used)
        .filter(|(entry, used)| *used != expected_matches(entry))
        .map(|(entry, used)| {
            format!(
                "{}: {} matched {}, expected {} ({})",
                entry.path,
                entry.needle,
                used,
                expected_matches(entry),
                entry.reason
            )
        })
        .collect();
    assert!(
        stale.is_empty(),
        "stale no-motif allowlist entries:\n{}",
        stale.join("\n")
    );
    assert!(
        failures.is_empty(),
        "unreviewed size or named-motif search branches:\n{}",
        failures.join("\n")
    );
}
