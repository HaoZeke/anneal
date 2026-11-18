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
        ("src/methods/cluster_hopping/config.rs", "if n_points < 98") => 2,
        ("src/soap.rs", "fivefold") => 27,
        ("src/soap.rs", "pentagon") => 15,
        ("src/methods/cluster_hopping/moves.rs", "fivefold") => 7,
        ("src/methods/cluster_hopping.rs", "icosahedral_share") => 2,
        ("src/catalog/packing.rs", "fivefold") => 5,
        ("src/catalog/packing.rs", "icosahedral") => 3,
        ("src/catalog.rs", "fivefold") => 3,
        ("src/catalog.rs", "icosahedral") => 2,
        ("src/known_basin.rs", "fivefold") => 3,
        ("src/soap.rs", "n > 13" | "n <= 13") => 2,
        ("src/methods/dls_published.rs", "mackay") => 4,
        ("src/methods/activation.rs", "pentagon") => 5,
        ("src/methods/activation.rs", "fivefold") => 6,
        ("src/csm.rs", "d5h" | "cyclic_about_z(5)") => 2,
        ("src/lattice.rs", "FaceCentredCubic" | "Icosahedral") => 2,
        ("src/twin.rs", "if n < 8") => 2,
        ("src/structure.rs", "Template::Icosahedral" | "Template::FaceCentredCubic") => 4,
        ("src/structure.rs", "FaceCentredCubic," | "Icosahedral,") => 2,
        ("src/corekey.rs", "fcc") => 2,
        ("src/corekey.rs", "five_fold" | "MotifClass") => 7,
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
    // In-flight removal: the surface-mode size-gate removal.
    Allow {
        path: "src/methods/cluster_hopping/config.rs",
        needle: "if n_points < 98",
        reason: "known surface-size branch; the surface-mode size-gate removal replaces it",
    },
    // In-flight removals: the SOAP fivefold cleanup.
    Allow {
        path: "src/soap.rs",
        needle: "fivefold",
        reason: "known fivefold search machinery; the SOAP fivefold cleanup removes it",
    },
    Allow {
        path: "src/soap.rs",
        needle: "pentagon",
        reason: "known fivefold search machinery; the SOAP fivefold cleanup removes it",
    },
    Allow {
        path: "src/methods/cluster_hopping/moves.rs",
        needle: "fivefold",
        reason: "known fivefold move wiring; the SOAP fivefold cleanup removes it",
    },
    Allow {
        path: "src/methods/cluster_hopping.rs",
        needle: "fivefold",
        reason: "known fivefold channel wiring; the SOAP fivefold cleanup removes it",
    },
    Allow {
        path: "src/methods/cluster_hopping.rs",
        needle: "icosahedral_share",
        reason: "known template channel wiring; the SOAP fivefold cleanup removes it",
    },
    Allow {
        path: "src/catalog/packing.rs",
        needle: "fivefold",
        reason: "known fivefold catalog channel; the SOAP fivefold cleanup removes it",
    },
    Allow {
        path: "src/catalog/packing.rs",
        needle: "icosahedral",
        reason: "known ICOSAHEDRAL_TEMPLATE path; the SOAP fivefold cleanup removes it",
    },
    Allow {
        path: "src/catalog/leave_learn.rs",
        needle: "fivefold",
        reason: "known fivefold arm; the SOAP fivefold cleanup removes it",
    },
    Allow {
        path: "src/catalog.rs",
        needle: "fivefold",
        reason: "re-export of fivefold paths removed in the SOAP fivefold cleanup",
    },
    Allow {
        path: "src/catalog.rs",
        needle: "icosahedral",
        reason: "re-export of template paths removed in the SOAP fivefold cleanup",
    },
    Allow {
        path: "src/featomic_hop.rs",
        needle: "fivefold",
        reason: "known fivefold move wiring; the SOAP fivefold cleanup removes it",
    },
    Allow {
        path: "src/known_basin.rs",
        needle: "fivefold",
        reason: "known fivefold leave arm wiring; the SOAP fivefold cleanup removes it",
    },
    Allow {
        path: "src/soap.rs",
        needle: "Template::FaceCentredCubic",
        reason: "known SOAP template prototype; the SOAP fivefold cleanup removes it",
    },
    Allow {
        path: "src/soap.rs",
        needle: "Template::Icosahedral",
        reason: "known SOAP template prototype; the SOAP fivefold cleanup removes it",
    },
    Allow {
        path: "src/soap.rs",
        needle: "n < 5",
        reason: "known fivefold size gate; the SOAP fivefold cleanup removes it",
    },
    Allow {
        path: "src/soap.rs",
        needle: "n > 13",
        reason: "known fivefold size gate; the SOAP fivefold cleanup removes it",
    },
    Allow {
        path: "src/soap.rs",
        needle: "n <= 13",
        reason: "known fivefold size gate; the SOAP fivefold cleanup removes it",
    },
    Allow {
        path: "src/soap.rs",
        needle: "ACTIVE_VOLUME_MIN",
        reason: "known SOAP active-region size floor; the SOAP fivefold cleanup removes it",
    },
    // Existing behavioural findings recorded in docs/no-builders-audit.md.
    Allow {
        path: "src/methods/gmin_hop.rs",
        needle: "temperature: if n < 18",
        reason: "published TIP4P baseline reproduces GMIN's size table",
    },
    Allow {
        path: "src/bank_rpc/server.rs",
        needle: "kmax_factor: if n >= 55",
        reason: "existing size-tuned IRA admission setting; replacement is audited",
    },
    Allow {
        path: "src/methods/dls_published.rs",
        needle: "mackay",
        reason: "published DLS baseline reproduces its named primary-site template",
    },
    Allow {
        path: "src/methods/activation.rs",
        needle: "pentagon",
        reason: "existing pentagonal opening builder; replacement is audited",
    },
    Allow {
        path: "src/methods/activation.rs",
        needle: "fivefold",
        reason: "existing fivefold opening builder; replacement is audited",
    },
    Allow {
        path: "src/csm.rs",
        needle: "d5h",
        reason: "existing fixed fivefold group projector; replacement is audited",
    },
    Allow {
        path: "src/csm.rs",
        needle: "D_5h",
        reason: "existing fixed fivefold group label; replacement is audited",
    },
    Allow {
        path: "src/csm.rs",
        needle: "cyclic_about_z(5)",
        reason: "existing fixed C5 group projector; replacement is audited",
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
        path: "src/corekey.rs",
        needle: "fcc",
        reason: "existing motif-class feature feeding restart policy; replacement is audited",
    },
    Allow {
        path: "src/corekey.rs",
        needle: "five_fold",
        reason: "existing fivefold class feeding restart policy; replacement is audited",
    },
    Allow {
        path: "src/corekey.rs",
        needle: "MotifClass",
        reason: "existing named-motif restart class; replacement is audited",
    },
    Allow {
        path: "src/corekey.rs",
        needle: "motif_class",
        reason: "existing named-motif restart class; replacement is audited",
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
        path: "src/methods/cluster_hopping/moves.rs",
        needle: "fcc oracle",
        reason: "diagnostic assertion text, not shipped search behaviour",
    },
    Allow {
        path: "src/methods/dls_published.rs",
        needle: "icosahedron",
        reason: "published DLS baseline builds a named primary-site template",
    },
    Allow {
        path: "src/methods/cluster_hopping.rs",
        needle: "RIGID_HOT_BODIES",
        reason: "existing rigid-body count switch; replacement is audited",
    },
    Allow {
        path: "src/methods/cluster_hopping.rs",
        needle: "STALL_POINTS",
        reason: "cfg(test) fixture size, not shipped search behaviour",
    },
    Allow {
        path: "src/catalog/packing.rs",
        needle: "MINIMUM_PACKING_ATOMS",
        reason: "existing fixed descriptor-size floor; replacement is audited",
    },
    Allow {
        path: "src/catalog/molecular.rs",
        needle: "WATER_HEXAMER_MOLECULES",
        reason: "paper-system preset constant; removal from search code is audited",
    },
];

fn rust_files(root: &Path) -> Vec<PathBuf> {
    fn visit(path: &Path, out: &mut Vec<PathBuf>) {
        for entry in fs::read_dir(path).expect("read source directory") {
            let path = entry.expect("source entry").path();
            if path.is_dir() {
                visit(&path, out);
            } else if path.extension().is_some_and(|extension| extension == "rs") {
                out.push(path);
            }
        }
    }
    let mut out = Vec::new();
    visit(root, &mut out);
    out.sort();
    out
}

fn source_code(line: &str) -> &str {
    line.split_once("//").map_or(line, |(code, _)| code)
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
    ["n", "n_points", "n_atoms", "atom_count", "natoms"]
        .iter()
        .any(|variable| {
            code.match_indices(variable).any(|(start, _)| {
                let end = start + variable.len();
                let left_boundary = start == 0
                    || !code.as_bytes()[start - 1].is_ascii_alphanumeric()
                        && code.as_bytes()[start - 1] != b'_';
                let right_boundary = end == code.len()
                    || !code.as_bytes()[end].is_ascii_alphanumeric()
                        && code.as_bytes()[end] != b'_';
                left_boundary
                    && right_boundary
                    && ["<=", "==", "!=", ">=", "<", ">"].iter().any(|comparison| {
                        let rest = code[end..].trim_start();
                        let Some(rest) = rest.strip_prefix(comparison) else {
                            return false;
                        };
                        let digits = rest
                            .trim_start()
                            .chars()
                            .take_while(char::is_ascii_digit)
                            .collect::<String>();
                        digits.parse::<usize>().is_ok_and(|number| number >= 5)
                    })
            })
        })
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

#[test]
fn detectors_cover_size_motif_and_named_count_examples() {
    assert!(size_hit("if n_points == 38 { choose_special_move(); }"));
    assert!(motif_hit("if fivefold_score > cutoff { choose_arm(); }"));
    assert!(motif_hit("let group = d5h_ops();"));
    assert!(named_size_constant_hit(
        "const MINIMUM_PACKING_ATOMS: usize = 13;"
    ));
    assert!(!size_hit("if samples < limit { collect_more(); }"));
}

#[test]
fn search_source_has_no_unreviewed_size_or_motif_branches() {
    let root = Path::new(env!("CARGO_MANIFEST_DIR"));
    let mut failures = Vec::new();
    let mut used = vec![0usize; ALLOW.len()];
    for file in rust_files(&root.join("src")) {
        let relative = file
            .strip_prefix(root)
            .expect("source below root")
            .to_string_lossy();
        let text = fs::read_to_string(&file).expect("read source");
        let mut cfg_test = false;
        for (index, line) in text.lines().enumerate() {
            if line.trim() == "#[cfg(test)]" {
                cfg_test = true;
                continue;
            }
            if cfg_test && line.trim_start().starts_with("mod tests") {
                break;
            }
            cfg_test = false;
            let code = source_code(line).trim();
            if code.is_empty()
                || (!motif_hit(code) && !size_hit(code) && !named_size_constant_hit(code))
            {
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
                failures.push(format!("{relative}:{}: {}", index + 1, line.trim()));
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
