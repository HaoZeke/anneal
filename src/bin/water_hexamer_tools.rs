// Copyright (c) anneal developers
//! Utilities for the water-hexamer benchmark campaign.
//!
//! Commands:
//!   - start --seed N --template FILE --out FILE
//!   - score --xyz FILE --isomer-dir DIR

use std::path::{Path, PathBuf};

use anneal_core::catalog::molecular::{
    MolecularCatalogPresetError, WATER_HEXAMER_MOLECULES, WATER_ISOMER_NAMES, assign_isomer,
    isomer_reference_gate, monomers_intact, reference_coordinates, water_groups,
};
use anneal_core::methods::cluster_hopping::repack_rigid_groups;
use ndarray::{Array1, ArrayView1};
use rand::SeedableRng;
use rand::rngs::StdRng;

fn main() {
    let mut args = std::env::args().skip(1).collect::<Vec<_>>();
    if args.is_empty() {
        usage("missing command");
    }
    let command = args.remove(0);
    match command.as_str() {
        "start" => command_start(&args),
        "score" => command_score(&args),
        other => usage(&format!("unknown command {other}")),
    }
}

fn usage(reason: &str) -> ! {
    eprintln!("{reason}");
    eprintln!("usage: water_hexamer_tools start --seed N --template FILE --out FILE");
    eprintln!("       water_hexamer_tools score --xyz FILE --isomer-dir DIR");
    std::process::exit(2);
}

fn command_start(args: &[String]) {
    let seed = parse_u64(args, "--seed").unwrap_or_else(|| usage("missing --seed"));
    let out = parse_path(args, "--out").unwrap_or_else(|| usage("missing --out"));
    let template_path =
        parse_path(args, "--template").unwrap_or_else(|| usage("missing --template"));
    let template_text = std::fs::read_to_string(&template_path)
        .unwrap_or_else(|error| panic!("read {}: {error}", template_path.display()));
    let parsed = parse_xyz(&template_text).unwrap_or_else(|| {
        panic!(
            "template {} is not a valid 18-atom xyz",
            template_path.display()
        )
    });
    let builtin = reference_coordinates(WATER_HEXAMER_MOLECULES).expect("template");
    let groups = water_groups(WATER_HEXAMER_MOLECULES).expect("groups");
    let (start, spread_steps) = prepared_water_start(Some(&parsed), &builtin, &groups, 1.0, seed);
    write_xyz(&out, &start, &format!("seed {seed} spread_steps {spread_steps}"))
        .unwrap_or_else(|error| panic!("write {}: {error}", out.display()));
    println!("seed: {seed}");
    println!("spread_steps: {spread_steps}");
    println!("out: {}", out.display());
}

fn command_score(args: &[String]) {
    let xyz = parse_path(args, "--xyz").unwrap_or_else(|| usage("missing --xyz"));
    let isomer_dir =
        parse_path(args, "--isomer-dir").unwrap_or_else(|| usage("missing --isomer-dir"));
    let xyz_text =
        std::fs::read_to_string(&xyz).unwrap_or_else(|error| panic!("read {}: {error}", xyz.display()));
    let geometry = parse_xyz(&xyz_text).unwrap_or_else(|| panic!("invalid xyz {}", xyz.display()));
    let intact = monomers_intact(&geometry, WATER_HEXAMER_MOLECULES).unwrap_or(false);
    let references = load_references(&isomer_dir);
    let reference_views = references
        .iter()
        .map(|(name, coordinates)| (name.as_str(), coordinates.as_slice()))
        .collect::<Vec<_>>();
    let gate = isomer_reference_gate(&reference_views, WATER_HEXAMER_MOLECULES).unwrap_or(None);

    if !intact {
        println!("hit: 0");
        println!("isomer: unassigned");
        println!("rmsd_A: none");
        println!(
            "gate_A: {}",
            gate.map(|value| format!("{value:.6}"))
                .unwrap_or_else(|| "unset".to_owned())
        );
        println!("reason: dissociated_monomer");
        return;
    }
    if reference_views.len() < 2 {
        println!("hit: 0");
        println!("isomer: unassigned");
        println!("rmsd_A: none");
        println!("gate_A: unset");
        println!("reason: fewer_than_two_references");
        return;
    }
    match assign_isomer(&geometry, WATER_HEXAMER_MOLECULES, &reference_views, gate) {
        Ok(matched) => {
            let name = matched.name.unwrap_or_else(|| "unassigned".to_owned());
            let hit = usize::from(name != "unassigned");
            println!("hit: {hit}");
            println!("isomer: {name}");
            println!(
                "rmsd_A: {}",
                matched
                    .rmsd
                    .map(|value| format!("{value:.6}"))
                    .unwrap_or_else(|| "none".to_owned())
            );
            println!(
                "gate_A: {}",
                gate.map(|value| format!("{value:.6}"))
                    .unwrap_or_else(|| "unset".to_owned())
            );
            println!(
                "reason: {}",
                if hit == 1 {
                    "inside_gate"
                } else if gate.is_none() {
                    "gate_unset"
                } else {
                    "outside_gate"
                }
            );
        }
        Err(MolecularCatalogPresetError::AssignmentNeedsIra) => {
            println!("hit: 0");
            println!("isomer: unassigned");
            println!("rmsd_A: none");
            println!("gate_A: unset");
            println!("reason: ira_missing");
        }
        Err(_error) => {
            println!("hit: 0");
            println!("isomer: unassigned");
            println!("rmsd_A: none");
            println!(
                "gate_A: {}",
                gate.map(|value| format!("{value:.6}"))
                    .unwrap_or_else(|| "unset".to_owned())
            );
            println!("reason: assignment_failed");
        }
    }
}

fn parse_u64(args: &[String], flag: &str) -> Option<u64> {
    parse_flag(args, flag).map(|value| {
        value
            .parse::<u64>()
            .unwrap_or_else(|error| panic!("{flag} {value:?}: {error}"))
    })
}

fn parse_path(args: &[String], flag: &str) -> Option<PathBuf> {
    parse_flag(args, flag).map(PathBuf::from)
}

fn parse_flag<'a>(args: &'a [String], flag: &str) -> Option<&'a str> {
    args.windows(2).find_map(|window| {
        (window[0] == flag)
            .then_some(window[1].as_str())
            .filter(|value| !value.starts_with("--"))
    })
}

fn write_xyz(path: &Path, coordinates: &Array1<f64>, comment: &str) -> std::io::Result<()> {
    if coordinates.len() != 9 * WATER_HEXAMER_MOLECULES {
        panic!("coordinate dimension mismatch");
    }
    if let Some(parent) = path.parent() {
        std::fs::create_dir_all(parent)?;
    }
    let mut text = String::from("18\n");
    text.push_str(comment);
    text.push('\n');
    for molecule in 0..WATER_HEXAMER_MOLECULES {
        let base = 9 * molecule;
        for (offset, symbol) in [(0, "O"), (3, "H"), (6, "H")] {
            text.push_str(&format!(
                "{symbol} {:.10} {:.10} {:.10}\n",
                coordinates[base + offset],
                coordinates[base + offset + 1],
                coordinates[base + offset + 2]
            ));
        }
    }
    std::fs::write(path, text)
}

fn load_references(dir: &Path) -> Vec<(String, Vec<f64>)> {
    let mut found = Vec::new();
    for name in WATER_ISOMER_NAMES {
        let path = dir.join(format!("{name}.xyz"));
        let Ok(text) = std::fs::read_to_string(&path) else {
            continue;
        };
        if let Some(coordinates) = parse_xyz(&text)
            && monomers_intact(&coordinates, WATER_HEXAMER_MOLECULES).unwrap_or(false)
        {
            found.push((name.to_owned(), coordinates));
        }
    }
    found
}

fn parse_xyz(text: &str) -> Option<Vec<f64>> {
    let mut lines = text.lines();
    let count: usize = lines.next()?.trim().parse().ok()?;
    if count != 3 * WATER_HEXAMER_MOLECULES {
        return None;
    }
    let _comment = lines.next()?;
    let mut coordinates = Vec::with_capacity(3 * count);
    for line in lines.take(count) {
        let mut fields = line.split_whitespace();
        let element = fields.next()?;
        if element != "O" && element != "H" {
            return None;
        }
        for _ in 0..3 {
            coordinates.push(fields.next()?.parse().ok()?);
        }
    }
    (coordinates.len() == 9 * WATER_HEXAMER_MOLECULES).then_some(coordinates)
}

fn monomer_centred(coordinates: &[f64], molecule: usize) -> [[f64; 3]; 3] {
    let base = 9 * molecule;
    let mut atoms = [[0.0; 3]; 3];
    let mut com = [0.0; 3];
    for atom in 0..3 {
        for axis in 0..3 {
            let value = coordinates[base + 3 * atom + axis];
            atoms[atom][axis] = value;
            com[axis] += value;
        }
    }
    for value in &mut com {
        *value /= 3.0;
    }
    for atom in &mut atoms {
        for axis in 0..3 {
            atom[axis] -= com[axis];
        }
    }
    atoms
}

fn water_template_matches(parsed: &[f64], builtin: &[f64]) -> bool {
    if parsed.len() != builtin.len() || builtin.len() != 9 * WATER_HEXAMER_MOLECULES {
        return false;
    }
    let reference = monomer_centred(builtin, 0);
    for molecule in 0..WATER_HEXAMER_MOLECULES {
        let found = monomer_centred(parsed, molecule);
        for atom in 0..3 {
            for axis in 0..3 {
                if (found[atom][axis] - reference[atom][axis]).abs() > 1e-6 {
                    return false;
                }
            }
        }
    }
    true
}

fn prepared_water_start(
    parsed: Option<&[f64]>,
    builtin: &[f64],
    groups: &[Vec<usize>],
    length_scale: f64,
    seed: u64,
) -> (Array1<f64>, usize) {
    let template = match parsed {
        Some(coordinates) if water_template_matches(coordinates, builtin) => builtin,
        Some(coordinates) => coordinates,
        None => builtin,
    };
    let mut rng = StdRng::seed_from_u64(seed);
    let packed = repack_rigid_groups(ArrayView1::from(template), groups, length_scale, &mut rng);
    separate_overlapped_start(packed, WATER_HEXAMER_MOLECULES)
}

fn separate_overlapped_start(mut start: Array1<f64>, n_molecules: usize) -> (Array1<f64>, usize) {
    let Ok(groups) = water_groups(n_molecules) else {
        return (start, 0);
    };
    let mut steps = 0usize;
    for _ in 0..64 {
        let intact = start
            .as_slice()
            .and_then(|coordinates| monomers_intact(coordinates, n_molecules).ok())
            .unwrap_or(false);
        if intact {
            return (start, steps);
        }
        steps += 1;
        let mut coms = vec![[0.0; 3]; groups.len()];
        for (group_index, atoms) in groups.iter().enumerate() {
            if atoms.is_empty() {
                continue;
            }
            for &atom in atoms {
                for axis in 0..3 {
                    coms[group_index][axis] += start[3 * atom + axis];
                }
            }
            let count = atoms.len() as f64;
            for value in &mut coms[group_index] {
                *value /= count;
            }
        }
        let mut global = [0.0; 3];
        for com in &coms {
            for axis in 0..3 {
                global[axis] += com[axis];
            }
        }
        let count = groups.len().max(1) as f64;
        for value in &mut global {
            *value /= count;
        }
        for (group_index, atoms) in groups.iter().enumerate() {
            let mut direction = [0.0; 3];
            for axis in 0..3 {
                direction[axis] = coms[group_index][axis] - global[axis];
            }
            let mut norm = direction.iter().map(|value| value * value).sum::<f64>();
            if norm < 1e-8 {
                direction = [
                    (group_index % 3) as f64 - 1.0,
                    (group_index / 3) as f64 - 1.0,
                    0.4,
                ];
                norm = direction.iter().map(|value| value * value).sum::<f64>();
            }
            let scale = 0.25 / norm.sqrt();
            for &atom in atoms {
                for axis in 0..3 {
                    start[3 * atom + axis] += direction[axis] * scale;
                }
            }
        }
    }
    (start, steps)
}
