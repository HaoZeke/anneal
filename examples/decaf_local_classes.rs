//! Per-center environment classes across structures, DECAF style.
//!
//! The global descriptor of a 75-atom cluster barely moves between an
//! icosahedral minimum at -392.7 and one at -396.0, which is exactly
//! the regime where per-atom environments still discriminate: a Marks
//! core differs from an icosahedral core atom by atom even when the
//! cloud mean does not. This tool pools the per-center descriptor rows
//! of every input structure, clusters them by leader clustering at a
//! chosen radius, and reports each structure's class histogram, the
//! pairwise histogram distances, and the classes that exist in one
//! structure and nowhere else. Inputs are xyz files or coordinator
//! journals (`catalog-requests-v5.bin`), from which the deepest
//! validated candidate per replica is taken.
//!
//!     decaf_local_classes [--radius R] INPUT [INPUT ...]

use anneal_core::catalog_rpc::{CatalogOperation, decode_request};
use anneal_core::soap::{SoapSpec, local_nu3_z};
use ndarray::Array1;
use std::collections::BTreeMap;
use std::io::Read;

struct Structure {
    label: String,
    energy: Option<f64>,
    coordinates: Array1<f64>,
}

fn read_xyz(path: &str) -> Vec<Structure> {
    let text = std::fs::read_to_string(path).expect("xyz must be readable");
    let values: Vec<f64> = text
        .lines()
        .skip(2)
        .flat_map(|line| {
            line.split_whitespace()
                .skip(1)
                .filter_map(|token| token.parse::<f64>().ok())
        })
        .collect();
    assert!(!values.is_empty(), "{path}: xyz body held no coordinates");
    vec![Structure {
        label: path.rsplit('/').next().unwrap_or(path).to_owned(),
        energy: None,
        coordinates: Array1::from(values),
    }]
}

fn read_journal(path: &str) -> Vec<Structure> {
    let mut file = std::fs::File::open(path).expect("journal must be readable");
    let mut best: BTreeMap<u32, (f64, Vec<f64>)> = BTreeMap::new();
    loop {
        let mut length_bytes = [0u8; 8];
        match file.read(&mut length_bytes) {
            Ok(0) => break,
            Ok(8) => {}
            Ok(_) | Err(_) => panic!("{path}: truncated frame length"),
        }
        let length =
            usize::try_from(u64::from_le_bytes(length_bytes)).expect("frame length fits usize");
        let mut bytes = vec![0u8; length];
        file.read_exact(&mut bytes)
            .expect("journal frame must be complete");
        let request = decode_request(&bytes).expect("journal frame must decode");
        let candidate = match &request.operation {
            CatalogOperation::RecordVisit { candidate }
            | CatalogOperation::OfferCandidate { candidate }
            | CatalogOperation::PopulationSubmit { candidate, .. } => candidate,
            _ => continue,
        };
        let replica = request.identity.replica;
        if best
            .get(&replica)
            .is_none_or(|(energy, _)| candidate.energy < *energy)
        {
            best.insert(replica, (candidate.energy, candidate.coordinates.clone()));
        }
    }
    let stem = path.rsplit('/').nth(2).unwrap_or(path).to_owned();
    best.into_iter()
        .map(|(replica, (energy, coordinates))| Structure {
            label: format!("{stem}/r{replica}"),
            energy: Some(energy),
            coordinates: Array1::from(coordinates),
        })
        .collect()
}

fn main() {
    let mut arguments: Vec<String> = std::env::args().skip(1).collect();
    let mut radius = None;
    if let Some(index) = arguments.iter().position(|a| a == "--radius") {
        radius = arguments
            .get(index + 1)
            .and_then(|value| value.parse::<f64>().ok());
        arguments.drain(index..index + 2);
    }
    if arguments.is_empty() {
        eprintln!("usage: decaf_local_classes [--radius R] INPUT [INPUT ...]");
        std::process::exit(2);
    }
    let structures: Vec<Structure> = arguments
        .iter()
        .flat_map(|input| {
            if input.ends_with(".bin") {
                read_journal(input)
            } else {
                read_xyz(input)
            }
        })
        .collect();

    // Per-center rows for every structure under one shared spec.
    let spec = SoapSpec::default();
    let rows: Vec<Vec<Array1<f64>>> = structures
        .iter()
        .map(|structure| {
            let local = local_nu3_z(structure.coordinates.view(), spec, None);
            (0..local.nrows())
                .map(|i| local.row(i).to_owned())
                .collect()
        })
        .collect();

    // A data-driven default radius: half the median distance between a
    // subsample of pooled environments, the scale below which two
    // environments read as the same local motif.
    let pooled: Vec<&Array1<f64>> = rows.iter().flatten().collect();
    let radius = radius.unwrap_or_else(|| {
        let step = (pooled.len() / 200).max(1);
        let sample: Vec<&&Array1<f64>> = pooled.iter().step_by(step).collect();
        let mut distances = Vec::new();
        for (i, a) in sample.iter().enumerate() {
            for b in sample.iter().skip(i + 1) {
                distances.push(
                    a.iter()
                        .zip(b.iter())
                        .map(|(p, q)| (p - q) * (p - q))
                        .sum::<f64>()
                        .sqrt(),
                );
            }
        }
        distances.sort_by(f64::total_cmp);
        0.5 * distances[distances.len() / 2]
    });

    // Leader clustering over the pooled environments.
    let mut leaders: Vec<Array1<f64>> = Vec::new();
    let mut histograms: Vec<BTreeMap<usize, usize>> = vec![BTreeMap::new(); structures.len()];
    for (structure_index, structure_rows) in rows.iter().enumerate() {
        for row in structure_rows {
            let mut assigned = None;
            for (class, leader) in leaders.iter().enumerate() {
                let distance = row
                    .iter()
                    .zip(leader.iter())
                    .map(|(p, q)| (p - q) * (p - q))
                    .sum::<f64>()
                    .sqrt();
                if distance <= radius {
                    assigned = Some(class);
                    break;
                }
            }
            let class = assigned.unwrap_or_else(|| {
                leaders.push(row.clone());
                leaders.len() - 1
            });
            *histograms[structure_index].entry(class).or_default() += 1;
        }
    }

    println!(
        "environments {} classes {} radius {radius:.6}",
        pooled.len(),
        leaders.len()
    );
    for (structure, histogram) in structures.iter().zip(&histograms) {
        let energy = structure
            .energy
            .map_or(String::new(), |e| format!(" e={e:.6}"));
        let mut classes: Vec<(usize, usize)> = histogram
            .iter()
            .map(|(class, count)| (*class, *count))
            .collect();
        classes.sort_by_key(|(_, count)| std::cmp::Reverse(*count));
        let rendered: Vec<String> = classes
            .iter()
            .map(|(class, count)| format!("{class}:{count}"))
            .collect();
        println!(
            "{}{energy} classes={} [{}]",
            structure.label,
            histogram.len(),
            rendered.join(" ")
        );
    }
    // Pairwise normalized-histogram L1 distances.
    println!("pairwise L1 (normalized histograms):");
    for i in 0..structures.len() {
        for j in (i + 1)..structures.len() {
            let total_i = histograms[i].values().sum::<usize>() as f64;
            let total_j = histograms[j].values().sum::<usize>() as f64;
            let classes: std::collections::BTreeSet<usize> = histograms[i]
                .keys()
                .chain(histograms[j].keys())
                .copied()
                .collect();
            let l1: f64 = classes
                .iter()
                .map(|class| {
                    let p = *histograms[i].get(class).unwrap_or(&0) as f64 / total_i;
                    let q = *histograms[j].get(class).unwrap_or(&0) as f64 / total_j;
                    (p - q).abs()
                })
                .sum();
            println!(
                "  {} vs {}: {l1:.4}",
                structures[i].label, structures[j].label
            );
        }
    }
    // Classes unique to a single structure: the discriminating motifs.
    for (index, structure) in structures.iter().enumerate() {
        let unique: Vec<usize> = histograms[index]
            .keys()
            .filter(|class| {
                histograms
                    .iter()
                    .enumerate()
                    .all(|(other, histogram)| other == index || !histogram.contains_key(*class))
            })
            .copied()
            .collect();
        if !unique.is_empty() {
            println!(
                "unique to {}: {} classes {:?}",
                structure.label,
                unique.len(),
                unique
            );
        }
    }
}
