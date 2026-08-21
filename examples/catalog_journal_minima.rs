//! Dump unique quenched Cartesian minima from a catalog journal.
//!
//!     catalog_journal_minima JOURNAL.bin > minima.min
//!
//! Each output row is energy followed by coordinates. Duplicates are
//! dropped by milli-epsilon energy plus a coarse coordinate hash.
//! Unconverged candidates are skipped.

use anneal_core::catalog_rpc::{CatalogOperation, decode_request};
use std::collections::BTreeMap;
use std::io::{self, Read, Write};

fn main() {
    let path = std::env::args()
        .nth(1)
        .expect("usage: catalog_journal_minima JOURNAL.bin");
    let mut file = std::fs::File::open(&path).expect("journal must be readable");
    let mut seen: BTreeMap<(i64, Vec<i32>), (f64, Vec<f64>)> = BTreeMap::new();
    let mut frames = 0u64;
    let mut candidates = 0u64;
    loop {
        let mut length_bytes = [0u8; 8];
        match file.read(&mut length_bytes) {
            Ok(0) => break,
            Ok(8) => {}
            Ok(_) | Err(_) => panic!("{path}: truncated frame length"),
        }
        let length = usize::try_from(u64::from_le_bytes(length_bytes)).expect("frame fits usize");
        let mut bytes = vec![0u8; length];
        file.read_exact(&mut bytes)
            .expect("journal frame must be complete");
        let request = decode_request(&bytes).expect("journal frame must decode");
        frames += 1;
        let candidate = match &request.operation {
            CatalogOperation::RecordVisit { candidate }
            | CatalogOperation::OfferCandidate { candidate }
            | CatalogOperation::PopulationSubmit { candidate, .. } => candidate,
            _ => continue,
        };
        if !candidate.quench_converged {
            continue;
        }
        if !candidate.energy.is_finite() || candidate.coordinates.is_empty() {
            continue;
        }
        if !candidate.coordinates.len().is_multiple_of(3) {
            continue;
        }
        candidates += 1;
        let energy_key = (candidate.energy * 1000.0).round() as i64;
        let hash: Vec<i32> = candidate
            .coordinates
            .iter()
            .step_by(6)
            .map(|c| (c * 100.0).round() as i32)
            .collect();
        let key = (energy_key, hash);
        match seen.get(&key) {
            Some((held, _)) if candidate.energy >= *held => {}
            _ => {
                seen.insert(key, (candidate.energy, candidate.coordinates.clone()));
            }
        }
    }
    let mut rows: Vec<(f64, Vec<f64>)> = seen.into_values().collect();
    rows.sort_by(|a, b| a.0.total_cmp(&b.0));
    let mut out = io::BufWriter::new(io::stdout().lock());
    for (energy, coords) in &rows {
        write!(out, "{energy:.10e}").unwrap();
        for c in coords {
            write!(out, " {c:.8e}").unwrap();
        }
        writeln!(out).unwrap();
    }
    eprintln!(
        "frames {frames} candidates {candidates} unique {}",
        rows.len()
    );
    if let (Some((lo, _)), Some((hi, _))) = (rows.first(), rows.last()) {
        eprintln!("E min {lo:.6} max {hi:.6}");
    }
}
