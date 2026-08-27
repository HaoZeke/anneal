//! Count catalog-journal operations and per-replica best offered energies.
//!
//!     catalog_journal_scan JOURNAL.bin

use anneal_core::catalog_rpc::{CatalogOperation, decode_request};
use std::collections::BTreeMap;
use std::io::Read;

fn main() {
    let path = std::env::args()
        .nth(1)
        .expect("usage: catalog_journal_scan JOURNAL.bin");
    let mut file = std::fs::File::open(&path).expect("journal must be readable");
    let mut ops: BTreeMap<&'static str, u64> = BTreeMap::new();
    let mut best: BTreeMap<u32, f64> = BTreeMap::new();
    let mut samples_incumbent = 0u64;
    let mut frames = 0u64;
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
        let name = match &request.operation {
            CatalogOperation::Snapshot => "snapshot",
            CatalogOperation::RecordVisit { candidate } => {
                record_best(&mut best, request.identity.replica, candidate.energy);
                "record_visit"
            }
            CatalogOperation::OfferCandidate { candidate } => {
                record_best(&mut best, request.identity.replica, candidate.energy);
                "offer"
            }
            CatalogOperation::Sample { draw } => {
                if *draw == u64::MAX {
                    samples_incumbent += 1;
                    "sample_incumbent"
                } else {
                    "sample"
                }
            }
            CatalogOperation::DescriptorHole { .. } => "descriptor_hole",
            CatalogOperation::BoundaryCrossing { .. } => "boundary_crossing",
            CatalogOperation::PolicyState { .. } => "policy_state",
            CatalogOperation::LedgerEvent { .. } => "ledger",
            CatalogOperation::LedgerBatch { .. } => "ledger_batch",
            CatalogOperation::PopulationSubmit { .. } => "population_submit",
            CatalogOperation::PopulationJoin { .. } => "population_join",
            CatalogOperation::PopulationAbstain { .. } => "population_abstain",
            CatalogOperation::PopulationPlan { .. } => "population_plan",
            _ => "other",
        };
        *ops.entry(name).or_insert(0) += 1;
    }
    println!("frames {frames}");
    for (name, count) in ops {
        println!("op {name} {count}");
    }
    println!("sample_incumbent {samples_incumbent}");
    for (replica, energy) in best {
        println!("best replica={replica} energy={energy:.6}");
    }
}

fn record_best(best: &mut BTreeMap<u32, f64>, replica: u32, energy: f64) {
    if best.get(&replica).is_none_or(|kept| energy < *kept) {
        best.insert(replica, energy);
    }
}
