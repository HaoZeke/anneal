//! Per-replica policy/offer energy stream from a catalog journal.
//!
//!     catalog_leave_audit JOURNAL.bin

use anneal_core::catalog_rpc::{CatalogOperation, decode_request};
use std::io::Read;

fn main() {
    let path = std::env::args()
        .nth(1)
        .expect("usage: catalog_leave_audit JOURNAL.bin");
    let mut file = std::fs::File::open(&path).expect("journal must be readable");
    let mut n_policy = 0u64;
    let mut n_offer = 0u64;
    let mut n_sample = 0u64;
    let mut up_after_policy = 0u64;
    let mut down_after_policy = 0u64;
    let mut last_policy_e: Vec<Option<f64>> = vec![None; 64];
    loop {
        let mut length_bytes = [0u8; 8];
        match file.read(&mut length_bytes) {
            Ok(0) => break,
            Ok(8) => {}
            Ok(_) | Err(_) => panic!("{path}: truncated frame length"),
        }
        let length = usize::try_from(u64::from_le_bytes(length_bytes)).expect("frame fits");
        let mut bytes = vec![0u8; length];
        file.read_exact(&mut bytes)
            .expect("journal frame must be complete");
        let request = decode_request(&bytes).expect("journal frame must decode");
        let replica = request.identity.replica as usize;
        match &request.operation {
            CatalogOperation::PolicyState {
                energy,
                leftover_lambda,
                ..
            } => {
                n_policy += 1;
                if replica < last_policy_e.len() {
                    last_policy_e[replica] = Some(*energy);
                }
                if n_policy <= 8 || n_policy % 50 == 0 {
                    println!(
                        "policy replica={replica} energy={energy:.6} lambda={leftover_lambda:.4}"
                    );
                }
            }
            CatalogOperation::OfferCandidate { candidate }
            | CatalogOperation::RecordVisit { candidate } => {
                n_offer += 1;
                let e = candidate.energy;
                if replica < last_policy_e.len()
                    && let Some(prev) = last_policy_e[replica]
                {
                    let delta = e - prev;
                    if delta > 0.5 {
                        up_after_policy += 1;
                        if up_after_policy <= 12 {
                            println!(
                                "offer_up replica={replica} from={prev:.6} to={e:.6} d={delta:+.3}"
                            );
                        }
                    } else if delta < -0.5 {
                        down_after_policy += 1;
                    }
                    last_policy_e[replica] = None;
                }
            }
            CatalogOperation::Sample { .. } => n_sample += 1,
            _ => {}
        }
    }
    println!("n_policy {n_policy}");
    println!("n_offer {n_offer}");
    println!("n_sample {n_sample}");
    println!("offer_up_after_policy {up_after_policy}");
    println!("offer_down_after_policy {down_after_policy}");
}
