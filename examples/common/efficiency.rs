//! First-encounter and improvement traces. Bank vs no-bank is this
//! quantity, not whether both arms finished.

use anneal_core::methods::cluster_hopping::Outcome;
use anneal_core::methods::cluster_search::{Encounter, first_encounter};

/// Print every recorded improvement and, when `TARGET_ENERGY` is set,
/// the first encounter against that target.
pub fn report_trace(out: &Outcome, spent: usize) {
    for &(hops, charged, basins, e) in &out.improvements {
        println!("    improve hops={hops} charged={charged} basins={basins} e={e:.8}");
    }
    let Some(target) = std::env::var("TARGET_ENERGY")
        .ok()
        .and_then(|v| v.parse::<f64>().ok())
    else {
        return;
    };
    let tol = std::env::var("TARGET_TOL")
        .ok()
        .and_then(|v| v.parse().ok())
        .unwrap_or(1e-3);
    match first_encounter(out, target, tol, spent) {
        Encounter::Found { charged, hops } => {
            println!("    encounter target={target:.8} found charged={charged} hops={hops}");
        }
        Encounter::Censored { charged } => {
            println!("    encounter target={target:.8} censored charged={charged}");
        }
    }
}

/// Whether this process is talking to a Cap'n bank.
pub fn bank_label() -> &'static str {
    match std::env::var("BANK_SHARING").as_deref() {
        Ok("shared") => return "shared",
        Ok("control") => return "control",
        _ => {}
    }
    match std::env::var("BANK_RPC") {
        Ok(s) if !s.is_empty() => "bank",
        _ => "nobank",
    }
}
