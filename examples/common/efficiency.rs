//! First-encounter and improvement traces. Bank vs no-bank is this
//! quantity, not whether both arms finished.

use std::time::Instant;

use anneal_core::methods::cluster_hopping::{Config, Outcome};
use anneal_core::methods::cluster_search::{Encounter, first_encounter};
use eindir_core::gradient::DifferentiableObjective;
use ndarray::ArrayView1;

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

/// Compacted-surface first phase from `TWO_PHASE_KAPPA` / `TWO_PHASE_MU`.
///
/// `TWO_PHASE_MU` is in energy per length squared of the preset scales
/// (`energy_scale / length_scale^2` converts it to the engine units).
pub fn apply_two_phase(cfg: &mut Config) {
    let envf = |key: &str| std::env::var(key).ok().and_then(|value| value.parse().ok());
    let kappa = envf("TWO_PHASE_KAPPA");
    let mu_raw = envf("TWO_PHASE_MU");
    if kappa.is_none() && mu_raw.is_none() {
        return;
    }
    let beta = envf("TWO_PHASE_BETA").unwrap_or(1.0);
    let mu = mu_raw.unwrap_or(0.0) * cfg.energy_scale / (cfg.length_scale * cfg.length_scale);
    let kappa = kappa.unwrap_or(0.0);
    cfg.two_phase = Some(anneal_core::methods::two_phase::TwoPhase {
        cutoff: anneal_core::methods::two_phase::Cutoff::Relative(kappa),
        beta: if kappa > 0.0 { beta } else { 0.0 },
        mu,
        anisotropic: false,
    });
    println!("  two-phase relaxation: {:?}", cfg.two_phase);
}

/// Wall time of one `value_and_gradient` call on `x`.
pub fn report_eval_wall<O>(objective: &O, x: ArrayView1<f64>, engine: &str)
where
    O: DifferentiableObjective<f64> + ?Sized,
{
    let repeats = std::env::var("EVAL_TIMING_REPEATS")
        .ok()
        .and_then(|value| value.parse().ok())
        .unwrap_or(5usize)
        .max(1);
    let start = Instant::now();
    for _ in 0..repeats {
        let _ = objective.value_and_gradient(x);
    }
    let mean = start.elapsed() / repeats as u32;
    println!(
        "  eval_wall engine={engine} repeats={repeats} mean_us={}",
        mean.as_micros()
    );
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
