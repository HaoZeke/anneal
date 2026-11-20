//! Stalled rigid-body archive walk.
//!
//! This cold typed path is reached only after a rigid search has exhausted
//! its normal cage walk.

use ndarray::{Array1, ArrayView1};
use rand::rngs::StdRng;
use rand::{Rng, SeedableRng};

use super::{ClusterMove, Config, Ledger};
use crate::allocate::FlooredThompson;

type ProposeStd = fn(&ClusterMove, ArrayView1<f64>, f64, f64, &mut StdRng) -> Array1<f64>;

/// Walk a stored cage after a rigid chain has stalled.
///
/// The hop calls this only when the rigid bridge is armed.
#[inline(never)]
#[cold]
pub(super) fn walk_armed_rigid_archive(
    cfg: &Config,
    ledger: &mut Ledger,
    relax: &mut (dyn FnMut(&mut Ledger, ArrayView1<f64>, usize) -> (f64, Array1<f64>) + Send),
    cage_archive: &mut crate::methods::stall_archive::StallArchive,
    kernels: &[ClusterMove],
    propose: ProposeStd,
    e: &mut f64,
    x: &mut Array1<f64>,
    hops: usize,
    cage_quantum: f64,
    orient_scale: f64,
    new_lowest: f64,
    archive_end: usize,
) {
    let key = super::rigid_cage_key(x.view(), cfg.n_points, cage_quantum);
    cage_archive.offer_cage(*e, x.view(), key);
    if let Some(best) = ledger.best_state.as_ref() {
        let best_key = super::rigid_cage_key(best.view(), cfg.n_points, cage_quantum);
        cage_archive.offer_cage(ledger.best, best.view(), best_key);
    }
    let (mut side_energy, mut side_state) = match ledger.best_state.as_ref() {
        Some(best) if best.len() == x.len() && ledger.best <= *e => (ledger.best, best.clone()),
        _ => (*e, x.clone()),
    };
    let mut side_rng = StdRng::seed_from_u64(
        (hops as u64)
            .wrapping_mul(0x9E37_79B9_7F4A_7C15)
            .wrapping_add(key)
            .wrapping_add(side_energy.to_bits()),
    );
    let mut bandit = FlooredThompson::new(kernels.len());
    let mut branch_best = side_energy;
    while ledger.spent() < archive_end && ledger.remaining() > 0 && !ledger.stopped() {
        let mut arm = bandit.select(&mut side_rng);
        let mut trial = if kernels[arm].reverses_directed_cycle() {
            let mut out = side_state.clone();
            if let Some(geometry) = cfg.rigid_body_geometry.as_ref() {
                super::cycles::reverse_chain(geometry, cfg.n_points, &mut out, &mut side_rng);
            }
            out
        } else {
            propose(
                &kernels[arm],
                side_state.view(),
                cfg.temperature,
                orient_scale,
                &mut side_rng,
            )
        };
        if super::hop_is_identity(side_state.view(), trial.view()) {
            for _ in 0..kernels.len() {
                arm = (arm + 1) % kernels.len();
                if !kernels[arm].reverses_directed_cycle() {
                    break;
                }
            }
            trial = propose(
                &kernels[arm],
                side_state.view(),
                cfg.temperature,
                orient_scale,
                &mut side_rng,
            );
        }
        super::recentre(&mut trial, cfg.n_points);
        super::contain_trial(cfg, &mut trial, cfg.n_points);
        let (quenched, relaxed) = relax(ledger, trial.view(), cfg.relax_steps);
        if !quenched.is_finite() || relaxed.len() != side_state.len() {
            break;
        }
        ledger.record(quenched, relaxed.view());
        let improved = quenched < branch_best - new_lowest;
        if improved {
            branch_best = quenched;
        }
        let rise = (quenched - side_energy) / cfg.temperature.max(1e-12);
        let accept = rise <= 0.0 || side_rng.random::<f64>() < (-rise).exp();
        bandit.update(arm, improved);
        if accept {
            side_energy = quenched;
            side_state = relaxed;
        }
    }
    if ledger.best < *e
        && let Some(best) = ledger.best_state.as_ref()
    {
        *e = ledger.best;
        *x = best.clone();
    }
}
