//! Stalled rigid-body archive walk.
//!
//! This cold typed path is reached only after a rigid search has gone a
//! full archive patience without a new lowest minimum on the main chain.

use ndarray::{Array1, ArrayView1};
use rand::rngs::StdRng;
use rand::{Rng, SeedableRng};

use super::{ClusterMove, Config, Ledger};
use crate::allocate::FlooredThompson;

type ProposeStd = fn(&ClusterMove, ArrayView1<f64>, f64, f64, &mut StdRng) -> Array1<f64>;

/// Walk a stored cage after a rigid chain has stalled.
///
/// The main walker and its generator stay untouched. A quench is recorded
/// only when it is finite, sane, stationary and clear of a centre clash.
/// The walk stops once that record reaches the ledger target, before
/// another evaluation.
#[inline(never)]
#[cold]
#[allow(clippy::too_many_arguments)]
pub(super) fn walk_armed_rigid_archive(
    cfg: &Config,
    ledger: &mut Ledger,
    relax: &mut (dyn FnMut(&mut Ledger, ArrayView1<f64>, usize) -> (f64, Array1<f64>) + Send),
    grad: &mut Option<&mut super::GradFn<'_>>,
    cage_archive: &mut crate::methods::stall_archive::StallArchive,
    kernels: &[ClusterMove],
    propose: ProposeStd,
    e: f64,
    x: ArrayView1<f64>,
    hops: usize,
    cage_quantum: f64,
    orient_scale: f64,
    new_lowest: f64,
    archive_end: usize,
) {
    let key = super::rigid_cage_key(x, cfg.n_points, cage_quantum);
    cage_archive.offer_cage(e, x, key);
    if let Some(best) = ledger.best_state.as_ref() {
        let best_key = super::rigid_cage_key(best.view(), cfg.n_points, cage_quantum);
        cage_archive.offer_cage(ledger.best, best.view(), best_key);
    }
    let (mut side_energy, mut side_state) = match cage_archive.escape_other_cage(key) {
        Some((energy, state)) if state.len() == x.len() => (energy, state),
        _ => match ledger.best_state.as_ref() {
            Some(best) if best.len() == x.len() && ledger.best <= e => (ledger.best, best.clone()),
            _ => (e, x.to_owned()),
        },
    };
    let mut side_rng = StdRng::seed_from_u64(
        (hops as u64)
            .wrapping_mul(0x9E37_79B9_7F4A_7C15)
            .wrapping_add(key)
            .wrapping_add(side_energy.to_bits()),
    );
    // The chain reversal reads the declared bonds, so a library that does
    // not offer the cycle arm on its main chain still gets one here.
    let geometry = cfg
        .rigid_body_geometry
        .as_ref()
        .filter(|geometry| super::cycles::has_contact_cycles(geometry));
    let chain_arm = geometry.is_some() && !kernels.iter().any(ClusterMove::reverses_directed_cycle);
    let arms = kernels.len() + usize::from(chain_arm);
    let reverses = |arm: usize| arm == kernels.len() || kernels[arm].reverses_directed_cycle();
    let mut bandit = FlooredThompson::new(arms);
    let mut branch_best = side_energy;
    let mut side_hops = 0usize;
    while ledger.spent() < archive_end
        && ledger.remaining() > 0
        && !ledger.stopped()
        && !ledger.target_reached()
    {
        if side_hops > 0 && side_hops.is_multiple_of(crate::methods::rigid_quench::STAGNATION_HOPS)
        {
            let next_key = super::rigid_cage_key(side_state.view(), cfg.n_points, cage_quantum);
            if let Some((energy, state)) = cage_archive.escape_other_cage(next_key)
                && state.len() == side_state.len()
            {
                side_energy = energy;
                side_state = state;
                branch_best = side_energy;
            }
        }
        side_hops += 1;
        let mut arm = bandit.select(&mut side_rng);
        let mut trial = if reverses(arm) {
            let mut out = side_state.clone();
            // A false return leaves `out` as it was. That hop is an
            // identity, so a kernel that moves the centres takes it.
            let turned = geometry.is_some_and(|geometry| {
                super::cycles::reverse_chain(geometry, cfg.n_points, &mut out, &mut side_rng)
            });
            if turned { out } else { side_state.clone() }
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
            for _ in 0..arms {
                arm = (arm + 1) % arms;
                if !reverses(arm) {
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
        super::record_auxiliary_rigid(cfg, ledger, grad, quenched, relaxed.view());
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
}
