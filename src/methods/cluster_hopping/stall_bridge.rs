//! Stalled rigid-body archive walk.
//!
//! The walk lives outside the hop loop. Keeping its body in that loop
//! changes how the monomorphised Tric step is partitioned and optimised,
//! including on a molecular search that never arms the bridge.

use ndarray::{Array1, ArrayView1};
use rand::Rng;
use rand::SeedableRng;
use rand::rngs::StdRng;

use super::{ClusterMove, Config, Ledger, MoveLibrary, RigidStallBook};
use crate::allocate::FlooredThompson;

/// Walk a stored cage after a rigid chain has stalled.
///
/// Not inlined. The hop loop calls this only when the rigid bridge is
/// armed, so a molecular or Tric step does not carry the walk.
#[inline(never)]
#[cold]
pub(super) fn walk_armed_rigid_archive(
    cfg: &Config,
    ledger: &mut Ledger,
    relax: &mut (dyn FnMut(&mut Ledger, ArrayView1<f64>, usize) -> (f64, Array1<f64>) + Send),
    book: &mut RigidStallBook,
    cage_archive: &mut crate::methods::stall_archive::StallArchive,
    kernels: &[ClusterMove],
    e: &mut f64,
    x: &mut Array1<f64>,
    hops: usize,
    cage_quantum: f64,
    orient_scale: f64,
    new_lowest: f64,
) {
    super::count_bridge_entry();
    if let Some(state) = book.side_lowest_state.as_ref() {
        ledger.record(book.side_lowest, state.view());
    }
    let key = super::rigid_cage_key(x.view(), cfg.n_points, cage_quantum);
    cage_archive.offer_cage(*e, x.view(), key);
    if let Some(best) = ledger.best_state.as_ref() {
        cage_archive.offer_cage(
            ledger.best,
            best.view(),
            super::rigid_cage_key(best.view(), cfg.n_points, cage_quantum),
        );
    }
    let (mut side_energy, mut side_state) = match ledger.best_state.as_ref() {
        Some(best) if best.len() == x.len() && ledger.best <= *e => (ledger.best, best.clone()),
        _ => (*e, x.clone()),
    };
    // A fresh generator, not a draw on the walker's.
    let mut side_rng = StdRng::seed_from_u64(
        (hops as u64)
            .wrapping_mul(0x9E37_79B9_7F4A_7C15)
            .wrapping_add(key)
            .wrapping_add(side_energy.to_bits()),
    );
    let side_scale = orient_scale;
    let mut side_bandit = FlooredThompson::new(kernels.len());
    let mut branch_best = side_energy;
    let mut branch_stale = 0usize;
    let mut branch_idx = 0usize;
    let rotate_arm = kernels.iter().position(|m| m.name() == "rrot");
    while ledger.remaining() > 0 && !ledger.stopped() {
        let stale_cap = if (2..7).contains(&branch_idx) {
            500
        } else {
            350
        };
        if branch_stale >= stale_cap {
            branch_idx += 1;
            branch_stale = 0;
            side_bandit = FlooredThompson::new(kernels.len());
            if branch_idx >= 7 && !branch_idx.is_multiple_of(2) {
                side_energy = book.initial_e;
                side_state = book.initial_x.clone();
                branch_best = side_energy;
                continue;
            }
            let angle = match cfg.move_library {
                MoveLibrary::RigidBody { rotate_step, .. } => {
                    super::RIGID_SHELL_STEPS * rotate_step
                }
                _ => 0.0,
            };
            if branch_idx < 7 && branch_idx % 3 != 1 && angle > 0.0 {
                let base_energy = ledger.best.min(side_energy);
                let mut base_state = ledger.best_state.as_ref().unwrap_or(&side_state).clone();
                if branch_idx >= 4
                    && let Some(rot_idx) = rotate_arm
                {
                    base_state = kernels[rot_idx].propose_scaled(
                        base_state.view(),
                        cfg.temperature,
                        8.0,
                        &mut side_rng,
                    );
                    super::recentre(&mut base_state, cfg.n_points);
                }
                let sign = if (branch_idx / 3).is_multiple_of(2) {
                    1.0
                } else {
                    -1.0
                };
                let mut turned: Option<(f64, Array1<f64>)> = None;
                for axis in 0..3 {
                    if ledger.remaining() == 0 || ledger.stopped() {
                        break;
                    }
                    let trial = super::rigid_shell_turn(
                        base_state.view(),
                        cfg.n_points,
                        axis,
                        sign * angle,
                    );
                    let (quenched, relaxed) = relax(ledger, trial.view(), cfg.relax_steps);
                    if quenched.is_finite()
                        && relaxed.len() == side_state.len()
                        && super::rigid_centres_clear(
                            relaxed.view(),
                            cfg.n_points,
                            0.5 * cfg.min_separation,
                        )
                    {
                        ledger.record(quenched, relaxed.view());
                        if (quenched - base_energy).abs() > 0.15
                            && turned.as_ref().is_none_or(|(held, _)| quenched < *held)
                        {
                            turned = Some((quenched, relaxed));
                        }
                    }
                }
                if let Some((quenched, relaxed)) = turned {
                    side_energy = quenched;
                    side_state = relaxed;
                    branch_best = quenched;
                    continue;
                }
            }
            let trial =
                super::random_rigid_cluster(cfg.n_points, cfg.min_separation, &mut side_rng);
            let (quenched, relaxed) =
                relax(ledger, trial.view(), cfg.relax_steps.saturating_mul(2));
            if !quenched.is_finite() || relaxed.len() != side_state.len() {
                if ledger.remaining() == 0 || ledger.stopped() {
                    break;
                }
                branch_stale = stale_cap;
                continue;
            }
            ledger.record(quenched, relaxed.view());
            side_energy = quenched;
            side_state = relaxed;
            branch_best = quenched;
            continue;
        }
        branch_stale += 1;
        let mut arm = side_bandit.select(&mut side_rng);
        let mut trial =
            if kernels[arm].name() == "rcycle" && (branch_idx >= 7 || side_rng.random_bool(0.5)) {
                let mut out = side_state.clone();
                if let Some(geom) = cfg.rigid_body_geometry.as_ref() {
                    super::cycles::reverse_chain(geom, cfg.n_points, &mut out, &mut side_rng);
                }
                out
            } else {
                kernels[arm].propose_scaled(
                    side_state.view(),
                    cfg.temperature,
                    side_scale,
                    &mut side_rng,
                )
            };
        if super::hop_is_identity(side_state.view(), trial.view()) {
            for _ in 0..kernels.len() {
                arm = (arm + 1) % kernels.len();
                if kernels[arm].name() != "rcycle" {
                    break;
                }
            }
            trial = kernels[arm].propose_scaled(
                side_state.view(),
                cfg.temperature,
                side_scale,
                &mut side_rng,
            );
        }
        super::recentre(&mut trial, cfg.n_points);
        super::contain_trial(cfg, &mut trial, cfg.n_points);
        let (quenched, relaxed) = relax(ledger, trial.view(), cfg.relax_steps);
        if !quenched.is_finite() || relaxed.len() != side_state.len() {
            if ledger.remaining() == 0 || ledger.stopped() {
                break;
            }
            continue;
        }
        ledger.record(quenched, relaxed.view());
        let improved_branch = quenched < branch_best - new_lowest;
        if improved_branch {
            branch_best = quenched;
            branch_stale = 0;
        }
        let side_temp =
            if (2..7).contains(&branch_idx) && side_energy < branch_best + 3.0 * cfg.temperature {
                cfg.temperature * (1.0 + (branch_stale as f64 / 100.0).min(2.0))
            } else {
                cfg.temperature
            };
        let rise = (quenched - side_energy) / side_temp.max(1e-12);
        let accept_side = rise <= 0.0 || side_rng.random::<f64>() < (-rise).exp();
        let parted = super::rigid_centres_parted(
            side_state.view(),
            relaxed.view(),
            cfg.n_points,
            0.12 * cfg.min_separation,
        );
        let reward =
            improved_branch || (accept_side && parted && quenched < side_energy - new_lowest);
        side_bandit.update(arm, reward);
        if accept_side {
            side_energy = quenched;
            side_state = relaxed;
        }
    }
    if let Some(best) = ledger.best_state.as_ref()
        && ledger.best < *e
    {
        *e = ledger.best;
        *x = best.clone();
    }
}
