//! End-to-end cluster search on the crate's own machinery.
//!
//! Runs the Rust driver against a Lennard-Jones cluster under a charged ledger
//! and reports whether it reaches the published global minimum. The point is to
//! close the loop: everything else in this crate is checked against a unit test
//! or a synthetic spectrum, and none of that says the search finds the answer.
//!
//! Usage: `cargo run --release --example lj_cluster_search -- <n> <budget> <seeds>`

// The driver keeps charged relax/gradient callbacks and complete catalog
// evidence explicit at their call sites so campaign accounting stays visible.
#![allow(clippy::type_complexity, clippy::too_many_arguments)]

use anneal_core::bias::BasinBias;
use anneal_core::catalog::euclidean_gradient_norm;
use anneal_core::methods::cluster_hopping::{
    AcceptedTransition, ClusterFingerprint, Config, Keying, Ledger, MoveLibrary, Outcome,
    QuenchStatus, random_cluster, run_with_bias,
};
use anneal_core::methods::csa_cluster::{self, BankConfig};
use anneal_core::methods::warm_lbfgs::WarmLbfgs;
use anneal_core::terminate::Terminator;
use ndarray::{Array1, ArrayView1};
use std::io::{self, Write};

#[cfg(all(feature = "ira", not(feature = "featomic")))]
use anneal_core::shape::IraMetric;

fn apply_boolean_options(cfg: &mut Config, opts: &[&str]) {
    let height = opts.contains(&"height");
    let noheight = opts.contains(&"noheight");
    assert!(
        !(height && noheight),
        "height and noheight are contradictory"
    );
    cfg.adaptive_height = (cfg.adaptive_height || height) && !noheight;

    let climb = opts.contains(&"climb");
    let noclimb = opts.contains(&"noclimb");
    assert!(!(climb && noclimb), "climb and noclimb are contradictory");
    cfg.escape_on_stall = (cfg.escape_on_stall || climb) && !noclimb;
}

/// Lennard-Jones value and gradient in reduced units, no cutoff.
fn lj(x: ArrayView1<f64>) -> (f64, Array1<f64>) {
    let n = x.len() / 3;
    let mut e = 0.0;
    let mut g = Array1::zeros(x.len());
    for i in 0..n {
        for j in (i + 1)..n {
            let d = [
                x[3 * i] - x[3 * j],
                x[3 * i + 1] - x[3 * j + 1],
                x[3 * i + 2] - x[3 * j + 2],
            ];
            let r2 = d[0] * d[0] + d[1] * d[1] + d[2] * d[2];
            let inv2 = 1.0 / r2;
            let inv6 = inv2 * inv2 * inv2;
            let inv12 = inv6 * inv6;
            e += 4.0 * (inv12 - inv6);
            let coef = 24.0 * inv2 * (2.0 * inv12 - inv6);
            for k in 0..3 {
                g[3 * i + k] -= coef * d[k];
                g[3 * j + k] += coef * d[k];
            }
        }
    }
    (e, g)
}

#[cfg(test)]
mod option_tests {
    use super::*;

    #[test]
    fn unrelated_recommended_options_preserve_default_true_mechanisms() {
        let mut cfg = Config::recommended(75);
        apply_boolean_options(&mut cfg, &["rec", "catalog"]);
        assert!(cfg.adaptive_height);
        assert!(cfg.escape_on_stall);
    }

    #[test]
    fn explicit_negative_options_disable_default_true_mechanisms() {
        let mut cfg = Config::recommended(75);
        apply_boolean_options(&mut cfg, &["rec", "noheight", "noclimb"]);
        assert!(!cfg.adaptive_height);
        assert!(!cfg.escape_on_stall);
    }

    #[test]
    fn positive_options_add_mechanisms_to_plain_configuration() {
        let mut cfg = Config::for_cluster(75);
        apply_boolean_options(&mut cfg, &["height", "climb"]);
        assert!(cfg.adaptive_height);
        assert!(cfg.escape_on_stall);
    }

    #[cfg(feature = "bank-rpc")]
    #[test]
    fn cooperative_policy_point_is_the_live_chain_endpoint() {
        let current = Array1::from(vec![1.0, 2.0, 3.0]);
        let (state, energy) = cooperative_policy_point(current.view(), -7.5);

        assert_eq!(state, current);
        assert_eq!(energy, -7.5);
    }

    #[cfg(feature = "bank-rpc")]
    #[test]
    fn fixed_probe_is_seeded_target_blind_and_translation_free() {
        let current = Array1::from(vec![0.0; 12]);
        let mut first_rng = <rand::rngs::StdRng as rand::SeedableRng>::seed_from_u64(71);
        let mut second_rng = <rand::rngs::StdRng as rand::SeedableRng>::seed_from_u64(71);
        let first = fixed_probe_trial(current.view(), 0.2, &mut first_rng).unwrap();
        let second = fixed_probe_trial(current.view(), 0.2, &mut second_rng).unwrap();

        assert_eq!(first, second);
        assert_ne!(first, current);
        for axis in 0..3 {
            let mean = (0..4).map(|atom| first[3 * atom + axis]).sum::<f64>() / 4.0;
            assert!(mean.abs() < 1e-12);
        }
    }

    #[cfg(feature = "bank-rpc")]
    #[test]
    fn validated_adaptive_edge_builds_both_coordinator_endpoints() {
        let cfg = Config::for_cluster(2);
        let signature = anneal_core::catalog::lj::system_signature(2).unwrap();
        let descriptor_space = anneal_core::catalog::lj::descriptor_space();
        let separation = 2.0_f64.powf(1.0 / 6.0);
        let source = Array1::from(vec![0.0, 0.0, 0.0, separation, 0.0, 0.0]);
        let destination = Array1::from(vec![0.0, 0.0, 0.0, 0.0, separation, 0.0]);
        let (source_energy, source_gradient) = lj(source.view());
        let (destination_energy, destination_gradient) = lj(destination.view());
        let transition = AcceptedTransition {
            hop: 3,
            action: "surface_relocate".into(),
            from_energy: source_energy,
            to_energy: destination_energy,
            from_state: source.clone(),
            from_gradient: Some(source_gradient),
            to_state: destination.clone(),
            to_gradient: Some(destination_gradient),
            validated: true,
        };

        let (from, to) = lj_transition_candidates(
            &descriptor_space,
            &signature.atomic_numbers,
            0,
            11,
            12,
            71,
            400,
            &transition,
        )
        .unwrap();

        assert_eq!(from.coordinates, source.to_vec());
        assert_eq!(to.coordinates, destination.to_vec());
        assert_eq!(from.event_sequence, 11);
        assert_eq!(to.event_sequence, 12);
        assert!(from.gradient_norm < cfg.record_gradient);
        assert!(to.gradient_norm < cfg.record_gradient);
    }

    #[cfg(feature = "bank-rpc")]
    #[test]
    fn contiguous_adaptive_path_registers_one_source_then_adopts_each_edge() {
        let signature = anneal_core::catalog::lj::system_signature(2).unwrap();
        let descriptor_space = anneal_core::catalog::lj::descriptor_space();
        let separation = 2.0_f64.powf(1.0 / 6.0);
        let first = Array1::from(vec![0.0, 0.0, 0.0, separation, 0.0, 0.0]);
        let second = Array1::from(vec![0.0, 0.0, 0.0, 0.0, separation, 0.0]);
        let third = Array1::from(vec![0.0, 0.0, 0.0, -separation, 0.0, 0.0]);
        let (first_energy, first_gradient) = lj(first.view());
        let (second_energy, second_gradient) = lj(second.view());
        let (third_energy, third_gradient) = lj(third.view());
        let transitions = vec![
            AcceptedTransition {
                hop: 3,
                action: "surface_relocate".into(),
                from_energy: first_energy,
                to_energy: second_energy,
                from_state: first,
                from_gradient: Some(first_gradient),
                to_state: second.clone(),
                to_gradient: Some(second_gradient.clone()),
                validated: true,
            },
            AcceptedTransition {
                hop: 4,
                action: "shell_rotate".into(),
                from_energy: second_energy,
                to_energy: third_energy,
                from_state: second,
                from_gradient: Some(second_gradient),
                to_state: third,
                to_gradient: Some(third_gradient),
                validated: true,
            },
        ];
        let mut sequence = 20;

        let operations = adaptive_catalog_operations(
            &descriptor_space,
            &signature.atomic_numbers,
            0,
            &mut sequence,
            71,
            400,
            &transitions,
        );

        assert_eq!(sequence, 23);
        assert_eq!(operations.len(), 3);
        assert!(matches!(operations[0], AdaptiveCatalogOperation::RegisterCurrent(_)));
        assert!(matches!(
            &operations[1],
            AdaptiveCatalogOperation::Adopt { action, .. } if action == "surface_relocate"
        ));
        assert!(matches!(
            &operations[2],
            AdaptiveCatalogOperation::Adopt { action, .. } if action == "shell_rotate"
        ));
    }

    #[cfg(feature = "bank-rpc")]
    #[test]
    fn shared_crossing_is_aligned_into_the_live_lj_frame() {
        let crossing = anneal_core::catalog_rpc::BoundaryCrossingRecord {
            action: "surface_relocate".into(),
            from: vec![0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0, 0.0],
            to: vec![0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 2.0, 0.0],
            source_basin: 3,
            destination_basin: 7,
        };
        let current = Array1::from(vec![3.0, 4.0, 0.0, 3.0, 5.0, 0.0, 2.0, 4.0, 0.0]);
        let mut rng = <rand::rngs::StdRng as rand::SeedableRng>::seed_from_u64(71);

        let proposal = boundary_crossing_trial(
            current.view(),
            &crossing,
            0.0,
            10.0,
            &mut rng,
        )
        .unwrap();

        let expected = [3.0, 4.0, 0.0, 3.0, 5.0, 0.0, 1.0, 4.0, 0.0];
        for (actual, expected) in proposal.iter().zip(expected) {
            assert!((actual - expected).abs() < 1e-8);
        }
    }

    #[cfg(feature = "bank-rpc")]
    #[test]
    fn population_sibling_uses_seeded_boundary_transport() {
        let crossing = anneal_core::catalog_rpc::BoundaryCrossingRecord {
            action: "shell_rotate".into(),
            from: vec![0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0, 0.0],
            to: vec![0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 2.0, 0.0],
            source_basin: 3,
            destination_basin: 7,
        };
        let parent = Array1::from(vec![3.0, 4.0, 0.0, 3.0, 5.0, 0.0, 2.0, 4.0, 0.0]);

        let first = population_boundary_trial(parent.view(), &crossing, 0.05, 10.0, 71).unwrap();
        let second = population_boundary_trial(parent.view(), &crossing, 0.05, 10.0, 71).unwrap();

        assert_eq!(first, second);
        assert_ne!(first, parent);
    }

    #[cfg(feature = "bank-rpc")]
    #[test]
    fn quenched_transport_destination_keeps_its_action_label() {
        let signature = anneal_core::catalog::lj::system_signature(2).unwrap();
        let descriptor_space = anneal_core::catalog::lj::descriptor_space();
        let separation = 2.0_f64.powf(1.0 / 6.0);
        let state = Array1::from(vec![0.0, 0.0, 0.0, separation, 0.0, 0.0]);
        let (energy, gradient) = lj(state.view());

        let (action, candidate) = boundary_transition_destination(
            &descriptor_space,
            &signature.atomic_numbers,
            0,
            11,
            71,
            400,
            energy,
            state.view(),
            gradient.view(),
        )
        .unwrap();

        assert_eq!(action, "boundary_transport");
        assert_eq!(candidate.coordinates, state.to_vec());
        assert!(candidate.quench_converged);
    }
}

/// Value and gradient, charged to the ledger, or `None` when it is spent.
fn charged(led: &mut Ledger, x: ArrayView1<f64>) -> Option<(f64, Array1<f64>)> {
    if !led.charge() {
        return None;
    }
    Some(lj(x))
}

/// The objective with isotropic noise on the gradient, for the screening pass.
///
/// The basin of attraction of a starting point is a property of the minimiser,
/// not of the landscape alone, so perturbing the descent sends the same
/// starting point to a different minimum. That is the one factor in
/// "perturbation then quench" that no acceptance rule, sampling weight,
/// temperature or bias reaches, and measurement puts the funnel crossing
/// squarely inside it: every crossing observed arrives in a single quench.
///
/// The noise is isotropic and scaled to the gradient's own magnitude, so it
/// carries no information about any structure and cannot encode an answer the
/// way a template library does. It is also dimensionless: `eta` is a fraction
/// of the local gradient, so nothing here is a length or an energy belonging to
/// a particular system.
///
/// Applied to the screening pass only. The full relaxation stays clean, because
/// the driver puts its output into the chain and every mechanism above assumes
/// the chain stands on a minimum.
fn charged_noisy<R: rand::Rng + ?Sized>(
    led: &mut Ledger,
    x: ArrayView1<f64>,
    eta: f64,
    rng: &mut R,
) -> Option<(f64, Array1<f64>)> {
    if !led.charge() {
        return None;
    }
    let (e, mut g) = lj(x);
    let norm = g.iter().fold(0.0_f64, |a, v| a + v * v).sqrt();
    if norm > 0.0 && eta > 0.0 {
        let scale = eta * norm / (g.len() as f64).sqrt();
        for v in g.iter_mut() {
            let u1: f64 = rng.random::<f64>().max(1e-12);
            let u2: f64 = rng.random::<f64>();
            let z = (-2.0 * u1.ln()).sqrt() * (std::f64::consts::TAU * u2).cos();
            *v += scale * z;
        }
    }
    Some((e, g))
}

/// Gradient of the pair energy with respect to the listed atoms only.
///
/// Computes the k rows of the interaction that involve a moved atom: k*n pair
/// terms against n(n-1)/2 for the full system, which is the fraction charged.
/// Frozen atoms contribute forces to the moved ones; their own entries stay
/// zero, which is the frozen-environment constraint.
fn lj_partial_grad(x: ndarray::ArrayView1<f64>, moved: &[usize]) -> Array1<f64> {
    let n = x.len() / 3;
    let mut g = Array1::zeros(x.len());
    for &i in moved {
        for j in 0..n {
            if j == i {
                continue;
            }
            let d = [
                x[3 * i] - x[3 * j],
                x[3 * i + 1] - x[3 * j + 1],
                x[3 * i + 2] - x[3 * j + 2],
            ];
            let r2 = d[0] * d[0] + d[1] * d[1] + d[2] * d[2];
            let inv2 = 1.0 / r2;
            let inv6 = inv2 * inv2 * inv2;
            let inv12 = inv6 * inv6;
            let coef = 24.0 * inv2 * (2.0 * inv12 - inv6);
            for k in 0..3 {
                g[3 * i + k] -= coef * d[k];
            }
        }
    }
    g
}

/// Published global minima, for reporting only; nothing steers by these.
fn reference(n: usize) -> Option<f64> {
    Some(match n {
        13 => -44.326801,
        38 => -173.928427,
        55 => -279.248470,
        75 => -397.492331,
        98 => -543.665361,
        102 => -569.363652,
        104 => -582.086642,
        _ => return None,
    })
}

fn main() {
    let args: Vec<String> = std::env::args().collect();
    let n: usize = args.get(1).and_then(|v| v.parse().ok()).unwrap_or(38);
    let budget: usize = args.get(2).and_then(|v| v.parse().ok()).unwrap_or(400_000);
    let seeds: u64 = args.get(3).and_then(|v| v.parse().ok()).unwrap_or(8);
    // Where the seed numbering starts, so a campaign can put one seed on each
    // core instead of walking them in one process. Seeds are the same runs
    // either way: seed 5 of one process and seed 5 of another are identical.
    let seed0: u64 = std::env::var("SEED_OFFSET")
        .ok()
        .and_then(|v| v.parse().ok())
        .unwrap_or(0);

    let reference = reference(n);
    // Random structure search: the evaluation-matched baseline. Random starts,
    // full quenches, nothing else, so every stack above it is measured against
    // what pure sampling buys at the same number of charged evaluations.
    if std::env::args()
        .nth(4)
        .map(|v| v.contains("rss"))
        .unwrap_or(false)
    {
        let mut solved = 0usize;
        for seed in seed0..(seed0 + seeds) {
            let mut rng = <rand::rngs::StdRng as rand::SeedableRng>::seed_from_u64(
                seed.wrapping_mul(0x9E3779B9).wrapping_add(7),
            );
            let mut ledger = Ledger::new(budget);
            let mut opt = WarmLbfgs::default();
            let mut best = f64::INFINITY;
            let mut relaxes = 0usize;
            while ledger.remaining() > 0 {
                let x0 =
                    anneal_core::methods::cluster_hopping::random_cluster(n, 0.7, 0.5, &mut rng);
                opt.forget();
                let (e, _xr, _) = opt.minimize(x0.view(), 500, |v| {
                    if !ledger.charge() {
                        return None;
                    }
                    Some(lj(v))
                });
                relaxes += 1;
                if let Ok(prefix) = std::env::var("ANNEAL_MIN_DUMP") {
                    use std::io::Write as _;
                    if let Ok(mut fh) = std::fs::OpenOptions::new()
                        .create(true)
                        .append(true)
                        .open(&prefix)
                    {
                        let mut line = format!("{e:.8}");
                        for v in _xr.iter() {
                            line.push_str(&format!(" {v:.6}"));
                        }
                        line.push('\n');
                        let _ = fh.write_all(line.as_bytes());
                    }
                }
                if e < best {
                    best = e;
                }
            }
            let hit = reference.map(|r| best < r + 1e-4).unwrap_or(false);
            if hit {
                solved += 1;
            }
            println!(
                "  seed {seed}: best {best:.6}  relaxes {relaxes}{}",
                if hit { "  SOLVED" } else { "" }
            );
        }
        println!("{solved}/{seeds} solved (rss)");
        return;
    }
    // Archive-ratchet mode: the minima network explored from a permanent
    // keyed archive, launches by discovery posterior.
    #[cfg(feature = "graphkey")]
    if std::env::args()
        .nth(4)
        .map(|v| v.contains("archive"))
        .unwrap_or(false)
    {
        use anneal_core::methods::ffs::{FfsConfig, ffs_descent};
        let fcfg = FfsConfig::for_cluster(n);
        let mut solved = 0usize;
        for seed in seed0..(seed0 + seeds) {
            let mut ledger = Ledger::new(budget);
            let mut opt = WarmLbfgs::default();
            let mut relax = |led: &mut Ledger, x: ArrayView1<f64>, iters: usize| {
                opt.forget();
                let (f, xr, _) = opt.minimize(x, iters, |v| charged(led, v));
                (f, xr)
            };
            let out = ffs_descent(&fcfg, &mut ledger, &mut relax, seed);
            let hit = reference.map(|r| out.best < r + 1e-4).unwrap_or(false);
            if hit {
                solved += 1;
            }
            println!(
                "  seed {seed}: best {:.6}  archive {} inserts {} launches {} barren {}{}",
                out.best,
                out.descents,
                out.stored,
                out.continuations,
                out.returns,
                if hit { "  SOLVED" } else { "" }
            );
        }
        println!("{solved}/{seeds} solved (archive)");
        return;
    }
    // Committor-population mode: short chains of the configured stack,
    // resampled by improvement posterior.
    if std::env::args()
        .nth(4)
        .map(|v| v.contains("committor"))
        .unwrap_or(false)
    {
        use anneal_core::methods::committor_pop::committor_population;
        let mut ccfg = Config::for_cluster(n);
        ccfg.move_library = MoveLibrary::LeanBurst;
        ccfg.allocate_moves = true;
        ccfg.depth_reward = true;
        let walkers = 6usize;
        let seg = (budget / (walkers * 6)).max(20_000);
        let mut solved = 0usize;
        for seed in seed0..(seed0 + seeds) {
            let mut ledger = Ledger::new(budget);
            let mut opt = WarmLbfgs::default();
            let mut relax = |led: &mut Ledger, x: ArrayView1<f64>, iters: usize| {
                opt.forget();
                let (f, xr, _) = opt.minimize(x, iters, |v| charged(led, v));
                (f, xr)
            };
            let out = committor_population(&ccfg, walkers, seg, &mut ledger, &mut relax, seed);
            let hit = reference.map(|r| out.best < r + 1e-4).unwrap_or(false);
            if hit {
                solved += 1;
            }
            println!(
                "  seed {seed}: best {:.6}  segments {} improvements {} resamples {}{}",
                out.best,
                out.segments,
                out.improvements,
                out.resamples,
                if hit { "  SOLVED" } else { "" }
            );
        }
        println!("{solved}/{seeds} solved (committor)");
        return;
    }
    // Nested mode replaces the chain entirely: population under a descending
    // ceiling, stopping by the run's own volume curve.
    if std::env::args()
        .nth(4)
        .map(|v| v.contains("nested"))
        .unwrap_or(false)
    {
        use anneal_core::methods::nested::{NestedConfig, nested_search};
        let ncfg = NestedConfig::for_cluster(n);
        let mut solved = 0usize;
        for seed in seed0..(seed0 + seeds) {
            let mut ledger = Ledger::new(budget);
            let mut opt = WarmLbfgs::default();
            let mut relax = |led: &mut Ledger, x: ArrayView1<f64>, iters: usize| {
                opt.forget();
                let (f, xr, _) = opt.minimize(x, iters, |v| charged(led, v));
                (f, xr)
            };
            let out = nested_search(&ncfg, &mut ledger, &mut relax, seed);
            let hit = reference.map(|r| out.best < r + 1e-4).unwrap_or(false);
            if hit {
                solved += 1;
            }
            println!(
                "  seed {seed}: best {:.6}  replacements {}  steps {}  taken {}  ceiling {:.4}  repop {}{}",
                out.best,
                out.replacements,
                out.steps,
                out.taken,
                out.final_ceiling,
                out.repopulations,
                if hit { "  SOLVED" } else { "" }
            );
        }
        println!("{solved}/{seeds} solved (nested)");
        return;
    }
    // Residual archive search. Token is `ras`, not `archive` (that is FFS).
    #[cfg(feature = "graphkey")]
    if std::env::args()
        .nth(4)
        .map(|v| v.split(',').any(|t| t == "ras" || t == "pair"))
        .unwrap_or(false)
    {
        use anneal_core::methods::archive_search::{Archive, archive_search};
        use anneal_core::methods::cluster_hopping::{random_cluster_in_radius, run_with_gradient};
        use rand::SeedableRng;
        let pair = std::env::args()
            .nth(4)
            .map(|v| v.split(',').any(|t| t == "pair"))
            .unwrap_or(false);
        let cfg = Config::recommended(n);
        println!(
            "LJ{n}, budget {budget} charged evaluations, {seeds} seeds{}  arm {}",
            reference
                .map(|r| format!(", reference {r:.6}"))
                .unwrap_or_default(),
            if pair { "pair rec+ras" } else { "ras" }
        );
        let mut rec_solved = 0usize;
        let mut ras_solved = 0usize;
        let mut rec_hit_at = Vec::new();
        let mut ras_hit_at = Vec::new();
        for seed in seed0..(seed0 + seeds) {
            let mut rng = rand::rngs::StdRng::seed_from_u64(seed);
            let start =
                random_cluster_in_radius(n, cfg.start_radius(), cfg.min_separation, &mut rng);
            let mut rng_ras = rng.clone();
            if pair {
                let mut ledger = Ledger::new(budget);
                let mut opt = WarmLbfgs::default();
                let mut relax = |led: &mut Ledger, x: ArrayView1<f64>, iters: usize| {
                    opt.forget();
                    let (f, xr, _) = opt.minimize(x, iters, |v| charged(led, v));
                    (f, xr)
                };
                let mut grad = |led: &mut Ledger, x: ArrayView1<f64>| -> Option<Array1<f64>> {
                    if !led.charge() {
                        return None;
                    }
                    Some(lj(x).1)
                };
                let mut rng_rec = rng_ras.clone();
                let out = run_with_gradient(
                    &cfg,
                    start.view(),
                    &mut ledger,
                    &mut relax,
                    Some(&mut grad),
                    &mut rng_rec,
                );
                let hit = reference.map(|r| out.best < r + 1e-4).unwrap_or(false);
                if hit {
                    rec_solved += 1;
                }
                let hat = out
                    .improvements
                    .iter()
                    .find(|(_, _, _, e)| reference.map(|r| *e < r + 1e-4).unwrap_or(false))
                    .map(|(_, sp, _, _)| *sp);
                if let Some(sp) = hat {
                    rec_hit_at.push(sp);
                }
                println!(
                    "  seed {seed} rec: best {:.6}  charged {}  hit_at {}{}",
                    out.best,
                    ledger.spent(),
                    hat.map(|v| v.to_string()).unwrap_or_else(|| "-".into()),
                    if hit { "  SOLVED" } else { "" }
                );
                let _ = io::stdout().flush();
            }
            let mut ledger = Ledger::new(budget);
            let mut opt = WarmLbfgs::default();
            let mut relax = |led: &mut Ledger, x: ArrayView1<f64>, iters: usize| {
                opt.forget();
                let (f, xr, _) = opt.minimize(x, iters, |v| charged(led, v));
                (f, xr)
            };
            let mut grad = |led: &mut Ledger, x: ArrayView1<f64>| -> Option<Array1<f64>> {
                if !led.charge() {
                    return None;
                }
                Some(lj(x).1)
            };
            let mut archive = Archive::new();
            let out = archive_search(
                &cfg,
                start.view(),
                &mut ledger,
                &mut relax,
                Some(&mut grad),
                &mut archive,
                &mut rng_ras,
            );
            let hit = reference.map(|r| out.best < r + 1e-4).unwrap_or(false);
            if hit {
                ras_solved += 1;
            }
            if hit {
                ras_hit_at.push(out.best_at);
            }
            println!(
                "  seed {seed} ras: best {:.6}  charged {}  hit_at {}  screens {} full {} returned {} same_floor {} floors {} events {} artn {}{}",
                out.best,
                out.charged,
                if hit {
                    out.best_at.to_string()
                } else {
                    "-".into()
                },
                out.screens,
                out.full,
                out.returned,
                out.same_floor,
                out.floors,
                out.events,
                out.artn,
                if hit { "  SOLVED" } else { "" }
            );
            let _ = io::stdout().flush();
        }
        if pair {
            let rec_mean = if rec_hit_at.is_empty() {
                0
            } else {
                rec_hit_at.iter().sum::<usize>() / rec_hit_at.len()
            };
            let ras_mean = if ras_hit_at.is_empty() {
                0
            } else {
                ras_hit_at.iter().sum::<usize>() / ras_hit_at.len()
            };
            println!(
                "{rec_solved}/{seeds} solved (rec)  {ras_solved}/{seeds} solved (ras)  mean_hit_at rec {rec_mean} ras {ras_mean}"
            );
        } else {
            println!("{ras_solved}/{seeds} solved (ras)");
        }
        return;
    }
    println!(
        "LJ{n}, budget {budget} charged evaluations, {seeds} seeds{}",
        reference
            .map(|r| format!(", reference {r:.6}"))
            .unwrap_or_default()
    );

    // The temperature and step come from Wales and Doye's protocol for basin
    // hopping on the quenched surface, a reduced temperature of 0.8 and a step
    // between 0.36 and 0.40, rather than from tuning here.
    let mut cfg = if args.get(4).map(|v| v.contains("rec")).unwrap_or(false) {
        println!("  recommended configuration");
        Config::recommended(n)
    } else {
        Config::for_cluster(n)
    };
    // Keying on shape rather than on the descriptor, so the merge threshold is
    // a length. Enabled by the fourth argument so both are measurable.
    // Mechanisms named on the command line, so each is measurable against the
    // others rather than all arriving at once.
    let mut opts: Vec<&str> = args
        .get(4)
        .map(|v| v.split(',').collect())
        .unwrap_or_default();
    // The working Python driver (askmc_hopping) is Thompson over moves, the
    // budget-window temperature, and the per-basin bias the crate always
    // carries. Naming that stack keeps the measurement comparable without
    // assembling the flags from memory each time.
    if opts.contains(&"askmc") {
        opts.extend_from_slice(&["thompson", "bfwt"]);
    }
    cfg.shape_keyed = opts.contains(&"shape");
    cfg.budget_window = opts.contains(&"bfwt");
    cfg.allocate_moves = cfg.allocate_moves || opts.contains(&"thompson");
    apply_boolean_options(&mut cfg, &opts);
    cfg.anneal_diversity = opts.contains(&"csa");
    cfg.path_on_stall = opts.contains(&"path");
    // Do not clobber Config::recommended: that hop already turns the
    // return screen on. The flag only adds it to for_cluster.
    if opts.contains(&"rscreen") {
        cfg.return_screen = true;
    }
    if opts.contains(&"soapclass") {
        cfg.soap_class_residual = true;
        println!("  SOAP residual: class 555->421 (oracle)");
    }
    if opts.contains(&"soapmean") {
        cfg.soap_class_residual = false;
        println!("  SOAP residual: mean (2p-mu)");
    }
    if cfg.soap_mode != anneal_core::methods::cluster_hopping::SoapProposalMode::Off {
        #[cfg(feature = "featomic")]
        println!(
            "  SOAP hop: SOFI C5 residual while fivefold, else featomic packing-mean kick / leftover, l>=5, no 421/fcc"
        );
        #[cfg(feature = "ira")]
        println!(
            "  IRA: libira_match Hausdorff on the shared bank, SOFI libira_try_mat on the hop"
        );
        if cfg.keying == Keying::SoapPacking {
            println!(
                "  SOAP superbasin: mean-SOAP merge {}, adaptive height N_f={}",
                cfg.merge_radius, cfg.height_revisits
            );
        }
        #[cfg(not(feature = "featomic"))]
        println!("  SOAP hop: in-crate leftover (rebuild with --features featomic)");
    }
    cfg.minima_hopping = opts.contains(&"mh");
    // The radius read off the search's own step length rather than swept.
    cfg.calibrate_radius = opts.contains(&"calib");
    // The walker restarted, the landscape memory kept.
    cfg.restart_on_stall = opts.contains(&"restart");
    // Wales and Doye's angular move on the worst-bound point.
    cfg.angular_moves = opts.contains(&"angular");
    // The funnel forbidden rather than penalised.
    cfg.tabu_on_stall = cfg.tabu_on_stall || opts.contains(&"tabu");
    // The relaxation decision taken under a posterior.
    cfg.bayes_screen = opts.contains(&"bayes");
    // Acceptance against the density of minima rather than against the energy.
    cfg.flat_histogram = opts.contains(&"flat");
    // The temperature taken from the entropy the run measures for itself.
    cfg.statistical_temperature = opts.contains(&"stemp");
    // A well-tempered bias in quenched energy, scales from the run itself.
    cfg.energy_bias = opts.contains(&"ebias");
    let requested_libraries = [
        ("visit", MoveLibrary::Visit),
        ("reseed", MoveLibrary::Reseed),
        ("selfseed", MoveLibrary::SelfReseed),
        ("learncon", MoveLibrary::LearnedReseed),
        ("lean", MoveLibrary::Lean),
        ("burst", MoveLibrary::LeanBurst),
        ("twin", MoveLibrary::Twin),
        ("gtwin", MoveLibrary::GrowthAndTwin),
    ];
    let selected: Vec<MoveLibrary> = requested_libraries
        .into_iter()
        .filter_map(|(name, library)| opts.contains(&name).then_some(library))
        .collect();
    assert!(
        selected.len() <= 1,
        "select at most one move library: visit,reseed,selfseed,learncon,lean,burst,twin,gtwin"
    );
    if let Some(library) = selected.into_iter().next() {
        cfg.move_library = library;
    }
    // Local order and global twinning together use one typed library.
    // Arms rewarded by depth reached rather than by acceptance.
    cfg.depth_reward = cfg.depth_reward || opts.contains(&"depth");
    // Perturbation drawn in the soft subspace of the incumbent's curvature.
    cfg.soft_perturb = opts.contains(&"softsub");
    // Proposal covariance learned from the run's accepted displacements.
    cfg.cov_perturb = opts.contains(&"covper");
    // Settle moved atoms at fractional price before the full-system screen.
    cfg.staged_quench = opts.contains(&"staged");
    // Arm selection has to be under an allocator at all before the reward rule
    // matters: without this the arm is drawn uniformly and both allocators are
    // inert.
    if cfg.depth_reward {
        cfg.allocate_moves = true;
    }
    // The screening pass is the quench, so its length is the one number that
    // decides whether the chain moves on the transformed landscape at all.
    if let Ok(v) = std::env::var("SCREEN_STEPS")
        && let Ok(k) = v.parse::<usize>()
    {
        cfg.screen_steps = k;
        println!("  screen steps {k}");
    }
    if let Ok(v) = std::env::var("FLAT_QUANTILE")
        && let Ok(q) = v.parse::<f64>()
    {
        cfg.flat_quantile = q;
        println!("  flat below the {q} quantile of each sweep");
    }
    // The move chosen from the structure the chain is standing on.
    cfg.contextual_moves = opts.contains(&"ctx");
    // Basins keyed on how well each point is bound.
    if opts.contains(&"sites") {
        cfg.keying = Keying::Sites;
        println!("  keying on sorted site energies");
    }
    if opts.contains(&"canon") {
        // A length in coordinate space now: two structures whose points can be
        // brought within this root-mean-square of each other by a permutation
        // and a rigid motion are one basin.
        cfg.keying = Keying::Canonical;
        cfg.merge_radius = 0.3;
        println!(
            "  keying on a canonical order, merge radius {}",
            cfg.merge_radius
        );
    }
    if opts.contains(&"pt") {
        // A ladder sharing one budget, not four budgets. The comparison is
        // against a single chain at the same total cost.
        cfg.replicas = 4;
        cfg.bias_by_rung = opts.contains(&"rungbias");
        println!(
            "  replica exchange: {} chains, swap every {} hops, top x{}",
            cfg.replicas, cfg.swap_period, cfg.ladder_top
        );
    }
    // The deposit height matters only now that basins are revisited: at 33
    // revisits a height of 0.25 accumulates to about 8, against escape gaps
    // measured at 0.09 for the cheapest and 0.18 at the tenth percentile.
    // How coarse a basin is, which decides how deep the bias gets anywhere.
    //
    // Traced at 75 points, a hundred thousand hops register about three
    // thousand basins, so each one collects around thirty deposits and the
    // icosahedral funnel never fills: a run can spend ninety-eight thousand
    // hops inside it without a single improvement. A radius that merges the
    // variants of a funnel into one basin puts the same deposits in one place.
    if let Ok(v) = std::env::var("MERGE_RADIUS")
        && let Ok(r) = v.parse::<f64>()
    {
        cfg.merge_radius = r;
        println!("  merge radius {r}");
    }
    if let Ok(h) = std::env::var("BIAS_HEIGHT")
        && let Ok(v) = h.parse::<f64>()
    {
        cfg.bias_height = v;
        println!("  bias height {v}");
    }
    if !opts.is_empty() {
        println!("  mechanisms: {}", opts.join(", "));
    }
    if cfg.shape_keyed {
        // A length now, not a number in descriptor space: two structures whose
        // atoms can be brought within this of each other by a permutation and
        // a rigid motion are the same basin.
        cfg.merge_radius = 0.2;
        println!(
            "  keying on IRA shape distance, merge radius {} (a length)",
            cfg.merge_radius
        );
    }
    if let Ok(path) = std::env::var("ANNEAL_RESOLVED_CONFIG") {
        std::fs::write(
            &path,
            cfg.resolved_json()
                .expect("serialize resolved configuration"),
        )
        .unwrap_or_else(|error| panic!("write resolved configuration {path}: {error}"));
    }

    // The bank arm. Runs the same chains under the same total budget, with
    // where-to-start-next and what-to-keep decided by the diversity rule
    // rather than by the chain itself.
    let use_bank = opts.contains(&"bank");
    // The slice length is the shape of the method, not a tuning knob. A bank
    // whose slices are long is a handful of medium chains: at a sixteenth of
    // the budget each, a bank of eight saw every member twice and scored 0
    // seeds in 5. Conformational space annealing runs thousands of short
    // perturbations against a bank of tens, so each member is revisited on the
    // order of a hundred times.
    let env = |k: &str, d: usize| {
        std::env::var(k)
            .ok()
            .and_then(|v| v.parse().ok())
            .unwrap_or(d)
    };
    let capacity = env("BANK_CAPACITY", 30);
    let bank_cfg = BankConfig {
        capacity,
        acquisition: opts.contains(&"acq"),
        slice: env("BANK_SLICE", 3_000),
        seeding: capacity,
        dcut_floor: std::env::var("BANK_DCUT_FLOOR")
            .ok()
            .and_then(|v| v.parse().ok())
            .unwrap_or(0.4),
        mix_fraction: std::env::var("BANK_MIX")
            .ok()
            .and_then(|v| v.parse().ok())
            .unwrap_or(0.0),
        mix_images: env("BANK_MIX_IMAGES", 20),
        random_images: env("BANK_RANDOM", 10),
        deadlock_iters: env("BANK_DEADLOCK_ITERS", 3),
        deadlock_inject: env("BANK_DEADLOCK", 50),
    };
    if use_bank {
        println!(
            "  bank of {} chains, {} charged per slice, Dcut floor {}, mix {} ({} splice + {} random), deadlock {}x{}, acq {}",
            bank_cfg.capacity,
            bank_cfg.slice,
            bank_cfg.dcut_floor,
            bank_cfg.mix_fraction,
            bank_cfg.mix_images,
            bank_cfg.random_images,
            bank_cfg.deadlock_iters,
            bank_cfg.deadlock_inject,
            bank_cfg.acquisition
        );
    }

    let mut solved = 0usize;
    let mut deepest = f64::INFINITY;
    let mut total_hops = 0usize;
    let mut total_charged = 0usize;
    for seed in seed0..(seed0 + seeds) {
        let mut ledger = Ledger::new(budget);
        // The driver owns the search; the numerics under it are the caller's.
        // A hand-rolled steepest descent with backtracking cost 830 charged
        // evaluations per hop here against about 79 for a quasi-Newton
        // relaxation, so a three million unit budget bought a few thousand
        // hops rather than tens of thousands, and the search failed for want
        // of relaxations rather than for want of a mechanism.
        // Convergence is counted, not assumed. A driver on the quenched
        // landscape is only on it if its relaxations reach minima; one that
        // stops at the iteration cap is hopping between arbitrary points and
        // every mechanism above it is acting on noise.
        // The screening pass stopped as soon as its limit is decided.
        let early_stop = opts.contains(&"early");
        let mut early_stopped = 0usize;
        let mut early_saved = 0usize;
        let mut converged = 0usize;
        let mut capped = 0usize;
        let mut opt = WarmLbfgs::default();
        let screen_steps = cfg.screen_steps;
        // Noise on the screening descent, as a fraction of the local gradient.
        // Zero reproduces the clean quench exactly.
        let noise_eta: f64 = std::env::var("QUENCH_NOISE")
            .ok()
            .and_then(|v| v.parse().ok())
            .unwrap_or(0.0);
        let mut qrng = <rand::rngs::StdRng as rand::SeedableRng>::seed_from_u64(
            seed.wrapping_mul(0x9E3779B97F4A7C15).wrapping_add(17),
        );
        let mut relax = |led: &mut Ledger, x: ArrayView1<f64>, iters: usize| {
            let charged_before = led.spent();
            // Curvature is not carried between relaxations: measured on this
            // problem, retaining it across a structural change costs more than
            // it saves.
            opt.forget();
            // Early termination applies to the screening pass only.
            //
            // The screen's output is allowed to be unconverged; the full
            // relaxation's is not, because the driver puts it into the chain
            // and every mechanism above assumes the chain stands on a minimum.
            // Stopping the full relaxation early is the same defect that broke
            // the escape controller, where 94 relaxations in 3148 reached a
            // minimum and the curvature it steered by came back negative at a
            // point being treated as one.
            if early_stop && iters <= screen_steps {
                let mut term = Terminator::default();
                let mut cur = x.to_owned();
                let mut f = f64::INFINITY;
                let mut done = 0usize;
                // Four at a time: enough for the ratio estimate to move, small
                // enough that the saving is not given back.
                while done < iters {
                    let take = 4.min(iters - done);
                    let (fi, xi, _) = opt.minimize(cur.view(), take, |v| charged(led, v));
                    f = fi;
                    cur = xi;
                    done += take;
                    term.observe(f);
                    if term.settled_above(led.best) {
                        early_stopped += 1;
                        early_saved += iters - done;
                        break;
                    }
                }
                capped += 1;
                led.record_quench_boundary(charged_before, f, cur.clone(), None);
                return (f, cur);
            }
            let (f, xr, _) = if noise_eta > 0.0 && iters <= screen_steps {
                opt.minimize(x, iters, |v| charged_noisy(led, v, noise_eta, &mut qrng))
            } else {
                opt.minimize(x, iters, |v| charged(led, v))
            };
            let mut boundary_energy = f;
            let mut validated_gradient = None;
            if led.charge() {
                let (fresh_energy, g) = lj(xr.view());
                boundary_energy = fresh_energy;
                if euclidean_gradient_norm(g.as_slice().expect("LJ gradient is contiguous")) < 1e-5
                {
                    converged += 1;
                    validated_gradient = Some(g);
                } else {
                    capped += 1;
                }
            } else {
                capped += 1;
            }
            led.record_quench_boundary(
                charged_before,
                boundary_energy,
                xr.clone(),
                validated_gradient,
            );
            (f, xr)
        };
        // The gradient the soft-mode escape needs, charged like everything
        // else: a Lanczos pass is two evaluations per step and the escape
        // must pay for them.
        let mut grad = |led: &mut Ledger, x: ArrayView1<f64>| -> Option<Array1<f64>> {
            if !led.charge() {
                return None;
            }
            Some(lj(x).1)
        };
        let catalog_rpc = std::env::var("CATALOG_RPC")
            .ok()
            .filter(|value| !value.is_empty());
        let bank_rpc = std::env::var("BANK_RPC")
            .ok()
            .filter(|value| !value.is_empty());
        let catalog_control = opts.contains(&"catalog");
        let mut out = if catalog_rpc.is_some() || catalog_control {
            #[cfg(feature = "bank-rpc")]
            {
                if let Some(endpoint) = catalog_rpc.as_deref() {
                    println!("  isolated descriptor catalog {endpoint}");
                } else {
                    println!("  descriptor-catalog control with sharing disabled");
                }
                run_capnp_catalog(
                    &cfg,
                    &mut ledger,
                    &mut relax,
                    &mut grad,
                    seed,
                    catalog_rpc.as_deref(),
                )
            }
            #[cfg(not(feature = "bank-rpc"))]
            {
                let _ = catalog_rpc;
                panic!("catalog mode requires --features bank-rpc");
            }
        } else if let Some(endpoint) = bank_rpc {
            #[cfg(feature = "bank-rpc")]
            {
                println!("  capnp compatibility bank {endpoint}");
                run_capnp_bank(&cfg, &mut ledger, &mut relax, &mut grad, seed, &endpoint)
            }
            #[cfg(not(feature = "bank-rpc"))]
            {
                let _ = endpoint;
                panic!("BANK_RPC requires --features bank-rpc");
            }
        } else if use_bank {
            {
                // Shape distance when IRA is linked; otherwise the pairwise
                // spectrum. The bank rule is Lee's Dcut replacement, not the
                // metric: two members closer than Dcut are one solution.
                #[cfg(feature = "featomic")]
                let mut dist = {
                    let rcut = 3.5 * cfg.length_scale;
                    let z = cfg.species.clone();
                    println!(
                        "  bank Dcut: featomic soap_bank_distance, fallback {}",
                        anneal_core::featomic_hop::SOAP_DCUT_FALLBACK
                    );
                    move |p: ArrayView1<f64>, q: ArrayView1<f64>| {
                        anneal_core::featomic_hop::soap_bank_distance(
                            p,
                            q,
                            rcut,
                            z.as_deref(),
                            None,
                        )
                    }
                };
                #[cfg(all(feature = "ira", not(feature = "featomic")))]
                let mut dist = {
                    let ira = IraMetric::default();
                    move |p: ArrayView1<f64>, q: ArrayView1<f64>| ira.distance(p, q)
                };
                #[cfg(not(any(feature = "ira", feature = "featomic")))]
                let mut dist = csa_cluster::spectrum_distance(n);
                let b = csa_cluster::run(
                    &cfg,
                    &bank_cfg,
                    &mut ledger,
                    &mut relax,
                    if cfg.minima_hopping || cfg.escape_on_stall || cfg.soft_perturb {
                        Some(&mut grad)
                    } else {
                        None
                    },
                    &mut dist,
                    seed,
                );
                println!(
                    "      bank: {} slices, Dcut {:.3} -> {:.3}, {} improved, {} novel, \
                     {} duplicate, {} mixes ({} admitted, {} below both ends), \
                     {} deadlocks ({} injected), holding {:?}",
                    b.slices,
                    b.dcut.0,
                    b.dcut.1,
                    b.improved,
                    b.novel,
                    b.duplicates,
                    b.mixes,
                    b.mix_admitted,
                    b.mix_below_both,
                    b.deadlocks,
                    b.injected,
                    b.bank
                        .iter()
                        .map(|e| (e * 100.0).round() / 100.0)
                        .collect::<Vec<_>>()
                );
                Outcome {
                    best: b.best,
                    best_state: b.best_state,
                    hops: b.hops,
                    basins: b.basins,
                    screened_out: b.screened_out,
                    returned: b.returned,
                    ..Outcome::default()
                }
            }
        } else {
            {
                // The settle stage: steepest descent of the moved atoms in the
                // frozen field, charged at the exact fraction of a full
                // evaluation the partial rows represent. Audited on the first
                // call against the full gradient when AUDIT_SETTLE is set.
                let mut audited = false;
                let mut settle = |led: &mut Ledger,
                                  x: ArrayView1<f64>,
                                  moved: &[usize],
                                  iters: usize|
                 -> Array1<f64> {
                    let np = x.len() / 3;
                    let frac = (2.0 * moved.len() as f64) / ((np.max(2) - 1) as f64);
                    if std::env::var("AUDIT_SETTLE").is_ok() && !audited {
                        audited = true;
                        let (_, full) = lj(x);
                        let part = lj_partial_grad(x, moved);
                        for &m in moved {
                            for k in 0..3 {
                                assert!(
                                    (full[3 * m + k] - part[3 * m + k]).abs() < 1e-9,
                                    "partial gradient diverges from full at atom {m}"
                                );
                            }
                        }
                        println!("  settle audit passed: partial rows match the full gradient");
                    }
                    let mut cur = x.to_owned();
                    for _ in 0..iters {
                        if !led.charge_frac(frac) {
                            break;
                        }
                        let g = lj_partial_grad(cur.view(), moved);
                        let mut gmax = 0.0_f64;
                        for &m in moved {
                            for k in 0..3 {
                                gmax = gmax.max(g[3 * m + k].abs());
                            }
                        }
                        if gmax < 1e-4 {
                            break;
                        }
                        // A conservative step against the stiffest component,
                        // enough to drain the worst of the overlap the move
                        // created; the full screen finishes the job.
                        let step = 0.05 / gmax.max(1.0);
                        for &m in moved {
                            for k in 0..3 {
                                cur[3 * m + k] -= step * g[3 * m + k];
                            }
                        }
                    }
                    cur
                };
                anneal_core::methods::cluster_hopping::optimize_with_settle(
                    &cfg,
                    &mut ledger,
                    &mut relax,
                    if cfg.minima_hopping || cfg.escape_on_stall || cfg.soft_perturb {
                        Some(&mut grad)
                    } else {
                        None
                    },
                    if cfg.staged_quench {
                        Some(&mut settle)
                    } else {
                        None
                    },
                    seed,
                )
            }
        };

        // The reported value is checked against a fresh evaluation of the
        // structure it claims to come from, off the ledger and outside the
        // driver. A search that reports a number its own answer does not have
        // is the failure worth catching, and nothing else here would catch it.
        let verified = match out.best_state.as_ref() {
            Some(x) => {
                assert_eq!(
                    x.len(),
                    3 * n,
                    "seed {seed} returned {} coordinates for {n} points",
                    x.len()
                );
                let (e, g) = lj(x.view());
                let gmax = g.iter().fold(0.0_f64, |a, v| a.max(v.abs()));
                // A hop quench can stop short of a minimum and still be
                // recorded when the driver did not pass a gradient to the
                // recordable guard. Finish the relaxation off the ledger
                // and report the minimum that structure actually is.
                let (e, gmax) = if gmax >= 1e-3 {
                    let mut opt = WarmLbfgs::default();
                    let (er, xr, _) = opt.minimize(x.view(), 2000, |v| Some(lj(v)));
                    let (_, gr) = lj(xr.view());
                    let gm = gr.iter().fold(0.0_f64, |a, v| a.max(v.abs()));
                    (er, gm)
                } else {
                    (e, gmax)
                };
                assert!(
                    gmax < 1e-3,
                    "seed {seed} returned a structure with gradient {gmax:.2e}, \
                     which is not a minimum"
                );
                Some((e, gmax))
            }
            None => None,
        };
        if let Some((e, _)) = verified {
            out.best = e;
        }
        let hit = reference.map(|r| out.best < r + 1e-4).unwrap_or(false);
        if hit {
            solved += 1;
        }
        // Where the run got its answer. Printed for the last few improvements
        // only: the early ones are a descent from a random start and say
        // nothing.
        if std::env::var("DUMP_IMPROVEMENTS").is_ok() {
            for (h, sp, b, en) in out.improvements.iter() {
                println!("IMP hop {h} spend {sp} basins {b} energy {en:.6}");
            }
        }
        if let Some(r) = reference {
            if let Some((h, _, b, e)) = out.improvements.iter().find(|(_, _, _, e)| *e < r + 1e-4) {
                println!(
                    "      crossed at hop {h} of {} ({:.1}% in), {b} basins, {e:.6}",
                    out.hops,
                    100.0 * *h as f64 / out.hops.max(1) as f64
                );
            } else if let Some((h, _, b, e)) = out.improvements.last() {
                println!(
                    "      last improvement at hop {h} of {} ({:.1}% in), {b} basins, {e:.6}",
                    out.hops,
                    100.0 * *h as f64 / out.hops.max(1) as f64
                );
            }
        }
        deepest = deepest.min(out.best);
        if seed == 0 && out.rungs.len() > 1 {
            for (t, b, en) in &out.rungs {
                println!("      rung T={t:.3}  basins {b:>5}  energy {en:>11.4}");
            }
        }
        total_hops += out.hops;
        total_charged += ledger.spent();
        println!(
            "  seed {seed}: best {:.6}  hops {}  screened {}  charged {}  \
             basins {} ({:.1} hops each)  returned {}  \
             swaps {}/{}  paths {} improved {} gain {:.3}  \
             escape {:.3} thr {:.4} same/known/new {}/{}/{} soft {}/{} sub {}/{} lmin {:.4} climbs {} gain {:.2} radius {:.3} step {:.3} restarts {} angular {}/{} R {:.3} tabu {} vetoed {} screen {}/{} expl {} obs {} ctx {:?}  \
             relaxed {converged}/{} converged  early {early_stopped} saved {early_saved}  \
             verified {}{}",
            out.best,
            out.hops,
            out.screened_out,
            ledger.spent(),
            out.basins,
            out.hops as f64 / out.basins.max(1) as f64,
            out.returned,
            out.swaps_accepted,
            out.swaps_tried,
            out.paths,
            out.path_improvements,
            out.path_gain,
            out.escape_scale,
            out.escape_threshold,
            out.visit_counts.0,
            out.visit_counts.1,
            out.visit_counts.2,
            out.soft_crossed,
            out.soft_escapes,
            out.soft_perturbs,
            out.soft_subspaces,
            out.soft_lambda,
            out.stall_escapes,
            out.stall_escape_gain,
            out.merge_radius,
            out.mean_step,
            out.restarts,
            out.angular.1,
            out.angular.0,
            out.angular.2,
            out.tabu.0,
            out.tabu.1,
            out.screen.1,
            out.screen.0,
            out.screen.2,
            out.screen.3,
            out.contextual.0,
            converged + capped,
            verified
                .map(|(e, gmax)| format!("{e:.6} |g| {gmax:.1e}"))
                .unwrap_or_else(|| "NO STATE".into()),
            if hit { "  SOLVED" } else { "" }
        );
    }
    // Both counts, since a force budget and a hop budget are different
    // contests and the literature reports hops.
    println!(
        "{solved}/{seeds} solved, deepest {deepest:.6}   \
         mean hops {:.0}, force per hop {:.0}",
        total_hops as f64 / seeds as f64,
        total_charged as f64 / total_hops.max(1) as f64
    );
    if let Some(r) = reference {
        println!("gap to reference {:+.6}", deepest - r);
    }
}

/// Good-Turing missing mass on shared packings. Each well height is
/// `visits * w0`. n1/N is the chance the next deposit is a new packing.
/// Saturated: enough observations and few singletons — the catalog is
/// as complete as aKMC's confidence test, so the next start should
/// leave rather than redraw the same cloud.
#[cfg(feature = "bank-rpc")]
fn catalog_saturated(wells: &[(Array1<f64>, f64)], w0: f64) -> bool {
    let w0 = w0.max(1e-9);
    let mut n = 0u32;
    let mut n1 = 0u32;
    for (_, h) in wells {
        let v = (*h / w0).round().max(0.0) as u32;
        if v == 0 {
            continue;
        }
        n += v;
        if v == 1 {
            n1 += 1;
        }
    }
    n >= 12 && (n1 as f64 / n as f64) < 0.20
}

#[cfg(feature = "bank-rpc")]
fn packing_is_known(x: ArrayView1<f64>, cfg: &Config, wells: &[Array1<f64>]) -> bool {
    if wells.is_empty() {
        return false;
    }
    let s = packing_of(x, cfg);
    let merge = {
        #[cfg(feature = "featomic")]
        {
            anneal_core::featomic_hop::SOAP_PACK_MERGE
        }
        #[cfg(not(feature = "featomic"))]
        {
            0.10
        }
    };
    wells.iter().any(|w| {
        if w.len() != s.len() || s.is_empty() {
            return false;
        }
        s.iter()
            .zip(w.iter())
            .map(|(a, b)| (a - b) * (a - b))
            .sum::<f64>()
            .sqrt()
            <= merge
    })
}

/// Move the current packing mean into a hole of the shared SOAP cloud.
///
/// Cartesian 0.35 pullbacks sit inside one packing; the quench is a
/// projector onto that packing. This inverts a far point on the unit
/// sphere through `∂μ/∂x` with a SOAP step, then relaxes once. A second
/// flow runs only if that quench landed back in the archive.
#[cfg(feature = "bank-rpc")]
fn leave_known_packing<R: rand::Rng + ?Sized>(
    x: ArrayView1<f64>,
    cfg: &Config,
    wells: &[Array1<f64>],
    ledger: &mut Ledger,
    relax: &mut dyn FnMut(&mut Ledger, ArrayView1<f64>, usize) -> (f64, Array1<f64>),
    rng: &mut R,
) -> Array1<f64> {
    #[cfg(feature = "featomic")]
    {
        let rcut = 3.5 * cfg.length_scale;
        let mut y = anneal_core::featomic_hop::step_into_hole(
            x,
            wells,
            anneal_core::featomic_hop::SOAP_PACK_MERGE,
            rcut,
            cfg.species.as_deref(),
            None,
            rng,
        );
        if ledger.remaining() < 8 {
            return y;
        }
        let steps = cfg.relax_steps.min(ledger.remaining());
        let (_e, q) = relax(ledger, y.view(), steps);
        if !packing_is_known(q.view(), cfg, wells) {
            return q;
        }
        y = anneal_core::featomic_hop::step_into_hole(
            q.view(),
            wells,
            anneal_core::featomic_hop::SOAP_PACK_MERGE * 1.5,
            rcut,
            cfg.species.as_deref(),
            None,
            rng,
        );
        y
    }
    #[cfg(not(feature = "featomic"))]
    {
        let _ = (cfg, wells, ledger, relax, rng);
        x.to_owned()
    }
}

#[cfg(feature = "bank-rpc")]
fn packing_of(x: ArrayView1<f64>, cfg: &Config) -> Array1<f64> {
    #[cfg(feature = "featomic")]
    {
        anneal_core::featomic_hop::soap_cloud_mean(
            x,
            3.5 * cfg.length_scale,
            cfg.species.as_deref(),
            None,
        )
    }
    #[cfg(not(feature = "featomic"))]
    {
        let _ = (x, cfg);
        Array1::zeros(0)
    }
}

#[cfg(feature = "bank-rpc")]
fn cooperative_policy_point(current: ArrayView1<f64>, current_energy: f64) -> (Array1<f64>, f64) {
    (current.to_owned(), current_energy)
}

#[cfg(feature = "bank-rpc")]
fn fixed_probe_trial<R: rand::Rng + ?Sized>(
    current: ArrayView1<f64>,
    scale: f64,
    rng: &mut R,
) -> Option<Array1<f64>> {
    if current.is_empty() || !current.len().is_multiple_of(3) || !scale.is_finite() || scale <= 0.0
    {
        return None;
    }
    let atoms = current.len() / 3;
    let mut displacement = Array1::zeros(current.len());
    for coordinate in &mut displacement {
        let u1 = rng.random::<f64>().max(1e-12);
        let u2 = rng.random::<f64>();
        *coordinate = scale * (-2.0 * u1.ln()).sqrt() * (std::f64::consts::TAU * u2).cos();
    }
    for axis in 0..3 {
        let mean = (0..atoms)
            .map(|atom| displacement[3 * atom + axis])
            .sum::<f64>()
            / atoms as f64;
        for atom in 0..atoms {
            displacement[3 * atom + axis] -= mean;
        }
    }
    Some(current.to_owned() + displacement)
}

#[cfg(feature = "bank-rpc")]
fn boundary_crossing_trial<R: rand::Rng + ?Sized>(
    current: ArrayView1<f64>,
    crossing: &anneal_core::catalog_rpc::BoundaryCrossingRecord,
    noise_scale: f64,
    trust_radius: f64,
    rng: &mut R,
) -> Option<Array1<f64>> {
    let crossing = anneal_core::boundary_transport::ObservedCrossing::new(
        Array1::from_vec(crossing.from.clone()),
        Array1::from_vec(crossing.to.clone()),
    )
    .ok()?;
    let mut noise = Array1::zeros(current.len());
    for coordinate in &mut noise {
        let u1 = rng.random::<f64>().max(1e-12);
        let u2 = rng.random::<f64>();
        *coordinate = (-2.0 * u1.ln()).sqrt() * (std::f64::consts::TAU * u2).cos();
    }
    anneal_core::boundary_transport::boundary_transport(
        current,
        &crossing,
        noise.view(),
        &anneal_core::boundary_transport::BoundaryTransportConfig {
            noise_scale,
            trust_radius,
            frozen_coordinates: Vec::new(),
            rigid_groups: Vec::new(),
        },
    )
    .ok()
}

#[cfg(feature = "bank-rpc")]
fn population_boundary_trial(
    current: ArrayView1<f64>,
    crossing: &anneal_core::catalog_rpc::BoundaryCrossingRecord,
    noise_scale: f64,
    trust_radius: f64,
    draw: u64,
) -> Option<Array1<f64>> {
    let mut rng = <rand::rngs::StdRng as rand::SeedableRng>::seed_from_u64(draw);
    boundary_crossing_trial(current, crossing, noise_scale, trust_radius, &mut rng)
}

/// One independently budgeted LJ replica against an isolated descriptor catalog.
#[cfg(feature = "bank-rpc")]
fn run_capnp_catalog(
    cfg: &Config,
    ledger: &mut Ledger,
    relax: &mut dyn FnMut(&mut Ledger, ArrayView1<f64>, usize) -> (f64, Array1<f64>),
    grad: &mut dyn FnMut(&mut Ledger, ArrayView1<f64>) -> Option<Array1<f64>>,
    seed: u64,
    endpoint: Option<&str>,
) -> Outcome {
    use anneal_core::catalog::lj::{descriptor_space, system_signature};
    use anneal_core::catalog_policy::PolicyAction;
    use anneal_core::catalog_rpc::client::{CatalogClient, ClientConfig};
    use anneal_core::catalog_rpc::{CatalogIdentity, TransitionDestination};
    use anneal_core::cooperative_search::ledger::ChargeKind;
    use anneal_core::cooperative_search::{
        CatalogBoundaryOutcome, CatalogSampleOutcome, CooperativeRun, PolicyEvidenceOutcome,
        PolicyRole, PopulationSynchronizationOutcome, ProposalFamily, RunManifest, SliceAdoption,
        SliceQuench, SliceTrace, SliceValidation, TransitionRecordOutcome,
    };
    use anneal_core::methods::feynman_kac::{
        population_family_position, population_rejuvenation_draw,
    };
    use rand::Rng;

    let campaign = required_catalog_env("CATALOG_CAMPAIGN");
    let ensemble = required_catalog_env("CATALOG_ENSEMBLE");
    let replica = required_catalog_env("CATALOG_REPLICA")
        .parse::<u32>()
        .expect("CATALOG_REPLICA must be an unsigned integer");
    let signature = system_signature(cfg.n_points).expect("LJ catalog signature must be valid");
    let descriptor_space = descriptor_space();
    let mut cooperative = CooperativeRun::new(
        [replica],
        u64::try_from(ledger.budget()).expect("LJ budget must fit the cooperative ledger"),
    )
    .expect("single-replica local ledger must be valid");
    if let Some(endpoint) = endpoint {
        let address = endpoint
            .parse()
            .expect("CATALOG_RPC must be a host:port socket address");
        let identity = CatalogIdentity {
            campaign: campaign.clone(),
            ensemble: ensemble.clone(),
            replica,
            signature_digest: signature.digest(),
        };
        match CatalogClient::connect(address, identity, ClientConfig::default()) {
            Ok(client) => cooperative
                .attach_client(replica, client)
                .expect("configured replica must accept its catalog client"),
            Err(error) => eprintln!(
                "catalog {endpoint} unavailable ({error}); local execution remains active"
            ),
        }
    }

    let mut run_cfg = cfg.clone();
    run_cfg.budget_window = true;
    let slice = std::env::var("CATALOG_SLICE")
        .or_else(|_| std::env::var("BANK_SLICE"))
        .ok()
        .and_then(|value| value.parse::<usize>().ok())
        .unwrap_or(500)
        .max(1);
    let probe_interval = std::env::var("CATALOG_PROBE_INTERVAL")
        .ok()
        .and_then(|value| value.parse::<u64>().ok())
        .unwrap_or(8)
        .max(1);
    let probe_scale = std::env::var("CATALOG_PROBE_SCALE")
        .ok()
        .and_then(|value| value.parse::<f64>().ok())
        .filter(|value| value.is_finite() && *value > 0.0)
        .unwrap_or(0.2 * run_cfg.length_scale);
    let transport_noise = std::env::var("CATALOG_TRANSPORT_NOISE")
        .ok()
        .and_then(|value| value.parse::<f64>().ok())
        .filter(|value| value.is_finite() && *value >= 0.0)
        .unwrap_or(0.05 * run_cfg.length_scale);
    let transport_radius = std::env::var("CATALOG_TRANSPORT_RADIUS")
        .ok()
        .and_then(|value| value.parse::<f64>().ok())
        .filter(|value| value.is_finite() && *value > 0.0)
        .unwrap_or(run_cfg.length_scale * (run_cfg.n_points as f64).sqrt());
    let minimum_population_interval = slice
        .checked_mul(2)
        .and_then(|value| value.checked_add(2))
        .expect("catalog slice must admit a charged-work population interval");
    let population_interval = std::env::var("CATALOG_POPULATION_INTERVAL")
        .ok()
        .and_then(|value| value.parse::<usize>().ok())
        .unwrap_or(50_000)
        .max(minimum_population_interval);
    let last_population_threshold = ledger.budget().saturating_sub(4);
    let mut rng = <rand::rngs::StdRng as rand::SeedableRng>::seed_from_u64(seed);
    let mut bias = BasinBias::new(
        ClusterFingerprint::of_config(&run_cfg, &Array1::zeros(0)),
        run_cfg.merge_radius,
        run_cfg.bias_height,
        run_cfg.bias_gamma,
    );
    let mut current = random_cluster(run_cfg.n_points, 0.7, run_cfg.min_separation, &mut rng);
    let mut current_energy = f64::NAN;
    let mut best = f64::INFINITY;
    let mut best_state = None;
    let mut accepted_transitions = Vec::new();
    let mut hops = 0usize;
    let mut basins = 0usize;
    let mut screened_out = 0usize;
    let mut returned = 0usize;
    let mut slices = 0u64;
    let mut candidate_sequence = 0u64;
    let mut population_epoch = 0u64;
    let mut stall = 0u32;

    while ledger.remaining() > 0 {
        if ledger.remaining() <= 4 {
            while ledger.charge() {
                let _ = lj(current.view());
                cooperative
                    .record_work(replica, ChargeKind::AuxiliaryEvaluation, 1)
                    .expect("terminal potential calls must enter the cooperative ledger");
            }
            break;
        }
        let slice_start = ledger.spent();
        let slice_budget = slice.min((ledger.remaining() - 2) / 2).max(1);
        let mut slice_ledger = Ledger::new(slice_budget);
        let output = run_with_bias(
            &run_cfg,
            current.view(),
            &mut slice_ledger,
            relax,
            Some(grad),
            &mut bias,
            &mut rng,
        );
        let slice_charged = slice_ledger.spent();
        ledger.charge_many(slice_charged);
        if let Some(state) = slice_ledger.best_state.as_ref() {
            ledger.record(slice_ledger.best, state.view());
        }
        let boundaries = slice_ledger.quench_boundaries().to_vec();
        let boundary_charged = boundaries
            .iter()
            .map(|boundary| boundary.charged_calls())
            .sum::<usize>();
        for boundary in &boundaries {
            cooperative
                .record_work(
                    replica,
                    match boundary.status() {
                        QuenchStatus::Validated => ChargeKind::AcceptedQuench,
                        QuenchStatus::Rejected => ChargeKind::RejectedQuench,
                    },
                    u64::try_from(boundary.charged_calls()).expect("quench charge must fit u64"),
                )
                .expect("each quench must enter the cooperative ledger exactly once");
        }
        let auxiliary_charged = slice_charged
            .checked_sub(boundary_charged)
            .expect("quench boundaries cannot exceed slice work");
        if auxiliary_charged > 0 {
            cooperative
                .record_work(
                    replica,
                    ChargeKind::AuxiliaryEvaluation,
                    u64::try_from(auxiliary_charged).expect("auxiliary charge must fit u64"),
                )
                .expect("proposal evaluations must enter the cooperative ledger");
        } else if slice_charged == 0 {
            cooperative
                .record_work(replica, ChargeKind::LocalProposal, 0)
                .expect("uncharged local work must enter the cooperative ledger");
        }
        let adaptive_operations = adaptive_catalog_operations(
            &descriptor_space,
            &signature.atomic_numbers,
            replica,
            &mut candidate_sequence,
            seed,
            ledger.spent(),
            &output.accepted_transitions,
        );
        let mut adaptive_path_active = false;
        for operation in adaptive_operations {
            match operation {
                AdaptiveCatalogOperation::RegisterCurrent(candidate) => {
                    if !ledger.charge() {
                        break;
                    }
                    cooperative
                        .record_work(replica, ChargeKind::FreshValidation, 1)
                        .expect("adaptive source validation must enter the ledger");
                    adaptive_path_active = cooperative
                        .record_current(replica, candidate)
                        .map(|outcome| outcome != TransitionRecordOutcome::Rejected)
                        .unwrap_or(false);
                }
                AdaptiveCatalogOperation::Adopt {
                    action,
                    destination,
                } => {
                    if !adaptive_path_active {
                        continue;
                    }
                    if !ledger.charge() {
                        break;
                    }
                    cooperative
                        .record_work(replica, ChargeKind::FreshValidation, 1)
                        .expect("adaptive destination validation must enter the ledger");
                    adaptive_path_active = cooperative
                        .record_transition(
                            replica,
                            action,
                            TransitionDestination::Resolved(destination),
                            true,
                        )
                        .map(|outcome| outcome != TransitionRecordOutcome::Rejected)
                        .unwrap_or(false);
                }
            }
        }
        let hop_offset = hops;
        accepted_transitions.extend(output.accepted_transitions.iter().cloned().map(
            |mut transition| {
                transition.hop = transition
                    .hop
                    .checked_add(hop_offset)
                    .expect("aggregate hop index must fit usize");
                transition
            },
        ));
        hops += output.hops;
        basins += output.basins;
        screened_out += output.screened_out;
        returned += output.returned;
        slices += 1;

        let mut slice_trace = SliceTrace {
            slice: slices,
            current_basin: None,
            active_relation: None,
            policy_role: PolicyRole::Unavailable,
            policy_reason: "policy_unavailable",
            proposal_family: ProposalFamily::Local,
            sampled_basin: None,
            descriptor_step_norm: None,
            cartesian_step_norm: None,
            validation: if boundaries
                .iter()
                .any(|boundary| boundary.status() == QuenchStatus::Validated)
            {
                SliceValidation::Accepted
            } else if boundaries.is_empty() {
                SliceValidation::NotAttempted
            } else {
                SliceValidation::Rejected
            },
            quench: if boundaries
                .iter()
                .any(|boundary| boundary.status() == QuenchStatus::Validated)
            {
                SliceQuench::Converged
            } else if boundaries.is_empty() {
                SliceQuench::NotAttempted
            } else {
                SliceQuench::Rejected
            },
            adoption: SliceAdoption::NotAttempted,
            novelty: None,
            energy: None,
            charged_work: 0,
        };

        let local_deepened = output.best < best;
        if local_deepened {
            best = output.best;
            best_state = output.best_state.clone();
            stall = 0;
        } else {
            stall = stall.saturating_add(1);
        }
        if let Some(state) = output.final_state.as_ref() {
            current = state.clone();
            current_energy = output.final_energy;
        } else if let Some(state) = output.best_state.as_ref() {
            current = state.clone();
            current_energy = output.best;
        }
        let (policy_state, policy_energy) =
            cooperative_policy_point(current.view(), current_energy);

        for boundary in boundaries
            .iter()
            .filter(|boundary| boundary.status() == QuenchStatus::Validated)
        {
            candidate_sequence = candidate_sequence
                .checked_add(1)
                .expect("candidate sequence must fit u64");
            cooperative
                .record_work(replica, ChargeKind::DescriptorEvaluation, 0)
                .expect("descriptor work must enter the cooperative ledger");
            let candidate = lj_catalog_candidate(
                &descriptor_space,
                &signature.atomic_numbers,
                replica,
                candidate_sequence,
                seed,
                ledger.spent(),
                boundary.energy(),
                boundary.state(),
                boundary
                    .gradient()
                    .expect("validated quench must retain its fresh gradient"),
            )
            .expect("validated LJ quench must produce a catalog candidate");
            assert!(
                ledger.charge(),
                "slice reservation must cover every receiving validation"
            );
            cooperative
                .record_work(replica, ChargeKind::FreshValidation, 1)
                .expect("receiving validation must enter the cooperative ledger");
            let _ = cooperative.offer_candidate(replica, candidate);
        }

        if slices.is_multiple_of(probe_interval) && ledger.remaining() >= 4 && ledger.charge() {
            let (source_energy, source_gradient) = lj(policy_state.view());
            cooperative
                .record_work(replica, ChargeKind::FreshValidation, 1)
                .expect("probe source validation must enter the cooperative ledger");
            candidate_sequence = candidate_sequence
                .checked_add(1)
                .expect("candidate sequence must fit u64");
            cooperative
                .record_work(replica, ChargeKind::DescriptorEvaluation, 0)
                .expect("probe source descriptor work must enter the cooperative ledger");
            let source_candidate = lj_catalog_candidate(
                &descriptor_space,
                &signature.atomic_numbers,
                replica,
                candidate_sequence,
                seed,
                ledger.spent(),
                source_energy,
                policy_state.view(),
                source_gradient.view(),
            );
            if let Some(source_candidate) = source_candidate
                && ledger.remaining() >= 3
                && ledger.charge()
            {
                cooperative
                    .record_work(replica, ChargeKind::FreshValidation, 1)
                    .expect("probe source receiving validation must enter the ledger");
                let registered = cooperative
                    .record_current(replica, source_candidate)
                    .expect("probe source registration must preserve local execution");
                if registered == TransitionRecordOutcome::Recorded
                    && let Some(trial) =
                        fixed_probe_trial(policy_state.view(), probe_scale, &mut rng)
                {
                    let probe_start = ledger.spent();
                    let (probe_energy, probe_state) =
                        relax(ledger, trial.view(), run_cfg.relax_steps);
                    let probe_gradient = grad(ledger, probe_state.view());
                    let probe_charged = ledger.spent().saturating_sub(probe_start);
                    candidate_sequence = candidate_sequence
                        .checked_add(1)
                        .expect("candidate sequence must fit u64");
                    let destination = probe_gradient.as_ref().and_then(|gradient| {
                        lj_catalog_candidate(
                            &descriptor_space,
                            &signature.atomic_numbers,
                            replica,
                            candidate_sequence,
                            seed,
                            ledger.spent(),
                            probe_energy,
                            probe_state.view(),
                            gradient.view(),
                        )
                    });
                    cooperative
                        .record_work(
                            replica,
                            if destination.is_some() {
                                ChargeKind::AcceptedQuench
                            } else {
                                ChargeKind::RejectedQuench
                            },
                            u64::try_from(probe_charged).expect("probe quench charge must fit u64"),
                        )
                        .expect("fixed-probe quench must enter the cooperative ledger");
                    match destination {
                        Some(candidate) if ledger.remaining() > 0 && ledger.charge() => {
                            cooperative
                                .record_work(replica, ChargeKind::FreshValidation, 1)
                                .expect("probe receiving validation must enter the ledger");
                            let _ = cooperative.record_transition(
                                replica,
                                "probe",
                                TransitionDestination::Resolved(candidate),
                                false,
                            );
                        }
                        _ => {
                            let _ = cooperative.record_transition(
                                replica,
                                "probe",
                                TransitionDestination::Unresolved,
                                false,
                            );
                        }
                    }
                }
            }
        }

        let Ok(descriptor) =
            descriptor_space.describe(policy_state.view(), Some(&signature.atomic_numbers))
        else {
            slice_trace.policy_reason = "descriptor_rejected";
            slice_trace.energy = current_energy.is_finite().then_some(current_energy);
            slice_trace.charged_work =
                u64::try_from(ledger.spent() - slice_start).expect("slice charge must fit u64");
            cooperative
                .record_slice(replica, slice_trace)
                .expect("descriptor failure must retain a complete slice diagnostic");
            continue;
        };
        cooperative
            .record_work(replica, ChargeKind::DescriptorEvaluation, 0)
            .expect("policy descriptor work must enter the cooperative ledger");
        let policy_energy = if policy_energy.is_finite() {
            policy_energy
        } else {
            slice_trace.policy_reason = "no_valid_energy";
            slice_trace.charged_work =
                u64::try_from(ledger.spent() - slice_start).expect("slice charge must fit u64");
            cooperative
                .record_slice(replica, slice_trace)
                .expect("invalid-energy slice must retain a complete diagnostic");
            continue;
        };
        let policy_outcome = cooperative
            .policy_input(
                replica,
                descriptor.values().to_vec(),
                policy_energy,
                stall,
                local_deepened,
            )
            .expect("coordinator policy evidence must preserve local invariants");
        let input = match policy_outcome {
            PolicyEvidenceOutcome::Remote(input) => input,
            PolicyEvidenceOutcome::Rejected => {
                slice_trace.policy_reason = "policy_rejected";
                slice_trace.energy = Some(policy_energy);
                slice_trace.charged_work =
                    u64::try_from(ledger.spent() - slice_start).expect("slice charge must fit u64");
                cooperative
                    .record_slice(replica, slice_trace)
                    .expect("rejected policy evidence must retain a complete diagnostic");
                continue;
            }
            PolicyEvidenceOutcome::LocalFallback => {
                slice_trace.policy_reason = "rpc_fallback";
                slice_trace.energy = Some(policy_energy);
                slice_trace.charged_work =
                    u64::try_from(ledger.spent() - slice_start).expect("slice charge must fit u64");
                cooperative
                    .record_slice(replica, slice_trace)
                    .expect("RPC fallback must retain a complete slice diagnostic");
                continue;
            }
            PolicyEvidenceOutcome::SharingDisabled => {
                slice_trace.policy_reason = "sharing_disabled";
                slice_trace.energy = Some(policy_energy);
                slice_trace.charged_work =
                    u64::try_from(ledger.spent() - slice_start).expect("slice charge must fit u64");
                cooperative
                    .record_slice(replica, slice_trace)
                    .expect("control-arm execution must retain a complete slice diagnostic");
                continue;
            }
        };
        let policy_trace = cooperative
            .events()
            .last()
            .and_then(|event| event.policy)
            .expect("remote policy evidence must be attached to its snapshot event");
        slice_trace.current_basin = policy_trace.local_basin;
        slice_trace.active_relation = Some(policy_trace.relation);
        slice_trace.novelty = Some(policy_trace.novelty);
        slice_trace.energy = Some(policy_energy);
        let decision = cooperative
            .decide(replica, input)
            .expect("pure policy decision must name the configured replica");
        slice_trace.policy_role = match decision.action {
            PolicyAction::ContinueLocal => PolicyRole::Local,
            PolicyAction::Exploit { .. } => PolicyRole::Exploit,
            PolicyAction::Explore => PolicyRole::Explore,
            PolicyAction::Leave => PolicyRole::Leave,
        };
        slice_trace.policy_reason = decision.reason.code();
        match decision.action {
            PolicyAction::ContinueLocal => {}
            PolicyAction::Exploit { win_only } => {
                slice_trace.proposal_family = ProposalFamily::CatalogSample;
                if let CatalogSampleOutcome::Candidate(candidate) = cooperative
                    .sample_candidate(replica, rng.random())
                    .expect("catalog sampling must preserve local execution")
                    && ledger.remaining() > 0
                    && ledger.charge()
                {
                    slice_trace.sampled_basin = candidate.census_basin;
                    slice_trace.descriptor_step_norm =
                        Some(vector_distance(descriptor.values(), &candidate.descriptor));
                    slice_trace.cartesian_step_norm = Some(vector_distance(
                        policy_state.as_slice().expect("LJ state is contiguous"),
                        &candidate.coordinates,
                    ));
                    cooperative
                        .record_work(replica, ChargeKind::FreshValidation, 1)
                        .expect("sample validation must enter the cooperative ledger");
                    if let Some((energy, coordinates)) = validate_sampled_lj(
                        &candidate,
                        &descriptor_space,
                        &signature.atomic_numbers,
                    ) {
                        slice_trace.validation = SliceValidation::Accepted;
                        if !win_only || energy < best {
                            slice_trace.adoption = SliceAdoption::Adopted;
                            current = coordinates.clone();
                            current_energy = energy;
                            if energy < best {
                                best = energy;
                                best_state = Some(coordinates);
                                stall = 0;
                            }
                        } else {
                            slice_trace.adoption = SliceAdoption::NotImproved;
                        }
                    } else {
                        slice_trace.validation = SliceValidation::Rejected;
                        slice_trace.adoption = SliceAdoption::Rejected;
                    }
                } else {
                    slice_trace.adoption = SliceAdoption::Rejected;
                }
            }
            PolicyAction::Explore | PolicyAction::Leave => {
                slice_trace.proposal_family = ProposalFamily::BoundaryTransport;
                if let CatalogBoundaryOutcome::Crossing(crossing) = cooperative
                    .boundary_crossing(replica, descriptor.values().to_vec(), rng.random())
                    .expect("boundary-crossing access must preserve local execution")
                {
                    slice_trace.sampled_basin = Some(crossing.destination_basin);
                    cooperative
                        .record_work(replica, ChargeKind::RemoteProposal, 0)
                        .expect("remote proposal work must enter the cooperative ledger");
                    if let Some(proposed) = boundary_crossing_trial(
                        policy_state.view(),
                        &crossing,
                        transport_noise,
                        transport_radius,
                        &mut rng,
                    ) {
                        slice_trace.cartesian_step_norm = Some(vector_distance(
                            policy_state.as_slice().expect("LJ state is contiguous"),
                            proposed.as_slice().expect("LJ proposal is contiguous"),
                        ));
                        slice_trace.adoption = SliceAdoption::Adopted;
                        current = proposed;
                        current_energy = f64::NAN;
                    } else {
                        slice_trace.adoption = SliceAdoption::Rejected;
                    }
                } else {
                    slice_trace.adoption = SliceAdoption::Rejected;
                }
            }
        }

        let population_threshold = usize::try_from(
            population_epoch
                .checked_add(1)
                .and_then(|epoch| {
                    epoch.checked_mul(
                        u64::try_from(population_interval)
                            .expect("population interval must fit u64"),
                    )
                })
                .expect("population charged-work threshold must fit u64"),
        )
        .expect("population charged-work threshold must fit usize");
        let mut population_representative = None;
        if population_threshold <= last_population_threshold
            && ledger.spent() >= population_threshold
            && ledger.remaining() >= 3
            && ledger.charge()
        {
            let (incumbent_energy, incumbent_gradient) = lj(policy_state.view());
            cooperative
                .record_work(replica, ChargeKind::FreshValidation, 1)
                .expect("incumbent validation must enter the cooperative ledger");
            candidate_sequence = candidate_sequence
                .checked_add(1)
                .expect("candidate sequence must fit u64");
            cooperative
                .record_work(replica, ChargeKind::DescriptorEvaluation, 0)
                .expect("incumbent descriptor work must enter the cooperative ledger");
            if let Some(candidate) = lj_catalog_candidate(
                &descriptor_space,
                &signature.atomic_numbers,
                replica,
                candidate_sequence,
                seed,
                ledger.spent(),
                incumbent_energy,
                policy_state.view(),
                incumbent_gradient.view(),
            ) {
                assert!(
                    ledger.charge(),
                    "population threshold must reserve incumbent receiving validation"
                );
                cooperative
                    .record_work(replica, ChargeKind::FreshValidation, 1)
                    .expect("incumbent receiving validation must enter the cooperative ledger");
                let _ = cooperative.offer_candidate(replica, candidate.clone());
                population_representative = Some(candidate);
            }
        }
        if population_threshold <= last_population_threshold
            && ledger.spent() >= population_threshold
            && let Some(representative) = population_representative.clone()
        {
            assert!(
                ledger.charge(),
                "slice reservation must cover population receiving validation"
            );
            cooperative
                .record_work(replica, ChargeKind::FreshValidation, 1)
                .expect("population validation must enter the cooperative ledger");
            let mut population = cooperative
                .submit_population(replica, population_epoch, representative)
                .expect("population submission must preserve cooperative invariants");
            while matches!(population, PopulationSynchronizationOutcome::Pending { .. }) {
                std::thread::sleep(std::time::Duration::from_millis(20));
                population = cooperative
                    .poll_population(replica, population_epoch)
                    .expect("population polling must preserve cooperative invariants");
            }
            match population {
                PopulationSynchronizationOutcome::Ready { parent, plan } => {
                    let family =
                        population_family_position(&plan.destinations, &plan.parents, replica)
                            .expect("validated population plan must address this replica");
                    let parent_coordinates = Array1::from_vec(parent.coordinates.clone());
                    let (next_state, next_energy) = if family.family_size() > 1 {
                        let draw = population_rejuvenation_draw(
                            seed,
                            population_epoch,
                            replica,
                            family.ordinal(),
                        );
                        match cooperative
                            .boundary_crossing(replica, parent.descriptor.clone(), draw)
                            .expect("population rejuvenation must preserve local execution")
                        {
                            CatalogBoundaryOutcome::Crossing(crossing) => {
                                cooperative
                                    .record_work(replica, ChargeKind::RemoteProposal, 0)
                                    .expect("rejuvenation proposal must enter the ledger");
                                match population_boundary_trial(
                                    parent_coordinates.view(),
                                    &crossing,
                                    transport_noise,
                                    transport_radius,
                                    draw,
                                ) {
                                    Some(proposed) => (proposed, f64::NAN),
                                    None => (parent_coordinates, parent.energy),
                                }
                            }
                            _ => (parent_coordinates, parent.energy),
                        }
                    } else {
                        (parent_coordinates, parent.energy)
                    };
                    current = next_state;
                    current_energy = next_energy;
                    population_epoch = population_epoch
                        .checked_add(1)
                        .expect("population epoch must fit u64");
                }
                PopulationSynchronizationOutcome::Rejected => {
                    panic!("coordinator rejected a catalog-validated population representative")
                }
                PopulationSynchronizationOutcome::LocalFallback
                | PopulationSynchronizationOutcome::SharingDisabled => {}
                PopulationSynchronizationOutcome::Pending { .. } => unreachable!(
                    "population polling loop exits only after the barrier leaves pending state"
                ),
            }
        }
        slice_trace.energy = current_energy.is_finite().then_some(current_energy);
        slice_trace.charged_work =
            u64::try_from(ledger.spent() - slice_start).expect("slice charge must fit u64");
        cooperative
            .record_slice(replica, slice_trace)
            .expect("every local slice must retain one complete diagnostic");
        println!(
            "      catalog slice {slices} spent {} best {best:.6} stall {stall}",
            ledger.spent()
        );
        let _ = io::stdout().flush();
    }

    let trace = cooperative.json_lines(&RunManifest {
        campaign,
        ensemble,
        sharing: endpoint.is_some(),
    });
    if let Ok(path) = std::env::var("CATALOG_TRACE") {
        let mut output = std::fs::OpenOptions::new()
            .create(true)
            .append(true)
            .open(path)
            .expect("CATALOG_TRACE must be writable");
        output
            .write_all(trace.as_bytes())
            .expect("catalog trace write must succeed");
    } else {
        eprint!("{trace}");
    }
    Outcome {
        best,
        best_state,
        final_state: Some(current),
        final_energy: current_energy,
        accepted_transitions,
        hops,
        basins,
        charged: ledger.spent(),
        screened_out,
        returned,
        ..Outcome::default()
    }
}

#[cfg(feature = "bank-rpc")]
fn vector_distance(left: &[f64], right: &[f64]) -> f64 {
    assert_eq!(left.len(), right.len(), "diagnostic vectors must align");
    left.iter()
        .zip(right)
        .map(|(left, right)| {
            let difference = left - right;
            difference * difference
        })
        .sum::<f64>()
        .sqrt()
}

#[cfg(feature = "bank-rpc")]
fn required_catalog_env(name: &str) -> String {
    std::env::var(name).unwrap_or_else(|_| panic!("{name} is required with CATALOG_RPC"))
}

#[cfg(feature = "bank-rpc")]
fn lj_catalog_candidate(
    descriptor_space: &anneal_core::descriptor_space::DescriptorSpace,
    species: &[u32],
    replica: u32,
    event_sequence: u64,
    seed: u64,
    charged_work: usize,
    energy: f64,
    coordinates: ArrayView1<f64>,
    gradient: ArrayView1<f64>,
) -> Option<anneal_core::catalog_rpc::CatalogCandidate> {
    let gradient_norm = euclidean_gradient_norm(
        gradient
            .as_slice()
            .expect("validated LJ gradient is contiguous"),
    );
    if !energy.is_finite() || !gradient_norm.is_finite() || gradient_norm > 1e-5 {
        return None;
    }
    let descriptor = descriptor_space.describe(coordinates, Some(species)).ok()?;
    Some(anneal_core::catalog_rpc::CatalogCandidate {
        producer_replica: replica,
        coordinates: coordinates.to_vec(),
        cell: None,
        energy,
        forces: gradient.iter().map(|value| -*value).collect(),
        gradient_norm,
        descriptor: descriptor.values().to_vec(),
        descriptor_schema_version: descriptor.schema_version(),
        quench_converged: true,
        charged_work: u64::try_from(charged_work).ok()?,
        event_sequence,
        seed,
        census_basin: None,
    })
}

#[cfg(feature = "bank-rpc")]
#[allow(clippy::too_many_arguments)]
fn lj_transition_candidates(
    descriptor_space: &anneal_core::descriptor_space::DescriptorSpace,
    species: &[u32],
    replica: u32,
    source_sequence: u64,
    destination_sequence: u64,
    seed: u64,
    charged_work: usize,
    transition: &AcceptedTransition,
) -> Option<(
    anneal_core::catalog_rpc::CatalogCandidate,
    anneal_core::catalog_rpc::CatalogCandidate,
)> {
    if !transition.validated {
        return None;
    }
    let source = lj_catalog_candidate(
        descriptor_space,
        species,
        replica,
        source_sequence,
        seed,
        charged_work,
        transition.from_energy,
        transition.from_state.view(),
        transition.from_gradient.as_ref()?.view(),
    )?;
    let destination = lj_catalog_candidate(
        descriptor_space,
        species,
        replica,
        destination_sequence,
        seed,
        charged_work,
        transition.to_energy,
        transition.to_state.view(),
        transition.to_gradient.as_ref()?.view(),
    )?;
    Some((source, destination))
}

#[cfg(feature = "bank-rpc")]
enum AdaptiveCatalogOperation {
    RegisterCurrent(anneal_core::catalog_rpc::CatalogCandidate),
    Adopt {
        action: String,
        destination: anneal_core::catalog_rpc::CatalogCandidate,
    },
}

#[cfg(feature = "bank-rpc")]
#[allow(clippy::too_many_arguments)]
fn adaptive_catalog_operations(
    descriptor_space: &anneal_core::descriptor_space::DescriptorSpace,
    species: &[u32],
    replica: u32,
    candidate_sequence: &mut u64,
    seed: u64,
    charged_work: usize,
    transitions: &[AcceptedTransition],
) -> Vec<AdaptiveCatalogOperation> {
    let mut operations = Vec::new();
    let mut registered_state: Option<Array1<f64>> = None;
    for transition in transitions {
        if !transition.validated
            || transition.from_gradient.is_none()
            || transition.to_gradient.is_none()
        {
            registered_state = None;
            continue;
        }
        let continues_registered_path = registered_state
            .as_ref()
            .is_some_and(|state| state == &transition.from_state);
        if !continues_registered_path {
            let source_sequence = candidate_sequence
                .checked_add(1)
                .expect("candidate sequence must fit u64");
            let Some(source) = lj_catalog_candidate(
                descriptor_space,
                species,
                replica,
                source_sequence,
                seed,
                charged_work,
                transition.from_energy,
                transition.from_state.view(),
                transition
                    .from_gradient
                    .as_ref()
                    .expect("validated transition source gradient checked")
                    .view(),
            ) else {
                registered_state = None;
                continue;
            };
            *candidate_sequence = source_sequence;
            operations.push(AdaptiveCatalogOperation::RegisterCurrent(source));
        }
        let destination_sequence = candidate_sequence
            .checked_add(1)
            .expect("candidate sequence must fit u64");
        let Some(destination) = lj_catalog_candidate(
            descriptor_space,
            species,
            replica,
            destination_sequence,
            seed,
            charged_work,
            transition.to_energy,
            transition.to_state.view(),
            transition
                .to_gradient
                .as_ref()
                .expect("validated transition destination gradient checked")
                .view(),
        ) else {
            registered_state = None;
            continue;
        };
        *candidate_sequence = destination_sequence;
        operations.push(AdaptiveCatalogOperation::Adopt {
            action: transition.action.clone(),
            destination,
        });
        registered_state = Some(transition.to_state.clone());
    }
    operations
}

#[cfg(feature = "bank-rpc")]
fn validate_sampled_lj(
    candidate: &anneal_core::catalog_rpc::CatalogCandidate,
    descriptor_space: &anneal_core::descriptor_space::DescriptorSpace,
    species: &[u32],
) -> Option<(f64, Array1<f64>)> {
    if candidate.coordinates.len() != 3 * species.len()
        || candidate.cell.is_some()
        || !candidate.quench_converged
        || minimum_pair_distance(&candidate.coordinates) < 0.5
    {
        return None;
    }
    let coordinates = Array1::from_vec(candidate.coordinates.clone());
    let descriptor = descriptor_space
        .describe(coordinates.view(), Some(species))
        .ok()?
        .values()
        .to_vec();
    if descriptor.len() != candidate.descriptor.len()
        || descriptor
            .iter()
            .zip(&candidate.descriptor)
            .any(|(left, right)| (left - right).abs() > 1e-10)
    {
        return None;
    }
    let (energy, gradient) = lj(coordinates.view());
    let gradient_norm = euclidean_gradient_norm(
        gradient
            .as_slice()
            .expect("fresh LJ gradient is contiguous"),
    );
    let energy_tolerance = 1e-9 + 1e-10 * energy.abs().max(candidate.energy.abs());
    if (energy - candidate.energy).abs() > energy_tolerance
        || gradient_norm > 1e-5
        || candidate.forces.len() != gradient.len()
        || candidate
            .forces
            .iter()
            .zip(gradient.iter())
            .any(|(force, gradient)| (force + gradient).abs() > 1e-9)
    {
        return None;
    }
    Some((energy, coordinates))
}

#[cfg(feature = "bank-rpc")]
fn minimum_pair_distance(coordinates: &[f64]) -> f64 {
    let mut minimum = f64::INFINITY;
    for first in 0..coordinates.len() / 3 {
        for second in first + 1..coordinates.len() / 3 {
            let squared = (0..3)
                .map(|axis| {
                    let difference = coordinates[3 * first + axis] - coordinates[3 * second + axis];
                    difference * difference
                })
                .sum::<f64>();
            minimum = minimum.min(squared.sqrt());
        }
    }
    minimum
}

/// One HQ chain against the Cap'n Proto bank: slice, offer, deposit, repeat.
#[cfg(feature = "bank-rpc")]
fn run_capnp_bank(
    cfg: &Config,
    ledger: &mut Ledger,
    relax: &mut dyn FnMut(&mut Ledger, ArrayView1<f64>, usize) -> (f64, Array1<f64>),
    grad: &mut dyn FnMut(&mut Ledger, ArrayView1<f64>) -> Option<Array1<f64>>,
    seed: u64,
    sock: &str,
) -> Outcome {
    use anneal_core::bank_rpc::BankClient;
    use anneal_core::diversity::DiversityAnnealer;
    use rand::Rng;
    let mut cfg = cfg.clone();
    cfg.budget_window = true;
    let cfg = &cfg;
    // Kubelet: the walk owns the hop. The bank is optional. A refused
    // connect (worker moved, login bank dead) is a solo leftover run.
    let mut client = match BankClient::connect(sock) {
        Ok(c) => Some(c),
        Err(e) => {
            println!("  bank {sock} down ({e}); own walk");
            None
        }
    };
    let sync_every = std::env::var("BANK_SYNC")
        .ok()
        .and_then(|v| v.parse().ok())
        .unwrap_or(8usize)
        .max(1);
    let mut bias = BasinBias::new(
        ClusterFingerprint::of_config(cfg, &Array1::zeros(0)),
        cfg.merge_radius,
        cfg.bias_height,
        cfg.bias_gamma,
    );
    let slice = std::env::var("BANK_SLICE")
        .ok()
        .and_then(|v| v.parse().ok())
        .unwrap_or(500);
    let mut rng = <rand::rngs::StdRng as rand::SeedableRng>::seed_from_u64(seed);
    let mut best = f64::INFINITY;
    let mut best_state: Option<Array1<f64>> = None;
    let mut hops = 0usize;
    let mut basins = 0usize;
    let mut screened_out = 0usize;
    let mut returned = 0usize;
    let mut slices = 0usize;
    let mut null_starts = 0usize;
    let total = ledger.remaining();
    let mut schedule: Option<DiversityAnnealer> = None;
    let mut well_pairs: Vec<(Array1<f64>, f64)> = Vec::new();
    let mut catalog_best = f64::INFINITY;
    let mut catalog_size = 0usize;
    let mut stall = 0u32;
    while ledger.remaining() > 0 {
        let progress = 1.0 - ledger.remaining() as f64 / total.max(1) as f64;
        let pull = slices == 0 || slices.is_multiple_of(sync_every);
        if pull && client.is_none() {
            client = BankClient::connect(sock).ok();
        }
        if pull && let Some(c) = client.as_mut() {
            match c.snapshot() {
                Ok(s) => {
                    for (soap, h) in &s.wells {
                        bias.import_well(soap.clone(), *h);
                    }
                    #[cfg(feature = "featomic")]
                    anneal_core::featomic_hop::set_packing_archive(
                        s.wells.iter().map(|(soap, _)| soap.clone()).collect(),
                    );
                    well_pairs = s.wells.clone();
                    catalog_size = s.size as usize;
                    if !s.energies.is_empty() {
                        catalog_best = s.energies.iter().copied().fold(f64::INFINITY, f64::min);
                    }
                    if s.size >= 2 {
                        let sched = schedule.get_or_insert_with(|| {
                            let floor = {
                                #[cfg(feature = "featomic")]
                                {
                                    anneal_core::featomic_hop::SOAP_PACK_MERGE
                                }
                                #[cfg(not(feature = "featomic"))]
                                {
                                    0.10
                                }
                            };
                            DiversityAnnealer::from_initial(s.dcut.max(floor))
                                .with_final_fraction(0.4)
                        });
                        let progress = 1.0 - ledger.remaining() as f64 / total.max(1) as f64;
                        if c.set_dcut(sched.threshold(progress)).is_err() {
                            client = None;
                        }
                    }
                }
                Err(_) => client = None,
            }
        }
        let wells: Vec<Array1<f64>> = well_pairs.iter().map(|(w, _)| w.clone()).collect();
        let mut start = if let Some(bx) = best_state.as_ref() {
            bx.clone()
        } else {
            random_cluster(cfg.n_points, 0.7, cfg.min_separation, &mut rng)
        };
        let mine = packing_of(start.view(), cfg);
        let gap = {
            #[cfg(feature = "featomic")]
            {
                anneal_core::featomic_hop::SOAP_PACK_GAP
            }
            #[cfg(not(feature = "featomic"))]
            {
                0.15
            }
        };
        let sat = catalog_saturated(&well_pairs, cfg.bias_height);
        let on_known = packing_is_known(start.view(), cfg, &wells);
        let swarm = anneal_core::swarm::policy(
            progress,
            best,
            best,
            catalog_best,
            catalog_size,
            sat,
            on_known && sat,
            stall,
        );
        if pull
            && swarm.pull
            && let Some(c) = client.as_mut()
        {
            match c.sample(rng.random()) {
                Ok(Some((e, x))) if x.len() == 3 * cfg.n_points => {
                    let theirs = packing_of(x.view(), cfg);
                    let dist = if mine.is_empty() || mine.len() != theirs.len() {
                        f64::INFINITY
                    } else {
                        mine.iter()
                            .zip(theirs.iter())
                            .map(|(a, b)| (a - b) * (a - b))
                            .sum::<f64>()
                            .sqrt()
                    };
                    let take = if swarm.win_only {
                        e < best - 0.05
                    } else {
                        e < best - 0.05 || (dist > gap && e < best + 1.0)
                    };
                    if take {
                        if e < best {
                            best = e;
                            best_state = Some(x.clone());
                        }
                        start = x;
                    }
                }
                Ok(_) => {}
                Err(_) => client = None,
            }
        }
        if swarm.leave {
            null_starts += 1;
            start = leave_known_packing(start.view(), cfg, &wells, ledger, relax, &mut rng);
        }
        let mut slice_led = Ledger::new(slice.min(ledger.remaining()));
        let out = run_with_bias(
            cfg,
            start.view(),
            &mut slice_led,
            relax,
            Some(grad),
            &mut bias,
            &mut rng,
        );
        ledger.charge_many(slice_led.spent());
        if let Some(st) = slice_led.best_state.as_ref() {
            ledger.record(slice_led.best, st.view());
        }
        hops += out.hops;
        basins += out.basins;
        screened_out += out.screened_out;
        returned += out.returned;
        slices += 1;
        let improved = out.best < best;
        if improved {
            best = out.best;
            best_state = out.best_state.clone();
            stall = 0;
        } else {
            stall = stall.saturating_add(1);
        }
        // Publish a win only. Depositing every slice is 48 chains
        // IRA-matching on the login node.
        if improved
            && let Some(st) = out.best_state.as_ref()
            && let Some(c) = client.as_mut()
        {
            let soap = packing_of(st.view(), cfg);
            if c.offer(out.best, st.view(), soap.view()).is_err()
                || c.deposit(soap.view(), cfg.bias_height).is_err()
            {
                client = None;
            }
        }
        println!(
            "      slice {slices} spent {} best {best:.6} null {null_starts}",
            ledger.spent()
        );
        let _ = std::io::stdout().flush();
    }
    println!(
        "      capnp bank: {slices} slices, {null_starts} archive-null starts, best {best:.6}"
    );
    Outcome {
        best,
        best_state,
        hops,
        basins,
        screened_out,
        returned,
        ..Outcome::default()
    }
}
