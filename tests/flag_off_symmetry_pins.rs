// Copyright (c) anneal developers
//! Shell-core and dual-core orbit completion with `detected_symmetry` off,
//! pinned by hash.
//!
//! The candidates of [`OrbitCore::Shells`] and [`OrbitCore::Both`], and short
//! walks of the dual-core cluster preset with each core, are the ones of
//! develop at `91fa4bbc`, where these hashes were recorded. Detected
//! symmetry is opt-in, so leaving it off must not move them.
//!
//! The walks force rgmin's scalar dots, as `lj_trajectory_pin` does.

use anneal_core::methods::cluster_hopping::{
    Config, Ledger, OrbitCore, random_cluster_in_radius, run_with_gradient,
};
use anneal_core::methods::warm_lbfgs::WarmLbfgs;
use anneal_core::potentials::PairPotential;
use ndarray::{Array1, ArrayView1};
use rand::rngs::StdRng;
use rand::{Rng, SeedableRng};
use sha2::{Digest, Sha256};

fn update(hasher: &mut Sha256, values: ArrayView1<f64>) {
    hasher.update((values.len() as u64).to_le_bytes());
    for value in values.iter() {
        hasher.update(value.to_le_bytes());
    }
}

fn quench(potential: &PairPotential, x: ArrayView1<f64>, iterations: usize) -> Array1<f64> {
    let mut optimiser = WarmLbfgs::default();
    let (_, relaxed, _) = optimiser.minimize(x, iterations, |point| {
        Some(potential.value_and_gradient(point))
    });
    relaxed
}

/// The 13-point icosahedron at the Lennard-Jones contact, each coordinate
/// moved by up to `noise`.
fn icosahedron(noise: f64, rng: &mut StdRng) -> Array1<f64> {
    let phi = (1.0 + 5.0_f64.sqrt()) / 2.0;
    // Host pow is not one function on glibc and libSystem. The edge is
    // the same portable call the soft math path uses.
    let edge = libm::pow(2.0, 1.0 / 6.0);
    let radius = edge * (phi * phi + 1.0).sqrt() / 2.0;
    let scale = radius / (1.0 + phi * phi).sqrt();
    let mut coords = vec![0.0, 0.0, 0.0];
    for s1 in [-1.0, 1.0] {
        for s2 in [-1.0, 1.0] {
            coords.extend([0.0, s1 * scale, s2 * phi * scale]);
            coords.extend([s1 * scale, s2 * phi * scale, 0.0]);
            coords.extend([s2 * phi * scale, 0.0, s1 * scale]);
        }
    }
    let mut x = Array1::from_vec(coords);
    x.mapv_inplace(|value| value + noise * (2.0 * rng.random::<f64>() - 1.0));
    x
}

/// A core of `rings` stacked `order`-fold rings, the centre atom when
/// `centred`, and `outer` points at random directions well outside it.
///
/// An order above six is outside the `[2, 3, 4, 5, 6]` scan of develop, so
/// these cores read a different group if that scan widens.
fn ring_core(
    order: usize,
    rings: usize,
    centred: bool,
    outer: usize,
    rng: &mut StdRng,
) -> Array1<f64> {
    let mut coords = Vec::new();
    if centred {
        coords.extend([0.0, 0.0, 0.0]);
    }
    for ring in 0..rings {
        let height = 0.9 * (ring as f64 - (rings as f64 - 1.0) / 2.0);
        let phase = std::f64::consts::PI * ring as f64 / order as f64;
        for step in 0..order {
            let angle = std::f64::consts::TAU * step as f64 / order as f64 + phase;
            // Host sine and cosine of one angle differ between glibc and
            // libSystem. These rings use the portable calls, as the
            // rotation that places a completed orbit does.
            coords.extend([1.12 * libm::cos(angle), 1.12 * libm::sin(angle), height]);
        }
    }
    for _ in 0..outer {
        let z = 2.0 * rng.random::<f64>() - 1.0;
        let azimuth = std::f64::consts::TAU * rng.random::<f64>();
        let radius = 2.6 + 0.3 * rng.random::<f64>();
        let ring = (1.0 - z * z).sqrt();
        coords.extend([
            radius * ring * libm::cos(azimuth),
            radius * ring * libm::sin(azimuth),
            radius * z,
        ]);
    }
    Array1::from_vec(coords)
}

/// Structures to complete: noisy icosahedra, cores with axes above order
/// six, and quenched random clusters.
fn structures() -> Vec<Array1<f64>> {
    let mut rng = StdRng::seed_from_u64(179);
    let mut out = Vec::new();
    for noise in [0.0, 0.02, 0.06] {
        out.push(icosahedron(noise, &mut rng));
    }
    for (order, rings, centred, outer) in [(7, 1, true, 8), (8, 1, true, 10), (7, 2, false, 9)] {
        out.push(ring_core(order, rings, centred, outer, &mut rng));
    }
    for (n, iterations) in [(13, 400), (19, 400), (26, 600), (38, 600), (38, 50)] {
        let cfg = Config::recommended(n);
        let potential = PairPotential::lennard_jones(n);
        let start = random_cluster_in_radius(n, cfg.start_radius(), cfg.min_separation, &mut rng);
        out.push(quench(&potential, start.view(), iterations));
    }
    out
}

fn candidate_hash(core: fn(f64) -> OrbitCore) -> (String, usize) {
    let mut hasher = Sha256::new();
    let mut produced = 0;
    for x in structures() {
        let n = x.len() / 3;
        let cfg = Config::for_cluster_dual_core(n);
        assert!(!cfg.detected_symmetry);
        let candidates = anneal_core::packing::orbit_candidates(
            x.view(),
            n,
            core(cfg.length_scale),
            cfg.symmetry_tolerance,
            cfg.symmetry_merge_radius,
            cfg.symmetrise_core_fraction,
            cfg.min_separation,
        );
        hasher.update((candidates.len() as u64).to_le_bytes());
        for y in &candidates {
            update(&mut hasher, y.view());
        }
        produced += candidates.len();
    }
    (format!("{:x}", hasher.finalize()), produced)
}

fn shells(length: f64) -> OrbitCore {
    OrbitCore::Shells(anneal_core::symmetrise::ShellCore::scaled(length))
}

fn both(length: f64) -> OrbitCore {
    OrbitCore::Both(anneal_core::symmetrise::ShellCore::scaled(length))
}

fn walk_hash(
    atoms: usize,
    seed: u64,
    budget: usize,
    core: fn(f64) -> OrbitCore,
) -> (String, usize) {
    let _ = rgmin::vecops::force_simd_path("scalar");
    let mut cfg = Config::for_cluster_dual_core(atoms);
    cfg.orbit_core = core(cfg.length_scale);
    assert!(!cfg.detected_symmetry);
    let potential = PairPotential::lennard_jones(atoms);
    let mut rng = StdRng::seed_from_u64(seed);
    let start = random_cluster_in_radius(atoms, cfg.start_radius(), cfg.min_separation, &mut rng);
    let mut ledger = Ledger::new(budget);
    let mut optimiser = WarmLbfgs::default();
    let mut relax = |led: &mut Ledger, x: ArrayView1<f64>, iterations: usize| {
        optimiser.forget();
        let (value, relaxed, _) = optimiser.minimize(x, iterations, |point| {
            if !led.charge() {
                return None;
            }
            Some(potential.value_and_gradient(point))
        });
        (value, relaxed)
    };
    let mut grad = |led: &mut Ledger, point: ArrayView1<f64>| {
        if !led.charge() {
            return None;
        }
        Some(potential.value_and_gradient(point).1)
    };
    let mut rng = StdRng::seed_from_u64(seed);
    let outcome = run_with_gradient(
        &cfg,
        start.view(),
        &mut ledger,
        &mut relax,
        Some(&mut grad),
        &mut rng,
    )
    .expect("finite start");
    let mut hasher = Sha256::new();
    hasher.update(outcome.best.to_le_bytes());
    hasher.update(outcome.final_energy.to_le_bytes());
    hasher.update((outcome.charged as u64).to_le_bytes());
    hasher.update((outcome.hops as u64).to_le_bytes());
    hasher.update((outcome.orbits.0 as u64).to_le_bytes());
    hasher.update(outcome.orbits.1.to_le_bytes());
    for edge in &outcome.accepted_transitions {
        hasher.update((edge.hop as u64).to_le_bytes());
        hasher.update(edge.action.as_bytes());
        hasher.update([0]);
        hasher.update(edge.from_energy.to_le_bytes());
        hasher.update(edge.to_energy.to_le_bytes());
        update(&mut hasher, edge.to_state.view());
    }
    (format!("{:x}", hasher.finalize()), outcome.orbits.0)
}

#[test]
fn shell_core_candidates_match_develop() {
    anneal_core::install_rgpot_kernels!();
    let (hash, produced) = candidate_hash(shells);
    assert!(produced > 0, "no shell-core completion to compare");
    assert_eq!(
        hash, "fd131ed0d756f065fbd008c584b12e9a21dc30a025b9bf57ad1fec2c765dff40",
        "shell-core candidates moved: {hash}"
    );
}

#[test]
fn dual_core_candidates_match_develop() {
    anneal_core::install_rgpot_kernels!();
    let (hash, produced) = candidate_hash(both);
    assert!(produced > 0, "no dual-core completion to compare");
    assert_eq!(
        hash, "f8a49913e79f0317777a5c918d919e40f761aabe29f77499c7693b22b6469aa8",
        "dual-core candidates moved: {hash}"
    );
}

#[test]
fn shell_core_walk_matches_develop() {
    anneal_core::install_rgpot_kernels!();
    let (hash, orbits) = walk_hash(38, 11, 6_000, shells);
    assert!(orbits > 0, "the walk completed no orbit");
    assert_eq!(
        hash, "757ddb08bfbd7d2d08af8fc8d967d08dd991740aaeb3fc75eb03c60143aa0326",
        "shell-core walk moved: {hash}"
    );
}

#[test]
fn dual_core_walk_matches_develop() {
    anneal_core::install_rgpot_kernels!();
    let (hash, orbits) = walk_hash(38, 11, 6_000, both);
    assert!(orbits > 0, "the walk completed no orbit");
    assert_eq!(
        hash, "36772bd929b3036c8939fb9149e590fd992bddd195d64b0d687e74e0a4c762aa",
        "dual-core walk moved: {hash}"
    );
}
