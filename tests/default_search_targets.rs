//! What the default search reaches on Lennard-Jones clusters.
//!
//! These replace the tests of the removed structure builders that asked
//! "does the lattice search reach the LJ13 icosahedron" and "does
//! occupation optimisation restore a displaced LJ98 or LJ104 surface".
//! The behaviour is now asked of `Config::recommended`, which has no
//! structure source: it must still end on the same minima within a
//! comparable call budget.

use anneal_core::methods::cluster_hopping::{Config, Ledger};
use anneal_core::methods::cluster_search::{search, search_from};
use anneal_core::potentials::{PairKind, PairPotential};
use anneal_core::{DEFAULT_HIT_TOL, energy_hits};
use ndarray::Array1;
use rand::{Rng, SeedableRng};

fn fixture(name: &str) -> Array1<f64> {
    let path = format!("{}/tests/fixtures/{name}.xyz", env!("CARGO_MANIFEST_DIR"));
    let text = std::fs::read_to_string(&path).unwrap_or_else(|e| panic!("{path}: {e}"));
    let mut coords = Vec::new();
    for line in text.lines().skip(2) {
        let f: Vec<f64> = line
            .split_whitespace()
            .skip(1)
            .map(|v| v.parse().expect("coordinate"))
            .collect();
        if f.len() == 3 {
            coords.extend(f);
        }
    }
    Array1::from(coords)
}

/// The eight least coordinated atoms scattered onto a sphere outside the
/// cluster.
fn displace_surface(x: &Array1<f64>, cutoff: f64, seed: u64) -> Array1<f64> {
    let n = x.len() / 3;
    let p = |a: usize| [x[3 * a], x[3 * a + 1], x[3 * a + 2]];
    let d2 = |a: [f64; 3], b: [f64; 3]| (0..3).map(|k| (a[k] - b[k]).powi(2)).sum::<f64>();
    let mut coordination = vec![0usize; n];
    let mut rmax: f64 = 0.0;
    for a in 0..n {
        rmax = rmax.max(d2([0.0; 3], p(a)).sqrt());
        for b in (a + 1)..n {
            if d2(p(a), p(b)) < cutoff * cutoff {
                coordination[a] += 1;
                coordination[b] += 1;
            }
        }
    }
    let mut order: Vec<usize> = (0..n).collect();
    order.sort_by_key(|&i| coordination[i]);
    let mut rng = rand::rngs::StdRng::seed_from_u64(seed);
    let mut out = x.clone();
    for &atom in order.iter().take(8) {
        let u: f64 = rng.random_range(-1.0..1.0);
        let phi: f64 = rng.random_range(0.0..std::f64::consts::TAU);
        let r = 1.15 * rmax;
        let s = (1.0 - u * u).sqrt();
        out[3 * atom] = r * s * phi.cos();
        out[3 * atom + 1] = r * s * phi.sin();
        out[3 * atom + 2] = r * u;
    }
    out
}

fn restored(name: &str, n: usize, budget: usize, seed: u64) -> f64 {
    anneal_core::install_rgpot_kernels!();
    let gm = fixture(name);
    assert_eq!(gm.len(), 3 * n);
    let cutoff = 1.35 * PairKind::LennardJones.r_min();
    let start = displace_surface(&gm, cutoff, 1);
    let cfg = Config::recommended(n);
    let pot = PairPotential::lennard_jones(n);
    let (displaced, _) = pot.value_and_gradient(start.view());
    let (minimum, _) = pot.value_and_gradient(gm.view());
    assert!(
        displaced > minimum + 10.0,
        "the start is not displaced from the minimum: {displaced} against {minimum}"
    );
    let mut ledger = Ledger::new(budget);
    let (out, _) = search_from(&pot, &cfg, &mut ledger, start.view(), seed).expect("finite");
    assert!(ledger.spent() <= budget);
    out.best
}

#[test]
fn the_default_search_reaches_the_lj13_icosahedron() {
    anneal_core::install_rgpot_kernels!();
    let cfg = Config::recommended(13);
    let pot = PairPotential::lennard_jones(13);
    let mut ledger = Ledger::new(20_000);
    let (out, _) = search(&pot, &cfg, &mut ledger, 1).expect("finite");
    assert!(
        energy_hits(out.best, -44.326801, DEFAULT_HIT_TOL),
        "LJ13 stopped at {:.6}",
        out.best
    );
    assert!(out.hops > 0 && ledger.spent() <= 20_000);
}

#[test]
fn the_default_search_restores_a_displaced_lj98_surface() {
    let best = restored("lj98_gm", 98, LJ98_BUDGET, 1);
    assert!(
        energy_hits(best, -543.665361, DEFAULT_HIT_TOL),
        "LJ98 stopped at {best:.6}"
    );
}

#[test]
fn the_default_search_restores_a_displaced_lj104_surface() {
    let best = restored("lj104_gm", 104, LJ104_BUDGET, 0);
    assert!(
        energy_hits(best, -582.086642, DEFAULT_HIT_TOL),
        "LJ104 stopped at {best:.6}"
    );
}

/// Eight seeds restore the LJ98 surface within 272 charged calls, so its
/// budget carries a margin of more than three over the slowest seed measured.
/// The LJ104 surface is restored within 40000 calls by 219 of walk seeds 0
/// to 239, and by 214 of them under the walk this one replaced; the others
/// stop short of it, most at -580.52. Seed 0 is one the walk restores under
/// both, so this pins the walk on a seed that reaches the minimum and not
/// the odds of reaching it.
const LJ98_BUDGET: usize = 3_000;
const LJ104_BUDGET: usize = 40_000;
