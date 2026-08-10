//! Head-to-head: observed-cloud SOAP on vs off, same seed and budget.
//!
//! Cluster control is the Elja paper-budget table, not this file.
//! This drives molecule and slab through `run_with_gradient`.

use anneal_core::methods::cluster_hopping::{Config, Ledger, optimize, run_with_gradient};
use anneal_core::methods::warm_lbfgs::WarmLbfgs;
use anneal_core::potentials::PairPotential;
use ndarray::{Array1, ArrayView1};
use rand::Rng;
use rand::SeedableRng;
use rand::rngs::StdRng;

fn packed_waters(m: usize, seed: u64) -> (Array1<f64>, Vec<u32>, Vec<Vec<usize>>) {
    let mut rng = StdRng::seed_from_u64(seed.wrapping_mul(0x9E37).wrapping_add(11));
    let proto = [
        [0.0, 0.0, 0.0],
        [0.96, 0.0, 0.0],
        [-0.24, 0.93, 0.0],
    ];
    let mut x = Vec::with_capacity(9 * m);
    let mut z = Vec::with_capacity(3 * m);
    let mut groups = Vec::with_capacity(m);
    let side = ((m as f64).cbrt().ceil() as usize).max(1);
    for g in 0..m {
        let ix = g % side;
        let iy = (g / side) % side;
        let iz = g / (side * side);
        let ox = ix as f64 * 3.4 + (rng.random::<f64>() - 0.5) * 0.6;
        let oy = iy as f64 * 3.4 + (rng.random::<f64>() - 0.5) * 0.6;
        let oz = iz as f64 * 3.4 + (rng.random::<f64>() - 0.5) * 0.6;
        let base = 3 * g;
        groups.push(vec![base, base + 1, base + 2]);
        z.extend_from_slice(&[8, 1, 1]);
        for p in &proto {
            x.push(p[0] + ox);
            x.push(p[1] + oy);
            x.push(p[2] + oz);
        }
    }
    (Array1::from(x), z, groups)
}

fn charged_relax<'a>(
    pot: &'a PairPotential,
    opt: &'a mut WarmLbfgs,
    frozen: Option<&'a [bool]>,
) -> impl FnMut(&mut Ledger, ArrayView1<f64>, usize) -> (f64, Array1<f64>) + 'a {
    move |led, x, iters| {
        opt.forget();
        let pin = x.to_owned();
        let (f, mut xr, _) = opt.minimize_watched(
            x,
            iters,
            |v| {
                if !led.charge() {
                    return None;
                }
                let (e, mut g) = pot.value_and_gradient(v);
                if let Some(mask) = frozen {
                    for (i, &is_frozen) in mask.iter().enumerate() {
                        if is_frozen {
                            for k in 0..3 {
                                g[3 * i + k] = 0.0;
                            }
                        }
                    }
                }
                Some((e, g))
            },
            |_, _| true,
        );
        if let Some(mask) = frozen {
            for (i, &is_frozen) in mask.iter().enumerate() {
                if is_frozen {
                    for k in 0..3 {
                        xr[3 * i + k] = pin[3 * i + k];
                    }
                }
            }
        }
        (f, xr)
    }
}

struct Row {
    best: f64,
    hops: usize,
    soap_draws: usize,
}

fn run_one(mut cfg: Config, start: ArrayView1<f64>, budget: usize, seed: u64, frozen: Option<&[bool]>) -> Row {
    let n = start.len() / 3;
    let pot = PairPotential::lennard_jones(n);
    let mut ledger = Ledger::new(budget);
    let mut opt = WarmLbfgs::default();
    let mut relax = charged_relax(&pot, &mut opt, frozen);
    let mut rng = StdRng::seed_from_u64(seed);
    let out = run_with_gradient(&cfg, start, &mut ledger, &mut relax, None, &mut rng);
    let soap_draws = out
        .arms
        .iter()
        .find(|(name, _, _, _)| name == "soap")
        .map(|(_, d, _, _)| *d)
        .unwrap_or(0);
    let _ = &mut cfg;
    Row {
        best: out.best,
        hops: out.hops,
        soap_draws,
    }
}

fn rec_on(species: Vec<u32>, groups: Vec<Vec<usize>>) -> Config {
    Config::recommended_molecular(species, groups, 1.0)
}

fn rec_off(species: Vec<u32>, groups: Vec<Vec<usize>>) -> Config {
    let mut cfg = Config::recommended_molecular(species, groups, 1.0);
    cfg.soap_hop = false;
    cfg
}

#[test]
fn soap_on_vs_off_water4_eight_seeds() {
    const M: usize = 4;
    const SEEDS: u64 = 8;
    const BUDGET: usize = 2_500;
    let mut on_better = 0usize;
    let mut off_better = 0usize;
    let mut tie = 0usize;
    let mut sum_on = 0.0;
    let mut sum_off = 0.0;
    let mut soap_draws = 0usize;
    for s in 0..SEEDS {
        let (start, z, g) = packed_waters(M, s);
        let a = run_one(rec_on(z.clone(), g.clone()), start.view(), BUDGET, 100 + s, None);
        let b = run_one(rec_off(z, g), start.view(), BUDGET, 100 + s, None);
        sum_on += a.best;
        sum_off += b.best;
        soap_draws += a.soap_draws;
        assert_eq!(b.soap_draws, 0, "soap_hop=false still drew SOAP");
        assert!(a.best.is_finite() && b.best.is_finite());
        let d = a.best - b.best;
        if d < -1e-6 {
            on_better += 1;
        } else if d > 1e-6 {
            off_better += 1;
        } else {
            tie += 1;
        }
        println!(
            "mol seed {s}: soap_on {:.6} hops {} draws {} | soap_off {:.6} hops {} | d {:+.6}",
            a.best, a.hops, a.soap_draws, b.best, b.hops, d
        );
    }
    let n = SEEDS as f64;
    println!(
        "mol summary: on_better {on_better}/{SEEDS} off_better {off_better}/{SEEDS} tie {tie} mean_on {:.6} mean_off {:.6} soap_draws {soap_draws}",
        sum_on / n,
        sum_off / n
    );
}

#[test]
fn soap_on_vs_off_slab_water4_eight_seeds() {
    const M: usize = 4;
    const SEEDS: u64 = 8;
    const BUDGET: usize = 2_500;
    let mut on_better = 0usize;
    let mut off_better = 0usize;
    let mut tie = 0usize;
    let mut sum_on = 0.0;
    let mut sum_off = 0.0;
    let mut soap_draws = 0usize;
    for s in 0..SEEDS {
        let (start, z, g) = packed_waters(M, s + 40);
        let n = z.len();
        let frozen: Vec<bool> = (0..n).map(|i| i >= 6).collect();
        let mut on = rec_on(z.clone(), g.clone());
        on.active_region = Some((vec![0, 1, 2, 3, 4, 5], 0));
        on.frozen = Some(frozen.clone());
        let mut off = rec_off(z, g);
        off.active_region = Some((vec![0, 1, 2, 3, 4, 5], 0));
        off.frozen = Some(frozen.clone());
        let a = run_one(on, start.view(), BUDGET, 200 + s, Some(frozen.as_slice()));
        let b = run_one(off, start.view(), BUDGET, 200 + s, Some(frozen.as_slice()));
        sum_on += a.best;
        sum_off += b.best;
        soap_draws += a.soap_draws;
        assert_eq!(b.soap_draws, 0);
        assert!(a.best.is_finite() && b.best.is_finite());
        let d = a.best - b.best;
        if d < -1e-6 {
            on_better += 1;
        } else if d > 1e-6 {
            off_better += 1;
        } else {
            tie += 1;
        }
        println!(
            "slab seed {s}: soap_on {:.6} hops {} draws {} | soap_off {:.6} hops {} | d {:+.6}",
            a.best, a.hops, a.soap_draws, b.best, b.hops, d
        );
    }
    let n = SEEDS as f64;
    println!(
        "slab summary: on_better {on_better}/{SEEDS} off_better {off_better}/{SEEDS} tie {tie} mean_on {:.6} mean_off {:.6} soap_draws {soap_draws}",
        sum_on / n,
        sum_off / n
    );
}

fn run_cluster(n: usize, budget: usize, seed: u64, soap: bool) -> Row {
    let mut cfg = Config::recommended(n);
    cfg.soap_hop = soap;
    let pot = PairPotential::lennard_jones(n);
    let mut ledger = Ledger::new(budget);
    let mut opt = WarmLbfgs::default();
    let mut relax = charged_relax(&pot, &mut opt, None);
    let out = optimize(&cfg, &mut ledger, &mut relax, seed);
    let soap_draws = out
        .arms
        .iter()
        .find(|(name, _, _, _)| name == "soap")
        .map(|(_, d, _, _)| *d)
        .unwrap_or(0);
    Row {
        best: out.best,
        hops: out.hops,
        soap_draws,
    }
}

#[test]
fn soap_on_vs_off_lj38_eight_seeds() {
    const N: usize = 38;
    const SEEDS: u64 = 8;
    const BUDGET: usize = 25_000;
    let mut on_better = 0usize;
    let mut off_better = 0usize;
    let mut tie = 0usize;
    let mut sum_on = 0.0;
    let mut sum_off = 0.0;
    let mut soap_draws = 0usize;
    for s in 0..SEEDS {
        let a = run_cluster(N, BUDGET, 300 + s, true);
        let b = run_cluster(N, BUDGET, 300 + s, false);
        sum_on += a.best;
        sum_off += b.best;
        soap_draws += a.soap_draws;
        assert_eq!(b.soap_draws, 0, "soap_hop=false still drew SOAP");
        assert!(a.best.is_finite() && b.best.is_finite());
        let d = a.best - b.best;
        if d < -1e-6 {
            on_better += 1;
        } else if d > 1e-6 {
            off_better += 1;
        } else {
            tie += 1;
        }
        println!(
            "lj38 seed {s}: soap_on {:.6} hops {} draws {} | soap_off {:.6} hops {} | d {:+.6}",
            a.best, a.hops, a.soap_draws, b.best, b.hops, d
        );
    }
    let n = SEEDS as f64;
    println!(
        "lj38 summary: on_better {on_better}/{SEEDS} off_better {off_better}/{SEEDS} tie {tie} mean_on {:.6} mean_off {:.6} soap_draws {soap_draws}",
        sum_on / n,
        sum_off / n
    );
}
