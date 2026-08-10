//! Head-to-head: observed-cloud SOAP on vs off, same seed and budget.
//!
//! Cluster control is the Elja paper-budget table, not this file.
//! This drives molecule and slab through `run_with_gradient`.

use anneal_core::methods::cluster_hopping::{Config, Ledger, run_with_gradient};
use anneal_core::methods::warm_lbfgs::WarmLbfgs;
use anneal_core::potentials::PairPotential;
use ndarray::{Array1, ArrayView1};
use rand::Rng;
use rand::SeedableRng;
use rand::rngs::StdRng;

fn water_proto() -> [[f64; 3]; 3] {
    [[0.0, 0.0, 0.0], [0.96, 0.0, 0.0], [-0.24, 0.93, 0.0]]
}

fn packed_waters(m: usize, seed: u64) -> (Array1<f64>, Vec<u32>, Vec<Vec<usize>>) {
    let mut rng = StdRng::seed_from_u64(seed.wrapping_mul(0x9E37).wrapping_add(11));
    let proto = water_proto();
    let mut x = Vec::with_capacity(9 * m);
    let mut z = Vec::with_capacity(3 * m);
    let mut groups = Vec::with_capacity(m);
    let side = ((m as f64).cbrt().ceil() as usize).max(1);
    for g in 0..m {
        let ix = g % side;
        let iy = (g / side) % side;
        let iz = g / (side * side);
        let ox = ix as f64 * 2.3 + (rng.random::<f64>() - 0.5) * 1.4;
        let oy = iy as f64 * 2.3 + (rng.random::<f64>() - 0.5) * 1.4;
        let oz = iz as f64 * 2.3 + (rng.random::<f64>() - 0.5) * 1.4;
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

/// Two adsorbate waters above a 2-water frozen layer. Contact spacing,
/// no overlapping nuclei. The two adsorbates sit in different sites so
/// the mobile ν=3 leftover is a real packing defect.
fn slab_waters(seed: u64) -> (Array1<f64>, Vec<u32>, Vec<Vec<usize>>) {
    let mut rng = StdRng::seed_from_u64(seed.wrapping_mul(0x9E37).wrapping_add(29));
    let proto = water_proto();
    let mut x = Vec::with_capacity(36);
    let mut z = Vec::with_capacity(12);
    let mut groups = Vec::with_capacity(4);
    let origins = [
        [
            (rng.random::<f64>() - 0.5) * 0.4,
            (rng.random::<f64>() - 0.5) * 0.4,
            3.3,
        ],
        [
            1.7 + (rng.random::<f64>() - 0.5) * 0.4,
            1.7 + (rng.random::<f64>() - 0.5) * 0.4,
            4.1,
        ],
        [0.0, 0.0, 0.0],
        [3.4, 0.0, 0.0],
    ];
    for (g, o) in origins.iter().enumerate() {
        groups.push(vec![3 * g, 3 * g + 1, 3 * g + 2]);
        z.extend_from_slice(&[8, 1, 1]);
        for p in &proto {
            x.push(p[0] + o[0]);
            x.push(p[1] + o[1]);
            x.push(p[2] + o[2]);
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

fn soap_rmsd(cfg: &Config) -> f64 {
    cfg.move_library
        .kernels(cfg)
        .into_iter()
        .find_map(|k| match k {
            anneal_core::methods::cluster_hopping::ClusterMove::Soap { rmsd, .. } => Some(rmsd),
            _ => None,
        })
        .unwrap_or(f64::NAN)
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
    let mut hops_on = 0usize;
    let mut hops_off = 0usize;
    {
        let ( _, z, g) = packed_waters(M, 0);
        println!("mol soap_rmsd {}", soap_rmsd(&rec_on(z, g)));
    }
    for s in 0..SEEDS {
        let (start, z, g) = packed_waters(M, s);
        let a = run_one(rec_on(z.clone(), g.clone()), start.view(), BUDGET, 100 + s, None);
        let b = run_one(rec_off(z, g), start.view(), BUDGET, 100 + s, None);
        sum_on += a.best;
        sum_off += b.best;
        soap_draws += a.soap_draws;
        hops_on += a.hops;
        hops_off += b.hops;
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
        "mol summary: on_better {on_better}/{SEEDS} off_better {off_better}/{SEEDS} tie {tie} mean_on {:.6} mean_off {:.6} soap_draws {soap_draws} hops_on {hops_on} hops_off {hops_off}",
        sum_on / n,
        sum_off / n
    );
    assert!(
        hops_on * 2 >= hops_off,
        "SOAP is a budget tax: hops_on {hops_on} hops_off {hops_off}"
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
    let mut hops_on = 0usize;
    let mut hops_off = 0usize;
    for s in 0..SEEDS {
        let (start, z, g) = slab_waters(s);
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
        hops_on += a.hops;
        hops_off += b.hops;
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
        "slab summary: on_better {on_better}/{SEEDS} off_better {off_better}/{SEEDS} tie {tie} mean_on {:.6} mean_off {:.6} soap_draws {soap_draws} hops_on {hops_on} hops_off {hops_off}",
        sum_on / n,
        sum_off / n
    );
}

fn ico_cluster(n: usize, seed: u64) -> Array1<f64> {
    let sites = anneal_core::lattice::grow(
        &anneal_core::structure::Template::Icosahedral.points(),
        n,
    );
    let nn = 2.0_f64.powf(1.0 / 6.0);
    let mut rng = StdRng::seed_from_u64(seed.wrapping_mul(0x51ED));
    let mut x = Array1::zeros(3 * n);
    for (i, p) in sites.iter().enumerate().take(n) {
        for k in 0..3 {
            x[3 * i + k] = p[k] * nn + (rng.random::<f64>() - 0.5) * 0.04;
        }
    }
    x
}

fn rec_cluster(n: usize, soap: bool) -> Config {
    let mut cfg = Config::recommended(n);
    cfg.soap_hop = soap;
    cfg
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
        let start = ico_cluster(N, s);
        let a = run_one(rec_cluster(N, true), start.view(), BUDGET, 300 + s, None);
        let b = run_one(rec_cluster(N, false), start.view(), BUDGET, 300 + s, None);
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
