//! Census of random-quench cores: which five-fold class a random LJ
//! cluster lands in after the compaction quench, and whether one
//! reoccupation of its surface reaches the global minimum.
//!
//! Usage: `lj_core_census [n] [seeds] [threads]`, `TARGET` overrides the
//! reference energy (default LJ98 -543.665361), `KAPPA` and `MU` the
//! compaction quench (0.7, 5.0).
use std::collections::BTreeMap;
use std::sync::Mutex;

use anneal_core::corekey::{MotifClass, motif_class};
use anneal_core::methods::cluster_hopping::{Ledger, random_cluster};
use anneal_core::methods::lattice_search::{LatticeSearchConfig, reoccupy};
use anneal_core::methods::two_phase::{largest_pair_distance, penalty};
use anneal_core::methods::warm_lbfgs::WarmLbfgs;
use anneal_core::potentials::PairPotential;
use ndarray::{Array1, ArrayView1};
use rand::SeedableRng;
use rand::rngs::StdRng;

fn env_f64(key: &str, default: f64) -> f64 {
    std::env::var(key)
        .ok()
        .and_then(|v| v.parse().ok())
        .unwrap_or(default)
}

struct Row {
    class0: MotifClass,
    e0: f64,
    class1: MotifClass,
    e1: f64,
    evals: usize,
}

fn quench(pot: &PairPotential, x: ArrayView1<f64>, evals: &mut usize) -> (f64, Array1<f64>) {
    let mut opt = WarmLbfgs::default();
    let (e, xr, _) = opt.minimize(x, 2000, |v| {
        *evals += 1;
        Some(pot.value_and_gradient(v))
    });
    (e, xr)
}

fn compaction_quench(
    pot: &PairPotential,
    x: ArrayView1<f64>,
    kappa: f64,
    mu: f64,
    evals: &mut usize,
) -> (f64, Array1<f64>) {
    let cutoff = kappa * largest_pair_distance(x);
    let mut opt = WarmLbfgs::default();
    let (_, xc, _) = opt.minimize(x, 2000, |v| {
        *evals += 1;
        let (e, g) = pot.value_and_gradient(v);
        let (pe, pg) = penalty(v, cutoff, 1.0, mu);
        Some((e + pe, g + pg))
    });
    quench(pot, xc.view(), evals)
}

fn main() {
    let args: Vec<String> = std::env::args().collect();
    let n: usize = args.get(1).and_then(|v| v.parse().ok()).unwrap_or(98);
    let seeds: u64 = args.get(2).and_then(|v| v.parse().ok()).unwrap_or(2000);
    let threads: usize = args.get(3).and_then(|v| v.parse().ok()).unwrap_or(16);
    let target = env_f64("TARGET", -543.665361);
    let kappa = env_f64("KAPPA", 0.7);
    let mu = env_f64("MU", 5.0);
    let rows: Mutex<Vec<Row>> = Mutex::new(Vec::new());
    std::thread::scope(|scope| {
        for t in 0..threads {
            let rows = &rows;
            scope.spawn(move || {
                let pot = PairPotential::lennard_jones(n);
                let cfg = LatticeSearchConfig::lennard_jones(n);
                let mut seed = t as u64;
                while seed < seeds {
                    let mut rng = StdRng::seed_from_u64(seed);
                    let mut evals = 0usize;
                    let x0 = random_cluster(n, 0.7, 0.5, &mut rng);
                    let (e0, xq) = compaction_quench(&pot, x0.view(), kappa, mu, &mut evals);
                    let class0 = motif_class(xq.view());
                    let mut ledger = Ledger::new(usize::MAX / 2);
                    let rebuilt = reoccupy(&cfg, &mut ledger, xq.view());
                    evals += ledger.spent();
                    let (e1, xr) = quench(&pot, rebuilt.view(), &mut evals);
                    let class1 = motif_class(xr.view());
                    rows.lock().expect("rows").push(Row {
                        class0,
                        e0,
                        class1,
                        e1,
                        evals,
                    });
                    seed += threads as u64;
                }
            });
        }
    });
    let rows = rows.into_inner().expect("rows");
    let mut by_class: BTreeMap<u8, (usize, usize, f64, f64, usize)> = BTreeMap::new();
    let mut hits_by_class1: BTreeMap<u8, usize> = BTreeMap::new();
    for r in &rows {
        let hit = r.e1 < target + 1e-4;
        let entry = by_class
            .entry(r.class0.index())
            .or_insert((0, 0, f64::INFINITY, 0.0, 0));
        entry.0 += 1;
        entry.1 += usize::from(hit);
        entry.2 = entry.2.min(r.e1);
        entry.3 += r.e1;
        entry.4 += r.evals;
        if hit {
            *hits_by_class1.entry(r.class1.index()).or_insert(0) += 1;
        }
    }
    println!(
        "LJ{n} census: {} random compaction quenches (kappa {kappa}, mu {mu}), one reoccupation each, target {target}",
        rows.len()
    );
    println!("class0 (after quench)  count  hits  best_e1  mean_e1  mean_evals");
    for (class, (count, hits, best, sum, evals)) in &by_class {
        println!(
            "{class:>6}  {count:>6}  {hits:>5}  {best:>10.4}  {:>10.4}  {:>7}",
            sum / *count as f64,
            evals / count
        );
    }
    println!("hits by class after reoccupation: {hits_by_class1:?}");
    let total_hits: usize = by_class.values().map(|v| v.1).sum();
    println!(
        "total hits {total_hits}/{} ; classes 0 none, 1 sparse, 2 moderate, 3 dense",
        rows.len()
    );
    let mean_e0: f64 = rows.iter().map(|r| r.e0).sum::<f64>() / rows.len().max(1) as f64;
    println!("mean energy after quench {mean_e0:.4}");
}
