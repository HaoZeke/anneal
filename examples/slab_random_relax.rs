//! Random-relaxation baseline for a multi-site slab (Kaappa, Garijo del
//! Rio, Jacobsen, Phys. Rev. B 103, 174114, 2021). Each start draws a
//! fresh adsorbate placement, quenches with the same engine the hop
//! search uses, and reports distinct minima (energy within 1e-4 eV) and
//! how many relaxations reach the lowest energy seen.
//!
//! Usage: slab_random_relax <con_file> <total_budget> [seed]

mod common;

use anneal_core::methods::cluster_hopping::Ledger;
use anneal_core::methods::warm_lbfgs::WarmLbfgs;
use common::rgpot_eindir::{RgpotObjective, emit_engine_manifest};
use common::slab::{Mobile, place_hydrogens, read_system};
use eindir_core::gradient::DifferentiableObjective;
use ndarray::ArrayView1;

const ENERGY_TOL: f64 = 1e-4;
const RELAX_STEPS: usize = 150;

fn cluster_energies(energies: &[f64]) -> Vec<(f64, usize)> {
    let mut sorted: Vec<f64> = energies
        .iter()
        .copied()
        .filter(|energy| energy.is_finite())
        .collect();
    sorted.sort_by(|a, b| a.total_cmp(b));
    let mut clusters: Vec<(f64, usize)> = Vec::new();
    for energy in sorted {
        if let Some((rep, count)) = clusters.last_mut() {
            if (energy - *rep).abs() <= ENERGY_TOL {
                *count += 1;
                continue;
            }
        }
        clusters.push((energy, 1));
    }
    clusters
}

fn main() {
    let args: Vec<String> = std::env::args().collect();
    let con = args
        .get(1)
        .cloned()
        .expect("usage: slab_random_relax <con_file> <total_budget> [seed]");
    let budget: usize = args.get(2).and_then(|v| v.parse().ok()).unwrap_or(400);
    let seed0: u64 = args
        .get(3)
        .and_then(|v| v.parse().ok())
        .or_else(|| {
            std::env::var("SEED_OFFSET")
                .ok()
                .and_then(|v| v.parse().ok())
        })
        .unwrap_or(0);
    let relax_steps = std::env::var("RELAX_STEPS")
        .ok()
        .and_then(|v| v.parse().ok())
        .unwrap_or(RELAX_STEPS);
    let (base_x, species, free, box_) = read_system(&con);
    let n = species.len();
    let atmnrs: Vec<i32> = species.iter().map(|&z| z as i32).collect();
    let pot = RgpotObjective::cuh2(&atmnrs, box_);
    emit_engine_manifest("cuh2");
    let inner = pot.wrapper();
    let mut active = vec![false; n];
    for &i in &free {
        active[i] = true;
    }
    let obj = Mobile {
        inner: &inner,
        active,
    };
    println!(
        "{con}: {n} atoms through eindir/rgpot cuh2, {} free, random-relax budget {budget}, seed0 {seed0}",
        free.len()
    );
    let mut ledger = Ledger::new(budget);
    let mut opt = WarmLbfgs::default();
    let mut energies: Vec<f64> = Vec::new();
    let mut start = 0u64;
    while ledger.remaining() > 0 {
        let seed = seed0.wrapping_add(start);
        let x0 = place_hydrogens(&base_x, &species, &free, box_, seed);
        opt.forget();
        let before = ledger.spent();
        let (energy, _xmin, _evals) = opt.minimize(x0.view(), relax_steps, |v: ArrayView1<f64>| {
            if !ledger.charge() {
                return None;
            }
            Some(obj.value_and_gradient(v))
        });
        let charged = ledger.spent() - before;
        if energy.is_finite() {
            energies.push(energy);
        }
        println!(
            "  relax {start}: e={energy:.6} charged={charged} remaining={}",
            ledger.remaining()
        );
        start += 1;
        if charged == 0 {
            break;
        }
    }
    let clusters = cluster_energies(&energies);
    let lowest = clusters.first().map(|(e, _)| *e);
    let hits = clusters.first().map(|(_, n)| *n).unwrap_or(0);
    println!(
        "distinct_minima {} (tol={ENERGY_TOL} eV)  starts {}  charged {}",
        clusters.len(),
        energies.len(),
        ledger.spent()
    );
    if let Some(energy) = lowest {
        println!(
            "lowest {energy:.6} hits {hits}/{}  fraction {:.3}",
            energies.len(),
            hits as f64 / energies.len().max(1) as f64
        );
    }
    for (energy, count) in &clusters {
        println!("  basin {energy:.6}  n={count}");
    }
    println!(
        "SUMMARY distinct={} lowest={} hits={}/{} charged={} starts={}",
        clusters.len(),
        lowest
            .map(|e| format!("{e:.8}"))
            .unwrap_or_else(|| "nan".into()),
        hits,
        energies.len(),
        ledger.spent(),
        energies.len()
    );
}
