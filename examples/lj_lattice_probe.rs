//! Where one random start and one dynamic lattice descent land.
//!
//! Usage: `lj_lattice_probe <n> <probes> <seed> [threads]`
//!
//! Each probe draws a random cluster, quenches it, and runs the lattice
//! descent of [`anneal_core::methods::lattice_search`] to its end, all on one
//! ledger per probe. The report gives the charged cost per probe, the
//! quenched energies reached, and the common-neighbour 555 fraction of each
//! end point, read after the run. The Cambridge reference scores hits after
//! the run and enters nothing.

use anneal_core::methods::cluster_hopping::{Ledger, random_cluster};
use anneal_core::methods::lattice_search::{Lattice, Quench};
use anneal_core::structure::cna_descriptor;
use ndarray::ArrayView1;
use rand::SeedableRng;
use rand::rngs::StdRng;
use rayon::prelude::*;

fn env_f64(key: &str, default: f64) -> f64 {
    std::env::var(key)
        .ok()
        .and_then(|v| v.parse().ok())
        .unwrap_or(default)
}

fn env_usize(key: &str, default: usize) -> usize {
    std::env::var(key)
        .ok()
        .and_then(|v| v.parse().ok())
        .unwrap_or(default)
}

fn reference(n: usize) -> Option<f64> {
    Some(match n {
        13 => -44.326801,
        38 => -173.928427,
        55 => -279.248470,
        75 => -397.492331,
        98 => -543.665361,
        _ => return None,
    })
}

struct Probe {
    start: f64,
    end: f64,
    charged: usize,
    first_calls: usize,
    searches: usize,
    f555: f64,
}

fn main() {
    let args: Vec<String> = std::env::args().collect();
    let n: usize = args.get(1).and_then(|v| v.parse().ok()).unwrap_or(75);
    let probes: usize = args.get(2).and_then(|v| v.parse().ok()).unwrap_or(200);
    let seed: u64 = args.get(3).and_then(|v| v.parse().ok()).unwrap_or(900);
    let threads: usize = args.get(4).and_then(|v| v.parse().ok()).unwrap_or(4);
    let quench = Quench {
        max_step: env_f64("QUENCH_STEP", 0.4),
        rms_tolerance: env_f64("QUENCH_TOL", 1e-5),
        memory: env_usize("QUENCH_MEMORY", 8),
        ..Quench::default()
    };
    let lattice = Lattice {
        candidates: env_usize("LATTICE_CANDIDATES", 4),
        ..Lattice::default()
    };
    let density = env_f64("DENSITY", 0.7);
    let reference = reference(n);
    let pool = rayon::ThreadPoolBuilder::new()
        .num_threads(threads)
        .build()
        .expect("thread pool");
    let results: Vec<Probe> = pool.install(|| {
        (0..probes)
            .into_par_iter()
            .map(|k| {
                let mut rng =
                    StdRng::seed_from_u64(seed.wrapping_mul(1_000_003).wrapping_add(k as u64));
                let x = random_cluster(n, density, 0.85, &mut rng);
                let mut ledger = Ledger::new(usize::MAX / 2);
                let first = quench
                    .relax(&mut ledger, x.as_slice().expect("contiguous"))
                    .expect("budget");
                let (end, state, stats) =
                    lattice.descend(&quench, &mut ledger, first.energy, &first.state);
                let cna = cna_descriptor(ArrayView1::from(&state), n, 1.39);
                Probe {
                    start: first.energy,
                    end,
                    charged: ledger.spent(),
                    first_calls: first.calls,
                    searches: stats.searches,
                    f555: cna[0],
                }
            })
            .collect()
    });
    let mut ends: Vec<f64> = results.iter().map(|p| p.end).collect();
    ends.sort_by(f64::total_cmp);
    let charged: usize = results.iter().map(|p| p.charged).sum();
    let hits = reference.map_or(0, |r| results.iter().filter(|p| p.end < r + 1e-4).count());
    let low555 = results.iter().filter(|p| p.f555 < 0.1).count();
    println!(
        "LJ{n}: {probes} probes from seed {seed}, quench step {} tol {} memory {}, density {density}",
        quench.max_step, quench.rms_tolerance, quench.memory
    );
    println!(
        "  charged per probe {:.0} (first quench {:.0}), searches per probe {:.2}, mean start {:.3}, hits {hits}, f555<0.1 in {low555}",
        charged as f64 / probes as f64,
        results.iter().map(|p| p.first_calls).sum::<usize>() as f64 / probes as f64,
        results.iter().map(|p| p.searches).sum::<usize>() as f64 / probes as f64,
        results.iter().map(|p| p.start).sum::<f64>() / probes as f64,
    );
    let q = |f: f64| ends[((ends.len() - 1) as f64 * f) as usize];
    println!(
        "  end energy: min {:.6} q05 {:.3} q25 {:.3} median {:.3} q75 {:.3}",
        ends[0],
        q(0.05),
        q(0.25),
        q(0.5),
        q(0.75)
    );
    let mut lowest: Vec<&Probe> = results.iter().collect();
    lowest.sort_by(|a, b| a.end.total_cmp(&b.end));
    for p in lowest.iter().take(env_usize("SHOW", 12)) {
        println!(
            "    {:.6}  f555 {:.3}  charged {}",
            p.end, p.f555, p.charged
        );
    }
}
