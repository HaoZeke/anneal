//! Adsorbate search on a substrate. The potential is an rgpot handle
//! viewed as an eindir objective. Anneal only runs [`cluster_search`].
//!
//! Usage: slab_adsorption <con_file> <budget> <seeds>

mod common;

use anneal_core::methods::cluster_hopping::{Config, Ledger, covalent_radius};
use anneal_core::methods::cluster_search::{search_from, verify};
use common::rgpot_eindir::RgpotObjective;
use eindir_core::Objective;
use eindir_core::bounds::Bounds;
use eindir_core::gradient::Gradient;
use ndarray::{Array1, ArrayView1};
use std::io::Write;
use std::path::Path;

fn read_system(path: &str) -> (Array1<f64>, Vec<u32>, Vec<usize>, [f64; 9]) {
    let frame = readcon_core::iterators::read_first_frame(Path::new(path))
        .expect("failed to read the con file");
    let mut pos = Vec::new();
    let mut species = Vec::new();
    let mut seeds = Vec::new();
    for (i, a) in frame.atom_data.iter().enumerate() {
        pos.extend_from_slice(&[a.x, a.y, a.z]);
        species.push(readcon_core::helpers::symbol_to_atomic_number(&a.symbol) as u32);
        if !a.is_fixed() {
            seeds.push(i);
        }
    }
    let boxl = frame.header.boxl;
    let box_ = [boxl[0], 0.0, 0.0, 0.0, boxl[1], 0.0, 0.0, 0.0, boxl[2]];
    (Array1::from(pos), species, seeds, box_)
}

fn symbol(z: u32) -> &'static str {
    match z {
        1 => "H",
        6 => "C",
        7 => "N",
        8 => "O",
        29 => "Cu",
        79 => "Au",
        _ => "X",
    }
}

/// Zero the gradient on frozen substrate atoms. The potential is unchanged.
struct Mobile<'a, O> {
    inner: &'a O,
    active: Vec<bool>,
}

impl<O: Objective<f64>> Objective<f64> for Mobile<'_, O> {
    fn dim(&self) -> usize {
        self.inner.dim()
    }
    fn bounds(&self) -> &Bounds<f64> {
        self.inner.bounds()
    }
    fn eval(&self, x: ArrayView1<f64>) -> f64 {
        self.inner.eval(x)
    }
}

impl<O: Gradient<f64>> Gradient<f64> for Mobile<'_, O> {
    fn dim(&self) -> usize {
        self.inner.dim()
    }
    fn grad(&self, x: ArrayView1<f64>) -> Array1<f64> {
        let mut g = self.inner.grad(x);
        for (i, on) in self.active.iter().enumerate() {
            if !on {
                for k in 0..3 {
                    g[3 * i + k] = 0.0;
                }
            }
        }
        g
    }
}

fn main() {
    let args: Vec<String> = std::env::args().collect();
    let con = args
        .get(1)
        .cloned()
        .expect("usage: slab_adsorption <con_file> <budget> <seeds>");
    let budget: usize = args.get(2).and_then(|v| v.parse().ok()).unwrap_or(25);
    let seeds: u64 = args.get(3).and_then(|v| v.parse().ok()).unwrap_or(1);
    let seed0: u64 = std::env::var("SEED_OFFSET")
        .ok()
        .and_then(|v| v.parse().ok())
        .unwrap_or(0);
    let (base_x, species, free_seeds, box_) = read_system(&con);
    let n = species.len();
    let atmnrs: Vec<i32> = species.iter().map(|&z| z as i32).collect();
    let groups = if free_seeds.is_empty() || free_seeds.len() == n {
        vec![(0..n).collect()]
    } else {
        vec![free_seeds.clone()]
    };
    let mut cfg = Config::recommended_molecular(species.clone(), groups, 1.0);
    cfg.active_region = Some((free_seeds.clone(), 1));
    if !free_seeds.is_empty() && free_seeds.len() < n {
        let adsorbate = free_seeds
            .iter()
            .map(|&i| covalent_radius(species[i]))
            .fold(0.0_f64, f64::max);
        cfg.length_scale = 2.0 * adsorbate;
    }
    cfg.screen_steps = 10;
    cfg.relax_steps = 150;
    let pot = RgpotObjective::cuh2(&atmnrs, box_);
    let inner = pot.wrapper();
    let mut active = vec![false; n];
    for &i in &free_seeds {
        active[i] = true;
    }
    let obj = Mobile {
        inner: &inner,
        active,
    };
    println!(
        "{con}: {n} atoms through eindir/rgpot cuh2, {} free, budget {budget}, seeds {seed0}..{}",
        free_seeds.len(),
        seed0 + seeds
    );
    for seed in seed0..seed0 + seeds {
        let mut ledger = Ledger::new(budget);
        let (out, stats) = search_from(&obj, &cfg, &mut ledger, base_x.view(), seed);
        let checked = verify(&obj, &out);
        println!(
            "  seed {seed}: best {:.6} eV  hops {}  charged {}  converged {}/{}{}",
            out.best,
            out.hops,
            ledger.spent(),
            stats.converged,
            stats.total(),
            checked
                .map(|(e, g)| format!("  verify e={e:.6} |g|_mobile={g:.3e}"))
                .unwrap_or_default()
        );
        if let Some(bx) = out.best_state {
            let path = format!("best_slab_eindir_s{seed}.xyz");
            let mut f = std::fs::File::create(&path).expect("xyz");
            writeln!(f, "{n}\nbest {:.6} eV", out.best).ok();
            for i in 0..n {
                writeln!(
                    f,
                    "{} {:.6} {:.6} {:.6}",
                    symbol(species[i]),
                    bx[3 * i],
                    bx[3 * i + 1],
                    bx[3 * i + 2]
                )
                .ok();
            }
            println!("  wrote {path}");
        }
    }
}
