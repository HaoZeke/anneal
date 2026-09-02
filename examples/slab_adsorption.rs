//! Adsorbate search on a substrate. The potential is an rgpot handle
//! viewed as an eindir objective. Anneal only runs [`cluster_search`].
//!
//! Usage: slab_adsorption <con_file> <budget> <seeds>

mod common;

use anneal_core::methods::cluster_hopping::{Config, Ledger, SoapProposalMode, covalent_radius};
use anneal_core::methods::cluster_search::{search_from_maybe_bank, verify};
use common::efficiency::{apply_two_phase, bank_label, report_eval_wall, report_trace};
use common::rgpot_eindir::{RgpotObjective, emit_engine_manifest};
use eindir_core::Objective;
use eindir_core::bounds::Bounds;
use eindir_core::gradient::{DifferentiableObjective, Gradient};
use ndarray::{Array1, ArrayView1};
use rand::Rng;
use rand::SeedableRng;
use rand::rngs::StdRng;
use std::io::Write;
use std::path::Path;

fn soap_mode_from_env() -> SoapProposalMode {
    match std::env::var("ANNEAL_SOAP_MODE").as_deref() {
        Ok("flexible") | Err(_) => SoapProposalMode::Flexible,
        Ok("rigid") => SoapProposalMode::Rigid,
        Ok("off") => SoapProposalMode::Off,
        Ok(value) => panic!("ANNEAL_SOAP_MODE must be flexible, rigid, or off; got {value}"),
    }
}

fn write_resolved_config(cfg: &Config) {
    let Ok(path) = std::env::var("ANNEAL_RESOLVED_CONFIG") else {
        return;
    };
    std::fs::write(
        &path,
        cfg.resolved_json()
            .expect("serialize resolved configuration"),
    )
    .unwrap_or_else(|error| panic!("write resolved configuration {path}: {error}"));
}

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

/// Place the free adsorbate at a seed-dependent site above the slab.
///
/// The con file is already a minimum. Eight seeds from that geometry
/// all report the start energy and never search. A random in-plane
/// site and height is a start the quench has to walk back from, so
/// first-encounter charged evaluations are a real comparison.
fn displace_adsorbate(
    base: &Array1<f64>,
    species: &[u32],
    free: &[usize],
    box_: [f64; 9],
    seed: u64,
) -> Array1<f64> {
    if free.is_empty() {
        return base.clone();
    }
    let mut rng = StdRng::seed_from_u64(seed.wrapping_mul(0x9E37).wrapping_add(3));
    let mut x = base.clone();
    let nfree = free.len() as f64;
    let mut c = [0.0; 3];
    for &i in free {
        for k in 0..3 {
            c[k] += x[3 * i + k];
        }
    }
    for v in c.iter_mut() {
        *v /= nfree;
    }
    let z_top = (0..species.len())
        .filter(|&i| species[i] != 1 && !free.contains(&i))
        .map(|i| base[3 * i + 2])
        .fold(f64::NEG_INFINITY, f64::max);
    let lx = box_[0].abs().max(1.0);
    let ly = box_[4].abs().max(1.0);
    let target = [
        rng.random::<f64>() * lx,
        rng.random::<f64>() * ly,
        z_top + 2.0 + rng.random::<f64>() * 5.0,
    ];
    let ang = rng.random::<f64>() * 2.0 * std::f64::consts::PI;
    let (sa, ca) = ang.sin_cos();
    for &i in free {
        let rel = [x[3 * i] - c[0], x[3 * i + 1] - c[1], x[3 * i + 2] - c[2]];
        x[3 * i] = target[0] + ca * rel[0] - sa * rel[1];
        x[3 * i + 1] = target[1] + sa * rel[0] + ca * rel[1];
        x[3 * i + 2] = target[2] + rel[2];
    }
    x
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

impl<O: Objective<f64> + Gradient<f64>> DifferentiableObjective<f64> for Mobile<'_, O> {}

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
    if !free_seeds.is_empty() && free_seeds.len() < n {
        let mut frozen = vec![true; n];
        for &i in &free_seeds {
            frozen[i] = false;
        }
        cfg.frozen = Some(frozen);
    }
    cfg.soap_mode = soap_mode_from_env();
    apply_two_phase(&mut cfg);
    write_resolved_config(&cfg);
    let pot = RgpotObjective::cuh2(&atmnrs, box_);
    emit_engine_manifest("cuh2");
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
        "{con}: {n} atoms through eindir/rgpot cuh2, {} free, arm {}, budget {budget}, seeds {seed0}..{}",
        free_seeds.len(),
        bank_label(),
        seed0 + seeds
    );
    for seed in seed0..seed0 + seeds {
        let x0 = displace_adsorbate(&base_x, &species, &free_seeds, box_, seed);
        if seed == seed0 {
            report_eval_wall(&obj, x0.view(), "cuh2");
        }
        let mut ledger = Ledger::new(budget);
        let (out, stats) = search_from_maybe_bank(&obj, &cfg, &mut ledger, x0.view(), seed);
        let checked = verify(&obj, &out);
        println!(
            "  seed {seed}: best {:.6} eV  hops {}  charged {}  basins {}  converged {}/{}  arm {}",
            out.best,
            out.hops,
            ledger.spent(),
            out.basins,
            stats.converged,
            stats.total(),
            bank_label()
        );
        println!(
            "    quench charged screen/full/check {}/{}/{}  screens {}  capped {}",
            stats.screen_charged,
            stats.full_charged,
            stats.check_charged,
            stats.screens,
            stats.capped
        );
        if let Some((e, g)) = checked {
            println!("    verify e={e:.6} |g|_mobile={g:.3e}");
        }
        report_trace(&out, ledger.spent());
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
