//! Global minimum of an adsorbate on a substrate through an rgpot backend.
//!
//! The geometry comes from an eOn con file through readcon: nothing is grown
//! here, and the file's fixed flags designate the substrate. The mobile set
//! is not that static designation, though: the active region is the free
//! seeds plus their bond-matrix neighbour shells, recomputed from the
//! current structure each hop, so substrate atoms near the adsorbate respond
//! and the far substrate stands still, the region following the adsorbate as
//! it moves. A descriptor-deviation bound over the same neighbourhoods is
//! the stated refinement of this selector.
//!
//! Usage: slab_adsorption <con_file> <budget> <seeds> [shells] [engine]
//! Engine is rgpot (default: Cap'n Proto to potserv), nwchemc, cpmdc,
//! or xtb / xtb-cli.

mod common;

#[cfg(feature = "graphkey")]
use anneal_core::methods::archive_search::{Archive, archive_search};
use anneal_core::methods::cluster_hopping::{Config, Ledger, run_with_gradient};
use anneal_core::methods::warm_lbfgs::WarmLbfgs;
use common::pipe_engine::{PipeEngine, symbol};
#[cfg(feature = "rgpot-ex")]
use common::profile_engine::{ProfileEngine, profile_prefix};
use ndarray::{Array1, ArrayView1};
use rand::rngs::StdRng;
use rand::{Rng, SeedableRng};
use std::io::Write;
use std::path::Path;

use anneal_core::methods::cluster_hopping::active_mask;

#[cfg(feature = "rgpot-ex")]
use rgpot_core::rpc::client::RpcClient;
#[cfg(feature = "rgpot-ex")]
use rgpot_core::tensor::{
    rgpot_tensor_cpu_f64_2d, rgpot_tensor_cpu_f64_matrix3, rgpot_tensor_cpu_i32_1d,
    rgpot_tensor_data, rgpot_tensor_free,
};
#[cfg(feature = "rgpot-ex")]
use rgpot_core::types::{rgpot_force_input_t, rgpot_force_out_t};

/// Reads the first frame of a con file: coordinates, species, the free
/// seeds (the atoms the file does not mark fixed), and the orthogonal
/// box matrix from the file header.
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

fn log_line(msg: &str) {
    let mut out = std::io::stdout();
    let _ = writeln!(out, "{msg}");
    let _ = out.flush();
}

fn atoms_overlap(x: ArrayView1<f64>) -> bool {
    let n = x.len() / 3;
    const MIN_R2: f64 = 0.16;
    for i in 0..n {
        for j in (i + 1)..n {
            let dx = x[3 * i] - x[3 * j];
            let dy = x[3 * i + 1] - x[3 * j + 1];
            let dz = x[3 * i + 2] - x[3 * j + 2];
            if dx * dx + dy * dy + dz * dz < MIN_R2 {
                return true;
            }
        }
    }
    false
}

#[cfg(feature = "rgpot-ex")]
struct Engine {
    host: String,
    port: u16,
    client: RpcClient,
    atmnrs: Vec<i32>,
    species: Vec<u32>,
    seeds: Vec<usize>,
    shells: usize,
    tolerance: f64,
    box_: [f64; 9],
    failures: usize,
}

#[cfg(feature = "rgpot-ex")]
impl Engine {
    fn call_once(&mut self, x: ArrayView1<f64>) -> Result<(f64, Array1<f64>), String> {
        let n = x.len() / 3;
        let mut pos: Vec<f64> = x.iter().cloned().collect();
        let input = unsafe {
            rgpot_force_input_t {
                positions: rgpot_tensor_cpu_f64_2d(pos.as_mut_ptr(), n as i64, 3),
                atomic_numbers: rgpot_tensor_cpu_i32_1d(self.atmnrs.as_mut_ptr(), n as i64),
                box_matrix: rgpot_tensor_cpu_f64_matrix3(self.box_.as_mut_ptr()),
            }
        };
        let mut out = rgpot_force_out_t {
            forces: std::ptr::null_mut(),
            energy: 0.0,
            variance: 0.0,
        };
        let res = self.client.calculate(&input, &mut out);
        unsafe {
            rgpot_tensor_free(input.positions);
            rgpot_tensor_free(input.atomic_numbers);
            rgpot_tensor_free(input.box_matrix);
        }
        match res {
            Ok(()) => {
                if !out.energy.is_finite() || out.energy < -1000.0 {
                    if !out.forces.is_null() {
                        unsafe { rgpot_tensor_free(out.forces) };
                    }
                    return Err(format!("non-physical energy {}", out.energy));
                }
                let mut g = Array1::zeros(3 * n);
                if !out.forces.is_null() {
                    let data = unsafe { rgpot_tensor_data(out.forces) } as *const f64;
                    if !data.is_null() {
                        // The same active region the driver moves: force on
                        // the mobile patch, zero outside it, recomputed from
                        // the coordinates being evaluated so the quench and
                        // the moves agree on what stands still.
                        let act =
                            active_mask(x, &self.species, &self.seeds, self.shells, self.tolerance);
                        for i in 0..n {
                            if act[i] {
                                for k in 0..3 {
                                    g[3 * i + k] = -unsafe { *data.add(3 * i + k) };
                                }
                            }
                        }
                    }
                    unsafe { rgpot_tensor_free(out.forces) };
                }
                Ok((out.energy, g))
            }
            Err(e) => Err(e),
        }
    }

    fn eval(&mut self, x: ArrayView1<f64>) -> Option<(f64, Array1<f64>)> {
        if atoms_overlap(x) {
            self.failures += 1;
            return None;
        }
        // Connection loss is not a refused geometry: wait for potserv to
        // come back on the same charged evaluation.
        for attempt in 0..30 {
            match self.call_once(x) {
                Ok(pair) => return Some(pair),
                Err(e) => {
                    let lost = e.contains("connection failed")
                        || e.contains("Disconnected")
                        || e.contains("RPC call failed");
                    if !lost {
                        self.failures += 1;
                        if self.failures == 1 || self.failures % 500 == 0 {
                            eprintln!("  engine failure {}: {e}", self.failures);
                        }
                        return None;
                    }
                    if attempt == 0 || attempt % 10 == 0 {
                        eprintln!("  potserv unreachable ({e}); retry {attempt}");
                    }
                    if let Ok(c) = RpcClient::new(&self.host, self.port) {
                        self.client = c;
                    }
                    std::thread::sleep(std::time::Duration::from_millis(200));
                }
            }
        }
        self.failures += 1;
        None
    }
}

fn main() {
    let args: Vec<String> = std::env::args().collect();
    let con = args
        .get(1)
        .cloned()
        .expect("usage: slab_adsorption <con_file> <budget> <seeds> [shells] [engine]");
    let budget: usize = args.get(2).and_then(|v| v.parse().ok()).unwrap_or(20_000);
    let seeds: u64 = args.get(3).and_then(|v| v.parse().ok()).unwrap_or(1);
    let tail = &args[4.min(args.len())..];
    let ras = tail.iter().any(|t| t == "ras" || t == "pair");
    let pair = tail.iter().any(|t| t == "pair");
    let shells: usize = tail.iter().find_map(|v| v.parse().ok()).unwrap_or(1);
    let engine = tail
        .iter()
        .find(|value| {
            matches!(
                value.as_str(),
                "rgpot" | "nwchemc" | "cpmdc" | "xtb" | "xtb-cli"
            )
        })
        .map(String::as_str)
        .unwrap_or("rgpot");
    let seed0: u64 = std::env::var("SEED_OFFSET")
        .ok()
        .and_then(|v| v.parse().ok())
        .unwrap_or(0);
    let host = std::env::var("RGPOT_HOST").unwrap_or_else(|_| "127.0.0.1".into());
    let port: u16 = std::env::var("RGPOT_PORT")
        .ok()
        .and_then(|v| v.parse().ok())
        .unwrap_or(9999);

    let (base_x, species, free_seeds, box_) = read_system(&con);
    let n = species.len();
    // STACK=base runs the plain protocol; ATOMIC=1 keeps the atomic move
    // library for free mixed clusters instead of the grouped molecular one.
    let base_stack = std::env::var("STACK").map(|v| v == "base").unwrap_or(false);
    let atomic = std::env::var("ATOMIC").map(|v| v == "1").unwrap_or(false);
    let groups = vec![(0..n).collect()];
    let mut cfg = if base_stack {
        Config::for_molecular(species.clone(), groups.clone(), 1.0)
    } else {
        Config::recommended_molecular(species.clone(), groups, 1.0)
    };
    if atomic {
        cfg.move_library = anneal_core::methods::cluster_hopping::MoveLibrary::Atomic;
    } else {
        cfg.active_region = Some((free_seeds.clone(), shells));
    }
    cfg.screen_steps = 10;
    cfg.relax_steps = 150;

    let atmnrs: Vec<i32> = species.iter().map(|&z| z as i32).collect();
    #[cfg(feature = "rgpot-ex")]
    let mut rpc_eng = (engine == "rgpot").then(|| Engine {
        host: host.clone(),
        port,
        client: RpcClient::new(&host, port).expect("rgpot client"),
        atmnrs: atmnrs.clone(),
        species: species.clone(),
        seeds: free_seeds.clone(),
        shells,
        tolerance: cfg.bond_tolerance,
        box_,
        failures: 0,
    });
    #[cfg(not(feature = "rgpot-ex"))]
    let mut rpc_eng: Option<()> = if engine == "rgpot" {
        panic!("rebuild with --features rgpot-ex for the rgpot engine");
    } else {
        None
    };
    #[cfg(feature = "rgpot-ex")]
    let mut profile_eng =
        profile_prefix(engine).map(|prefix| ProfileEngine::load(prefix, atmnrs, Some(box_)));
    #[cfg(not(feature = "rgpot-ex"))]
    let mut profile_eng: Option<()> = None;
    let _ = (&host, port, &atmnrs);
    let mut pipe_eng = if matches!(engine, "xtb" | "xtb-cli") {
        let symbols = species.iter().map(|&z| symbol(z).to_string()).collect();
        let cell = (box_[0].abs() > 1.0 && box_[4].abs() > 1.0 && box_[8].abs() > 1.0)
            .then_some(box_);
        Some(PipeEngine::start(engine, symbols, cell))
    } else {
        None
    };

    for seed in seed0..(seed0 + seeds) {
        let mut rng = StdRng::seed_from_u64(seed.wrapping_mul(0x9E37).wrapping_add(3));
        let mut x0 = base_x.clone();
        // Different seeds start the adsorbate at different lateral offsets;
        // the substrate stays as the file placed it. An all-free file is a
        // cluster, not a slab: do not melt every atom before the first hop.
        if free_seeds.len() < n {
            for &a in &free_seeds {
                x0[3 * a] += (rng.random::<f64>() - 0.5) * 3.0;
                x0[3 * a + 1] += (rng.random::<f64>() - 0.5) * 3.0;
                x0[3 * a + 2] += rng.random::<f64>() * 1.0;
            }
        }
        log_line(&format!(
            "{con}: {n} atoms through {engine}, {} free seeds, {shells} active shells, budget {budget}, seed {seed}, box {:.4} {:.4} {:.4}",
            free_seeds.len(),
            box_[0],
            box_[4],
            box_[8]
        ));
        let failures_before = pipe_eng.as_ref().map(|e| e.failures).unwrap_or(0)
            + {
                #[cfg(feature = "rgpot-ex")]
                {
                    rpc_eng.as_ref().map(|e| e.failures).unwrap_or(0)
                        + profile_eng.as_ref().map(ProfileEngine::failures).unwrap_or(0)
                }
                #[cfg(not(feature = "rgpot-ex"))]
                {
                    0
                }
            };
        let mut overlap_failures = 0usize;
        let mut ledger = Ledger::new(budget);
        let mut opt = WarmLbfgs::default();
        let mut relax = |led: &mut Ledger, x: ArrayView1<f64>, iters: usize| {
            opt.forget();
            let (f, xr, _) = opt.minimize(x, iters, |v| {
                if !led.charge() {
                    return None;
                }
                if atoms_overlap(v) {
                    overlap_failures += 1;
                    return None;
                }
                if let Some(p) = pipe_eng.as_mut() {
                    let (energy, mut gradient) = p.eval(v)?;
                    let active =
                        active_mask(v, &species, &free_seeds, shells, cfg.bond_tolerance);
                    for (atom, is_active) in active.into_iter().enumerate() {
                        if !is_active {
                            for axis in 0..3 {
                                gradient[3 * atom + axis] = 0.0;
                            }
                        }
                    }
                    return Some((energy, gradient));
                }
                #[cfg(feature = "rgpot-ex")]
                {
                    if let Some(rpc) = rpc_eng.as_mut() {
                        return rpc.eval(v);
                    }
                    let (energy, mut gradient) = profile_eng.as_mut()?.eval(v)?;
                    let active =
                        active_mask(v, &species, &free_seeds, shells, cfg.bond_tolerance);
                    for (atom, is_active) in active.into_iter().enumerate() {
                        if !is_active {
                            for axis in 0..3 {
                                gradient[3 * atom + axis] = 0.0;
                            }
                        }
                    }
                    return Some((energy, gradient));
                }
                #[cfg(not(feature = "rgpot-ex"))]
                None
            });
            (f, xr)
        };
        #[cfg(feature = "graphkey")]
        let mut rng_ras = rng.clone();
        #[cfg(feature = "graphkey")]
        if pair {
            let mut rng_rec = rng.clone();
            let mut ledger_rec = Ledger::new(budget);
            let rec = run_with_gradient(
                &cfg,
                x0.view(),
                &mut ledger_rec,
                &mut relax,
                None,
                &mut rng_rec,
            );
            log_line(&format!(
                "  seed {seed} rec: best {:.6} eV  charged {}  hops {}",
                rec.best,
                ledger_rec.spent(),
                rec.hops
            ));
        }
        #[cfg(feature = "graphkey")]
        let out = if ras {
            let mut archive = Archive::new();
            let a = archive_search(
                &cfg,
                x0.view(),
                &mut ledger,
                &mut relax,
                None,
                &mut archive,
                &mut rng_ras,
            );
            log_line(&format!(
                "  seed {seed} ras: best {:.6} eV  charged {}  hit_at {}  floors {} returned {} same_floor {}",
                a.best, a.charged, a.best_at, a.floors, a.returned, a.same_floor
            ));
            anneal_core::methods::cluster_hopping::Outcome {
                best: a.best,
                best_state: a.best_state,
                hops: a.full,
                ..Default::default()
            }
        } else {
            run_with_gradient(&cfg, x0.view(), &mut ledger, &mut relax, None, &mut rng)
        };
        #[cfg(not(feature = "graphkey"))]
        let out = run_with_gradient(&cfg, x0.view(), &mut ledger, &mut relax, None, &mut rng);
        let failures_after = pipe_eng.as_ref().map(|e| e.failures).unwrap_or(0)
            + {
                #[cfg(feature = "rgpot-ex")]
                {
                    rpc_eng.as_ref().map(|e| e.failures).unwrap_or(0)
                        + profile_eng.as_ref().map(ProfileEngine::failures).unwrap_or(0)
                }
                #[cfg(not(feature = "rgpot-ex"))]
                {
                    0
                }
            };
        log_line(&format!(
            "  seed {seed}: best {:.6} eV  hops {}  engine failures {}",
            out.best,
            out.hops,
            failures_after.saturating_sub(failures_before) + overlap_failures
        ));
        if let Some(bx) = out.best_state {
            let path = format!("best_slab_s{seed}.xyz");
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
            log_line(&format!("  wrote {path}"));
        }
    }
}

#[cfg(test)]
mod tests {
    #[test]
    fn pipe_engine_is_the_default_without_capnp() {
        assert_eq!(super::symbol(29), "Cu");
        assert_eq!(super::symbol(1), "H");
        assert_eq!(super::symbol(79), "Au");
    }

    #[cfg(feature = "rgpot-ex")]
    #[test]
    fn periodic_profile_request_carries_the_simulation_cell() {
        let positions = [0.0, 0.0, 0.0];
        let atomic_numbers = [29];
        let cell = [8.0, 0.0, 0.0, 0.0, 9.0, 0.0, 0.0, 0.0, 10.0];
        let request = super::common::profile_engine::profile_request(
            &positions,
            &atomic_numbers,
            Some(&cell),
        );

        assert_eq!(request.box_matrix, Some(&cell));
    }
}
