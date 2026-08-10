//! Global minimum of an adsorbate on a substrate through the rgpot server.
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
//! Usage: slab_adsorption <con_file> <budget> <seeds> [shells]
//! Env: RGPOT_HOST / RGPOT_PORT for the server (CuH2 for the copper case).

use anneal_core::methods::cluster_hopping::{run_with_gradient, Config, Ledger};
use anneal_core::methods::warm_lbfgs::WarmLbfgs;
use ndarray::{Array1, ArrayView1};
use rand::rngs::StdRng;
use rand::{Rng, SeedableRng};
use std::io::Write;
use std::path::Path;

use anneal_core::methods::cluster_hopping::active_mask;

use rgpot_core::rpc::client::RpcClient;
use rgpot_core::tensor::{
    rgpot_tensor_cpu_f64_2d, rgpot_tensor_cpu_f64_matrix3, rgpot_tensor_cpu_i32_1d,
    rgpot_tensor_data, rgpot_tensor_free,
};
use rgpot_core::types::{rgpot_force_input_t, rgpot_force_out_t};

/// Reads the first frame of a con file: coordinates, species, and the free
/// seeds (the atoms the file does not mark fixed).
fn read_system(path: &str) -> (Array1<f64>, Vec<u32>, Vec<usize>) {
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
    (Array1::from(pos), species, seeds)
}

struct Engine {
    client: RpcClient,
    atmnrs: Vec<i32>,
    species: Vec<u32>,
    seeds: Vec<usize>,
    shells: usize,
    tolerance: f64,
    box_: [f64; 9],
    failures: usize,
}

impl Engine {
    fn eval(&mut self, x: ArrayView1<f64>) -> Option<(f64, Array1<f64>)> {
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
                Some((out.energy, g))
            }
            Err(_) => {
                self.failures += 1;
                None
            }
        }
    }
}

fn main() {
    let args: Vec<String> = std::env::args().collect();
    let con = args.get(1).cloned().expect("usage: slab_adsorption <con_file> <budget> <seeds> [shells]");
    let budget: usize = args.get(2).and_then(|v| v.parse().ok()).unwrap_or(20_000);
    let seeds: u64 = args.get(3).and_then(|v| v.parse().ok()).unwrap_or(1);
    let shells: usize = args.get(4).and_then(|v| v.parse().ok()).unwrap_or(1);
    let seed0: u64 = std::env::var("SEED_OFFSET")
        .ok()
        .and_then(|v| v.parse().ok())
        .unwrap_or(0);
    let host = std::env::var("RGPOT_HOST").unwrap_or_else(|_| "127.0.0.1".into());
    let port: u16 = std::env::var("RGPOT_PORT")
        .ok()
        .and_then(|v| v.parse().ok())
        .unwrap_or(9999);

    for seed in seed0..(seed0 + seeds) {
        let mut rng = StdRng::seed_from_u64(seed.wrapping_mul(0x9E37).wrapping_add(3));
        let (mut x0, species, free_seeds) = read_system(&con);
        let n = species.len();
        // Different seeds start the adsorbate at different lateral offsets;
        // the substrate stays as the file placed it.
        for &a in &free_seeds {
            x0[3 * a] += (rng.random::<f64>() - 0.5) * 3.0;
            x0[3 * a + 1] += (rng.random::<f64>() - 0.5) * 3.0;
            x0[3 * a + 2] += rng.random::<f64>() * 1.0;
        }
        println!("{con}: {n} atoms, {} free seeds, {shells} active shells, budget {budget}, seed {seed}",
                 free_seeds.len());
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
        let mut eng = Engine {
            client: RpcClient::new(&host, port).expect("rgpot client"),
            atmnrs: species.iter().map(|&z| z as i32).collect(),
            species: species.clone(),
            seeds: free_seeds.clone(),
            shells,
            tolerance: cfg.bond_tolerance,
            box_: [60.0, 0.0, 0.0, 0.0, 60.0, 0.0, 0.0, 0.0, 60.0],
            failures: 0,
        };
        let mut ledger = Ledger::new(budget);
        let mut opt = WarmLbfgs::default();
        let mut relax = |led: &mut Ledger, x: ArrayView1<f64>, iters: usize| {
            opt.forget();
            let (f, xr, _) = opt.minimize(x, iters, |v| {
                if !led.charge() {
                    return None;
                }
                eng.eval(v)
            });
            (f, xr)
        };
        let out = run_with_gradient(&cfg, x0.view(), &mut ledger, &mut relax, None, &mut rng);
        println!(
            "  seed {seed}: best {:.6} eV  hops {}  engine failures {}",
            out.best, out.hops, eng.failures
        );
        if let Some(bx) = out.best_state {
            let path = format!("best_cuh2_s{seed}.xyz");
            let mut f = std::fs::File::create(&path).expect("xyz");
            writeln!(f, "{n}\nbest {:.6} eV", out.best).ok();
            for i in 0..n {
                writeln!(
                    f,
                    "{} {:.6} {:.6} {:.6}",
                    if species[i] == 29 { "Cu" } else { "H" },
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
