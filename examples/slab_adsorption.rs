//! Global minimum of H2 on a copper slab through the rgpot CuH2 backend.
//!
//! The surface-science shape of the search: a frozen substrate, free
//! adsorbate, species-aware bond-matrix connectivity so the copper slab and
//! the hydrogen molecule group themselves, and the frozen mask keeping the
//! moves and the quench off the substrate. The engine is the rgpot server
//! started with the CuH2 potential, the same server ABI as every other
//! backend, so the driver holds no engine code.
//!
//! Usage: slab_adsorption <budget> <seeds>
//! Env: RGPOT_HOST / RGPOT_PORT for the server (started with CuH2).

use anneal_core::methods::cluster_hopping::{run_with_gradient, Config, Ledger};
use anneal_core::methods::warm_lbfgs::WarmLbfgs;
use ndarray::{Array1, ArrayView1};
use rand::rngs::StdRng;
use rand::{Rng, SeedableRng};
use std::io::Write;

use rgpot_core::rpc::client::RpcClient;
use rgpot_core::tensor::{
    rgpot_tensor_cpu_f64_2d, rgpot_tensor_cpu_f64_matrix3, rgpot_tensor_cpu_i32_1d,
    rgpot_tensor_data, rgpot_tensor_free,
};
use rgpot_core::types::{rgpot_force_input_t, rgpot_force_out_t};

/// Nearest-neighbour spacing of copper, Angstrom.
const CU_NN: f64 = 2.556;

/// A three-layer fcc(100)-like slab, `nx` by `ny` sites per layer, plus one
/// H2 molecule floating above. Returns coordinates, species and frozen mask:
/// the bottom two layers frozen, the top layer and the hydrogens free.
fn build_system(nx: usize, ny: usize, rng: &mut StdRng) -> (Array1<f64>, Vec<u32>, Vec<bool>) {
    let a = CU_NN;
    let mut pos = Vec::new();
    let mut species = Vec::new();
    let mut frozen = Vec::new();
    for layer in 0..3 {
        let z = layer as f64 * a * 0.7071;
        let off = if layer % 2 == 1 { a * 0.5 } else { 0.0 };
        for i in 0..nx {
            for j in 0..ny {
                pos.extend_from_slice(&[i as f64 * a + off, j as f64 * a + off, z]);
                species.push(29);
                frozen.push(layer < 2);
            }
        }
    }
    let top_z = 2.0 * a * 0.7071;
    let cx = (nx as f64 - 1.0) * a * 0.5 + rng.random::<f64>() * a;
    let cy = (ny as f64 - 1.0) * a * 0.5 + rng.random::<f64>() * a;
    let h = 2.5 + rng.random::<f64>() * 1.5;
    pos.extend_from_slice(&[cx - 0.37, cy, top_z + h, cx + 0.37, cy, top_z + h]);
    species.push(1);
    species.push(1);
    frozen.push(false);
    frozen.push(false);
    (Array1::from(pos), species, frozen)
}

struct Engine {
    client: RpcClient,
    atmnrs: Vec<i32>,
    box_: [f64; 9],
    frozen: Vec<bool>,
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
                        for i in 0..n {
                            // Zero force on frozen atoms: the quench then
                            // leaves the substrate exactly where it stands.
                            if !self.frozen[i] {
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
    let budget: usize = args.get(1).and_then(|v| v.parse().ok()).unwrap_or(20_000);
    let seeds: u64 = args.get(2).and_then(|v| v.parse().ok()).unwrap_or(1);
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
        let (x0, species, frozen) = build_system(4, 4, &mut rng);
        let n = species.len();
        println!("H2 on a Cu slab: {n} atoms, budget {budget}, seed {seed}");
        let mut cfg = Config::recommended(n);
        cfg.burst_moves = false;
        cfg.molecular_groups = Some(vec![(0..n).collect()]);
        cfg.species = Some(species.clone());
        cfg.frozen = Some(frozen.clone());
        cfg.group_cutoff = 3.4;
        cfg.screen_steps = 10;
        cfg.relax_steps = 150;
        let mut eng = Engine {
            client: RpcClient::new(&host, port).expect("rgpot client"),
            atmnrs: species.iter().map(|&z| z as i32).collect(),
            box_: [60.0, 0.0, 0.0, 0.0, 60.0, 0.0, 0.0, 0.0, 60.0],
            frozen,
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
