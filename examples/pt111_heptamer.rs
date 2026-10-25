//! Pt(111) heptamer on the OptBench Morse slab.
//!
//! Built on the slab driver: the `.con` file supplies the cell, the
//! coordinates, and which atoms are frozen. The Morse parameters are read
//! from the OptBench problem file (`morse/smooth/linear`, the scalar pair
//! sum). Nothing in this file is a typed reference energy or a typed
//! saddle coordinate.
//!
//! rgmin quenches through the cluster search. Each pair of quenched
//! endpoints starts an rgsaddle dimer along the mobile displacement.
//! Reference `.con` files under the reference root are evaluated with the
//! same potential. `mid.con` is a band image and is not a saddle.
//!
//! ```text
//! pt111_heptamer <pos.con> <problem.toml> <reference_root> <budget> <seeds> [seed0] [recommended|plain]
//! ```
//!
//! `SCREEN_STEPS`, `RELAX_STEPS`, `DIMER_STEPS`, `DIMER_PAIRS` and
//! `DIMER_ROTATIONS` override the search and the dimer. `PAPER_OUT` is the
//! directory for one manifest per seed.

#[cfg(feature = "rgsaddle")]
mod common;

#[cfg(feature = "rgsaddle")]
use std::path::{Path, PathBuf};

#[cfg(feature = "rgsaddle")]
use anneal_core::methods::cluster_hopping::{Config, Ledger, covalent_radius};
#[cfg(feature = "rgsaddle")]
use anneal_core::methods::cluster_search::search_from;
#[cfg(feature = "rgsaddle")]
use anneal_core::run_manifest::RunManifest;
#[cfg(feature = "rgsaddle")]
use common::slab::{adsorbate_groups, place_adsorbates, read_system};
#[cfg(any(test, feature = "rgsaddle"))]
use eindir_core::Objective;
#[cfg(any(test, feature = "rgsaddle"))]
use eindir_core::bounds::Bounds;
#[cfg(any(test, feature = "rgsaddle"))]
use eindir_core::gradient::{DifferentiableObjective, Gradient};
#[cfg(any(test, feature = "rgsaddle"))]
use ndarray::{Array1, ArrayView1};

#[cfg(feature = "rgsaddle")]
const ENERGY_WINDOW_EV: f64 = 0.05;
#[cfg(feature = "rgsaddle")]
const RMSD_WINDOW_A: f64 = 0.1;

#[cfg(any(test, feature = "rgsaddle"))]
#[derive(Clone, Copy, Debug)]
struct MorseParams {
    depth: f64,
    alpha: f64,
    r0: f64,
    cutoff: f64,
    barrier_below_ev: f64,
    connected_saddles: Option<u64>,
    periodic: [bool; 3],
}

#[cfg(any(test, feature = "rgsaddle"))]
fn parse_problem(text: &str) -> Result<MorseParams, String> {
    let mut depth = None;
    let mut alpha = None;
    let mut r0 = None;
    let mut cutoff = None;
    let mut barrier = None;
    let mut connected = None;
    let mut periodic = None;
    for line in text.lines() {
        let Some((key, value)) = line.split_once('=') else {
            continue;
        };
        let key = key.trim();
        let value = value.trim().trim_matches('"');
        match key {
            "D" => depth = value.parse().ok(),
            "alpha" => alpha = value.parse().ok(),
            "r0" => r0 = value.parse().ok(),
            "cutoff" => cutoff = value.parse().ok(),
            "barrier_below_eV" => barrier = value.parse().ok(),
            "connected_saddles" => connected = value.parse().ok(),
            "pbc" => {
                let body = value
                    .trim_start_matches('[')
                    .trim_end_matches(']')
                    .split(',')
                    .map(str::trim)
                    .collect::<Vec<_>>();
                if body.len() == 3 {
                    let flag = |s: &str| match s {
                        "true" => Some(true),
                        "false" => Some(false),
                        _ => None,
                    };
                    if let (Some(x), Some(y), Some(z)) =
                        (flag(body[0]), flag(body[1]), flag(body[2]))
                    {
                        periodic = Some([x, y, z]);
                    }
                }
            }
            _ => {}
        }
    }
    let params = MorseParams {
        depth: depth.ok_or("problem file has no Morse D")?,
        alpha: alpha.ok_or("problem file has no Morse alpha")?,
        r0: r0.ok_or("problem file has no Morse r0")?,
        cutoff: cutoff.ok_or("problem file has no cutoff")?,
        barrier_below_ev: barrier.ok_or("problem file has no barrier_below_eV")?,
        connected_saddles: connected,
        periodic: periodic.unwrap_or([true, true, false]),
    };
    if !(params.depth.is_finite()
        && params.alpha.is_finite()
        && params.r0.is_finite()
        && params.cutoff.is_finite()
        && params.depth > 0.0
        && params.alpha > 0.0
        && params.r0 > 0.0
        && params.cutoff > 0.0)
    {
        return Err("Morse parameters must be finite and positive".into());
    }
    Ok(params)
}

/// `morse/smooth/linear`: Morse minus its cutoff value minus `(r-rc)` times
/// the cutoff derivative. Pairs are `i < j`. A periodic axis rounds half to
/// even, which is `nearbyint` in the OptBench pair sum.
#[cfg(any(test, feature = "rgsaddle"))]
fn morse_smooth(x: ArrayView1<f64>, params: &MorseParams, cell: [f64; 3]) -> (f64, Array1<f64>) {
    let n = x.len() / 3;
    let cutoff2 = params.cutoff * params.cutoff;
    let at_cut = (-params.alpha * (params.cutoff - params.r0)).exp();
    let offset = params.depth * (at_cut * at_cut - 2.0 * at_cut);
    let der_cut = -2.0 * params.alpha * params.depth * (at_cut * at_cut - at_cut);
    let mut energy = 0.0;
    let mut gradient = Array1::zeros(3 * n);
    for i in 0..n {
        for j in (i + 1)..n {
            let mut d = [
                x[3 * i] - x[3 * j],
                x[3 * i + 1] - x[3 * j + 1],
                x[3 * i + 2] - x[3 * j + 2],
            ];
            for k in 0..3 {
                if params.periodic[k] && cell[k] > 0.0 {
                    d[k] -= cell[k] * (d[k] / cell[k]).round_ties_even();
                }
            }
            let r2 = d[0] * d[0] + d[1] * d[1] + d[2] * d[2];
            if !(r2 > 0.0 && r2 < cutoff2) {
                continue;
            }
            let radius = r2.sqrt();
            let decay = (-params.alpha * (radius - params.r0)).exp();
            let morse = params.depth * (decay * decay - 2.0 * decay);
            energy += morse - offset - (radius - params.cutoff) * der_cut;
            let slope = -2.0 * params.alpha * params.depth * (decay * decay - decay) - der_cut;
            let scale = slope / radius;
            for k in 0..3 {
                let g = scale * d[k];
                gradient[3 * i + k] += g;
                gradient[3 * j + k] -= g;
            }
        }
    }
    (energy, gradient)
}

#[cfg(any(test, feature = "rgsaddle"))]
struct Surface {
    params: MorseParams,
    cell: [f64; 3],
    active: Vec<bool>,
    template: Array1<f64>,
    bounds: Bounds<f64>,
}

#[cfg(any(test, feature = "rgsaddle"))]
impl Surface {
    fn energy_gradient(&self, x: ArrayView1<f64>) -> (f64, Array1<f64>) {
        let mut placed = x.to_owned();
        for (i, on) in self.active.iter().enumerate() {
            if !on {
                for k in 0..3 {
                    placed[3 * i + k] = self.template[3 * i + k];
                }
            }
        }
        let (energy, mut gradient) = morse_smooth(placed.view(), &self.params, self.cell);
        for (i, on) in self.active.iter().enumerate() {
            if !on {
                for k in 0..3 {
                    gradient[3 * i + k] = 0.0;
                }
            }
        }
        (energy, gradient)
    }
}

#[cfg(any(test, feature = "rgsaddle"))]
impl Objective<f64> for Surface {
    fn dim(&self) -> usize {
        self.template.len()
    }
    fn bounds(&self) -> &Bounds<f64> {
        &self.bounds
    }
    fn eval(&self, x: ArrayView1<f64>) -> f64 {
        self.energy_gradient(x).0
    }
}

#[cfg(any(test, feature = "rgsaddle"))]
impl Gradient<f64> for Surface {
    fn dim(&self) -> usize {
        self.template.len()
    }
    fn grad(&self, x: ArrayView1<f64>) -> Array1<f64> {
        self.energy_gradient(x).1
    }
}

#[cfg(any(test, feature = "rgsaddle"))]
impl DifferentiableObjective<f64> for Surface {
    fn value_and_gradient(&self, x: ArrayView1<f64>) -> (f64, Array1<f64>) {
        self.energy_gradient(x)
    }
}

#[cfg(feature = "rgsaddle")]
fn env_usize(key: &str, default: usize) -> usize {
    std::env::var(key)
        .ok()
        .and_then(|v| v.parse().ok())
        .unwrap_or(default)
}

#[cfg(feature = "rgsaddle")]
fn hardware_model() -> String {
    std::fs::read_to_string("/proc/cpuinfo")
        .ok()
        .and_then(|text| {
            text.lines()
                .find(|line| line.starts_with("model name"))
                .and_then(|line| line.split(':').nth(1))
                .map(|value| value.trim().to_owned())
        })
        .filter(|value| !value.is_empty())
        .unwrap_or_else(|| "unknown".to_owned())
}

#[cfg(feature = "rgsaddle")]
fn command_text(program: &str, args: &[&str]) -> String {
    std::process::Command::new(program)
        .args(args)
        .output()
        .ok()
        .filter(|output| output.status.success())
        .map(|output| String::from_utf8_lossy(&output.stdout).trim().to_owned())
        .filter(|text| !text.is_empty())
        .unwrap_or_else(|| "unavailable".to_owned())
}

#[cfg(feature = "rgsaddle")]
fn provenance_json(arm: &str, seed: u64, budget: usize) -> String {
    let cores = std::thread::available_parallelism()
        .map(|n| n.get())
        .unwrap_or(0);
    let threads = std::env::var("RAYON_NUM_THREADS").unwrap_or_else(|_| cores.to_string());
    format!(
        "{{\n  \"arm\": \"{arm}\",\n  \"seed\": {seed},\n  \"budget\": {budget},\n  \"hardware_model\": \"{}\",\n  \"cores\": {cores},\n  \"threads\": \"{threads}\",\n  \"compiler\": \"{}\",\n  \"commit\": \"{}\",\n  \"package\": \"anneal-core {}\"\n}}\n",
        hardware_model().replace('"', "'"),
        command_text("rustc", &["-vV"])
            .replace('"', "'")
            .replace('\n', " | "),
        command_text("git", &["rev-parse", "HEAD"]),
        env!("CARGO_PKG_VERSION")
    )
}

#[cfg(feature = "rgsaddle")]
fn mobile_of(full: &Array1<f64>, mobile: &[usize]) -> Array1<f64> {
    let mut out = Array1::zeros(3 * mobile.len());
    for (slot, &atom) in mobile.iter().enumerate() {
        for k in 0..3 {
            out[3 * slot + k] = full[3 * atom + k];
        }
    }
    out
}

#[cfg(all(feature = "rgsaddle", feature = "ira"))]
fn rmsd(a: ArrayView1<f64>, b: ArrayView1<f64>) -> f64 {
    let n = a.len() / 3;
    if n == 0 || a.len() != b.len() {
        return f64::INFINITY;
    }
    let mut sum = 0.0;
    for i in 0..a.len() {
        let d = a[i] - b[i];
        sum += d * d;
    }
    (sum / n as f64).sqrt()
}

#[cfg(feature = "rgsaddle")]
fn permutation_rmsd(reference: ArrayView1<f64>, query: ArrayView1<f64>) -> f64 {
    let n = reference.len() / 3;
    if n == 0 || query.len() != reference.len() || n > 8 {
        return f64::INFINITY;
    }
    let mut order: Vec<usize> = (0..n).collect();
    let mut best = f64::INFINITY;
    fn walk(
        reference: ArrayView1<f64>,
        query: ArrayView1<f64>,
        order: &mut [usize],
        slot: usize,
        best: &mut f64,
    ) {
        if slot == order.len() {
            let mut sum = 0.0;
            for (i, &src) in order.iter().enumerate() {
                for k in 0..3 {
                    let d = reference[3 * i + k] - query[3 * src + k];
                    sum += d * d;
                }
            }
            let value = (sum / order.len() as f64).sqrt();
            if value < *best {
                *best = value;
            }
            return;
        }
        for i in slot..order.len() {
            order.swap(slot, i);
            walk(reference, query, order, slot + 1, best);
            order.swap(slot, i);
        }
    }
    walk(reference, query, &mut order, 0, &mut best);
    best
}

#[cfg(feature = "rgsaddle")]
fn island_rmsd(reference: ArrayView1<f64>, query: ArrayView1<f64>) -> (f64, &'static str) {
    #[cfg(feature = "ira")]
    {
        let order = anneal_core::shape::CanonicalOrder::new(reference.to_owned(), 1.8);
        if let Some(canon) = order.canonicalise(query) {
            return (rmsd(reference, canon.view()), "ira");
        }
    }
    (permutation_rmsd(reference, query), "cartesian-permutation")
}

#[cfg(feature = "rgsaddle")]
struct Geometry {
    energy: f64,
    mobile: Array1<f64>,
    saddle: bool,
}

#[cfg(feature = "rgsaddle")]
fn is_saddle_path(path: &Path) -> bool {
    path.components()
        .any(|component| matches!(component.as_os_str().to_str(), Some("saddle" | "saddles")))
}

#[cfg(feature = "rgsaddle")]
fn collect_con(dir: &Path, out: &mut Vec<PathBuf>) {
    let Ok(entries) = std::fs::read_dir(dir) else {
        return;
    };
    for entry in entries.flatten() {
        let path = entry.path();
        if path.is_dir() {
            collect_con(&path, out);
        } else if path.extension().and_then(|ext| ext.to_str()) == Some("con")
            && path.file_name().and_then(|name| name.to_str()) != Some("mid.con")
        {
            out.push(path);
        }
    }
}

#[cfg(feature = "rgsaddle")]
fn load_references(root: &Path, params: &MorseParams) -> Vec<Geometry> {
    let mut files = Vec::new();
    collect_con(root, &mut files);
    files.sort();
    let mut loaded = Vec::new();
    for path in files {
        let (pos, _species, free, box_) = read_system(path.to_str().unwrap_or(""));
        let island = island_layer(&pos, &free);
        if island.is_empty() {
            continue;
        }
        let cell = [box_[0], box_[4], box_[8]];
        let (energy, _) = morse_smooth(pos.view(), params, cell);
        loaded.push(Geometry {
            energy,
            mobile: mobile_of(&pos, &island),
            saddle: is_saddle_path(&path),
        });
    }
    loaded
}

#[cfg(feature = "rgsaddle")]
fn matched(found_e: f64, found: ArrayView1<f64>, reference: &Geometry) -> bool {
    let (deviation, _) = island_rmsd(reference.mobile.view(), found);
    (found_e - reference.energy).abs() <= ENERGY_WINDOW_EV && deviation <= RMSD_WINDOW_A
}

#[cfg(any(test, feature = "rgsaddle"))]
/// Free atoms split into layers when neighbouring z values differ by more
/// than this. The heptamer is the smallest layer. The other free layers are
/// the movable substrate.
const LAYER_GAP_A: f64 = 1.0;

#[cfg(any(test, feature = "rgsaddle"))]
fn island_layer(base: &Array1<f64>, free: &[usize]) -> Vec<usize> {
    let mut ordered = free.to_vec();
    ordered.sort_by(|&a, &b| {
        base[3 * a + 2]
            .partial_cmp(&base[3 * b + 2])
            .unwrap_or(std::cmp::Ordering::Equal)
    });
    let mut layers: Vec<Vec<usize>> = Vec::new();
    for atom in ordered {
        let split = layers.last().is_none_or(|layer| {
            let previous = *layer.last().expect("layer");
            base[3 * atom + 2] - base[3 * previous + 2] > LAYER_GAP_A
        });
        if split {
            layers.push(vec![atom]);
        } else if let Some(layer) = layers.last_mut() {
            layer.push(atom);
        }
    }
    layers
        .into_iter()
        .min_by(|left, right| {
            left.len().cmp(&right.len()).then_with(|| {
                let z = |layer: &Vec<usize>| layer.iter().map(|&i| base[3 * i + 2]).sum::<f64>();
                z(right)
                    .partial_cmp(&z(left))
                    .unwrap_or(std::cmp::Ordering::Equal)
            })
        })
        .unwrap_or_default()
}

#[cfg(feature = "rgsaddle")]
fn configure(species: &[u32], island: &[usize], base: &Array1<f64>, arm: &str) -> Config {
    let groups = adsorbate_groups(base, species, island);
    let mut cfg = if arm == "plain" {
        Config::for_molecular(species.to_vec(), groups, 1.0)
    } else {
        Config::recommended_molecular(species.to_vec(), groups, 1.0)
    };
    cfg.active_region = Some((island.to_vec(), 0));
    if !island.is_empty() {
        let adsorbate = island
            .iter()
            .map(|&i| covalent_radius(species[i]))
            .fold(0.0_f64, f64::max);
        cfg.length_scale = 2.0 * adsorbate;
    }
    cfg.screen_steps = env_usize("SCREEN_STEPS", 10);
    cfg.relax_steps = env_usize("RELAX_STEPS", 150);
    cfg
}

fn main() {
    #[cfg(not(feature = "rgsaddle"))]
    {
        eprintln!("pt111_heptamer requires the rgsaddle feature");
        std::process::exit(2);
    }
    #[cfg(feature = "rgsaddle")]
    drive();
}
