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

fn main() {}
