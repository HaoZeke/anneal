//! Where a cluster search spends its force calls, on an optbench start set.
//!
//! Runs [`anneal_core::methods::cluster_search::search_from`] (the path
//! `anneal.cluster_search(..., start=...)` takes) from each start file, with
//! the optbench lane-B counting rule: one fresh coordinate is one call and
//! an immediate repeat is free. A start ends at its first converged hit,
//! a fresh evaluation within `0.001` of the target with every gradient
//! component at most `1e-3`, or at the cap. Prints the call count and the
//! relaxation and hop counters for each start, then PAR-2.
//!
//! This is a development profile, not a board score: the board number is
//! the one `scripts/optbench/anneal_globalopt.py` writes through the
//! optbench harness.
//!
//! Usage:
//! `cargo run --release --example globalopt_profile -- <communicating|recommended> <target> <cap> <seed0> <species|-> <start>...`
//!
//! A start is whitespace-separated Cartesian coordinates. `species` is a
//! file of one label per point (1 or 2) for binary Lennard-Jones with
//! sigma 1 and 1.3, arithmetic-mean mixing and cutoff 20; `-` is plain
//! Lennard-Jones with cutoff 40.

use anneal_core::methods::cluster_hopping::{Config, Ledger};
use anneal_core::methods::cluster_search::search_from;
use eindir_core::Objective;
use eindir_core::bounds::Bounds;
use eindir_core::gradient::{DifferentiableObjective, Gradient};
use ndarray::{Array1, ArrayView1};
use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::{Arc, Mutex};

enum Model {
    LennardJones,
    Binary(Vec<f64>),
    /// GMIN rigid TIP4P (optbench.tip4p), on anneal's chart: centres, then
    /// active rotation vectors, the negatives of GMIN's angle-axis vectors.
    Tip4p([[f64; 3]; 4]),
}

const TIP4P_C6: f64 = 2552.24;
const TIP4P_C12: f64 = 2510.4e3;
const TIP4P_COULOMB: f64 = 1389.354848;
const TIP4P_CHARGES: [f64; 4] = [0.0, 0.52, 0.52, -1.04];

/// O, H, H, M of optbench.tip4p.reference_sites, centre-of-mass frame.
fn tip4p_sites() -> [[f64; 3]; 4] {
    let half = 0.5 * 104.52_f64.to_radians();
    let r = 0.9572;
    let mut sites = [
        [0.0, 0.0, 0.0],
        [0.0, half.sin() * r, half.cos() * r],
        [0.0, -half.sin() * r, half.cos() * r],
        [0.0, 0.0, 0.15],
    ];
    let masses = [16.0, 1.0, 1.0, 0.0];
    let total: f64 = masses.iter().sum();
    let centre: [f64; 3] =
        std::array::from_fn(|k| (0..4).map(|s| masses[s] * sites[s][k]).sum::<f64>() / total);
    for site in sites.iter_mut() {
        for k in 0..3 {
            site[k] -= centre[k];
        }
    }
    sites
}

/// GMIN `source/tip.f90` rotation of `p` by angle-axis `w` and d r / d w.
fn gmin_rotate(w: [f64; 3], p: [f64; 3]) -> ([f64; 3], [[f64; 3]; 3]) {
    let alpha2 = w[0] * w[0] + w[1] * w[1] + w[2] * w[2];
    let alpha = alpha2.sqrt();
    let small = alpha < 1e-8;
    let (cosine, c3, sine_over, d_cosine, d_c3, d_sine) = if small {
        (1.0, -0.5, 1.0, 0.0, 0.0, 0.0)
    } else {
        let (sine, cosine) = alpha.sin_cos();
        (
            cosine,
            (cosine - 1.0) / alpha2,
            sine / alpha,
            -sine,
            (-sine * alpha2 - (cosine - 1.0) * 2.0 * alpha) / (alpha2 * alpha2),
            (cosine * alpha - sine) / alpha2,
        )
    };
    let d_alpha: [f64; 3] = std::array::from_fn(|k| if small { 0.0 } else { w[k] / alpha });
    let projection = p[0] * w[0] + p[1] * w[1] + p[2] * w[2];
    let cross = [
        w[1] * p[2] - w[2] * p[1],
        w[2] * p[0] - w[0] * p[2],
        w[0] * p[1] - w[1] * p[0],
    ];
    let out =
        std::array::from_fn(|a| cosine * p[a] - c3 * projection * w[a] - sine_over * cross[a]);
    let mut d = [[0.0; 3]; 3];
    for k in 0..3 {
        let mut unit = [0.0; 3];
        unit[k] = 1.0;
        let ucross = [
            unit[1] * p[2] - unit[2] * p[1],
            unit[2] * p[0] - unit[0] * p[2],
            unit[0] * p[1] - unit[1] * p[0],
        ];
        for a in 0..3 {
            d[a][k] = d_cosine * d_alpha[k] * p[a]
                - d_c3 * d_alpha[k] * projection * w[a]
                - c3 * p[k] * w[a]
                - c3 * projection * unit[a]
                - d_sine * d_alpha[k] * cross[a]
                - sine_over * ucross[a];
        }
    }
    (out, d)
}

fn tip4p_energy_gradient(sites: &[[f64; 3]; 4], x: &[f64]) -> (f64, Vec<f64>) {
    let n = x.len() / 6;
    let mut placed = vec![[[0.0; 3]; 4]; n];
    let mut deriv = vec![[[[0.0; 3]; 3]; 4]; n];
    for m in 0..n {
        let w = [
            -x[3 * n + 3 * m],
            -x[3 * n + 3 * m + 1],
            -x[3 * n + 3 * m + 2],
        ];
        for s in 0..4 {
            let (r, d) = gmin_rotate(w, sites[s]);
            for a in 0..3 {
                placed[m][s][a] = x[3 * m + a] + r[a];
            }
            deriv[m][s] = d;
        }
    }
    let mut site_grad = vec![[[0.0; 3]; 4]; n];
    let mut e = 0.0;
    for l in 0..n {
        for r in (l + 1)..n {
            let d: [f64; 3] = std::array::from_fn(|a| placed[l][0][a] - placed[r][0][a]);
            let inv = 1.0 / (d[0] * d[0] + d[1] * d[1] + d[2] * d[2]);
            let r6 = inv * inv * inv;
            let r12 = r6 * r6;
            e += TIP4P_C12 * r12 - TIP4P_C6 * r6;
            let slope = (-12.0 * TIP4P_C12 * r12 + 6.0 * TIP4P_C6 * r6) * inv;
            for a in 0..3 {
                site_grad[l][0][a] += slope * d[a];
                site_grad[r][0][a] -= slope * d[a];
            }
            for sa in 1..4 {
                for sb in 1..4 {
                    let d: [f64; 3] = std::array::from_fn(|a| placed[l][sa][a] - placed[r][sb][a]);
                    let r2 = d[0] * d[0] + d[1] * d[1] + d[2] * d[2];
                    let pair = TIP4P_COULOMB * TIP4P_CHARGES[sa] * TIP4P_CHARGES[sb] / r2.sqrt();
                    e += pair;
                    let slope = -pair / r2;
                    for a in 0..3 {
                        site_grad[l][sa][a] += slope * d[a];
                        site_grad[r][sb][a] -= slope * d[a];
                    }
                }
            }
        }
    }
    let mut g = vec![0.0; 6 * n];
    for m in 0..n {
        for a in 0..3 {
            g[3 * m + a] = (0..4).map(|s| site_grad[m][s][a]).sum();
        }
        for k in 0..3 {
            let mut total = 0.0;
            for s in 0..4 {
                for a in 0..3 {
                    total += deriv[m][s][a][k] * site_grad[m][s][a];
                }
            }
            g[3 * n + 3 * m + k] = -total;
        }
    }
    (e, g)
}

struct Counted {
    dim: usize,
    model: Model,
    bounds: Bounds<f64>,
    target: f64,
    state: Mutex<(Option<Array1<f64>>, usize, Option<usize>)>,
    stop: Arc<AtomicBool>,
}

impl Counted {
    fn energy_gradient(&self, x: ArrayView1<f64>) -> (f64, Array1<f64>) {
        let x = x.to_vec();
        if let Model::Tip4p(sites) = &self.model {
            let (e, g) = tip4p_energy_gradient(sites, &x);
            return (e, Array1::from(g));
        }
        let (sigma, cutoff2) = match &self.model {
            Model::Binary(sigma) => (Some(sigma), 400.0),
            _ => (None, 1600.0),
        };
        let n = self.dim / 3;
        let mut e = 0.0;
        let mut g = vec![0.0; 3 * n];
        for i in 0..n {
            let (xi, rest) = x[3 * i..].split_at(3);
            let mut gi = [0.0; 3];
            for (jj, xj) in rest.as_chunks::<3>().0.iter().enumerate() {
                let j = i + 1 + jj;
                let d = [xi[0] - xj[0], xi[1] - xj[1], xi[2] - xj[2]];
                let r2 = d[0] * d[0] + d[1] * d[1] + d[2] * d[2];
                if r2.partial_cmp(&0.0) != Some(std::cmp::Ordering::Greater)
                    || r2.partial_cmp(&cutoff2) != Some(std::cmp::Ordering::Less)
                {
                    continue;
                }
                let s2 = match sigma {
                    Some(s) => 0.25 * (s[i] + s[j]) * (s[i] + s[j]),
                    None => 1.0,
                };
                let inv = 1.0 / r2;
                let sr2 = s2 * inv;
                let sr6 = sr2 * sr2 * sr2;
                let sr12 = sr6 * sr6;
                e += sr12 - sr6;
                let slope = 24.0 * (sr6 - 2.0 * sr12) * inv;
                for k in 0..3 {
                    gi[k] += slope * d[k];
                    g[3 * j + k] -= slope * d[k];
                }
            }
            for k in 0..3 {
                g[3 * i + k] += gi[k];
            }
        }
        (4.0 * e, Array1::from(g))
    }

    fn fresh(&self, x: ArrayView1<f64>) -> (f64, Array1<f64>) {
        let (e, g) = self.energy_gradient(x);
        let mut state = self.state.lock().unwrap();
        if state.0.as_ref().is_none_or(|last| last.view() != x) {
            state.0 = Some(x.to_owned());
            state.1 += 1;
            let worst = g.iter().fold(0.0_f64, |a, v| a.max(v.abs()));
            if state.2.is_none() && e <= self.target + 1e-3 && worst <= 1e-3 {
                state.2 = Some(state.1);
                self.stop.store(true, Ordering::Relaxed);
            }
        }
        (e, g)
    }
}

impl Objective<f64> for Counted {
    fn dim(&self) -> usize {
        self.dim
    }
    fn bounds(&self) -> &Bounds<f64> {
        &self.bounds
    }
    fn eval(&self, x: ArrayView1<f64>) -> f64 {
        self.fresh(x).0
    }
}

impl Gradient<f64> for Counted {
    fn grad(&self, x: ArrayView1<f64>) -> Array1<f64> {
        self.fresh(x).1
    }
    fn dim(&self) -> usize {
        self.dim
    }
}

impl DifferentiableObjective<f64> for Counted {
    fn value_and_gradient(&self, x: ArrayView1<f64>) -> (f64, Array1<f64>) {
        self.fresh(x)
    }
}

/// `ANNEAL_PROFILE_SET="field=value,..."` overrides preset fields, so an
/// arm can be profiled without a rebuild.
fn apply_overrides(cfg: &mut Config) {
    let Ok(spec) = std::env::var("ANNEAL_PROFILE_SET") else {
        return;
    };
    for item in spec.split(',').filter(|s| !s.is_empty()) {
        let (key, value) = item.split_once('=').expect("field=value");
        let flag = || matches!(value, "1" | "true");
        let real = || value.parse::<f64>().expect("number");
        let count = || value.parse::<usize>().expect("integer");
        match key {
            "orbit_complete_on_new" => cfg.orbit_complete_on_new = flag(),
            "point_symmetrise_on_new" => cfg.point_symmetrise_on_new = flag(),
            "symmetrise_on_stall" => cfg.symmetrise_on_stall = flag(),
            "symmetrise_patience" => cfg.symmetrise_patience = count(),
            "return_screen" => cfg.return_screen = flag(),
            "allocate_moves" => cfg.allocate_moves = flag(),
            "depth_reward" => cfg.depth_reward = flag(),
            "tabu_on_stall" => cfg.tabu_on_stall = flag(),
            "angular_moves" => cfg.angular_moves = flag(),
            "jump_on_stall" => cfg.jump_on_stall = flag(),
            "jump_patience" => cfg.jump_patience = count(),
            "relax_steps" => cfg.relax_steps = count(),
            "screen_steps" => cfg.screen_steps = count(),
            "temperature" => cfg.temperature = real(),
            "screen_margin" => cfg.screen_margin = real(),
            "calibrated_screen" => cfg.calibrated_screen = flag(),
            "finish_improving" => cfg.finish_improving = flag(),
            "record_gradient" => cfg.record_gradient = real(),
            "surfaces" if value == "0" => cfg.surfaces.clear(),
            "surfaces" => {
                cfg.surfaces = vec![anneal_core::methods::two_phase::TwoPhase::relative(
                    real(),
                    1.0,
                )]
            }
            other => panic!("ANNEAL_PROFILE_SET has no field {other}"),
        }
    }
}

fn read_floats(path: &str) -> Vec<f64> {
    std::fs::read_to_string(path)
        .unwrap_or_else(|e| panic!("{path}: {e}"))
        .split_whitespace()
        .map(|t| t.parse().unwrap_or_else(|e| panic!("{path}: {t}: {e}")))
        .collect()
}

fn main() {
    let args: Vec<String> = std::env::args().skip(1).collect();
    assert!(args.len() >= 6, "see the usage line in the source");
    let preset = args[0].clone();
    let target: f64 = args[1].parse().unwrap();
    let cap: usize = args[2].parse().unwrap();
    let seed0: u64 = args[3].parse().unwrap();
    let tip4p = args[4] == "tip4p";
    let sigma = (args[4] != "-" && !tip4p).then(|| {
        read_floats(&args[4])
            .into_iter()
            .map(|label| if label == 1.0 { 1.0 } else { 1.3 })
            .collect::<Vec<_>>()
    });
    let starts: Vec<(usize, String)> = args[5..].iter().cloned().enumerate().collect();
    let queue = Mutex::new(starts.into_iter());
    let rows = Mutex::new(Vec::new());
    let workers = std::thread::available_parallelism().map_or(1, |n| n.get());
    std::thread::scope(|scope| {
        for _ in 0..workers {
            scope.spawn(|| {
                loop {
                    let Some((index, path)) = queue.lock().unwrap().next() else {
                        break;
                    };
                    let mut x0 = Array1::from(read_floats(&path));
                    let dim = x0.len();
                    let n = if tip4p { dim / 6 } else { dim / 3 };
                    if tip4p {
                        // GMIN's angle-axis vectors rotate passively.
                        x0.slice_mut(ndarray::s![3 * n..]).mapv_inplace(|v| -v);
                    }
                    let mut cfg = match (preset.as_str(), tip4p) {
                        ("communicating", false) => Config::communicating(n),
                        ("recommended", false) => Config::recommended(n),
                        ("communicating", true) => Config::communicating_tip4p(n),
                        ("recommended", true) => Config::recommended_tip4p(n),
                        ("base", false) => Config::for_cluster(n),
                        ("base", true) => Config::for_tip4p(n),
                        (other, _) => panic!("no preset {other}"),
                    };
                    if tip4p {
                        cfg.rigid_body_geometry = Some(
                            anneal_core::rigid_body::RigidBodyGeometry::new(
                                tip4p_sites()[..3].to_vec(),
                                vec![8, 1, 1],
                            )
                            .unwrap(),
                        );
                    }
                    apply_overrides(&mut cfg);
                    let extent = if tip4p {
                        4.0 * 3.15365 * (n as f64).cbrt()
                    } else {
                        4.0 * 2.0_f64.powf(1.0 / 6.0) * (n as f64).cbrt()
                    };
                    let stop = Arc::new(AtomicBool::new(false));
                    let objective = Counted {
                        dim,
                        model: match (&sigma, tip4p) {
                            (_, true) => Model::Tip4p(tip4p_sites()),
                            (Some(s), _) => Model::Binary(s.clone()),
                            (None, _) => Model::LennardJones,
                        },
                        bounds: Bounds::new(
                            Array1::from_elem(dim, -extent),
                            Array1::from_elem(dim, extent),
                            0.0,
                        ),
                        target,
                        state: Mutex::new((None, 0, None)),
                        stop: Arc::clone(&stop),
                    };
                    let start_energy = objective.energy_gradient(x0.view()).0;
                    let mut ledger = Ledger::with_stop(cap, stop);
                    let started = std::time::Instant::now();
                    let (out, stats) = search_from(
                        &objective,
                        &cfg,
                        &mut ledger,
                        x0.view(),
                        seed0 + index as u64,
                    );
                    let (_, calls, hit) = *objective.state.lock().unwrap();
                    let line = format!(
                        "{path} start={start_energy:.9} hit={} calls={} best={:.6} hops={} basins={} screened={} returned={} visits={:?} relax={} screens={} converged={} capped={} screen_calls={} full_calls={} check_calls={} secs={:.1}",
                        hit.is_some(),
                        hit.unwrap_or(calls),
                        out.best,
                        out.hops,
                        out.basins,
                        out.screened_out,
                        out.returned,
                        out.visit_counts,
                        stats.total(),
                        stats.screens,
                        stats.converged,
                        stats.capped,
                        stats.screen_charged,
                        stats.full_charged,
                        stats.check_charged,
                        started.elapsed().as_secs_f64(),
                    );
                    println!("{line}");
                    rows.lock().unwrap().push((hit, calls));
                }
            });
        }
    });
    let rows = rows.into_inner().unwrap();
    let total = rows.len() as f64;
    let hits = rows.iter().filter(|(hit, _)| hit.is_some()).count();
    let par2 = rows
        .iter()
        .map(|(hit, _)| hit.map_or(2.0 * cap as f64, |c| c as f64))
        .sum::<f64>()
        / total;
    let mut solved: Vec<usize> = rows.iter().filter_map(|(hit, _)| *hit).collect();
    solved.sort_unstable();
    let median = solved.get(solved.len() / 2).copied().unwrap_or(0);
    println!(
        "preset={preset} starts={} hits={hits} par2={par2:.2} median_hit={median}",
        rows.len()
    );
}
