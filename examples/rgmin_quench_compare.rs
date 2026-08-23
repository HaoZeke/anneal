//! Does a different rgmin method change an LJ75 Leave landing?
//!
//! Hopping already quenches with [`rgmin::Lbfgs`] through
//! [`anneal_core::methods::warm_lbfgs::WarmLbfgs`]. This binary asks
//! whether FIRE 2.0 (Bitzek 2006 / Guénolé 2020; the method current
//! rgmin ships and the pin does not) lands a different
//! packing from the same start, or only spends a different number of
//! evaluations.
//!
//!     rgmin_quench_compare [RELAX_STEPS]
//!
//! Starts: sealed ico, sealed Marks, fivefold residual 0.35 from ico,
//! and three first-rung packing-map covers. Each start is quenched
//! with L-BFGS and with FIRE 2.0.

use anneal_core::catalog::{PACKING_LINK, PackingBook, packing_link_labels};
use anneal_core::known_basin;
use anneal_core::methods::warm_lbfgs::WarmLbfgs;
use anneal_core::potentials::{PairKind, PairPotential};
use ndarray::{Array1, ArrayView1};

fn load_xyz(text: &str) -> Array1<f64> {
    let coordinates = text
        .lines()
        .skip(2)
        .filter(|line| !line.trim().is_empty())
        .flat_map(|line| {
            line.split_whitespace()
                .skip(1)
                .take(3)
                .map(|value| value.parse::<f64>().expect("fixture coordinate parses"))
                .collect::<Vec<f64>>()
        })
        .collect::<Vec<f64>>();
    Array1::from(coordinates)
}

fn ginf(g: ArrayView1<f64>) -> f64 {
    g.iter().fold(0.0_f64, |a, v| a.max(v.abs()))
}

fn community_of(ico: &[f64], marks: &[f64], trial: &[f64]) -> usize {
    let mut book = PackingBook::default();
    for state in [ico, marks, trial] {
        book.observe(state);
    }
    let histograms: Vec<Vec<f64>> = [ico, marks, trial]
        .into_iter()
        .filter_map(|state| book.histogram(state))
        .collect();
    if histograms.len() != 3 {
        return 2;
    }
    let labels = packing_link_labels(&histograms, PACKING_LINK);
    if labels[2] == labels[0] {
        0
    } else if labels[2] == labels[1] {
        1
    } else {
        2
    }
}

/// FIRE 2.0 (Guénolé 2020): mix, then semi-implicit Euler.
///
/// Same integrator current rgmin runs as `Method::Fire`.
/// The anneal pin is older and does not export it, so the loop lives
/// here until that pin moves. Bitzek et al., Phys. Rev. Lett. 97, 170201.
fn fire2(
    potential: &PairPotential,
    x0: ArrayView1<f64>,
    max_iter: usize,
) -> (f64, Array1<f64>, usize, f64) {
    let mut x = x0.to_owned();
    let (mut energy, mut grad) = potential.value_and_gradient(x.view());
    let mut evals = 1usize;
    let mut vel = Array1::<f64>::zeros(x.len());
    let mut dt = 0.1_f64;
    let dt_max = 0.25_f64;
    let mut alpha = 0.1_f64;
    let mut n_pos = 0usize;
    for _ in 0..max_iter {
        let g = ginf(grad.view());
        if g < 1e-6 {
            break;
        }
        let force = grad.mapv(|v| -v);
        let power = force
            .iter()
            .zip(vel.iter())
            .map(|(f, v)| f * v)
            .sum::<f64>();
        if power > 0.0 {
            let fnorm = force.iter().map(|v| v * v).sum::<f64>().sqrt();
            let vnorm = vel.iter().map(|v| v * v).sum::<f64>().sqrt();
            if fnorm > 0.0 {
                for i in 0..vel.len() {
                    vel[i] = (1.0 - alpha) * vel[i] + alpha * vnorm * force[i] / fnorm;
                }
            }
            n_pos += 1;
            if n_pos > 5 {
                dt = (dt * 1.1).min(dt_max);
                alpha *= 0.99;
            }
        } else {
            vel.fill(0.0);
            alpha = 0.1;
            n_pos = 0;
            dt *= 0.5;
        }
        for i in 0..vel.len() {
            vel[i] += force[i] * dt;
            x[i] += vel[i] * dt;
        }
        let (e, g) = potential.value_and_gradient(x.view());
        energy = e;
        grad = g;
        evals += 1;
        if !energy.is_finite() {
            break;
        }
    }
    let g = ginf(grad.view());
    (energy, x, evals, g)
}

fn lbfgs(
    potential: &PairPotential,
    x0: ArrayView1<f64>,
    max_iter: usize,
) -> (f64, Array1<f64>, usize, f64) {
    let mut opt = WarmLbfgs::default();
    let (energy, x, evals) = opt.minimize(x0, max_iter, |v| Some(potential.value_and_gradient(v)));
    let g = ginf(potential.value_and_gradient(x.view()).1.view());
    (energy, x, evals, g)
}

fn main() {
    let steps: usize = std::env::args()
        .nth(1)
        .and_then(|value| value.parse().ok())
        .unwrap_or(200);
    let ico_fix = load_xyz(include_str!("../tests/fixtures/lj75_ico.xyz"));
    let marks_fix = load_xyz(include_str!("../tests/fixtures/lj75_marks.xyz"));
    let potential = PairPotential::new(75, PairKind::LennardJones, 40.0);
    let (ico_e, ico, _, ico_g) = lbfgs(&potential, ico_fix.view(), 600);
    let (marks_e, marks, _, marks_g) = lbfgs(&potential, marks_fix.view(), 600);
    let ico_slice = ico.as_slice().expect("contiguous").to_vec();
    let marks_slice = marks.as_slice().expect("contiguous").to_vec();
    println!(
        "{{\"kind\":\"floors\",\"ico\":{ico_e:.6},\"ico_g\":{ico_g:.3e},\"marks\":{marks_e:.6},\"marks_g\":{marks_g:.3e},\"steps\":{steps}}}"
    );

    let depth = ico_e.abs() / 75.0;
    let references = vec![ico_slice.clone()];
    let fivefold = anneal_core::soap::step_away_fivefold_measured(ico.view(), 0.35);
    let mut starts: Vec<(&str, Array1<f64>)> = vec![
        ("ico", ico.clone()),
        ("marks", marks.clone()),
        ("fivefold_035", fivefold),
    ];
    for cover in 0..3 {
        let start = known_basin::leave_packing_rung_to(
            ico.view(),
            cover,
            known_basin::rung_barrier(depth, 1),
            &references,
            None,
            None,
            |v| Some(potential.value_and_gradient(v).0),
        );
        starts.push((["cover0", "cover1", "cover2"][cover], start));
    }

    for (label, start) in &starts {
        for (method, quench) in [
            (
                "lbfgs",
                lbfgs
                    as fn(&PairPotential, ArrayView1<f64>, usize) -> (f64, Array1<f64>, usize, f64),
            ),
            ("fire2", fire2),
        ] {
            let (energy, landed, evals, g) = quench(&potential, start.view(), steps);
            let community = landed
                .as_slice()
                .map_or(2, |slice| community_of(&ico_slice, &marks_slice, slice));
            println!(
                "{{\"kind\":\"quench\",\"start\":\"{label}\",\"method\":\"{method}\",\"energy\":{energy:.6},\"evals\":{evals},\"ginf\":{g:.3e},\"community\":{community}}}"
            );
            let _ = std::io::Write::flush(&mut std::io::stdout());
        }
    }
}
