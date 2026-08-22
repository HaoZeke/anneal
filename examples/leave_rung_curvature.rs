//! The Leave rung from measured curvature, not from a constant.
//!
//!     leave_rung_curvature
//!
//! At a minimum the gradient vanishes, so a displacement `d` raises the
//! energy by `dᵀHd/2` to second order. Along the softest non-rigid mode with
//! curvature `λ` a root-mean-square step `δ` over `N` atoms has squared
//! length `Nδ²`, so it reaches `λNδ²/2`. Setting that equal to a barrier `Δ`
//! gives the step that can cross it,
//!
//!     δ = sqrt(2Δ / (λN)),
//!
//! which is `Hop.rung_reaches_barrier` in `proofs/lean/Hop/LeavePacking.lean`.
//! Nothing in it is chosen: `λ` is measured by Lanczos on the structure the
//! Leave starts from and `Δ` is the barrier the run is trying to clear.
//!
//! Reported here for both sealed LJ75 minima, against the Wales and Doye
//! ico-Marks barriers of 8.69 and 7.48 epsilon, together with the energy the
//! harmonic estimate actually delivers when the step is taken, so the size of
//! the anharmonic error is on the record rather than assumed away.
//!
//! Wales, D. J.; Doye, J. P. K. *J. Phys. Chem. A* **1997**, *101*, 5111.
//! <https://doi.org/10.1021/jp970984n>

use anneal_core::curvature::curvature_features;
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

fn report(label: &str, potential: &PairPotential, start: ArrayView1<f64>) {
    let mut opt = WarmLbfgs::default();
    let (energy, x, _) = opt.minimize(start, 2000, |v: ArrayView1<f64>| {
        Some(potential.value_and_gradient(v))
    });
    let atoms = x.len() / 3;
    let Some(features) = curvature_features(
        x.view(),
        |v: ArrayView1<f64>| Some(potential.value_and_gradient(v).1),
        64,
        1e-4,
    ) else {
        println!("{{\"kind\":\"rung\",\"label\":\"{label}\",\"curvature\":false}}");
        return;
    };
    println!(
        "{{\"kind\":\"rung_curvature\",\"label\":\"{label}\",\"energy\":{energy:.6},\"lambda_min\":{:.6},\"lambda_second\":{:.6},\"gap\":{:.6},\"participation\":{:.4},\"evaluations\":{}}}",
        features.lambda_min,
        features.lambda_second,
        features.gap,
        features.participation,
        features.evaluations
    );
    let n = atoms as f64;
    for barrier in [1.0_f64, 7.48, 8.69] {
        if features.lambda_min <= 0.0 {
            continue;
        }
        // δ = sqrt(2Δ / (λN)) : Hop.rung_reaches_barrier.
        let delta = (2.0 * barrier / (features.lambda_min * n)).sqrt();
        // What the step actually costs on the potential, along the mode the
        // curvature came from. The harmonic estimate is the barrier itself.
        let scale = n.sqrt() * delta;
        let mut y = x.clone();
        y.scaled_add(scale, &features.mode);
        let measured = potential.value_and_gradient(y.view()).0 - energy;
        println!(
            "{{\"kind\":\"rung\",\"label\":\"{label}\",\"barrier\":{barrier},\"delta\":{delta:.4},\"harmonic\":{barrier:.4},\"measured\":{measured:.4}}}"
        );
    }
}

fn main() {
    let potential = PairPotential::new(75, PairKind::LennardJones, 40.0);
    report(
        "lj75_ico",
        &potential,
        load_xyz(include_str!("../tests/fixtures/lj75_ico.xyz")).view(),
    );
    report(
        "lj75_marks",
        &potential,
        load_xyz(include_str!("../tests/fixtures/lj75_marks.xyz")).view(),
    );
    let lj38 = PairPotential::new(38, PairKind::LennardJones, 40.0);
    report(
        "lj38_ico",
        &lj38,
        load_xyz(include_str!("../tests/fixtures/lj38_ico.xyz")).view(),
    );
    report(
        "lj38_oh",
        &lj38,
        load_xyz(include_str!("../tests/fixtures/lj38_fcc.xyz")).view(),
    );
}
