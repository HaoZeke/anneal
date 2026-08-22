//! Measure the DECAF packing grain: within-funnel spread versus ico-Marks.
//!
//!     decaf_packing_grain [ISOMERS]
//!
//! [`anneal_core::catalog::PACKING_MERGE`] is the cell grain, and the live
//! LJ75 book puts tens of icosahedral cells above it. A packing grain has to
//! sit above the largest within-funnel L1 of quenched isomers and below the
//! ico-Marks L1. Two metrics are reported per pair:
//!
//! * `throwaway`, what [`anneal_core::catalog::different_decaf_family`] and
//!   the Leave accept path use: grow the codebook on the origin alone, then
//!   histogram the trial against it. Environments the origin never saw fold
//!   into one shared bin, so the distance is asymmetric and counts novelty of
//!   environment rather than packing.
//! * `shared`, the live [`anneal_core::catalog::PackingBook`] form: grow the
//!   codebook on both structures, then compare the two histograms. Symmetric,
//!   and no shared unseen bin.
//!
//! Isomers are bucketed by quenched energy above the icosahedral floor, so the
//! spread of the shelf the live book actually holds is separated from the
//! amorphous minima a large perturbation reaches.

use anneal_core::catalog::{PACKING_MERGE, PackingBook, packing_distance};
use anneal_core::methods::warm_lbfgs::WarmLbfgs;
use anneal_core::potentials::{PairKind, PairPotential};
use ndarray::{Array1, ArrayView1};
use rand::SeedableRng;
use rand::rngs::StdRng;
use rand_distr::{Distribution, Normal};

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

/// L1 against a book grown on `origin` alone: the Leave accept path.
fn throwaway_l1(origin: &[f64], trial: &[f64]) -> Option<f64> {
    let mut book = PackingBook::default();
    book.observe(origin)?;
    let home = book.histogram(origin)?;
    let away = book.histogram(trial)?;
    Some(packing_distance(&home, &away))
}

/// L1 against a codebook grown on both structures: the live book form.
fn shared_l1(left: &[f64], right: &[f64]) -> Option<f64> {
    let mut book = PackingBook::default();
    book.observe(left)?;
    book.observe(right)?;
    let a = book.histogram(left)?;
    let b = book.histogram(right)?;
    Some(packing_distance(&a, &b))
}

fn quench(potential: &PairPotential, x: ArrayView1<f64>, steps: usize) -> (f64, Array1<f64>) {
    let mut opt = WarmLbfgs::default();
    let (energy, relaxed, _) = opt.minimize(x, steps, |v| Some(potential.value_and_gradient(v)));
    (energy, relaxed)
}

fn perturbed(x: ArrayView1<f64>, sigma: f64, rng: &mut StdRng) -> Array1<f64> {
    let normal = Normal::new(0.0, sigma).expect("sigma is positive");
    let mut y = x.to_owned();
    for value in y.iter_mut() {
        *value += normal.sample(rng);
    }
    y
}

#[derive(Clone, Copy)]
struct Sample {
    above: f64,
    ico_throwaway: f64,
    ico_shared: f64,
    marks_shared: f64,
}

fn report(label: &str, rows: &[Sample]) {
    if rows.is_empty() {
        println!("{{\"kind\":\"decaf_grain_bucket\",\"bucket\":\"{label}\",\"n\":0}}");
        return;
    }
    let stat = |mut values: Vec<f64>| {
        values.sort_by(|a, b| a.partial_cmp(b).expect("L1 is finite"));
        let median = values[values.len() / 2];
        let max = *values.last().expect("non-empty");
        let min = values[0];
        (min, median, max)
    };
    let (_, ico_t_med, ico_t_max) = stat(rows.iter().map(|r| r.ico_throwaway).collect());
    let (_, ico_s_med, ico_s_max) = stat(rows.iter().map(|r| r.ico_shared).collect());
    let (marks_s_min, marks_s_med, _) = stat(rows.iter().map(|r| r.marks_shared).collect());
    // A usable grain needs every shelf isomer nearer ico than Marks is.
    let nearer_marks = rows.iter().filter(|r| r.marks_shared < r.ico_shared).count();
    println!(
        "{{\"kind\":\"decaf_grain_bucket\",\"bucket\":\"{label}\",\"n\":{},\"ico_throwaway_median\":{ico_t_med:.4},\"ico_throwaway_max\":{ico_t_max:.4},\"ico_shared_median\":{ico_s_med:.4},\"ico_shared_max\":{ico_s_max:.4},\"marks_shared_min\":{marks_s_min:.4},\"marks_shared_median\":{marks_s_med:.4},\"nearer_marks\":{nearer_marks}}}",
        rows.len()
    );
}

fn main() {
    let isomers: usize = std::env::args()
        .nth(1)
        .and_then(|value| value.parse().ok())
        .unwrap_or(160);
    let lj75_ico = load_xyz(include_str!("../tests/fixtures/lj75_ico.xyz"));
    let lj75_marks = load_xyz(include_str!("../tests/fixtures/lj75_marks.xyz"));
    let lj38_ico = load_xyz(include_str!("../tests/fixtures/lj38_ico.xyz"));
    let lj38_fcc = load_xyz(include_str!("../tests/fixtures/lj38_fcc.xyz"));

    let ico = lj75_ico.as_slice().expect("fixture is contiguous");
    let marks = lj75_marks.as_slice().expect("fixture is contiguous");
    let lj38_a = lj38_ico.as_slice().expect("fixture is contiguous");
    let lj38_b = lj38_fcc.as_slice().expect("fixture is contiguous");
    println!(
        "{{\"kind\":\"decaf_grain_reference\",\"merge\":{PACKING_MERGE},\"lj75_ico_marks_throwaway\":{:.4},\"lj75_marks_ico_throwaway\":{:.4},\"lj75_ico_marks_shared\":{:.4},\"lj38_ico_fcc_throwaway\":{:.4},\"lj38_ico_fcc_shared\":{:.4}}}",
        throwaway_l1(ico, marks).expect("LJ75 histograms exist"),
        throwaway_l1(marks, ico).expect("LJ75 histograms exist"),
        shared_l1(ico, marks).expect("LJ75 histograms exist"),
        throwaway_l1(lj38_a, lj38_b).expect("LJ38 histograms exist"),
        shared_l1(lj38_a, lj38_b).expect("LJ38 histograms exist"),
    );

    let potential = PairPotential::new(75, PairKind::LennardJones, 40.0);
    let (ico_energy, _) = quench(&potential, lj75_ico.view(), 400);
    let (marks_energy, _) = quench(&potential, lj75_marks.view(), 400);
    println!(
        "{{\"kind\":\"decaf_grain_floors\",\"ico\":{ico_energy:.6},\"marks\":{marks_energy:.6}}}"
    );
    let mut rng = StdRng::seed_from_u64(20260821);
    let mut rows: Vec<Sample> = Vec::new();
    let mut below_ico = 0usize;
    for index in 0..isomers {
        let sigma = 0.10 + 0.05 * ((index % 12) as f64);
        let start = perturbed(lj75_ico.view(), sigma, &mut rng);
        let (energy, relaxed) = quench(&potential, start.view(), 900);
        if !energy.is_finite() || energy > -300.0 {
            continue;
        }
        let trial = relaxed.as_slice().expect("quench is contiguous");
        let (Some(ico_throwaway), Some(ico_shared), Some(marks_shared)) = (
            throwaway_l1(ico, trial),
            shared_l1(ico, trial),
            shared_l1(marks, trial),
        ) else {
            continue;
        };
        if energy < ico_energy - 1e-6 {
            below_ico += 1;
            println!(
                "{{\"kind\":\"decaf_grain_below_ico\",\"sigma\":{sigma:.2},\"energy\":{energy:.6},\"ico_shared\":{ico_shared:.4},\"marks_shared\":{marks_shared:.4}}}"
            );
        }
        rows.push(Sample {
            above: energy - ico_energy,
            ico_throwaway,
            ico_shared,
            marks_shared,
        });
    }
    println!(
        "{{\"kind\":\"decaf_grain_sample\",\"n\":{},\"below_ico\":{below_ico}}}",
        rows.len()
    );
    for (label, hi) in [("0-1", 1.0), ("1-2", 2.0), ("2-4", 4.0), ("4-8", 8.0)] {
        let lo = match label {
            "0-1" => f64::NEG_INFINITY,
            "1-2" => 1.0,
            "2-4" => 2.0,
            _ => 4.0,
        };
        let bucket: Vec<Sample> = rows
            .iter()
            .copied()
            .filter(|r| r.above > lo && r.above <= hi)
            .collect();
        report(label, &bucket);
    }
    report("all", &rows);
}
