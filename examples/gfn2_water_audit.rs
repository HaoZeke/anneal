//! Numerical audit of the GFN2-xTB/rgpot objective used by water searches.
//!
//! Usage:
//!   gfn2_water_audit gradient <input.xyz>
//!   gfn2_water_audit fire <input.xyz> <output.xyz>

#[path = "common/rgpot_eindir.rs"]
mod rgpot_eindir;

use std::fs;
use std::sync::Mutex;

use eindir_core::gradient::DifferentiableObjective;
use ndarray::{Array1, ArrayView1};
use rgmin::{Control, FireKind, Method, Oracle, Solver, Status};
use rgpot_eindir::RgpotObjective;

const FD_STEP_A: f64 = 1.0e-4;
const FIRE_GTOL_EV_PER_A: f64 = 0.0001;

fn read_xyz(path: &str) -> (Vec<String>, Array1<f64>) {
    let text = fs::read_to_string(path).unwrap_or_else(|error| panic!("read {path}: {error}"));
    let mut lines = text.lines();
    let count: usize = lines
        .next()
        .expect("xyz atom count")
        .trim()
        .parse()
        .expect("xyz atom count is an integer");
    let _comment = lines.next();
    let mut symbols = Vec::with_capacity(count);
    let mut coordinates = Vec::with_capacity(3 * count);
    for line in lines.take(count) {
        let fields: Vec<_> = line.split_whitespace().collect();
        assert!(
            fields.len() >= 4,
            "xyz atom line has fewer than four fields"
        );
        symbols.push(fields[0].to_string());
        coordinates.extend(
            fields[1..4]
                .iter()
                .map(|value| value.parse::<f64>().expect("xyz coordinate")),
        );
    }
    assert_eq!(symbols.len(), count, "xyz ended before all atoms were read");
    (symbols, Array1::from(coordinates))
}

fn atomic_numbers(symbols: &[String]) -> Vec<i32> {
    symbols
        .iter()
        .map(|symbol| match symbol.as_str() {
            "H" => 1,
            "O" => 8,
            other => panic!("unsupported atom {other}"),
        })
        .collect()
}

fn objective(
    symbols: &[String],
) -> Oracle<impl Fn(ArrayView1<f64>) -> (f64, Array1<f64>) + Send + Sync + use<>> {
    let pot = Mutex::new(RgpotObjective::xtb_fresh(
        &atomic_numbers(symbols),
        [0.0; 9],
    ));
    let dim = 3 * symbols.len();
    Oracle::unbounded(dim, move |coordinates| {
        let coordinates = coordinates
            .as_slice()
            .expect("audit coordinates are contiguous");
        let evaluation = pot
            .lock()
            .expect("xtb objective lock")
            .fresh_evaluation(coordinates)
            .expect("xtb evaluation");
        (
            evaluation.energy,
            Array1::from_iter(evaluation.forces.into_iter().map(|force| -force)),
        )
    })
}

fn gradient_audit(symbols: &[String], coordinates: &Array1<f64>) {
    let objective = objective(symbols);
    let (energy, analytic) = objective.value_and_gradient(coordinates.view());
    let components = [
        0,
        1,
        2,
        coordinates.len() / 4,
        coordinates.len() / 2,
        3 * coordinates.len() / 4,
        coordinates.len() - 3,
        coordinates.len() - 2,
        coordinates.len() - 1,
    ];
    let mut max_error = 0.0_f64;
    println!("energy_eV={energy:.12}");
    println!("component,analytic_eV_per_A,finite_difference_eV_per_A,absolute_error");
    for component in components {
        let mut plus = coordinates.clone();
        let mut minus = coordinates.clone();
        plus[component] += FD_STEP_A;
        minus[component] -= FD_STEP_A;
        let plus_energy = objective.value_and_gradient(plus.view()).0;
        let minus_energy = objective.value_and_gradient(minus.view()).0;
        let finite_difference = (plus_energy - minus_energy) / (2.0 * FD_STEP_A);
        let error = (analytic[component] - finite_difference).abs();
        max_error = max_error.max(error);
        println!(
            "{component},{:.12},{finite_difference:.12},{error:.6e}",
            analytic[component]
        );
    }
    println!("max_absolute_error_eV_per_A={max_error:.6e}");
    assert!(
        max_error < 1.0e-3,
        "rgpot objective gradient disagrees with central finite differences"
    );
}

fn fire_quench(symbols: &[String], mut coordinates: Array1<f64>, output: &str) {
    let objective = objective(symbols);
    println!("gtol_eV_per_A={FIRE_GTOL_EV_PER_A:.6e}");
    let control = Control {
        maxiter: 10_000,
        gtol: FIRE_GTOL_EV_PER_A,
        istep: 0.1,
        maxmove: Some(0.1),
        ftol_rel: None,
        noise_floor: None,
    };
    let mut solver = Solver::new(
        Method::Fire { kind: FireKind::V2 },
        control,
        coordinates.len(),
    );
    let report = loop {
        let report = solver
            .step(&objective, &mut coordinates)
            .expect("rgmin FIRE step");
        if report.status != Status::Iterating {
            break report;
        }
    };
    println!(
        "status={:?} steps={} energy_eV={:.12} gradient_norm_eV_per_A={:.6e}",
        report.status, report.steps, report.value, report.grad_norm
    );
    assert_eq!(report.status, Status::Tolerance, "FIRE did not converge");
    let mut xyz = format!("{}\nrgmin FIRE: {:.12} eV\n", symbols.len(), report.value);
    for (atom, position) in symbols
        .iter()
        .zip(report.coords.as_slice().unwrap().chunks(3))
    {
        xyz.push_str(&format!(
            "{atom} {:.12} {:.12} {:.12}\n",
            position[0], position[1], position[2]
        ));
    }
    fs::write(output, xyz).unwrap_or_else(|error| panic!("write {output}: {error}"));
}

fn main() {
    anneal_core::install_rgpot_kernels!();
    let arguments: Vec<String> = std::env::args().collect();
    let input = arguments.get(2).expect("input xyz");
    let (symbols, coordinates) = read_xyz(input);
    match arguments.get(1).map(String::as_str) {
        Some("gradient") => gradient_audit(&symbols, &coordinates),
        Some("fire") => {
            let output = arguments.get(3).expect("output xyz");
            fire_quench(&symbols, coordinates, output);
        }
        _ => {
            panic!("usage: gfn2_water_audit {{gradient <input.xyz>|fire <input.xyz> <output.xyz>}}")
        }
    }
}
