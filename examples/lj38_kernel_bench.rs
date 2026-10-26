//! Times one energy-and-gradient evaluation of reduced LJ38.
//!
//! The geometry is a fixed Fibonacci shell. The same binary reports the
//! in-process kernel, so a before and after run is the comparison.

use std::fs;
use std::time::Instant;

use anneal_core::potentials::PairPotential;
use ndarray::Array1;

const N: usize = 38;

fn fibonacci(n: usize) -> Array1<f64> {
    let mut x = Array1::zeros(3 * n);
    let golden = std::f64::consts::PI * (3.0 - 5.0_f64.sqrt());
    for i in 0..n {
        let z = 1.0 - 2.0 * (i as f64 + 0.5) / n as f64;
        let radius = (1.0 - z * z).max(0.0).sqrt();
        let theta = golden * i as f64;
        let shell = 2.5;
        x[3 * i] = shell * radius * theta.cos();
        x[3 * i + 1] = shell * radius * theta.sin();
        x[3 * i + 2] = shell * z;
    }
    x
}

fn cpu() -> String {
    fs::read_to_string("/proc/cpuinfo")
        .unwrap_or_default()
        .lines()
        .find_map(|line| line.strip_prefix("model name\t: "))
        .unwrap_or("unknown")
        .to_string()
}

fn main() {
    let potential = PairPotential::lennard_jones(N);
    let x = fibonacci(N);
    let (energy, gradient) = potential.value_and_gradient(x.view());
    let gradient_sum: f64 = gradient.iter().sum();
    let warmup = 50;
    let calls = 2000;
    for _ in 0..warmup {
        let _ = potential.value_and_gradient(x.view());
    }
    let start = Instant::now();
    for _ in 0..calls {
        let _ = potential.value_and_gradient(x.view());
    }
    let seconds = start.elapsed().as_secs_f64();
    println!("cpu: {}", cpu());
    println!("atoms: {N}");
    println!("energy: {energy:.16}");
    println!("energy_hex: {:x}", energy.to_bits());
    println!("gradient_sum: {gradient_sum:.16}");
    println!("gradient_sum_hex: {:x}", gradient_sum.to_bits());
    println!("g0_hex: {:x}", gradient[0].to_bits());
    println!("calls: {calls}");
    println!("seconds: {seconds:.6}");
    println!(
        "energy_gradient_calls_per_second: {:.3}",
        calls as f64 / seconds
    );
}
