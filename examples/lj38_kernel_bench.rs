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
    let gradient_bits = gradient.iter().fold(0u64, |acc, value| {
        acc.wrapping_mul(0x9E37_79B1_85EB_CA87)
            .wrapping_add(value.to_bits())
    });
    println!("gradient_bits: {gradient_bits:x}");
    let mut pair_energy = 0.0;
    let n = x.len() / 3;
    for i in 0..n {
        for j in (i + 1)..n {
            let mut r2 = 0.0;
            for k in 0..3 {
                let d = x[3 * i + k] - x[3 * j + k];
                r2 += d * d;
            }
            pair_energy += potential.kind().pair(r2).0;
        }
    }
    println!("pair_energy: {pair_energy:.16}");
    println!("pair_energy_hex: {:x}", pair_energy.to_bits());
    let pair_calls = 2000;
    let pair_start = Instant::now();
    for _ in 0..pair_calls {
        let mut acc = 0.0;
        for i in 0..n {
            for j in (i + 1)..n {
                let mut r2 = 0.0;
                for k in 0..3 {
                    let d = x[3 * i + k] - x[3 * j + k];
                    r2 += d * d;
                }
                acc += potential.kind().pair(r2).0;
            }
        }
        std::hint::black_box(acc);
    }
    let pair_seconds = pair_start.elapsed().as_secs_f64();
    println!("pair_loop_calls: {pair_calls}");
    println!("pair_loop_seconds: {pair_seconds:.6}");
    println!(
        "pair_loop_calls_per_second: {:.3}",
        pair_calls as f64 / pair_seconds
    );
    println!("calls: {calls}");
    println!("seconds: {seconds:.6}");
    println!(
        "energy_gradient_calls_per_second: {:.3}",
        calls as f64 / seconds
    );
}
