// Copyright (c) anneal developers
//! Symmetry detection microbench for the nm3-scan-safe4 gate.
//!
//! Exercises `anneal_core::symmetrise::detect_all` and `detect_orders` on the
//! exact five geometries (`random33`, `random55`, `random38`, `ring55`,
//! `packed80`) without warm-up iterations and emits per-case timings together
//! with a bit-exact digest of all returned `Candidate` values.

use anneal_core::symmetrise::{Candidate, detect_all, detect_orders};
use ndarray::Array1;
use std::hint::black_box;
use std::time::Instant;

fn bench_cloud(n: usize, seed: u64) -> Array1<f64> {
    let mut state = seed;
    let mut next = || {
        state ^= state << 13;
        state ^= state >> 7;
        state ^= state << 17;
        (state >> 11) as f64 / ((1u64 << 53) as f64) - 0.5
    };
    let radius = 1.15 * (n as f64).cbrt();
    Array1::from(
        (0..3 * n)
            .map(|_| 2.0 * radius * next())
            .collect::<Vec<_>>(),
    )
}

fn bench_ring(n: usize, radius: f64) -> Array1<f64> {
    let mut coords = vec![0.0; 3 * n];
    for atom in 0..n {
        let angle = 2.0 * std::f64::consts::PI * (atom as f64) / (n as f64);
        coords[3 * atom] = radius * angle.cos();
        coords[3 * atom + 1] = radius * angle.sin();
    }
    Array1::from(coords)
}

fn bench_packed80() -> Array1<f64> {
    let mut coords = Vec::new();
    for i in -4..=4_i32 {
        for j in -4..=4_i32 {
            for k in -4..=4_i32 {
                let r2 = i * i + j * j + k * k;
                if r2 > 0 && r2 <= 6 {
                    coords.push(i as f64);
                    coords.push(j as f64);
                    coords.push(k as f64);
                }
            }
        }
    }
    assert_eq!(coords.len(), 3 * 80, "packed80 must have 80 atoms");
    Array1::from(coords)
}

fn fixture(geom: &str) -> (Array1<f64>, usize, Vec<usize>) {
    match geom {
        "random33" => (bench_cloud(33, 1), 33, (2..=12).collect()),
        "random55" => (bench_cloud(55, 2), 55, (2..=12).collect()),
        "random38" => (bench_cloud(38, 7), 38, (2..=38).collect()),
        "ring55" => (bench_ring(55, 2.2), 55, (2..=12).collect()),
        "packed80" => (bench_packed80(), 80, (2..=12).collect()),
        other => panic!("unknown geometry {other}"),
    }
}

fn fnv1a_candidates(cands: &[Candidate]) -> u64 {
    let mut h: u64 = 0xcbf2_9ce4_8422_2325;
    let mut mix = |bytes: &[u8]| {
        for &b in bytes {
            h ^= u64::from(b);
            h = h.wrapping_mul(0x0100_0000_01b3);
        }
    };
    mix(&(cands.len() as u64).to_le_bytes());
    for c in cands {
        mix(&(c.order as u64).to_le_bytes());
        mix(&[u8::from(c.improper)]);
        for a in c.axis {
            mix(&a.to_bits().to_le_bytes());
        }
        mix(&c.deviation.to_bits().to_le_bytes());
    }
    h
}

fn main() {
    let args: Vec<String> = std::env::args().skip(1).collect();
    assert!(
        args.len() == 2 || args.len() == 3,
        "usage: microbench <case> <iters> [--dump]"
    );
    let spec = args[0].as_str();
    let iters: usize = args[1].parse().expect("iters");
    let dump = args.get(2).map(String::as_str) == Some("--dump");
    let (op, geom) = spec.split_once(':').expect("case must be <op>:<geom>");
    let (coords, n, orders) = fixture(geom);
    let tolerance = 0.1_f64;

    if dump {
        let cands = match op {
            "detect_all" => detect_all(coords.view(), n, &orders, tolerance),
            "detect_orders" => detect_orders(coords.view(), n, &orders, tolerance),
            other => panic!("unknown op {other}"),
        };
        println!("count={}", cands.len());
        println!("fnv1a={:016x}", fnv1a_candidates(&cands));
        for (idx, c) in cands.iter().enumerate() {
            println!(
                "{idx},{},{},{:016x},{:016x},{:016x},{:016x}",
                c.order,
                u8::from(c.improper),
                c.axis[0].to_bits(),
                c.axis[1].to_bits(),
                c.axis[2].to_bits(),
                c.deviation.to_bits(),
            );
        }
        return;
    }

    assert!(iters > 0, "iters must be positive");
    let mut last_hash = 0_u64;
    let mut last_count = 0_usize;
    let start = Instant::now();
    match op {
        "detect_all" => {
            for _ in 0..iters {
                let cands = detect_all(
                    black_box(coords.view()),
                    black_box(n),
                    black_box(&orders),
                    black_box(tolerance),
                );
                last_count = cands.len();
                last_hash = fnv1a_candidates(black_box(&cands));
                black_box(last_hash);
            }
        }
        "detect_orders" => {
            for _ in 0..iters {
                let cands = detect_orders(
                    black_box(coords.view()),
                    black_box(n),
                    black_box(&orders),
                    black_box(tolerance),
                );
                last_count = cands.len();
                last_hash = fnv1a_candidates(black_box(&cands));
                black_box(last_hash);
            }
        }
        other => panic!("unknown op {other}"),
    }
    let total_s = start.elapsed().as_secs_f64();
    let per_call_s = total_s / (iters as f64);
    println!("{spec},{iters},{total_s:.9},{per_call_s:.12e},{last_hash:016x},{last_count}");
}
