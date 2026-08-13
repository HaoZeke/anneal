//! Differential test of the ported components against Python reference output.
//!
//! A port is verified by producing the same output for the same input, not by
//! reaching a similar success rate downstream. An aggregate statistic cannot
//! localise a discrepancy: two implementations differing anywhere give two
//! different numbers, and nothing says where.
//!
//! The reference values come from running the Python components on a fixed
//! input sequence.

use anneal_core::allocate::BudgetWindowTemperature;
use anneal_core::bias::{AdaptiveHeight, BasinBias, Bias, Fingerprint};
use ndarray::{Array1, ArrayView1};

/// Passes a descriptor through, so basin identity is exactly the input.
struct IdentityKey;

impl Fingerprint for IdentityKey {
    fn describe(&self, x: ArrayView1<f64>) -> Array1<f64> {
        x.to_owned()
    }
}

fn close(a: f64, b: f64, tol: f64) -> bool {
    (a - b).abs() <= tol * a.abs().max(b.abs()).max(1.0)
}

fn main() {
    let mut failures = 0usize;

    // 1. Budget-window temperature.
    let rejections = [0.0, 2.5, 1.0, 0.0, 4.0, 0.3, 7.0, 0.0, 1.5, 2.0];
    let gaps = [3.0, 0.5, 0.01, 12.0, 0.2, 1.0, 0.001, 5.0, 0.75, 0.05];
    let remaining = [
        1000000usize,
        500000,
        100000,
        50000,
        10000,
        5000,
        1000,
        100,
        10,
        1,
    ];
    let expected = [
        0.013157895,
        0.003810288,
        0.005993250,
        0.052631579,
        0.016027165,
        0.017688690,
        0.041625859,
        0.062101654,
        0.122648918,
        0.263209643,
    ];
    let mut law = BudgetWindowTemperature::new(114, 0.5);
    println!("{:>4} {:>16} {:>16}  match", "step", "rust", "python");
    for k in 0..10 {
        law.observe_rejection(rejections[k]);
        let t = law.temperature(gaps[k], remaining[k]);
        let ok = close(t, expected[k], 1e-7);
        if !ok {
            failures += 1;
        }
        println!(
            "{k:>4} {t:>16.9} {:>16.9}  {}",
            expected[k],
            if ok { "yes" } else { "NO" }
        );
    }
    println!("escape_forced rust {} python 6", law.escape_forced);
    if law.escape_forced != 6 {
        failures += 1;
    }

    // 2. Adaptive deposit height on the same gap stream.
    let gaps: Vec<f64> = (0..400)
        .map(|i| {
            let u = ((i * 7919 + 13) % 1000) as f64 / 1000.0;
            -(1.0 - u).ln() * 2.5
        })
        .collect();
    let expected_h = [
        0.028071519,
        0.048693555,
        0.005280677,
        0.054643506,
        0.017210625,
        0.020515250,
        0.056899093,
        0.026367973,
    ];
    let mut h = AdaptiveHeight::new(0.1, 4.0, 0.25);
    let mut seen = Vec::new();
    for (i, g) in gaps.iter().enumerate() {
        h.observe(*g);
        if (i + 1) % 50 == 0 {
            seen.push(h.height());
        }
    }
    println!("\n{:>4} {:>16} {:>16}  match", "n", "rust", "python");
    for (k, e) in expected_h.iter().enumerate() {
        let ok = close(seen[k], *e, 1e-7);
        if !ok {
            failures += 1;
        }
        println!(
            "{:>4} {:>16.9} {e:>16.9}  {}",
            (k + 1) * 50,
            seen[k],
            if ok { "yes" } else { "NO" }
        );
    }
    // 3. Per-basin well-tempered accumulation on a fixed visit sequence.
    //
    // Each visit is given a descriptor that is a distinct constant, so the
    // basin identity is exactly the sequence and the comparison isolates the
    // deposit rule from the fingerprint and the merge radius.
    let visits: [usize; 30] = [
        0, 0, 0, 1, 1, 0, 2, 2, 2, 2, 0, 1, 3, 3, 0, 2, 4, 4, 4, 0, 1, 2, 3, 4, 0, 0, 1, 1, 2, 2,
    ];
    let expected_final = [
        1.822052920,
        1.303884480,
        1.656837268,
        0.706314509,
        0.915847728,
    ];
    let mut bias = BasinBias::new(IdentityKey, 1e-9, 0.25, 5.0);
    for v in &visits {
        let s = Array1::from(vec![*v as f64]);
        bias.deposit(s.view(), 1.0);
    }
    println!("\n{:>6} {:>16} {:>16}  match", "basin", "rust", "python");
    for (k, expected) in expected_final.iter().enumerate() {
        let s = Array1::from(vec![k as f64]);
        let got = bias.potential(s.view());
        let ok = close(got, *expected, 1e-7);
        if !ok {
            failures += 1;
        }
        println!(
            "{k:>6} {got:>16.9} {:>16.9}  {}",
            expected,
            if ok { "yes" } else { "NO" }
        );
    }

    println!("\nmismatches: {failures}");
    if failures > 0 {
        std::process::exit(1);
    }
}
